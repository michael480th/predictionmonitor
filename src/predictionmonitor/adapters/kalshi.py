"""Kalshi adapter (Trade API v2).

Docs: https://trading-api.readme.io  (market/event reads are public)

The `/markets` endpoint uses cursor pagination and returns binary (Yes/No)
markets with prices quoted in cents [1, 99]. We normalize those to implied
probabilities in [0, 1].

Auth is optional for reads. If `KALSHI_API_KEY_ID` and a private key are set in
the environment we sign requests (RSA-PSS over `timestamp + METHOD + path`), per
Kalshi's scheme; otherwise we send unauthenticated public reads.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from datetime import datetime
from typing import Any, Iterator, Optional

from predictionmonitor.adapters.base import Adapter
from predictionmonitor.http import get_json
from predictionmonitor.schema import (
    Market,
    Outcome,
    PricePoint,
    Trade,
    iso_from_unix,
    parse_float,
)

log = logging.getLogger(__name__)

_MAX_PAGE = 1000

# Kalshi candlestick period_interval is constrained to these minute steps.
_ALLOWED_PERIODS = (1, 60, 1440)


def _cents_to_prob(cents: Any) -> Optional[float]:
    val = parse_float(cents)
    if val is None:
        return None
    return round(val / 100.0, 4)


def _dollars_to_prob(value: Any) -> Optional[float]:
    val = parse_float(value)
    if val is None:
        return None
    return round(val, 4)


def _epoch_seconds(raw: dict[str, Any]) -> Optional[float]:
    """Best-effort trade/candle timestamp in unix seconds."""
    ts = parse_float(raw.get("created_ts"))
    if ts is not None:
        return ts
    iso = raw.get("created_time")
    if iso:
        try:
            return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


class KalshiAdapter(Adapter):
    name = "kalshi"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._key_id = os.environ.get("KALSHI_API_KEY_ID") or None
        self._private_key = self._load_private_key()
        if self._key_id and self._private_key:
            log.info("Kalshi adapter using signed (authenticated) requests")

    @staticmethod
    def _load_private_key():
        """Load an RSA private key from env (inline PEM or file path)."""
        pem = os.environ.get("KALSHI_API_PRIVATE_KEY")
        path = os.environ.get("KALSHI_API_PRIVATE_KEY_PATH")
        if not pem and path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                pem = fh.read()
        if not pem:
            return None
        try:
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )

            return load_pem_private_key(pem.encode("utf-8"), password=None)
        except Exception as exc:  # cryptography missing or bad key
            log.warning("Could not load Kalshi private key (%s); using public reads", exc)
            return None

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build Kalshi signed-request headers, or {} if creds are absent."""
        if not (self._key_id and self._private_key):
            return {}
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            ts = str(int(time.time() * 1000))
            message = f"{ts}{method}{path}".encode("utf-8")
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return {
                "KALSHI-ACCESS-KEY": self._key_id,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
                "KALSHI-ACCESS-TIMESTAMP": ts,
            }
        except Exception as exc:
            log.warning("Failed to sign Kalshi request (%s); sending unsigned", exc)
            return {}

    def iter_markets(self) -> Iterator[Market]:
        limit = min(self.page_size, _MAX_PAGE)
        path = "/markets"  # path used for request signing
        cursor: Optional[str] = None
        for _page in range(self.max_pages):
            params: dict[str, Any] = {"limit": limit}
            if self.only_open:
                params["status"] = "open"
            if cursor:
                params["cursor"] = cursor
            data = get_json(
                self.session,
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout,
                headers=self._auth_headers("GET", f"/trade-api/v2{path}"),
            )
            rows = data.get("markets") or []
            if not rows:
                return
            for raw in rows:
                market = self._to_market(raw)
                if market is not None:
                    yield market
            cursor = data.get("cursor") or None
            if not cursor:
                return

    def _build_outcomes(self, raw: dict[str, Any]) -> list[Outcome]:
        # Prefer last traded price; fall back to bid/ask midpoint.
        yes_prob = _cents_to_prob(raw.get("last_price"))
        if yes_prob is None:
            bid = parse_float(raw.get("yes_bid"))
            ask = parse_float(raw.get("yes_ask"))
            if bid is not None and ask is not None:
                yes_prob = round((bid + ask) / 200.0, 4)
        no_prob = round(1.0 - yes_prob, 4) if yes_prob is not None else None
        return [Outcome(name="Yes", price=yes_prob), Outcome(name="No", price=no_prob)]

    def _to_market(self, raw: dict[str, Any]) -> Optional[Market]:
        ticker = raw.get("ticker")
        if not ticker:
            return None
        title = raw.get("title") or ticker
        subtitle = raw.get("subtitle")
        full_title = f"{title} — {subtitle}" if subtitle else title

        event_ticker = raw.get("event_ticker")
        status_raw = (raw.get("status") or "").lower()
        status = "open" if status_raw in {"open", "active"} else (
            "closed" if status_raw else "unknown"
        )

        page = event_ticker or ticker
        url = f"https://kalshi.com/markets/{page}"

        # Candlesticks (Phase 3) are addressed by series_ticker. Kalshi market
        # records don't always include it, but it is the ticker's first segment.
        series_ticker = raw.get("series_ticker") or (
            ticker.split("-")[0] if "-" in ticker else None
        )
        platform_meta: dict[str, Any] = {}
        if series_ticker:
            platform_meta["series_ticker"] = str(series_ticker)
        if event_ticker:
            platform_meta["event_ticker"] = str(event_ticker)

        return Market(
            platform=self.name,
            market_id=str(ticker),
            title=str(full_title),
            url=url,
            status=status,
            outcomes=self._build_outcomes(raw),
            event_id=str(event_ticker) if event_ticker else None,
            event_title=raw.get("title"),
            description=subtitle,
            category=raw.get("category"),
            tags=[],
            volume=parse_float(raw.get("volume")),
            volume_unit="contracts",
            liquidity=parse_float(raw.get("liquidity")),
            open_interest=parse_float(raw.get("open_interest")),
            open_time=raw.get("open_time"),
            close_time=raw.get("close_time"),
            platform_meta=platform_meta,
        )

    # ------------------------------------------------------------------
    # Phase 3 activity collection
    # ------------------------------------------------------------------
    @staticmethod
    def _period_interval(fidelity_minutes: int) -> int:
        """Snap a requested resolution to Kalshi's allowed minute steps."""
        for step in _ALLOWED_PERIODS:
            if fidelity_minutes <= step:
                return step
        return _ALLOWED_PERIODS[-1]

    @staticmethod
    def _candle_price(candle: dict[str, Any]) -> Optional[float]:
        """Closing implied probability from a candlestick, cents or dollars."""
        price = candle.get("price")
        if not isinstance(price, dict):
            return None
        for key in ("close", "mean"):
            if price.get(f"{key}_dollars") is not None:
                return _dollars_to_prob(price[f"{key}_dollars"])
            if price.get(key) is not None:
                return _cents_to_prob(price[key])
        return None

    @staticmethod
    def _trade_price(raw: dict[str, Any]) -> Optional[float]:
        """Yes-side implied probability from a trade, cents or dollars."""
        if raw.get("yes_price_dollars") is not None:
            return _dollars_to_prob(raw["yes_price_dollars"])
        return _cents_to_prob(raw.get("yes_price"))

    def fetch_price_history(
        self, market: Market, *, window_days: int = 14, fidelity_minutes: int = 60
    ) -> list[PricePoint]:
        """Candlestick price + volume series via the Kalshi market history API."""
        series = market.platform_meta.get("series_ticker")
        if not series:
            raise ValueError("market has no series_ticker; cannot fetch candlesticks")

        ticker = market.market_id
        period = self._period_interval(fidelity_minutes)
        end_ts = int(time.time())
        start_ts = end_ts - window_days * 86400
        path = f"/series/{series}/markets/{ticker}/candlesticks"
        data = get_json(
            self.session,
            f"{self.base_url}{path}",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period},
            timeout=self.timeout,
            headers=self._auth_headers("GET", f"/trade-api/v2{path}"),
        )
        points: list[PricePoint] = []
        for candle in data.get("candlesticks") or []:
            ts = iso_from_unix(candle.get("end_period_ts"))
            if ts is None:
                continue
            points.append(
                PricePoint(
                    t=ts,
                    price=self._candle_price(candle),
                    volume=parse_float(candle.get("volume")),
                )
            )
        return points

    def fetch_trades(
        self, market: Market, *, window_days: int = 14, max_trades: int = 2000
    ) -> list[Trade]:
        """Recent public trades for the market (anonymous: no wallet)."""
        ticker = market.market_id
        cutoff = time.time() - window_days * 86400
        path = "/markets/trades"
        out: list[Trade] = []
        cursor: Optional[str] = None
        for _ in range(self.max_pages):
            params: dict[str, Any] = {"ticker": ticker, "limit": min(self.page_size, _MAX_PAGE)}
            if cursor:
                params["cursor"] = cursor
            data = get_json(
                self.session,
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout,
                headers=self._auth_headers("GET", f"/trade-api/v2{path}"),
            )
            rows = data.get("trades") or []
            if not rows:
                break
            stop = False
            for r in rows:
                ts = _epoch_seconds(r)
                if ts is not None and ts < cutoff:
                    stop = True
                    break
                out.append(
                    Trade(
                        t=iso_from_unix(ts) or "",
                        price=self._trade_price(r),
                        size=parse_float(r.get("count_fp") or r.get("count")),
                        side=(r.get("taker_side") or None),
                        wallet=None,  # Kalshi is anonymous
                    )
                )
                if len(out) >= max_trades:
                    return out
            cursor = data.get("cursor") or None
            if stop or not cursor:
                break
        return out
