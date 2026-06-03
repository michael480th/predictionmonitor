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
from typing import Any, Iterator, Optional

from predictionmonitor.adapters.base import Adapter
from predictionmonitor.http import get_json
from predictionmonitor.schema import Market, Outcome, parse_float

log = logging.getLogger(__name__)

_MAX_PAGE = 1000


def _cents_to_prob(cents: Any) -> Optional[float]:
    val = parse_float(cents)
    if val is None:
        return None
    return round(val / 100.0, 4)


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
        )
