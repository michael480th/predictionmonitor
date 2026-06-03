"""Polymarket adapter (Gamma API).

Docs: https://docs.polymarket.com  (Gamma markets endpoint is public, no auth)

The Gamma `/markets` endpoint paginates with `limit`/`offset`. Outcomes and
their prices arrive as JSON-encoded strings that we parse and zip together.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator, Optional

from predictionmonitor.adapters.base import Adapter
from predictionmonitor.http import get_json
from predictionmonitor.schema import (
    Market,
    Outcome,
    PricePoint,
    Trade,
    cluster_key,
    iso_from_unix,
    parse_float,
)

log = logging.getLogger(__name__)

# Gamma caps page size around 500.
_MAX_PAGE = 500

# Activity lives on different hosts than the Gamma catalog API.
_DEFAULT_CLOB_URL = "https://clob.polymarket.com"
_DEFAULT_DATA_URL = "https://data-api.polymarket.com"


class PolymarketAdapter(Adapter):
    name = "polymarket"

    def iter_markets(self) -> Iterator[Market]:
        limit = min(self.page_size, _MAX_PAGE)
        offset = 0
        for _page in range(self.max_pages):
            params: dict[str, Any] = {"limit": limit, "offset": offset}
            if self.only_open:
                # Open == not yet closed and currently active/tradable.
                params["closed"] = "false"
                params["active"] = "true"
            data = get_json(
                self.session,
                f"{self.base_url}/markets",
                params=params,
                timeout=self.timeout,
            )
            # Gamma returns either a bare list or {"data": [...]}.
            rows = data.get("data") if isinstance(data, dict) else data
            if not rows:
                return
            for raw in rows:
                market = self._to_market(raw)
                if market is not None:
                    yield market
            if len(rows) < limit:
                return
            offset += limit

    @staticmethod
    def _parse_json_list(value: Any) -> list:
        """Outcomes/prices come as a JSON string or already-decoded list."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    def _build_outcomes(self, raw: dict[str, Any]) -> list[Outcome]:
        names = self._parse_json_list(raw.get("outcomes"))
        prices = self._parse_json_list(raw.get("outcomePrices"))
        outcomes: list[Outcome] = []
        for i, name in enumerate(names):
            price = parse_float(prices[i]) if i < len(prices) else None
            outcomes.append(Outcome(name=str(name), price=price))
        return outcomes

    def _to_market(self, raw: dict[str, Any]) -> Optional[Market]:
        market_id = raw.get("id") or raw.get("conditionId")
        question = raw.get("question") or raw.get("title")
        if not market_id or not question:
            return None

        events = raw.get("events") or []
        event = events[0] if isinstance(events, list) and events else {}
        event_slug = event.get("slug")
        market_slug = raw.get("slug")
        slug = event_slug or market_slug
        url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"

        closed = bool(raw.get("closed"))
        active = raw.get("active")
        if closed:
            status = "closed"
        elif active is False:
            status = "closed"
        else:
            status = "open"

        # Category/tags: Gamma exposes these inconsistently; gather what's there.
        tags: list[str] = []
        for t in raw.get("tags") or event.get("tags") or []:
            if isinstance(t, dict):
                label = t.get("label") or t.get("slug")
                if label:
                    tags.append(str(label))
            elif t:
                tags.append(str(t))

        # Identifiers Phase 3 needs: conditionId (for trades) and the per-outcome
        # CLOB token ids (for price history). Gamma returns the latter as a
        # JSON-encoded string list aligned with `outcomes`.
        platform_meta: dict[str, Any] = {}
        condition_id = raw.get("conditionId")
        if condition_id:
            platform_meta["condition_id"] = str(condition_id)
        token_ids = self._parse_json_list(raw.get("clobTokenIds"))
        if token_ids:
            platform_meta["clob_token_ids"] = [str(t) for t in token_ids]
        if market_slug:
            platform_meta["slug"] = str(market_slug)

        return Market(
            platform=self.name,
            market_id=str(market_id),
            title=str(question),
            url=url,
            status=status,
            outcomes=self._build_outcomes(raw),
            event_id=str(event.get("id")) if event.get("id") else None,
            event_title=event.get("title"),
            description=raw.get("description"),
            category=raw.get("category") or event.get("category"),
            tags=tags,
            volume=parse_float(raw.get("volumeNum") or raw.get("volume")),
            volume_unit="usd",
            liquidity=parse_float(raw.get("liquidityNum") or raw.get("liquidity")),
            open_time=raw.get("startDate"),
            close_time=raw.get("endDate"),
            platform_meta=platform_meta,
        )

    # ------------------------------------------------------------------
    # Phase 3 activity collection
    # ------------------------------------------------------------------
    def fetch_price_history(
        self, market: Market, *, window_days: int = 14, fidelity_minutes: int = 60
    ) -> list[PricePoint]:
        """Price history for the market's first (Yes) outcome via the CLOB API.

        Polymarket prices history is per CLOB token; we use the first token,
        whose price is the market's implied "yes" probability. No per-point
        volume is exposed here.
        """
        token_ids = market.platform_meta.get("clob_token_ids") or []
        if not token_ids:
            raise ValueError("market has no clob_token_ids; cannot fetch price history")

        clob_url = self.config.get("clob_base_url", _DEFAULT_CLOB_URL).rstrip("/")
        end_ts = int(time.time())
        start_ts = end_ts - window_days * 86400
        data = get_json(
            self.session,
            f"{clob_url}/prices-history",
            params={
                "market": token_ids[0],
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": fidelity_minutes,
            },
            timeout=self.timeout,
        )
        history = data.get("history") if isinstance(data, dict) else data
        points: list[PricePoint] = []
        for h in history or []:
            ts = iso_from_unix(h.get("t"))
            if ts is None:
                continue
            points.append(PricePoint(t=ts, price=parse_float(h.get("p"))))
        return points

    def fetch_trades(
        self, market: Market, *, window_days: int = 14, max_trades: int = 2000
    ) -> list[Trade]:
        """Recent trades for the market via the Data API, newest first.

        Each trade carries an on-chain proxy wallet, which we immediately reduce
        to an opaque cluster key (raw addresses never leave this method).
        """
        condition_id = market.platform_meta.get("condition_id")
        if not condition_id:
            raise ValueError("market has no condition_id; cannot fetch trades")

        data_url = self.config.get("data_base_url", _DEFAULT_DATA_URL).rstrip("/")
        cutoff = time.time() - window_days * 86400
        page = min(self.page_size, 500)
        out: list[Trade] = []
        offset = 0
        for _ in range(self.max_pages):
            rows = get_json(
                self.session,
                f"{data_url}/trades",
                params={"market": condition_id, "limit": page, "offset": offset},
                timeout=self.timeout,
            )
            rows = rows if isinstance(rows, list) else (rows.get("data") or [])
            if not rows:
                break
            stop = False
            for r in rows:
                ts = parse_float(r.get("timestamp"))
                if ts is not None and ts < cutoff:
                    stop = True
                    break
                wallet = r.get("proxyWallet") or r.get("maker") or r.get("wallet")
                out.append(
                    Trade(
                        t=iso_from_unix(r.get("timestamp")) or "",
                        price=parse_float(r.get("price")),
                        size=parse_float(r.get("size")),
                        side=(r.get("side") or "").lower() or None,
                        wallet=cluster_key(wallet) if wallet else None,
                    )
                )
                if len(out) >= max_trades:
                    return out
            if stop or len(rows) < page:
                break
            offset += page
        return out
