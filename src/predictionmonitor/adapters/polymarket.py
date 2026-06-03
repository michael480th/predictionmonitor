"""Polymarket adapter (Gamma API).

Docs: https://docs.polymarket.com  (Gamma markets endpoint is public, no auth)

The Gamma `/markets` endpoint paginates with `limit`/`offset`. Outcomes and
their prices arrive as JSON-encoded strings that we parse and zip together.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Optional

from predictionmonitor.adapters.base import Adapter
from predictionmonitor.http import get_json
from predictionmonitor.schema import Market, Outcome, parse_float

log = logging.getLogger(__name__)

# Gamma caps page size around 500.
_MAX_PAGE = 500


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
        )
