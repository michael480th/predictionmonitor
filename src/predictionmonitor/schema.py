"""Normalized data model shared across platform adapters.

Every adapter maps its platform-native market payload into a :class:`Market`
so downstream phases (relevance filtering, anomaly detection) never need to
know which platform a market came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Outcome:
    """A single tradable outcome within a market and its current price.

    `price` is the implied probability in [0, 1] (e.g. 0.62 == 62%), or None
    when the platform did not provide one.
    """

    name: str
    price: Optional[float] = None


@dataclass
class Market:
    """A normalized prediction market, platform-agnostic.

    Monetary fields (`volume`, `liquidity`) are best-effort and expressed in the
    platform's native unit (USD for Polymarket; contracts/cents for Kalshi).
    Keep `volume_unit` so later phases compare like with like.
    """

    platform: str                      # "polymarket" | "kalshi"
    market_id: str                     # platform-native unique id
    title: str                         # the question being traded
    url: str
    status: str                        # "open" | "closed" | "unknown"
    outcomes: list[Outcome] = field(default_factory=list)

    event_id: Optional[str] = None     # grouping id (an "event" spans markets)
    event_title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    volume: Optional[float] = None
    volume_unit: Optional[str] = None  # "usd" | "contracts"
    liquidity: Optional[float] = None
    open_interest: Optional[float] = None

    open_time: Optional[str] = None    # ISO 8601
    close_time: Optional[str] = None   # ISO 8601

    fetched_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def search_text(self) -> str:
        """Concatenated text used by Phase 2 relevance scoring."""
        parts = [self.title, self.event_title or "", self.description or "",
                 self.category or "", " ".join(self.tags)]
        return " ".join(p for p in parts if p).lower()


def parse_float(value: Any) -> Optional[float]:
    """Coerce mixed API number representations to float, tolerantly."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
