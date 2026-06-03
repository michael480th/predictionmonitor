"""Normalized data model shared across platform adapters.

Every adapter maps its platform-native market payload into a :class:`Market`
so downstream phases (relevance filtering, anomaly detection) never need to
know which platform a market came from.
"""

from __future__ import annotations

import hashlib
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

    # Platform-specific identifiers downstream phases need but the normalized
    # model doesn't otherwise model (e.g. Polymarket conditionId / CLOB token
    # ids for price history, Kalshi series_ticker for candlesticks). Opaque to
    # everything except the adapter that produced it.
    platform_meta: dict[str, Any] = field(default_factory=dict)

    fetched_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Market":
        """Reconstruct a Market from its serialized form (e.g. catalog JSON)."""
        fields = dict(data)
        fields["outcomes"] = [
            Outcome(name=o.get("name", ""), price=o.get("price"))
            for o in (data.get("outcomes") or [])
        ]
        # Drop any unknown keys so older/newer files stay loadable.
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in fields.items() if k in allowed})

    @property
    def strong_text(self) -> str:
        """High-signal fields: the question itself and its event title."""
        parts = [self.title, self.event_title or ""]
        return " ".join(p for p in parts if p).lower()

    @property
    def weak_text(self) -> str:
        """Lower-signal fields: description/category/tags (often boilerplate)."""
        parts = [self.description or "", self.category or "", " ".join(self.tags)]
        return " ".join(p for p in parts if p).lower()

    @property
    def search_text(self) -> str:
        """All text used by Phase 2 relevance scoring (strong + weak)."""
        return " ".join(p for p in (self.strong_text, self.weak_text) if p)


def parse_float(value: Any) -> Optional[float]:
    """Coerce mixed API number representations to float, tolerantly."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iso_from_unix(ts: Any) -> Optional[str]:
    """Convert a unix timestamp (seconds) to an ISO 8601 UTC string, or None."""
    val = parse_float(ts)
    if val is None:
        return None
    return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()


def cluster_key(address: Any) -> str:
    """Opaque, stable key for a public on-chain wallet address.

    Per the project's guardrails, Polymarket wallet addresses (public,
    pseudonymous) are used *only* as opaque cluster keys for pattern detection
    and are **never** stored raw or attributed to a person. Hashing the
    lowercased address gives a key that is stable across markets and days (so
    Phase 4 can cluster the same wallet's activity) while the raw address never
    lands in any stored report.
    """
    norm = str(address or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Phase 3: activity time series (price/volume/trade samples for one market).
# --------------------------------------------------------------------------


@dataclass
class PricePoint:
    """One observation of a market's implied probability over time.

    `price` is the implied probability in [0, 1]; `volume` is the traded
    volume in the sampling period when the platform reports it (Kalshi
    candlesticks do; Polymarket's price history does not), else None.
    """

    t: str                              # ISO 8601 UTC timestamp
    price: Optional[float]
    volume: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    """A single executed trade, normalized across platforms.

    `wallet` is an *opaque cluster key* (see :func:`cluster_key`), never a raw
    address, and is None for anonymous platforms (Kalshi).
    """

    t: str                              # ISO 8601 UTC timestamp
    price: Optional[float]              # implied probability in [0, 1]
    size: Optional[float]              # contracts/shares traded
    side: Optional[str] = None         # normalized: "buy"|"sell"|"yes"|"no"
    wallet: Optional[str] = None       # opaque cluster key, or None if anonymous

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
