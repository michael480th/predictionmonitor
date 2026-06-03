"""Platform adapters: each turns a platform's API into normalized Markets."""

from predictionmonitor.adapters.base import Adapter
from predictionmonitor.adapters.kalshi import KalshiAdapter
from predictionmonitor.adapters.polymarket import PolymarketAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    "polymarket": PolymarketAdapter,
    "kalshi": KalshiAdapter,
}

__all__ = ["Adapter", "PolymarketAdapter", "KalshiAdapter", "ADAPTERS"]
