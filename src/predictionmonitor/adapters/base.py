"""Adapter interface contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import requests

from predictionmonitor.http import build_session
from predictionmonitor.schema import Market, PricePoint, Trade


class Adapter(ABC):
    """Base class for a prediction-platform adapter.

    Subclasses implement :meth:`iter_markets`, which yields normalized
    :class:`Market` objects, transparently handling pagination. The catalog
    orchestrator owns max-markets capping and aggregation.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        base_url: str,
        session: Optional[requests.Session] = None,
        timeout: float = 30.0,
        page_size: int = 500,
        max_pages: int = 200,
        only_open: bool = True,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or build_session()
        self.timeout = timeout
        self.page_size = page_size
        self.max_pages = max_pages
        self.only_open = only_open
        # Per-platform settings block (e.g. activity endpoint base URLs).
        self.config = config or {}

    @abstractmethod
    def iter_markets(self) -> Iterator[Market]:
        """Yield normalized markets, paginating until exhausted or capped."""
        raise NotImplementedError

    # Subclasses implement the platform-specific mapping.
    @abstractmethod
    def _to_market(self, raw: dict[str, Any]) -> Optional[Market]:
        """Map one raw API record to a Market (or None to skip it)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Phase 3: activity collection. Defaults are "unsupported" (empty) so a
    # platform can opt in by overriding just the pieces it exposes.
    # ------------------------------------------------------------------
    def fetch_price_history(
        self, market: Market, *, window_days: int = 14, fidelity_minutes: int = 60
    ) -> list[PricePoint]:
        """Return a price/probability time series for the last `window_days`."""
        return []

    def fetch_trades(
        self, market: Market, *, window_days: int = 14, max_trades: int = 2000
    ) -> list[Trade]:
        """Return up to `max_trades` recent trades within `window_days`."""
        return []
