"""Adapter interface contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import requests

from predictionmonitor.http import build_session
from predictionmonitor.schema import Market


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or build_session()
        self.timeout = timeout
        self.page_size = page_size
        self.max_pages = max_pages
        self.only_open = only_open

    @abstractmethod
    def iter_markets(self) -> Iterator[Market]:
        """Yield normalized markets, paginating until exhausted or capped."""
        raise NotImplementedError

    # Subclasses implement the platform-specific mapping.
    @abstractmethod
    def _to_market(self, raw: dict[str, Any]) -> Optional[Market]:
        """Map one raw API record to a Market (or None to skip it)."""
        raise NotImplementedError
