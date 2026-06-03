"""Shared HTTP session: timeouts, retries with exponential backoff, UA header."""

from __future__ import annotations

from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "predictionmonitor/0.1"


def build_session(
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    max_retries: int = 4,
    backoff_factor: float = 1.0,
) -> requests.Session:
    """A requests Session that retries idempotent GETs on transient errors.

    Retries cover connection errors and 429/5xx responses with exponential
    backoff (backoff_factor * 2**(n-1) seconds), matching the project's
    network-resilience policy.
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    timeout: float = 30.0,
    headers: Optional[dict[str, str]] = None,
) -> Any:
    """GET a URL and return parsed JSON, raising for non-2xx responses."""
    resp = session.get(url, params=params, timeout=timeout, headers=headers)
    resp.raise_for_status()
    return resp.json()
