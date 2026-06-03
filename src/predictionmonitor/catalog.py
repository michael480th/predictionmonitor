"""Catalog orchestration: run adapters, aggregate normalized markets, persist."""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any, Iterable, Optional

import yaml

from predictionmonitor.adapters import ADAPTERS
from predictionmonitor.http import build_session
from predictionmonitor.schema import Market

log = logging.getLogger(__name__)

_DEFAULT_SETTINGS_PATH = os.path.join("config", "settings.yml")


def load_settings(path: str = _DEFAULT_SETTINGS_PATH) -> dict[str, Any]:
    """Load settings.yml, returning {} if it is missing."""
    if not os.path.exists(path):
        log.warning("settings file %s not found; using built-in defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_adapter(platform: str, settings: dict[str, Any]):
    http_cfg = settings.get("http", {})
    ing_cfg = settings.get("ingestion", {})
    plat_cfg = settings.get("platforms", {}).get(platform, {})

    session = build_session(
        user_agent=http_cfg.get("user_agent", "predictionmonitor/0.1"),
        max_retries=http_cfg.get("max_retries", 4),
        backoff_factor=http_cfg.get("backoff_factor", 1.0),
    )
    adapter_cls = ADAPTERS[platform]
    return adapter_cls(
        base_url=plat_cfg["base_url"],
        session=session,
        timeout=http_cfg.get("timeout_seconds", 30),
        page_size=ing_cfg.get("page_size", 500),
        max_pages=ing_cfg.get("max_pages", 200),
        only_open=ing_cfg.get("only_open", True),
    )


def collect_markets(
    platform: str,
    settings: dict[str, Any],
    max_markets: Optional[int] = None,
) -> list[Market]:
    """Fetch and normalize all markets from one platform, with optional cap."""
    adapter = _build_adapter(platform, settings)
    out: list[Market] = []
    for market in adapter.iter_markets():
        out.append(market)
        if max_markets is not None and len(out) >= max_markets:
            log.info("%s: reached max_markets cap (%d)", platform, max_markets)
            break
    log.info("%s: collected %d markets", platform, len(out))
    return out


def run_catalog(
    platforms: Iterable[str],
    settings: dict[str, Any],
    max_markets: Optional[int] = None,
) -> dict[str, Any]:
    """Run catalog ingestion across platforms and assemble a result dict.

    Each platform is isolated: a failure in one is recorded as an error and does
    not abort the others (so a partial catalog still lands).
    """
    markets: list[Market] = []
    errors: dict[str, str] = {}
    per_platform: dict[str, int] = {}

    for platform in platforms:
        try:
            collected = collect_markets(platform, settings, max_markets=max_markets)
            markets.extend(collected)
            per_platform[platform] = len(collected)
        except Exception as exc:  # network/parse/etc. — keep going
            log.exception("ingestion failed for %s", platform)
            errors[platform] = f"{type(exc).__name__}: {exc}"
            per_platform[platform] = 0

    return {
        "generated_at": date.today().isoformat(),
        "platforms": list(platforms),
        "counts": per_platform,
        "total": len(markets),
        "errors": errors,
        "markets": [m.to_dict() for m in markets],
    }


def write_catalog(result: dict[str, Any], output_dir: str = "reports") -> str:
    """Persist a catalog result to reports/catalog-YYYY-MM-DD.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"catalog-{result['generated_at']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    return path


def latest_catalog_path(output_dir: str = "reports") -> Optional[str]:
    """Return the newest reports/catalog-*.json by filename, or None."""
    import glob

    matches = sorted(glob.glob(os.path.join(output_dir, "catalog-*.json")))
    return matches[-1] if matches else None


def load_catalog_markets(path: str) -> list[Market]:
    """Load a saved catalog JSON back into Market objects."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [Market.from_dict(m) for m in data.get("markets", [])]
