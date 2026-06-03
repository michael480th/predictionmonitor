"""Phase 5: daily pipeline orchestration + top-level digest report.

Runs the whole scan end to end in one process — catalog -> relevance filter ->
activity collection -> anomaly/lead scoring — and writes a single human-facing
digest (`report-YYYY-MM-DD.md`) that links the per-stage artifacts. This is what
the daily GitHub Actions cron invokes; it's also runnable locally as
`python -m predictionmonitor daily`.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from predictionmonitor.activity import run_activity, write_activity
from predictionmonitor.anomaly import run_leads, write_leads
from predictionmonitor.catalog import run_catalog, write_catalog
from predictionmonitor.relevance import (
    description_weight,
    filter_markets,
    load_taxonomy,
    relevance_thresholds,
)
from predictionmonitor.schema import Market
from predictionmonitor.watchlist import write_watchlist

log = logging.getLogger(__name__)


def enabled_platforms(settings: dict[str, Any]) -> list[str]:
    plats = (settings or {}).get("platforms", {})
    enabled = [name for name, cfg in plats.items() if cfg.get("enabled", True)]
    return enabled or ["polymarket", "kalshi"]


def run_daily(
    settings: dict[str, Any],
    *,
    platforms: Optional[list[str]] = None,
    max_markets: Optional[int] = None,
    window_days: Optional[int] = None,
    include_review: bool = False,
    taxonomy_path: str = "config/taxonomy.yml",
    output_dir: str = "reports",
) -> dict[str, Any]:
    """Run catalog -> filter -> activity -> leads, writing all artifacts.

    Returns a digest summary dict (also written to report-YYYY-MM-DD.md).
    """
    plats = platforms or enabled_platforms(settings)

    # 1. Catalog.
    log.info("daily: cataloging %s", ", ".join(plats))
    catalog_result = run_catalog(plats, settings, max_markets=max_markets)
    catalog_path = write_catalog(catalog_result, output_dir=output_dir)
    markets = [Market.from_dict(m) for m in catalog_result.get("markets", [])]

    # 2. Relevance filter -> watchlist.
    taxonomy = load_taxonomy(taxonomy_path)
    watch_th, review_th = relevance_thresholds(settings)
    watchlist_result = filter_markets(
        markets,
        taxonomy,
        watch_threshold=watch_th,
        review_threshold=review_th,
        description_weight=description_weight(settings),
    )
    wl_json, wl_md = write_watchlist(watchlist_result, output_dir=output_dir)

    # 3. Activity collection for the watchlisted markets.
    activity_result = run_activity(
        watchlist_result,
        markets,
        settings,
        window_days=window_days,
        include_review=include_review,
    )
    act_json, act_md = write_activity(activity_result, output_dir=output_dir)

    # 4. Anomaly detection -> leads.
    leads_result = run_leads(activity_result, settings)
    leads_json, leads_md = write_leads(leads_result, output_dir=output_dir)

    summary = {
        "generated_at": date.today().isoformat(),
        "platforms": plats,
        "catalog": {
            "total": catalog_result.get("total", 0),
            "counts": catalog_result.get("counts", {}),
            "errors": catalog_result.get("errors", {}),
        },
        "relevance": watchlist_result.get("counts", {}),
        "activity": activity_result.get("counts", {}),
        "window_days": activity_result.get("window_days"),
        "leads": {
            "event_counts": leads_result.get("event_counts", {}),
            "market_counts": leads_result.get("counts", {}),
            "top_events": [
                e for e in leads_result.get("events", []) if e["tier"] != "low"
            ],
        },
        "artifacts": {
            "catalog": catalog_path,
            "watchlist_md": wl_md,
            "watchlist_json": wl_json,
            "activity_md": act_md,
            "activity_json": act_json,
            "leads_md": leads_md,
            "leads_json": leads_json,
        },
    }
    digest_path = write_digest(summary, output_dir=output_dir)
    summary["artifacts"]["digest"] = digest_path
    return summary


# --------------------------------------------------------------------------
# Digest report
# --------------------------------------------------------------------------


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _signal_text(signals: list[dict[str, Any]]) -> str:
    parts = []
    for s in signals:
        sigma = (s.get("detail") or {}).get("sigma")
        extra = f", {sigma}σ" if sigma is not None else ""
        parts.append(f"{s['label']} {s['value']} (≥{s['threshold']}{extra})")
    return "; ".join(parts) or "—"


def render_digest(summary: dict[str, Any]) -> str:
    """Render the top-level daily digest as Markdown."""
    today = summary["generated_at"]
    cat = summary["catalog"]
    rel = summary["relevance"]
    act = summary["activity"]
    leads = summary["leads"]
    ec = leads["event_counts"]
    arts = summary["artifacts"]

    cat_breakdown = ", ".join(
        f"{p} {n}" for p, n in (cat.get("counts") or {}).items()
    ) or "—"

    lines = [
        f"# FMCC Daily Scan — {today}",
        "",
        "> **Leads, not findings.** This is an automated daily indicator over "
        "*public* prediction-market data. Flagged markets are statistically "
        "unusual and worth a look — never evidence of wrongdoing, and never "
        "attributed to a person.",
        "",
        "## At a glance",
        "",
        "| Stage | Result |",
        "|-------|--------|",
        f"| Catalog | {cat.get('total', 0)} open markets ({cat_breakdown}) |",
        f"| FMCC relevance | {rel.get('watch', 0)} watch · {rel.get('review', 0)} "
        f"review · {rel.get('excluded', 0)} excluded |",
        f"| Activity | {act.get('markets', 0)} markets, "
        f"{summary.get('window_days')}-day window |",
        f"| Anomaly leads | **{ec.get('high', 0)} high** · {ec.get('medium', 0)} "
        f"medium events |",
        "",
    ]

    if cat.get("errors"):
        errs = "; ".join(f"{k}: {v}" for k, v in cat["errors"].items())
        lines += [f"> ⚠️ Catalog errors: {errs}", ""]

    lines += ["## Top leads", ""]
    top = leads.get("top_events", [])
    if not top:
        lines.append("_No anomalous activity flagged today._")
    else:
        for e in top:
            sib = (
                f" · {e['n_flagged']}/{e['n_markets']} markets"
                if e["n_markets"] > 1
                else ""
            )
            title = (e["event_title"] or "").replace("|", "\\|")
            lines.append(
                f"- **{e['lead_score']:.2f} [{e['tier']}]** "
                f"[{title}]({e['url']}){sib}  \n"
                f"  {e['headline_market']} — {_signal_text(e['top_signals'])}"
            )
    lines += [
        "",
        "## Full reports",
        "",
        f"- Watchlist: `{_basename(arts['watchlist_md'])}`",
        f"- Activity: `{_basename(arts['activity_json'])}`",
        f"- Leads: `{_basename(arts['leads_md'])}`",
        "",
        "_Generated by predictionmonitor. Thresholds and taxonomy are in "
        "`config/`._",
    ]
    return "\n".join(lines) + "\n"


def write_digest(summary: dict[str, Any], output_dir: str = "reports") -> str:
    import os

    os.makedirs(output_dir, exist_ok=True)
    today = summary.get("generated_at", date.today().isoformat())
    path = os.path.join(output_dir, f"report-{today}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_digest(summary))
    return path
