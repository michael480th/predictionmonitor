"""Persist Phase 2 relevance results as JSON + a reviewer-friendly Markdown report."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any


def write_watchlist(result: dict[str, Any], output_dir: str = "reports") -> tuple[str, str]:
    """Write watchlist-YYYY-MM-DD.json and .md. Returns (json_path, md_path)."""
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().isoformat()
    json_path = os.path.join(output_dir, f"watchlist-{today}.json")
    md_path = os.path.join(output_dir, f"watchlist-{today}.md")

    payload = {"generated_at": today, **result}
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(today, result))

    return json_path, md_path


def _render_rows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_None._\n"
    lines = ["| Score | Platform | Market | Why flagged |",
             "|------:|----------|--------|-------------|"]
    for r in items:
        reason = "; ".join(
            f"{b['label']} ({', '.join(b['matched_keywords'])})"
            for b in r.get("matched_buckets", [])
        ) or "—"
        title = r["title"].replace("|", "\\|")
        lines.append(
            f"| {r['score']:.1f} | {r['platform']} | "
            f"[{title}]({r['url']}) | {reason} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(today: str, result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    th = result.get("thresholds", {})
    return (
        f"# FMCC Prediction-Market Watchlist — {today}\n\n"
        "> **Lead, not a finding.** These are markets whose subject matter is "
        "relevant to Freddie Mac. Relevance does not imply any wrongdoing; it "
        "marks markets worth monitoring for unusual trading in later phases.\n\n"
        f"**Thresholds:** watch ≥ {th.get('watch')}, review ≥ {th.get('review')}\n\n"
        f"**Counts:** {counts.get('watch', 0)} watch · "
        f"{counts.get('review', 0)} review · "
        f"{counts.get('ignored', 0)} ignored · "
        f"{counts.get('excluded', 0)} excluded\n\n"
        "## Watchlist\n\n"
        f"{_render_rows(result.get('watch', []))}\n"
        "## Review (borderline — human triage)\n\n"
        f"{_render_rows(result.get('review', []))}"
    )
