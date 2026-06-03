"""Append-only event history + cross-day timeline.

Each daily run appends one compact record per flagged event to a small,
*tracked* JSON-lines file (`history/events.jsonl`) — unlike the bulky, gitignored
per-run reports, this is meant to be committed so the timeline accumulates
across cron runs. The timeline renders events-per-day over time.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from predictionmonitor.viz import daily_bars, escape, TIER_COLORS

DEFAULT_HISTORY_PATH = os.path.join("history", "events.jsonl")


def event_records(leads_result: dict[str, Any], *, run_date: str) -> list[dict[str, Any]]:
    """One flat record per flagged (high/medium) event for the history log."""
    records = []
    for e in leads_result.get("events", []):
        if e["tier"] == "low":
            continue
        at = None
        for s in e.get("top_signals", []):
            if s.get("name") == "price_jump":
                at = (s.get("detail") or {}).get("at")
                break
        records.append(
            {
                "date": run_date,
                "platform": e["platform"],
                "event_id": e.get("event_id"),
                "event_title": e.get("event_title"),
                "tier": e["tier"],
                "lead_score": e["lead_score"],
                "n_flagged": e.get("n_flagged"),
                "n_markets": e.get("n_markets"),
                "jump_at": at,
                "url": e.get("url"),
            }
        )
    return records


def load_history(path: str = DEFAULT_HISTORY_PATH) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_run(
    leads_result: dict[str, Any],
    *,
    run_date: str | None = None,
    path: str = DEFAULT_HISTORY_PATH,
) -> int:
    """Append this run's events, replacing any prior entries for the same date.

    Re-running on the same day overwrites that day's rows (idempotent), so a
    manual re-run doesn't double-count. Returns the number of records written.
    """
    run_date = run_date or date.today().isoformat()
    existing = [r for r in load_history(path) if r.get("date") != run_date]
    new_records = event_records(leads_result, run_date=run_date)
    combined = existing + new_records
    combined.sort(key=lambda r: (r.get("date", ""), -(r.get("lead_score") or 0)))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in combined:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new_records)


def _by_day(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days: dict[str, dict[str, int]] = {}
    for r in history:
        d = days.setdefault(r["date"], {"date": r["date"], "high": 0, "medium": 0})
        if r.get("tier") in ("high", "medium"):
            d[r["tier"]] += 1
    return [days[k] for k in sorted(days)]


def render_timeline(history: list[dict[str, Any]]) -> str:
    """Render the cross-day timeline as a self-contained HTML page."""
    days = _by_day(history)
    total_days = len(days)
    total_events = len(history)

    # Most-recurring events (flagged on the most distinct days).
    recur: dict[tuple, dict[str, Any]] = {}
    for r in history:
        key = (r["platform"], r.get("event_id") or r.get("event_title"))
        slot = recur.setdefault(
            key, {"title": r.get("event_title"), "url": r.get("url"), "days": set(),
                   "max_score": 0.0}
        )
        slot["days"].add(r["date"])
        slot["max_score"] = max(slot["max_score"], r.get("lead_score") or 0)
    top = sorted(recur.values(), key=lambda s: (len(s["days"]), s["max_score"]),
                 reverse=True)[:12]

    rows = "".join(
        f'<tr><td>{len(s["days"])}</td><td>{s["max_score"]:.2f}</td>'
        f'<td><a href="{escape(s["url"])}">{escape(s["title"])}</a></td></tr>'
        for s in top
    ) or '<tr><td colspan="3">No events recorded yet.</td></tr>'

    css = (
        "body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;"
        "color:#2c3e50;background:#f5f6f8;margin:0}.wrap{max-width:900px;"
        "margin:0 auto;padding:24px}.section{background:#fff;border:1px solid "
        "#e1e4e8;border-radius:10px;padding:16px 18px;margin:16px 0;overflow-x:auto}"
        "h1{font-size:22px;margin:0}.k{color:#7f8c8d;font-size:13px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;"
        "border-top:1px solid #f0f0f0;text-align:left}th{color:#7f8c8d;"
        "font-size:12px;text-transform:uppercase}a{color:#1a5fb4;"
        "text-decoration:none}.lg span{display:inline-block;width:12px;height:12px;"
        "border-radius:3px;margin:0 4px 0 12px;vertical-align:middle}"
    )
    legend = (
        f'<span class="lg"><span style="background:{TIER_COLORS["high"]}"></span>'
        f'high<span style="background:{TIER_COLORS["medium"]}"></span>medium</span>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>FMCC Leads Timeline</title><style>{css}</style></head><body>"
        '<div class="wrap"><h1>FMCC Leads — Timeline</h1>'
        f'<div class="k">{total_events} flagged events across {total_days} '
        f"scan day(s)</div>"
        f'<div class="section"><h2 style="margin-top:0;font-size:16px">'
        f"Flagged events per day {legend}</h2>{daily_bars(days)}</div>"
        '<div class="section"><h2 style="margin-top:0;font-size:16px">'
        "Most recurring events</h2><table><tr><th>Days flagged</th>"
        "<th>Peak lead</th><th>Event</th></tr>"
        f"{rows}</table></div></div></body></html>"
    )


def write_timeline(history: list[dict[str, Any]], output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "timeline.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_timeline(history))
    return path
