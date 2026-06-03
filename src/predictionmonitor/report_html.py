"""Render the daily leads as a self-contained, visual HTML report.

Joins the anomaly leads (which events/markets fired, and when the jump
happened) with the activity price series (for sparklines) into one HTML string
with inline CSS + SVG — no external assets, opens straight in a browser or as a
CI artifact.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any, Optional

from predictionmonitor import history as history_mod
from predictionmonitor.viz import (
    TIER_COLORS,
    _parse_iso,
    escape,
    event_timeline,
    sparkline,
)

_CSS = """
:root { --high:#c0392b; --medium:#e08e0b; --ink:#2c3e50; --muted:#7f8c8d; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color: var(--ink); margin: 0; background: #f5f6f8; }
.wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 4px; font-size: 24px; }
.note { background:#fff7e6; border:1px solid #f0d399; border-radius:8px;
        padding:10px 14px; color:#6b5400; font-size:13px; margin:14px 0; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
.card { background:#fff; border:1px solid #e1e4e8; border-radius:10px;
        padding:12px 16px; flex:1; min-width:150px; }
.card .k { color:var(--muted); font-size:12px; text-transform:uppercase;
           letter-spacing:.04em; }
.card .v { font-size:22px; font-weight:600; margin-top:2px; }
.section { background:#fff; border:1px solid #e1e4e8; border-radius:10px;
           padding:16px 18px; margin:16px 0; }
.event { border-top:1px solid #eee; padding:12px 0; }
.event:first-of-type { border-top:0; }
.event h3 { margin:0 0 2px; font-size:15px; }
.badge { display:inline-block; color:#fff; border-radius:6px; padding:1px 8px;
         font-size:11px; font-weight:600; vertical-align:middle; margin-right:6px; }
.meta { color:var(--muted); font-size:12px; }
table.mk { width:100%; border-collapse:collapse; margin-top:8px; }
table.mk td { padding:6px 6px; border-top:1px solid #f0f0f0; vertical-align:middle; }
table.mk td.spark { width:180px; }
table.mk td.sc { width:48px; text-align:right; font-variant-numeric:tabular-nums;
                 font-weight:600; }
.sig { color:var(--muted); font-size:12px; }
.trades { margin:8px 0 2px; font-size:12.5px; }
.trades ul { margin:4px 0 0; padding-left:18px; }
.trades li { margin:2px 0; }
.trades .tx { font-weight:600; }
a { color:#1a5fb4; text-decoration:none; }
a:hover { text-decoration:underline; }
.legend { margin:6px 0 0; padding:0; list-style:none; columns:2; font-size:13px; }
.legend li { margin:2px 0; }
.dot { display:inline-block; width:16px; height:16px; border-radius:50%;
       color:#fff; text-align:center; line-height:16px; font-size:10px;
       margin-right:6px; }
footer { color:var(--muted); font-size:12px; margin:18px 0; text-align:center; }
"""


def _points_index(activity_result: dict[str, Any]) -> dict[tuple, list[dict]]:
    return {
        (a.get("platform"), a.get("market_id")): a.get("price_points") or []
        for a in activity_result.get("activity", [])
    }


def _series_and_highlight(points: list[dict], signals: list[dict]):
    """(price list, index of the detected jump) for a market's sparkline."""
    values = [p.get("price") for p in points]
    at = None
    for s in signals:
        if s.get("name") == "price_jump":
            at = (s.get("detail") or {}).get("at")
            break
    hi = None
    if at is not None:
        for i, p in enumerate(points):
            if p.get("t") == at:
                hi = i
                break
    return values, hi


def _tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, TIER_COLORS["low"])
    return f'<span class="badge" style="background:{color}">{escape(tier)}</span>'


def _card(label: str, value: Any) -> str:
    return f'<div class="card"><div class="k">{escape(label)}</div>' \
           f'<div class="v">{value}</div></div>'


def _fmt_num(x: Any) -> str:
    """Compact number formatting (no scientific notation)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "?"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return str(int(v)) if v == int(v) else f"{v:.4g}"


def _flagged_trades_html(trades: list[dict[str, Any]]) -> str:
    """The outlier trades for an event, linking straight to the tx + wallet."""
    if not trades:
        return ""
    items = []
    for tr in trades[:5]:
        side = escape((tr.get("side") or "").upper())
        desc = f'{_fmt_num(tr.get("size"))} @ {_fmt_num(tr.get("price"))} {side}'.strip()
        links = []
        if tr.get("tx_url"):
            links.append(f'<a class="tx" href="{escape(tr["tx_url"])}">tx</a>')
        if tr.get("account_url"):
            links.append(f'<a href="{escape(tr["account_url"])}">wallet</a>')
        link_txt = " · ".join(links) or "—"
        when = escape((tr.get("t") or "")[:16].replace("T", " "))
        items.append(
            f'<li>{desc} — {link_txt} <span class="meta">{when}</span></li>'
        )
    return ('<div class="trades"><b>Outlier trades</b> '
            "(largest in window — open to investigate):"
            f'<ul>{"".join(items)}</ul></div>')


def render_report(
    leads_result: dict[str, Any],
    activity_result: dict[str, Any],
    *,
    summary: Optional[dict[str, Any]] = None,
    today: Optional[str] = None,
    history: Optional[list[dict[str, Any]]] = None,
) -> str:
    today = today or leads_result.get("generated_at") or date.today().isoformat()
    points_by_id = _points_index(activity_result)
    window_days = leads_result.get("source_window_days") or activity_result.get(
        "window_days"
    )

    events = leads_result.get("events", [])
    flagged = [e for e in events if e["tier"] != "low"]

    # Number events for the timeline <-> legend cross-reference.
    timeline_events = []
    for n, e in enumerate(flagged, start=1):
        e["_n"] = n
        at = None
        for s in e.get("top_signals", []):
            if s.get("name") == "price_jump":
                at = (s.get("detail") or {}).get("at")
                break
        timeline_events.append(
            {"n": n, "at": at, "tier": e["tier"], "score": e["lead_score"],
             "label": e["event_title"]}
        )

    # Frame the timeline to the full N-day window (N = the report window), so
    # sparse days are visible rather than the axis collapsing onto the data.
    n_days = window_days or 30
    all_times = [
        p.get("t") for pts in points_by_id.values() for p in pts if p.get("t")
    ]
    parsed = [t for t in (_parse_iso(x) for x in all_times) if t is not None]
    end_ts = max(parsed) if parsed else time.time()
    start_ts = end_ts - n_days * 86400

    ec = leads_result.get("event_counts", {})
    cards = [_card("Anomaly leads", f'{ec.get("high", 0)} high · {ec.get("medium", 0)} med')]
    if summary:
        cat = summary.get("catalog", {})
        rel = summary.get("relevance", {})
        act = summary.get("activity", {})
        cards = [
            _card("Catalog", cat.get("total", 0)),
            _card("FMCC relevant", f'{rel.get("watch", 0)} watch · {rel.get("review", 0)} rev'),
            _card("Activity", f'{act.get("markets", 0)} mkts / {window_days}d'),
        ] + cards

    out = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>FMCC Daily Scan — {escape(today)}</title>",
        f"<style>{_CSS}{history_mod.TIMELINE_CSS}</style></head><body>"
        '<div class="wrap">',
        f"<h1>FMCC Prediction-Market Scan</h1>",
        f'<div class="meta">{escape(today)} · {window_days}-day window</div>',
        '<div class="note"><strong>Leads, not findings.</strong> Statistically '
        "unusual activity on <em>public</em> markets — a prompt to look closer, "
        "never evidence of wrongdoing or attribution to a person.</div>",
        '<div class="cards">' + "".join(cards) + "</div>",
    ]

    # Within-window timeline section.
    out.append('<div class="section"><h2 style="margin-top:0;font-size:16px">'
               f"When the jumps happened (last {n_days} days)</h2>")
    out.append(event_timeline(timeline_events, start_ts=start_ts, end_ts=end_ts))
    if flagged:
        legend = ['<ul class="legend">']
        for e in flagged:
            color = TIER_COLORS.get(e["tier"], TIER_COLORS["low"])
            legend.append(
                f'<li><span class="dot" style="background:{color}">{e["_n"]}</span>'
                f'<a href="{escape(e["url"])}">{escape(e["event_title"])}</a> '
                f'<span class="meta">({e["lead_score"]:.2f})</span></li>'
            )
        legend.append("</ul>")
        out.append("".join(legend))
    out.append("</div>")

    # Lead detail, grouped by tier then event.
    for tier, heading in (("high", "High-priority leads"),
                          ("medium", "Medium-priority leads")):
        tier_events = [e for e in flagged if e["tier"] == tier]
        if not tier_events:
            continue
        out.append(f'<div class="section"><h2 style="margin-top:0;font-size:16px">'
                   f"{escape(heading)}</h2>")
        for e in tier_events:
            sib = (f' · {e["n_flagged"]}/{e["n_markets"]} markets'
                   if e["n_markets"] > 1 else "")
            out.append(
                f'<div class="event"><h3>{_tier_badge(tier)}'
                f'<span class="dot" style="background:{TIER_COLORS[tier]}">'
                f'{e["_n"]}</span> '
                f'<a href="{escape(e["url"])}">{escape(e["event_title"])}</a></h3>'
                f'<div class="meta">lead {e["lead_score"]:.2f}{sib}</div>'
                '<table class="mk">'
            )
            for m in e["members"]:
                pts = points_by_id.get((e["platform"], m["market_id"]), [])
                values, hi = _series_and_highlight(pts, m.get("signals", []))
                spark = sparkline(values, highlight_index=hi)
                sig_txt = "; ".join(
                    f'{s["label"]} {s["value"]}'
                    + (f' ({s["detail"]["sigma"]}σ)'
                       if (s.get("detail") or {}).get("sigma") is not None else "")
                    for s in m.get("signals", [])
                ) or "—"
                out.append(
                    f'<tr><td class="sc">{m["lead_score"]:.2f}</td>'
                    f'<td class="spark">{spark}</td>'
                    f'<td><a href="{escape(m["url"])}">{escape(m["title"])}</a>'
                    f'<div class="sig">{escape_sig(sig_txt)}</div></td></tr>'
                )
            out.append("</table>")
            out.append(_flagged_trades_html(e.get("flagged_trades", [])))
            out.append("</div>")
        out.append("</div>")

    # Cross-day timeline (accumulates across scans) — same page, scroll down.
    if history:
        out.append('<h2 style="font-size:18px;margin:22px 0 6px">'
                   "Activity over time (all scans)</h2>")
        out.append(history_mod.timeline_body(history))

    out.append('<footer>Generated by predictionmonitor · thresholds &amp; '
               "taxonomy in <code>config/</code></footer>")
    out.append("</div></body></html>")
    return "".join(out)


def escape_sig(text: str) -> str:
    # signal text is already built from numbers/labels; escape the σ-safe string.
    return escape(text)


def write_html_report(
    leads_result: dict[str, Any],
    activity_result: dict[str, Any],
    *,
    summary: Optional[dict[str, Any]] = None,
    history: Optional[list[dict[str, Any]]] = None,
    output_dir: str = "reports",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    today = leads_result.get("generated_at") or date.today().isoformat()
    path = os.path.join(output_dir, f"report-{today}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(leads_result, activity_result, summary=summary,
                               today=today, history=history))
    return path
