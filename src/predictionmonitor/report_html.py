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
from predictionmonitor.anomaly import (
    DEFAULT_SIGNAL_THRESHOLDS,
    _SIGNAL_LABELS,
    signal_value_text,
)
from predictionmonitor.schema import format_usd
from predictionmonitor.viz import (
    TIER_COLORS,
    _parse_iso,
    escape,
    event_timeline,
    price_volume_chart,
)

# Plain-language explanation of each signal — what the unusual pattern is and why
# it can be a sign of insider/informed trading. Kept beside the detector so the
# report's explainer stays in sync with what we actually compute.
_SIGNAL_WHY = {
    "price_jump": "The market's odds lurched in a single step — far more than its "
    "normal jitter. A classic footprint of someone trading on news before it's "
    "public.",
    "abs_move": "The odds travelled a long way across the window in one direction "
    "— sustained conviction, not random noise.",
    "volume_spike": "Trading suddenly ran far hotter than this market's usual pace "
    "— money showing up in a burst, often around a catalyst.",
    "wallet_concentration": "A single wallet drove an outsized share of the volume "
    "— one concentrated, confident bet rather than a crowd.",
    "material_trade": "A single trade moved real money — far above the few-hundred-"
    "dollar norm on these thin markets. The clearest tripwire that material "
    "capital just entered, whoever placed it.",
    "material_wallet": "One wallet's total stake crossed a material dollar threshold "
    "across the window — a sizeable position building up, not idle churn.",
}


def _threshold_text(thresholds: dict[str, Any]) -> dict[str, str]:
    """Human phrasing of each signal's firing threshold (from the live config)."""
    t = {**DEFAULT_SIGNAL_THRESHOLDS, **(thresholds or {})}
    return {
        "price_jump": f"Fires on a ≥{t['price_jump_abs']:.2f} ({t['price_jump_abs'] * 100:.0f}-point) "
        f"one-step move that is also ≥{t['price_jump_z']:.0f}σ of normal.",
        "abs_move": f"Fires on a ≥{t['abs_move']:.2f} ({t['abs_move'] * 100:.0f}-point) net move "
        "over the window.",
        "volume_spike": f"Fires when peak volume is ≥{t['volume_spike']:.0f}× the median period.",
        "wallet_concentration": f"Fires when one wallet holds ≥{t['wallet_concentration'] * 100:.0f}% "
        "of the traded volume.",
        "material_trade": f"Fires when the largest single trade is "
        f"≥{format_usd(t['material_trade_usd'])} in notional value.",
        "material_wallet": f"Fires when one wallet's total flow is "
        f"≥{format_usd(t['material_wallet_usd'])} across the window.",
    }

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
.arb { display:inline-block; background:#eef1f4; color:#5b6b7b; border:1px solid #dde2e7;
       border-radius:5px; padding:0 6px; font-size:11px; font-weight:600; }
.demoted li { margin:4px 0; font-size:12.5px; color:#4b5b6b; }
.demoted .who { font-weight:600; color:var(--ink); }
table.ov { width:100%; border-collapse:collapse; font-size:13px; }
table.ov th { color:var(--muted); font-weight:600; font-size:11px; letter-spacing:.03em;
              text-transform:uppercase; text-align:left; padding:4px 8px; }
table.ov td { padding:6px 8px; border-top:1px solid #f0f0f0; vertical-align:top; }
table.ov td.amt { font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }
.why { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; }
.why .w { flex:1; min-width:210px; background:#fbfcfd; border:1px solid #eef0f2;
          border-radius:8px; padding:10px 12px; }
.why .w h4 { margin:0 0 4px; font-size:13px; }
.why .w p { margin:0 0 6px; font-size:12.5px; color:#4b5b6b; }
.why .w .thr { font-size:11.5px; color:var(--muted); }
.mkt { border-top:1px solid #f3f3f3; padding:10px 0 4px; }
.mkt:first-of-type { border-top:0; }
.mkt .hd { display:flex; justify-content:space-between; gap:10px; align-items:baseline; }
.mkt .hd .sc { font-variant-numeric:tabular-nums; font-weight:600; color:var(--muted); }
.chart { margin-top:6px; overflow-x:auto; }
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


def _trades_index(activity_result: dict[str, Any]) -> dict[tuple, list[dict]]:
    return {
        (a.get("platform"), a.get("market_id")): a.get("trades") or []
        for a in activity_result.get("activity", [])
    }


def _jump_at(signals: list[dict]) -> Optional[str]:
    for s in signals or []:
        if s.get("name") == "price_jump":
            return (s.get("detail") or {}).get("at")
    return None


def _overview_html(flagged: list[dict[str, Any]]) -> str:
    """Top-of-page table of the biggest flagged trades, across all leads.

    Each row links the trader to their activity, the outcome to the bet, and the
    market to its detail section further down the page.
    """
    rows: list[tuple[dict, dict]] = []
    for e in flagged:
        for tr in e.get("flagged_trades", []):
            if (tr.get("usd") or 0) > 0:
                rows.append((tr, e))
    rows.sort(key=lambda r: r[0].get("usd") or 0, reverse=True)
    rows = rows[:12]
    if not rows:
        return ('<div class="section"><h2 style="margin-top:0;font-size:16px">'
                "Trades we're flagging</h2><div class=\"meta\">No individual "
                "trades crossed the size floor in this window.</div></div>")

    body = [
        '<div class="section"><h2 style="margin-top:0;font-size:16px">'
        "Trades we're flagging</h2>"
        '<div class="meta">Largest individual trades behind today\'s leads — '
        "biggest first. Click a trader for their activity, an outcome for the "
        "bet, or a lead to jump to its charts.</div>"
        '<table class="ov"><tr><th>Size</th><th>Trade</th><th>Lead</th>'
        "<th>When</th></tr>"
    ]
    for tr, e in rows:
        amount = escape(format_usd(tr.get("usd")) or "—")
        actor = escape(tr.get("actor_label") or "A wallet")
        activity = tr.get("account_url")
        name = f'<a href="{escape(activity)}">{actor}</a>' if activity else actor
        action = escape(tr.get("action") or "traded")
        outcome, market = tr.get("outcome"), tr.get("market_url")
        if outcome and market:
            oc = f' of <a href="{escape(market)}">{escape(outcome)}</a>'
        elif outcome:
            oc = f" of {escape(outcome)}"
        else:
            oc = ""
        receipt = (f' · <a href="{escape(tr["tx_url"])}">receipt</a>'
                   if tr.get("tx_url") else "")
        lead = f'<a href="#lead-{e["_n"]}">{escape(e["event_title"])}</a>'
        when = escape((tr.get("t") or "")[:10])
        body.append(
            f'<tr><td class="amt">{amount}</td>'
            f'<td>{name} {action}{oc}{receipt}</td>'
            f'<td>{lead}</td><td class="meta">{when}</td></tr>'
        )
    body.append("</table></div>")
    return "".join(body)


def _explainer_html(thresholds: dict[str, Any]) -> str:
    """The 'what we're scanning for' section: each insider-trading sign, plainly."""
    thr = _threshold_text(thresholds)
    cards = ['<div class="why">']
    for name, label in _SIGNAL_LABELS.items():
        cards.append(
            f'<div class="w"><h4>{escape(label)}</h4>'
            f'<p>{escape(_SIGNAL_WHY.get(name, ""))}</p>'
            f'<div class="thr">{escape(thr.get(name, ""))}</div></div>'
        )
    cards.append("</div>")
    return (
        '<div class="section"><h2 style="margin-top:0;font-size:16px">'
        "What we're scanning for</h2>"
        '<div class="meta">These are the patterns that, on a public market, can '
        "hint at informed or insider trading. Each lead below lists which of "
        "these fired and by how much — they are prompts to look closer, never "
        "proof.</div>"
        + "".join(cards)
        + "</div>"
    )


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
    """The biggest trades behind a lead, in plain English.

    Reads like "Brave-Honey bought $1,610 of Yes". Every fact is on the line; the
    links just verify at the source — the trader name opens that wallet's activity
    feed, the outcome opens the bet's page, and "receipt" opens the on-chain tx.
    """
    if not trades:
        return ""
    items = []
    for tr in trades[:5]:
        actor = escape(tr.get("actor_label") or "A wallet")
        action = escape(tr.get("action") or "traded")
        amount = escape(format_usd(tr.get("usd")) or f'{_fmt_num(tr.get("size"))} shares')
        activity = tr.get("account_url")
        name = (f'<a class="tx" href="{escape(activity)}">{actor}</a>'
                if activity else f'<span class="tx">{actor}</span>')
        outcome, market = tr.get("outcome"), tr.get("market_url")
        if outcome and market:
            of = f' of <a href="{escape(market)}">{escape(outcome)}</a>'
        elif outcome:
            of = f" of {escape(outcome)}"
        elif market:
            of = f' — <a href="{escape(market)}">open bet</a>'
        else:
            of = ""
        receipt = (f' · <a href="{escape(tr["tx_url"])}">receipt</a>'
                   if tr.get("tx_url") else "")
        when = escape((tr.get("t") or "")[:16].replace("T", " "))
        # A trade the arb classifier flagged as structural (sweep/hedge) is
        # labeled so a reviewer doesn't read it as a suspicious actor.
        tag = ""
        if tr.get("arb"):
            note = escape(tr.get("arb_note") or "structural arbitrage")
            tag = f' <span class="arb" title="{note}">structural arb</span>'
        items.append(
            f'<li>{name} {action} <b>{amount}</b>{of}{tag} '
            f'<span class="meta">· {when}{receipt}</span></li>'
        )
    return ('<div class="trades"><b>Largest trades</b> '
            "(trader → activity, outcome → the bet):"
            f'<ul>{"".join(items)}</ul></div>')


def _arb_demoted_html(events: list[dict[str, Any]]) -> str:
    """Transparency block: events whose leads were auto-demoted as structural arb."""
    demoted = [e for e in events if (e.get("arb") or {}).get("demoted")]
    if not demoted:
        return ""
    rows = []
    for e in demoted:
        title = escape(e.get("event_title") or "")
        actors = "; ".join(
            f'<span class="who">{escape(w["label"])}</span> — {escape(w["reason"])}'
            for w in e["arb"]["wallets"][:3]
        )
        rows.append(
            f'<li><a href="{escape(e["url"])}">{title}</a> '
            f'<span class="meta">({escape(e["platform"])} · '
            f'{escape(e.get("pre_arb_tier", ""))} → {escape(e["tier"])})</span>'
            f"<br>{actors}</li>"
        )
    return (
        '<div class="section"><h2 style="margin-top:0;font-size:16px">'
        "Auto-demoted: likely arbitrage / market-making</h2>"
        '<div class="meta" style="margin-bottom:8px">These events were flagged '
        "<em>only</em> by wallet/trade signals that turned out to be a "
        "<em>structural</em> pattern — sweeping near-certain outcomes across a "
        "partition, or holding both sides (near-risk-free arbitrage, not a "
        "directional bet). That signal was discounted, dropping them out of the "
        "leads above; any independent price/volume signal would have kept the "
        "lead. Shown here for transparency.</div>"
        f'<ul class="demoted">{"".join(rows)}</ul></div>'
    )


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
    trades_by_id = _trades_index(activity_result)
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

    # Top: an at-a-glance table of the trades we're flagging, then a plain-language
    # explainer of the insider-trading signs we scan for.
    out.append(_overview_html(flagged))
    out.append(_explainer_html(leads_result.get("signal_thresholds", {})))

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
                f'<div class="event" id="lead-{e["_n"]}"><h3>{_tier_badge(tier)}'
                f'<span class="dot" style="background:{TIER_COLORS[tier]}">'
                f'{e["_n"]}</span> '
                f'<a href="{escape(e["url"])}">{escape(e["event_title"])}</a></h3>'
                f'<div class="meta">lead {e["lead_score"]:.2f}{sib}</div>'
            )
            for m in e["members"]:
                pts = points_by_id.get((e["platform"], m["market_id"]), [])
                trs = trades_by_id.get((e["platform"], m["market_id"]), [])
                sig_txt = "; ".join(
                    f'{s["label"]} {signal_value_text(s["name"], s["value"])}'
                    + (f' ({s["detail"]["sigma"]}σ)'
                       if (s.get("detail") or {}).get("sigma") is not None else "")
                    for s in m.get("signals", [])
                ) or "—"
                chart = price_volume_chart(
                    pts, trs, start_ts=start_ts, end_ts=end_ts,
                    highlight_at=_jump_at(m.get("signals", [])),
                )
                out.append(
                    f'<div class="mkt"><div class="hd">'
                    f'<a href="{escape(m["url"])}">{escape(m["title"])}</a>'
                    f'<span class="sc">{m["lead_score"]:.2f}</span></div>'
                    f'<div class="sig">{escape_sig(sig_txt)}</div>'
                    f'<div class="chart">{chart}</div></div>'
                )
            out.append(_flagged_trades_html(e.get("flagged_trades", [])))
            out.append("</div>")
        out.append("</div>")

    # Transparency: events whose leads were auto-demoted as structural arb.
    out.append(_arb_demoted_html(events))

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
