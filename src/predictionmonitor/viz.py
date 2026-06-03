"""Dependency-free SVG building blocks for the visual reports.

We deliberately avoid matplotlib/plotly so the project keeps its tiny
requests+PyYAML footprint and the output is a self-contained string that renders
in any browser (and as a GitHub Actions artifact). Everything here is pure
functions returning SVG/HTML fragments; coordinates are rounded so output is
stable and testable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape as _html_escape
from typing import Any, Optional

# Tier -> colour, shared across charts and the HTML shell.
TIER_COLORS = {"high": "#c0392b", "medium": "#e08e0b", "low": "#7f8c8d"}


def escape(text: Any) -> str:
    return _html_escape(str(text if text is not None else ""))


def _r(x: float) -> float:
    return round(x, 1)


def _parse_iso(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def sparkline(
    values: list[Optional[float]],
    *,
    width: int = 170,
    height: int = 38,
    pad: int = 4,
    highlight_index: Optional[int] = None,
    color: str = "#2c3e50",
    highlight_color: str = "#c0392b",
) -> str:
    """A tiny line chart of a price/probability series.

    `values` may contain None (skipped). `highlight_index` (an index into the
    *original* list) is marked with a dot — used to point at the detected jump.
    The y-axis auto-scales to the series so small-but-real moves stay visible.
    """
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return (
            f'<svg width="{width}" height="{height}" '
            f'role="img" aria-label="no data"></svg>'
        )

    xs = [i for i, _ in pts]
    ys = [v for _, v in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1
    spany = (maxy - miny) or 1

    def fx(i: float) -> float:
        return _r(pad + (i - minx) / spanx * (width - 2 * pad))

    def fy(v: float) -> float:
        return _r(height - pad - (v - miny) / spany * (height - 2 * pad))

    d = "M " + " L ".join(f"{fx(i)} {fy(v)}" for i, v in pts)
    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="price sparkline">',
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>',
    ]
    if highlight_index is not None and 0 <= highlight_index < len(values):
        hv = values[highlight_index]
        if hv is not None:
            parts.append(
                f'<circle cx="{fx(highlight_index)}" cy="{fy(hv)}" r="2.8" '
                f'fill="{highlight_color}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def event_timeline(
    events: list[dict[str, Any]],
    *,
    width: int = 760,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> str:
    """A lollipop timeline: each event a marker at its time, height ∝ score.

    `events`: dicts with ``at`` (ISO time), ``label``, ``tier``, ``score`` and an
    ``n`` index used as the marker number (kept in sync with a legend the caller
    renders). Returns an SVG string; empty-state when nothing has a time.
    """
    timed = []
    for e in events:
        t = _parse_iso(e.get("at"))
        if t is not None:
            timed.append((t, e))
    if not timed:
        return (
            '<svg width="%d" height="40" role="img" aria-label="timeline">'
            '<text x="8" y="24" font-size="12" fill="#7f8c8d">'
            "No timed events.</text></svg>" % width
        )

    times = [t for t, _ in timed]
    lo = start_ts if start_ts is not None else min(times)
    hi = end_ts if end_ts is not None else max(times)
    if hi <= lo:
        lo, hi = lo - 86400, hi + 86400
    span = hi - lo

    height = 150
    axis_y = height - 28
    pad = 36
    plot_w = width - 2 * pad
    max_score = max((e.get("score") or 0) for _, e in timed) or 1.0

    def fx(t: float) -> float:
        return _r(pad + (t - lo) / span * plot_w)

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="event timeline">',
        f'<line x1="{pad}" y1="{axis_y}" x2="{width - pad}" y2="{axis_y}" '
        f'stroke="#bdc3c7" stroke-width="1"/>',
    ]

    # Date ticks (up to ~6 evenly spaced).
    ticks = 5
    for k in range(ticks + 1):
        t = lo + span * k / ticks
        x = fx(t)
        label = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%b %d")
        parts.append(
            f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{axis_y + 4}" '
            f'stroke="#bdc3c7"/>'
            f'<text x="{x}" y="{axis_y + 16}" font-size="10" fill="#7f8c8d" '
            f'text-anchor="middle">{label}</text>'
        )

    # Lollipops.
    for t, e in sorted(timed, key=lambda te: te[0]):
        x = fx(t)
        score = e.get("score") or 0
        stem = _r((axis_y - 16) * min(score / max_score, 1.0))
        top_y = _r(axis_y - 6 - stem)
        color = TIER_COLORS.get(e.get("tier"), TIER_COLORS["low"])
        parts.append(
            f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{top_y}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{x}" cy="{top_y}" r="9" fill="{color}"/>'
            f'<text x="{x}" y="{top_y + 3.5}" font-size="10" fill="#fff" '
            f'text-anchor="middle">{escape(e.get("n"))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def daily_bars(
    days: list[dict[str, Any]],
    *,
    width: int = 760,
    bar_w: int = 16,
    gap: int = 6,
) -> str:
    """Stacked daily bars of flagged events over time (high over medium).

    `days`: dicts with ``date`` (YYYY-MM-DD), ``high``, ``medium`` counts,
    already sorted ascending. Width auto-expands to fit all days.
    """
    if not days:
        return ('<svg width="%d" height="40" role="img"><text x="8" y="24" '
                'font-size="12" fill="#7f8c8d">No history yet.</text></svg>'
                % width)

    n = len(days)
    needed = max(width, n * (bar_w + gap) + 60)
    height = 170
    base_y = height - 26
    top_pad = 14
    max_count = max((d.get("high", 0) + d.get("medium", 0)) for d in days) or 1
    plot_h = base_y - top_pad

    def bh(c: int) -> float:
        return _r(c / max_count * plot_h)

    parts = [
        f'<svg width="{needed}" height="{height}" viewBox="0 0 {needed} {height}" '
        f'role="img" aria-label="events per day">',
        f'<line x1="40" y1="{base_y}" x2="{needed - 8}" y2="{base_y}" '
        f'stroke="#bdc3c7"/>',
    ]
    for c in (0, max_count):
        y = _r(base_y - bh(c))
        parts.append(
            f'<text x="34" y="{y + 3}" font-size="10" fill="#7f8c8d" '
            f'text-anchor="end">{c}</text>'
        )

    x = 48
    show_every = max(1, n // 12)  # avoid crowding date labels
    for i, d in enumerate(days):
        hi = d.get("high", 0)
        med = d.get("medium", 0)
        h_hi = bh(hi)
        h_med = bh(med)
        y_med = _r(base_y - h_med)
        y_hi = _r(y_med - h_hi)
        if med:
            parts.append(
                f'<rect x="{x}" y="{y_med}" width="{bar_w}" height="{h_med}" '
                f'fill="{TIER_COLORS["medium"]}"/>'
            )
        if hi:
            parts.append(
                f'<rect x="{x}" y="{y_hi}" width="{bar_w}" height="{h_hi}" '
                f'fill="{TIER_COLORS["high"]}"/>'
            )
        if i % show_every == 0:
            label = d.get("date", "")[5:]  # MM-DD
            parts.append(
                f'<text x="{_r(x + bar_w / 2)}" y="{base_y + 14}" font-size="9" '
                f'fill="#7f8c8d" text-anchor="middle">{escape(label)}</text>'
            )
        x += bar_w + gap
    parts.append("</svg>")
    return "".join(parts)
