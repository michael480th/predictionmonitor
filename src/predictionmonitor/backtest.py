"""Phase 6: backtest / threshold calibration.

There is no labelled "suspicious" ground truth for pseudonymous market trades,
so this phase is a **calibration harness** rather than a precision/recall
backtest. It re-measures the raw signal values across every collected market
(optionally pooled over many days of `activity-*.json`), shows their
distributions, sweeps each threshold to reveal how lead counts respond, and
suggests data-driven thresholds (e.g. the 90th percentile) — turning the
hand-picked defaults in `config/settings.yml` into tuned ones.

It re-scores stored activity only; it never re-fetches, so it's fast and
deterministic.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import statistics
from collections import Counter
from datetime import date
from typing import Any, Optional

from predictionmonitor.anomaly import anomaly_config, jump_stats, score_activity

log = logging.getLogger(__name__)

# Signal value keys we measure/sweep (price_jump_z is a gate, reported too).
SIGNAL_KEYS = ["price_jump_abs", "abs_move", "volume_spike", "wallet_concentration"]

# Candidate threshold values for the sweeps.
_SWEEP_CANDIDATES = {
    "price_jump_abs": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "abs_move": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
    "volume_spike": [2, 3, 5, 8, 10],
    "wallet_concentration": [0.3, 0.4, 0.5, 0.6, 0.8],
}


def measure_activity(activity: dict[str, Any]) -> dict[str, Optional[float]]:
    """Raw signal measurements for one market, independent of any threshold."""
    points = activity.get("price_points") or []
    prices = [p["price"] for p in points if p.get("price") is not None]
    m: dict[str, Optional[float]] = {k: None for k in SIGNAL_KEYS}
    m["price_jump_z"] = None

    if len(prices) >= 2:
        m["abs_move"] = round(abs(prices[-1] - prices[0]), 4)
    js = jump_stats(prices)
    if js is not None:
        max_step, z, _ = js
        m["price_jump_abs"] = round(max_step, 4)
        m["price_jump_z"] = z  # may be inf when the rest of the series is flat

    vols = [p["volume"] for p in points if p.get("volume")]
    if len(vols) >= 3:
        med = statistics.median(vols)
        if med > 0:
            m["volume_spike"] = round(max(vols) / med, 2)

    share = (activity.get("stats") or {}).get("top_wallet_share")
    if share is not None:
        m["wallet_concentration"] = round(share, 4)
    return m


def _pct(sorted_vals: list[float], q: float) -> float:
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def distribution(values: list[Optional[float]]) -> dict[str, Any]:
    vals = sorted(v for v in values if v is not None and v != float("inf"))
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(vals[0], 4),
        "p50": round(_pct(vals, 0.50), 4),
        "p90": round(_pct(vals, 0.90), 4),
        "p95": round(_pct(vals, 0.95), 4),
        "max": round(vals[-1], 4),
    }


def _fires(measure: dict, key: str, value: float, thresholds: dict) -> bool:
    mv = measure.get(key)
    if mv is None:
        return False
    if key == "price_jump_abs":
        # price_jump is gated by σ too — hold the current σ gate fixed.
        z = measure.get("price_jump_z")
        if z is None or z < thresholds["price_jump_z"]:
            return False
    return mv >= value


def _tier_counts(activities, weights, thresholds, tiers) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for a in activities:
        r = score_activity(a, weights=weights, thresholds=thresholds, tiers=tiers)
        counts[r.tier] += 1
    return counts


def run_backtest(
    activities: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    n_inputs: int = 1,
) -> dict[str, Any]:
    weights, thresholds, tiers = anomaly_config(settings)
    measures = [measure_activity(a) for a in activities]

    dists = {k: distribution([m[k] for m in measures]) for k in SIGNAL_KEYS}
    dists["price_jump_z"] = distribution([m["price_jump_z"] for m in measures])

    # How often each signal currently fires, and the resulting tier mix.
    fires = Counter()
    for a in activities:
        r = score_activity(a, weights=weights, thresholds=thresholds, tiers=tiers)
        fires.update(s.name for s in r.signals)
    current_tiers = _tier_counts(activities, weights, thresholds, tiers)

    sweeps = {
        key: [
            {"value": c, "fires": sum(_fires(m, key, c, thresholds) for m in measures)}
            for c in cands
        ]
        for key, cands in _SWEEP_CANDIDATES.items()
    }

    # Suggest thresholds at the 90th percentile of observed values (≈ flag the
    # top decile), where there's enough data to be meaningful.
    suggested = {}
    for key in SIGNAL_KEYS:
        d = dists[key]
        if d.get("n", 0) >= 10:
            suggested[key] = d["p90"]
    suggested_thresholds = {**thresholds, **suggested}
    suggested_tiers = _tier_counts(
        activities, weights, suggested_thresholds, tiers
    )

    return {
        "generated_at": date.today().isoformat(),
        "n_inputs": n_inputs,
        "n_markets": len(activities),
        "current": {
            "thresholds": thresholds,
            "tiers": current_tiers,
            "fires": dict(fires),
        },
        "distributions": dists,
        "sweeps": sweeps,
        "suggested": {
            "thresholds": suggested,
            "tiers_if_applied": suggested_tiers,
        },
    }


# --------------------------------------------------------------------------
# Loading activity files + report
# --------------------------------------------------------------------------


def load_activities(paths: list[str]) -> list[dict[str, Any]]:
    """Pool the `activity` lists from one or more activity-*.json files."""
    out: list[dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            out.extend(json.load(fh).get("activity", []))
    return out


def find_activity_files(output_dir: str = "reports") -> list[str]:
    return sorted(glob.glob(os.path.join(output_dir, "activity-*.json")))


def _dist_row(name: str, d: dict[str, Any]) -> str:
    if not d.get("n"):
        return f"| {name} | 0 | — | — | — | — | — |"
    return (
        f"| {name} | {d['n']} | {d['min']} | {d['p50']} | {d['p90']} | "
        f"{d['p95']} | {d['max']} |"
    )


def render_markdown(result: dict[str, Any]) -> str:
    cur = result["current"]
    dists = result["distributions"]
    sug = result["suggested"]

    lines = [
        f"# FMCC Lead Calibration — {result['generated_at']}",
        "",
        f"Pooled **{result['n_markets']} markets** from {result['n_inputs']} "
        "activity file(s). No labelled ground truth exists for pseudonymous "
        "trades, so this tunes thresholds by the *distribution* of observed "
        "signal values, not precision/recall.",
        "",
        "## Signal value distributions",
        "",
        "| Signal | n | min | p50 | p90 | p95 | max |",
        "|--------|--:|----:|----:|----:|----:|----:|",
        _dist_row("price_jump_abs", dists["price_jump_abs"]),
        _dist_row("price_jump_z (σ)", dists["price_jump_z"]),
        _dist_row("abs_move", dists["abs_move"]),
        _dist_row("volume_spike", dists["volume_spike"]),
        _dist_row("wallet_concentration", dists["wallet_concentration"]),
        "",
        "## Current settings",
        "",
        f"- Thresholds: `{cur['thresholds']}`",
        f"- Tier mix: **{cur['tiers']['high']} high** · "
        f"{cur['tiers']['medium']} medium · {cur['tiers']['low']} low",
        f"- Signal fire counts: {cur['fires'] or '—'}",
        "",
        "## Threshold sweeps (markets that would fire each signal)",
        "",
    ]
    for key, rows in result["sweeps"].items():
        cells = " · ".join(f"{r['value']}→{r['fires']}" for r in rows)
        lines.append(f"- **{key}**: {cells}")

    lines += [
        "",
        "## Suggested thresholds (p90 of observed values)",
        "",
    ]
    if sug["thresholds"]:
        for k, v in sug["thresholds"].items():
            lines.append(f"- `{k}`: **{v}** (currently {cur['thresholds'].get(k)})")
        t = sug["tiers_if_applied"]
        lines.append(
            f"\nProjected tier mix if applied: **{t['high']} high** · "
            f"{t['medium']} medium · {t['low']} low."
        )
    else:
        lines.append(
            "_Not enough data (need ≥10 markets with a measurable value per "
            "signal) to suggest thresholds. Pool more activity files._"
        )
    lines += [
        "",
        "_Edit `config/settings.yml` → `anomaly.thresholds` to apply. "
        "Re-run `backtest` after collecting more days for a stabler estimate._",
    ]
    return "\n".join(lines) + "\n"


def write_backtest(result: dict[str, Any], output_dir: str = "reports") -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    today = result.get("generated_at", date.today().isoformat())
    json_path = os.path.join(output_dir, f"backtest-{today}.json")
    md_path = os.path.join(output_dir, f"backtest-{today}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(result))
    return json_path, md_path
