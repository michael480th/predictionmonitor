"""Phase 4: anomaly detection + lead scoring.

Phase 3 collected recent activity (price/volume/trade/wallet series) for each
FMCC-relevant market. This phase turns that activity into **leads for Compliance
to investigate** by flagging statistically unusual patterns.

Like Phase 2's relevance scorer, the detector is deliberately simple and
**explainable**: each market accrues a `lead_score` from a small set of named
signals, and every flagged market carries exactly which signals fired, their
measured value, and the threshold they crossed. Nothing here identifies a
trader — signals are computed over public price/volume series and over opaque
wallet cluster keys only.

Signals (each guarded by data availability, so anonymous/partial markets simply
contribute fewer signals rather than erroring):

- ``price_jump``           biggest single-step move, in σ of the step series
- ``abs_move``             cumulative |price change| over the window
- ``volume_spike``         peak period volume / median period volume
- ``wallet_concentration`` share of trade volume in the single largest wallet
"""

from __future__ import annotations

import glob
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

log = logging.getLogger(__name__)

# Signal weights — how much each anomaly contributes to the lead score.
DEFAULT_WEIGHTS: dict[str, float] = {
    "price_jump": 1.5,
    "abs_move": 1.0,
    "volume_spike": 1.0,
    "wallet_concentration": 1.5,
}

# A signal fires once its measured value reaches its threshold.
DEFAULT_SIGNAL_THRESHOLDS: dict[str, float] = {
    "price_jump_abs": 0.1,        # min single-step move (probability points)
    "price_jump_z": 6.0,          # AND that move must be this many σ of normal
    "abs_move": 0.25,             # cumulative |Δ probability| over the window
    "volume_spike": 5.0,          # peak / median period volume
    "wallet_concentration": 0.5,  # top wallet's share of trade volume
}

# Lead tiers by total score.
DEFAULT_LEAD_TIERS: dict[str, float] = {"high": 3.0, "medium": 1.5}

# A fired signal's strength is capped so one extreme signal can't dominate.
_MAX_STRENGTH = 3.0

_SIGNAL_LABELS = {
    "price_jump": "Abrupt price jump",
    "abs_move": "Large net move",
    "volume_spike": "Volume spike (×median)",
    "wallet_concentration": "Single-wallet concentration",
}


@dataclass
class Signal:
    name: str
    label: str
    value: float
    threshold: float
    weight: float
    detail: dict[str, Any] = field(default_factory=dict)  # extra context for humans

    @property
    def strength(self) -> float:
        if not self.threshold:
            return 0.0
        return min(self.value / self.threshold, _MAX_STRENGTH)

    @property
    def contribution(self) -> float:
        return round(self.weight * self.strength, 4)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "threshold": self.threshold,
            "weight": self.weight,
            "contribution": self.contribution,
        }
        if self.detail:
            d["detail"] = self.detail
        return d


@dataclass
class LeadResult:
    platform: str
    market_id: str
    title: str
    url: str
    relevance_decision: str          # carried from the watchlist (watch|review)
    relevance_score: float
    lead_score: float
    tier: str                        # high|medium|low
    signals: list[Signal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "market_id": self.market_id,
            "title": self.title,
            "url": self.url,
            "relevance_decision": self.relevance_decision,
            "relevance_score": self.relevance_score,
            "lead_score": self.lead_score,
            "tier": self.tier,
            "signals": [s.to_dict() for s in self.signals],
        }

    @property
    def reason(self) -> str:
        if not self.signals:
            return "no anomaly signals fired"
        return "; ".join(
            f"{s.label}={s.value} (≥{s.threshold})" for s in self.signals
        )


# --------------------------------------------------------------------------
# Signal detectors (operate on the serialized activity dict)
# --------------------------------------------------------------------------


# Signal name -> the threshold key that gates it (defaults to the name itself).
_THRESHOLD_KEYS = {"price_jump": "price_jump_abs"}


def _mk_signal(name: str, value: float, thresholds: dict, weights: dict) -> Signal:
    return Signal(
        name=name,
        label=_SIGNAL_LABELS[name],
        value=value,
        threshold=float(thresholds[_THRESHOLD_KEYS.get(name, name)]),
        weight=float(weights[name]),
    )


def _price_signals(points: list[dict], thresholds: dict, weights: dict) -> list[Signal]:
    prices = [p["price"] for p in points if p.get("price") is not None]
    signals: list[Signal] = []
    if len(prices) < 3:
        return signals

    steps = [b - a for a, b in zip(prices, prices[1:])]
    # A jump must be BOTH materially large in absolute terms AND statistically
    # unusual. At fine resolution most steps are ~0, so a tiny tick can look
    # like many σ — the absolute floor keeps "frozen market ticks 0.01->0.03"
    # from scoring. The σ test (measured against the *other* steps, so the jump
    # doesn't mask itself) keeps a slow steady climb from scoring. The lead
    # strength scales with the move size; σ rides along as explanatory detail.
    if len(steps) >= 3:
        peak_idx = max(range(len(steps)), key=lambda i: abs(steps[i]))
        max_step = abs(steps[peak_idx])
        baseline = steps[:peak_idx] + steps[peak_idx + 1:]
        sd = statistics.pstdev(baseline)
        z = (max_step / sd) if sd > 0 else float("inf")
        if max_step >= thresholds["price_jump_abs"] and z >= thresholds["price_jump_z"]:
            sig = _mk_signal("price_jump", round(max_step, 4), thresholds, weights)
            sig.detail = {"sigma": round(z, 1) if z != float("inf") else None}
            signals.append(sig)

    abs_move = abs(prices[-1] - prices[0])
    if abs_move >= thresholds["abs_move"]:
        signals.append(_mk_signal("abs_move", round(abs_move, 4), thresholds, weights))
    return signals


def _volume_signal(points: list[dict], thresholds: dict, weights: dict) -> list[Signal]:
    vols = [p["volume"] for p in points if p.get("volume")]
    if len(vols) < 3:
        return []
    median = statistics.median(vols)
    if median <= 0:
        return []
    ratio = max(vols) / median
    if ratio >= thresholds["volume_spike"]:
        return [_mk_signal("volume_spike", round(ratio, 2), thresholds, weights)]
    return []


def _wallet_signal(stats: dict, thresholds: dict, weights: dict) -> list[Signal]:
    share = stats.get("top_wallet_share")
    if share is None or share < thresholds["wallet_concentration"]:
        return []
    return [_mk_signal("wallet_concentration", round(share, 4), thresholds, weights)]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _tier(score: float, tiers: dict[str, float]) -> str:
    if score >= tiers.get("high", DEFAULT_LEAD_TIERS["high"]):
        return "high"
    if score >= tiers.get("medium", DEFAULT_LEAD_TIERS["medium"]):
        return "medium"
    return "low"


def score_activity(
    activity: dict[str, Any],
    *,
    weights: dict[str, float],
    thresholds: dict[str, float],
    tiers: dict[str, float],
) -> LeadResult:
    """Score one market's collected activity into a lead."""
    points = activity.get("price_points") or []
    stats = activity.get("stats") or {}

    signals: list[Signal] = []
    signals += _price_signals(points, thresholds, weights)
    signals += _volume_signal(points, thresholds, weights)
    signals += _wallet_signal(stats, thresholds, weights)
    signals.sort(key=lambda s: s.contribution, reverse=True)

    lead_score = round(sum(s.contribution for s in signals), 4)
    return LeadResult(
        platform=activity.get("platform", ""),
        market_id=activity.get("market_id", ""),
        title=activity.get("title", ""),
        url=activity.get("url", ""),
        relevance_decision=activity.get("decision", ""),
        relevance_score=activity.get("score", 0.0),
        lead_score=lead_score,
        tier=_tier(lead_score, tiers),
        signals=signals,
    )


def anomaly_config(settings: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Resolve (weights, thresholds, tiers) from settings, with defaults."""
    cfg = (settings or {}).get("anomaly", {})
    weights = {**DEFAULT_WEIGHTS, **(cfg.get("weights") or {})}
    thresholds = {**DEFAULT_SIGNAL_THRESHOLDS, **(cfg.get("thresholds") or {})}
    tiers = {**DEFAULT_LEAD_TIERS, **(cfg.get("tiers") or {})}
    return weights, thresholds, tiers


def run_leads(
    activity_result: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Score every collected market and bucket the results into lead tiers."""
    weights, thresholds, tiers = anomaly_config(settings)

    leads = [
        score_activity(a, weights=weights, thresholds=thresholds, tiers=tiers)
        for a in activity_result.get("activity", [])
    ]
    leads.sort(key=lambda r: r.lead_score, reverse=True)

    counts = {"high": 0, "medium": 0, "low": 0}
    for r in leads:
        counts[r.tier] += 1

    return {
        "generated_at": date.today().isoformat(),
        "source_window_days": activity_result.get("window_days"),
        "thresholds": thresholds,
        "weights": weights,
        "tiers": tiers,
        "counts": counts,
        "leads": [r.to_dict() for r in leads],
    }


# --------------------------------------------------------------------------
# Loading activity + writing the leads report
# --------------------------------------------------------------------------


def latest_activity_path(output_dir: str = "reports") -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(output_dir, "activity-*.json")))
    return matches[-1] if matches else None


def load_activity(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_leads(result: dict[str, Any], output_dir: str = "reports") -> tuple[str, str]:
    """Write leads-YYYY-MM-DD.json and .md. Returns (json_path, md_path)."""
    os.makedirs(output_dir, exist_ok=True)
    today = result.get("generated_at", date.today().isoformat())
    json_path = os.path.join(output_dir, f"leads-{today}.json")
    md_path = os.path.join(output_dir, f"leads-{today}.md")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(today, result))
    return json_path, md_path


def _render_rows(leads: list[dict[str, Any]]) -> str:
    if not leads:
        return "_None._\n"
    lines = [
        "| Lead | Tier | Platform | Market | Signals |",
        "|-----:|------|----------|--------|---------|",
    ]
    for r in leads:
        title = r["title"].replace("|", "\\|")
        parts = []
        for s in r["signals"]:
            extra = ""
            sigma = (s.get("detail") or {}).get("sigma")
            if sigma is not None:
                extra = f", {sigma}σ"
            parts.append(f"{s['label']} {s['value']} (≥{s['threshold']}{extra})")
        why = "; ".join(parts) or "—"
        lines.append(
            f"| {r['lead_score']:.2f} | {r['tier']} | {r['platform']} | "
            f"[{title}]({r['url']}) | {why} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(today: str, result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    leads = result.get("leads", [])
    high = [r for r in leads if r["tier"] == "high"]
    medium = [r for r in leads if r["tier"] == "medium"]

    return (
        f"# FMCC Prediction-Market Leads — {today}\n\n"
        "> **Lead, not a finding.** Each row flags FMCC-relevant market activity "
        "that is *statistically unusual* over the collection window — a prompt "
        "for Compliance to look closer, **not** evidence of wrongdoing or any "
        "attribution to a person. Signals are computed from public price/volume "
        "series and opaque wallet cluster keys only.\n\n"
        f"**Window:** {result.get('source_window_days')} days · "
        f"**High:** {counts.get('high', 0)} · "
        f"**Medium:** {counts.get('medium', 0)} · "
        f"**Low/none:** {counts.get('low', 0)}\n\n"
        "## High-priority leads\n\n"
        f"{_render_rows(high)}\n"
        "## Medium-priority leads\n\n"
        f"{_render_rows(medium)}"
    )
