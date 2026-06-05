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

from predictionmonitor.arb import (
    annotate_events,
    apply_arb_adjustments,
    arb_config,
)

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
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    # The biggest individual trades behind this lead, carrying tx/wallet links
    # so a reviewer can open the actual outlier transaction (empty for anonymous
    # platforms or when the trades feed was unavailable).
    flagged_trades: list[dict[str, Any]] = field(default_factory=list)
    # Set when Phase 7 dropped a structural (arb/market-maker) signal from this
    # lead and re-scored it (see arb.py); `original_tier` records the tier before
    # that adjustment so the report can show what was demoted.
    arb_adjusted: bool = False
    original_tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "market_id": self.market_id,
            "title": self.title,
            "url": self.url,
            "event_id": self.event_id,
            "event_title": self.event_title,
            "relevance_decision": self.relevance_decision,
            "relevance_score": self.relevance_score,
            "lead_score": self.lead_score,
            "tier": self.tier,
            "signals": [s.to_dict() for s in self.signals],
            "flagged_trades": self.flagged_trades,
            "arb_adjusted": self.arb_adjusted,
            "original_tier": self.original_tier,
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


def jump_stats(prices: list[float]):
    """(max |step|, σ of that step vs the others, index of the step) or None.

    The biggest move is measured against the spread of the *other* steps so a
    lone spike doesn't mask itself in its own volatility. Shared by detection
    and the Phase 6 backtest so both use identical math. σ is ``inf`` when the
    rest of the series is flat (sd == 0).
    """
    if len(prices) < 4:  # need >= 3 steps for a leave-one-out baseline
        return None
    steps = [b - a for a, b in zip(prices, prices[1:])]
    peak_idx = max(range(len(steps)), key=lambda i: abs(steps[i]))
    max_step = abs(steps[peak_idx])
    if max_step == 0:  # perfectly flat series — no jump to speak of
        return None
    baseline = steps[:peak_idx] + steps[peak_idx + 1:]
    sd = statistics.pstdev(baseline)
    z = (max_step / sd) if sd > 0 else float("inf")
    return max_step, z, peak_idx


def _price_signals(points: list[dict], thresholds: dict, weights: dict) -> list[Signal]:
    pairs = [(p.get("t"), p["price"]) for p in points if p.get("price") is not None]
    prices = [v for _, v in pairs]
    times = [t for t, _ in pairs]
    signals: list[Signal] = []
    if len(prices) < 3:
        return signals

    # A jump must be BOTH materially large in absolute terms AND statistically
    # unusual. At fine resolution most steps are ~0, so a tiny tick can look
    # like many σ — the absolute floor keeps "frozen market ticks 0.01->0.03"
    # from scoring. The σ test keeps a slow steady climb from scoring. The lead
    # strength scales with the move size; σ rides along as explanatory detail.
    js = jump_stats(prices)
    if js is not None:
        max_step, z, peak_idx = js
        if max_step >= thresholds["price_jump_abs"] and z >= thresholds["price_jump_z"]:
            sig = _mk_signal("price_jump", round(max_step, 4), thresholds, weights)
            sig.detail = {
                "sigma": round(z, 1) if z != float("inf") else None,
                "at": times[peak_idx + 1],
            }
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
        event_id=activity.get("event_id"),
        event_title=activity.get("event_title"),
        relevance_decision=activity.get("decision", ""),
        relevance_score=activity.get("score", 0.0),
        lead_score=lead_score,
        tier=_tier(lead_score, tiers),
        signals=signals,
        flagged_trades=(stats.get("suspicious_trades") or []),
    )


def _group_events(leads: list[LeadResult]) -> list[dict[str, Any]]:
    """Collapse sibling markets into one lead per event.

    Many sub-markets of a single event (e.g. a Fed-rate ladder) re-price
    together, so reporting them as one lead per event keeps a single news event
    from flooding the report. An event's headline is its highest-scoring market;
    the flagged siblings travel with it as members.
    """
    groups: dict[tuple, list[LeadResult]] = {}
    order: list[tuple] = []
    for r in leads:
        key = (r.platform, r.event_id or f"market:{r.market_id}")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    rank = {"low": 0, "medium": 1, "high": 2}
    events: list[dict[str, Any]] = []
    for key in order:
        members = sorted(groups[key], key=lambda r: r.lead_score, reverse=True)
        head = members[0]
        flagged = [m for m in members if m.tier != "low"]
        # Best tier the event would have had before any arb adjustment, so the
        # report can tell a true demotion from an event that merely contains arb.
        pre_arb_tier = max(
            (m.original_tier or m.tier for m in members), key=lambda t: rank[t]
        )
        events.append(
            {
                "platform": head.platform,
                "event_id": head.event_id,
                "market_id": head.market_id,   # for arb keying of single-market events
                "event_title": head.event_title or head.title,
                "url": head.url,
                "lead_score": head.lead_score,
                "tier": head.tier,
                "pre_arb_tier": pre_arb_tier,
                "n_markets": len(members),
                "n_flagged": len(flagged),
                "headline_market": head.title,
                "top_signals": [s.to_dict() for s in head.signals],
                "flagged_trades": head.flagged_trades,
                "members": [
                    {
                        "market_id": m.market_id,
                        "title": m.title,
                        "url": m.url,
                        "lead_score": m.lead_score,
                        "tier": m.tier,
                        "signals": [s.to_dict() for s in m.signals],
                    }
                    for m in flagged
                ],
            }
        )
    events.sort(key=lambda e: e["lead_score"], reverse=True)
    return events


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
    for r in leads:  # remember the pre-arb tier so demotions are reportable
        r.original_tier = r.tier

    # Phase 7: demote the wallet/trade-based portion of leads that are really
    # structural arbitrage / market-making (e.g. a wallet sweeping near-certain
    # "No" across every bucket of a partition), re-scoring + re-tiering them.
    acfg = arb_config(settings)
    arb_by_event: dict = {}
    if acfg.get("enabled", True):
        arb_by_event = apply_arb_adjustments(
            leads, activity_result, config=acfg, tiers=tiers, retier=_tier
        )

    leads.sort(key=lambda r: r.lead_score, reverse=True)

    events = _group_events(leads)
    n_arb_events = annotate_events(events, arb_by_event) if arb_by_event else 0

    counts = {"high": 0, "medium": 0, "low": 0}
    for r in leads:
        counts[r.tier] += 1
    event_counts = {"high": 0, "medium": 0, "low": 0}
    for e in events:
        event_counts[e["tier"]] += 1

    return {
        "generated_at": date.today().isoformat(),
        "source_window_days": activity_result.get("window_days"),
        "thresholds": thresholds,
        "weights": weights,
        "tiers": tiers,
        "arb_config": acfg,
        "n_arb_events": n_arb_events,
        "counts": counts,                # per market
        "event_counts": event_counts,    # per event (grouped)
        "events": events,
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


def _signals_text(signals: list[dict[str, Any]]) -> str:
    parts = []
    for s in signals:
        sigma = (s.get("detail") or {}).get("sigma")
        extra = f", {sigma}σ" if sigma is not None else ""
        parts.append(f"{s['label']} {s['value']} (≥{s['threshold']}{extra})")
    return "; ".join(parts) or "—"


def _render_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return "_None._\n"
    blocks: list[str] = []
    for e in events:
        title = (e["event_title"] or "").replace("|", "\\|")
        sib = ""
        if e["n_markets"] > 1:
            sib = f" · {e['n_flagged']} of {e['n_markets']} markets flagged"
        header = (
            f"**{e['lead_score']:.2f} — [{title}]({e['url']})** "
            f"({e['platform']}{sib})  \n"
            f"Top: _{e['headline_market']}_ — {_signals_text(e['top_signals'])}"
        )
        # If several sibling markets fired, list them for the reviewer.
        if e["n_markets"] > 1 and len(e["members"]) > 1:
            rows = ["", "", "| Lead | Market | Signals |", "|-----:|--------|---------|"]
            for m in e["members"]:
                mt = m["title"].replace("|", "\\|")
                rows.append(
                    f"| {m['lead_score']:.2f} | [{mt}]({m['url']}) | "
                    f"{_signals_text(m['signals'])} |"
                )
            header += "\n" + "\n".join(rows)
        blocks.append(header)
    return "\n\n".join(blocks) + "\n"


def _render_arb_section(events: list[dict[str, Any]]) -> str:
    """List events whose leads were auto-demoted as structural arbitrage."""
    demoted = [e for e in events if (e.get("arb") or {}).get("demoted")]
    if not demoted:
        return ""
    lines = [
        "\n## Auto-demoted: likely arbitrage / market-making\n",
        "These events were flagged **only** by wallet/trade signals that turned "
        "out to be a *structural* pattern — a wallet sweeping near-certain "
        "outcomes across a partition, or holding both sides (near-risk-free arb, "
        "not a directional bet). That signal was discounted, dropping them out "
        "of the leads above. Shown for transparency; any independent price/volume "
        "signal would have kept the lead.\n",
    ]
    for e in demoted:
        title = (e["event_title"] or "").replace("|", "\\|")
        actors = "; ".join(
            f"**{w['label']}** — {w['reason']}" for w in e["arb"]["wallets"][:3]
        )
        lines.append(
            f"- [{title}]({e['url']}) ({e['platform']}, "
            f"_{e['pre_arb_tier']}_ → _{e['tier']}_): {actors}"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(today: str, result: dict[str, Any]) -> str:
    ec = result.get("event_counts", {})
    mc = result.get("counts", {})
    events = result.get("events", [])
    high = [e for e in events if e["tier"] == "high"]
    medium = [e for e in events if e["tier"] == "medium"]

    return (
        f"# FMCC Prediction-Market Leads — {today}\n\n"
        "> **Lead, not a finding.** Each entry flags FMCC-relevant market "
        "activity that is *statistically unusual* over the collection window — a "
        "prompt for Compliance to look closer, **not** evidence of wrongdoing or "
        "any attribution to a person. Signals are computed from public "
        "price/volume series and opaque wallet cluster keys only.\n\n"
        f"**Window:** {result.get('source_window_days')} days · "
        f"**Events:** {ec.get('high', 0)} high / {ec.get('medium', 0)} medium / "
        f"{ec.get('low', 0)} low "
        f"(across {mc.get('high', 0)} + {mc.get('medium', 0)} flagged markets)\n\n"
        "Leads are grouped by event: sibling markets of one event (e.g. a "
        "rate ladder) that re-price together appear as a single lead.\n\n"
        "## High-priority leads\n\n"
        f"{_render_events(high)}\n"
        "## Medium-priority leads\n\n"
        f"{_render_events(medium)}"
        f"{_render_arb_section(events)}"
    )
