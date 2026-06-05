"""Phase 7: arbitrage / market-maker classification, to demote false-positive leads.

A large class of leads are *structurally* explainable and not insider activity:
arbitrageurs and market-makers who sweep **every outcome** of a partition at
**near-certain prices**, or hold **both sides** of a market to lock a spread.
The canonical example is a wallet buying "No" on all market-cap buckets of an
IPO event at ~99¢ — economically one near-risk-free position, not a directional
bet by someone who knows something.

This module recognizes that structure from the trade data Phase 3 already
collects (opaque wallet cluster keys + outcome + price across sibling markets)
and **demotes** the wallet/trade-based portion of a lead:

- the ``wallet_concentration`` signal is dropped when the dominant wallet is an
  arb sweeper (the lead is re-scored and re-tiered) — so a lead that was *only*
  "one wallet dominated volume" by an arbitrageur falls out of the report;
- the arb's trades are **tagged** ``structural`` in the flagged-trades list so a
  reviewer isn't misled into reading them as suspicious actors.

Crucially it is **surgical**: independent price/volume signals (a genuine
abrupt price jump) are left untouched, because the arb's near-certain trades
don't cause them. We only auto-demote on high-precision *structural* evidence,
never merely because a wallet is active — keeping real directional leads intact.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_ARB_CONFIG: dict[str, Any] = {
    "enabled": True,
    # A price at/above this (for a buy) is "near-certain" — almost no edge left.
    "near_certain_price": 0.90,
    # A wallet must trade at least this many sibling markets of one event to
    # count as a partition "sweep".
    "min_sweep_markets": 3,
    # ...and at least this share of its trades in the event must be near-certain
    # buys, so a wallet making real directional bets isn't mislabeled.
    "near_certain_share": 0.5,
}


def arb_config(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve the arb config from settings, with defaults."""
    cfg = (settings or {}).get("arb", {})
    return {**DEFAULT_ARB_CONFIG, **(cfg or {})}


def _event_key(platform: str, event_id: Optional[str], market_id: str) -> tuple:
    """Same keying as anomaly._group_events, so the two stay aligned."""
    return (platform, event_id or f"market:{market_id}")


def identify_arb_wallets(
    sibling_activities: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Find arb/market-maker wallets among the sibling markets of one event.

    Returns ``{wallet_cluster_key: evidence}``. A wallet is flagged when it shows
    one of the high-precision structural patterns:

    - **partition sweep** — bought near-certain outcomes across ``min_sweep_markets``
      or more sibling markets (most of its trades near-certain);
    - **both sides of one market** — bought two different outcomes of the same
      market (a locked spread);
    - **spanning outcomes** — held multiple distinct outcomes across many
      sibling markets.
    """
    near = float(config["near_certain_price"])
    min_mkts = int(config["min_sweep_markets"])
    min_share = float(config["near_certain_share"])

    per_wallet: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "markets": set(),
            "trades": 0,
            "near_buys": 0,
            "outcomes": set(),
            "outcomes_by_market": defaultdict(set),
            "label": None,
        }
    )

    for act in sibling_activities:
        mid = act.get("market_id")
        for t in act.get("trades", []) or []:
            w = t.get("wallet")
            if not w:
                continue
            d = per_wallet[w]
            d["markets"].add(mid)
            d["trades"] += 1
            if d["label"] is None:
                d["label"] = t.get("actor_label")
            outcome = (t.get("outcome") or "").strip().lower()
            if outcome:
                d["outcomes"].add(outcome)
                d["outcomes_by_market"][mid].add(outcome)
            price = t.get("price")
            if t.get("action") == "bought" and price is not None and price >= near:
                d["near_buys"] += 1

    arb: dict[str, dict[str, Any]] = {}
    for w, d in per_wallet.items():
        n_mkts = len(d["markets"])
        near_share = (d["near_buys"] / d["trades"]) if d["trades"] else 0.0
        both_sides = any(len(s) >= 2 for s in d["outcomes_by_market"].values())
        sweep = n_mkts >= min_mkts and near_share >= min_share
        spans = len(d["outcomes"]) >= 2 and n_mkts >= min_mkts

        if not (sweep or both_sides or spans):
            continue

        reasons: list[str] = []
        if sweep:
            reasons.append(
                f"bought near-certain (≥{near:.0%}) outcomes across "
                f"{n_mkts} sibling markets"
            )
        if both_sides:
            reasons.append("bought both sides of the same market")
        if spans and not sweep:
            reasons.append(f"held {len(d['outcomes'])} outcomes across {n_mkts} markets")

        arb[w] = {
            "label": d["label"] or "a wallet",
            "reason": "; ".join(reasons),
            "n_markets": n_mkts,
            "near_certain_buys": d["near_buys"],
            "trades": d["trades"],
            "outcomes": sorted(d["outcomes"]),
        }
    return arb


def apply_arb_adjustments(
    leads: list,
    activity_result: dict[str, Any],
    *,
    config: dict[str, Any],
    tiers: dict[str, float],
    retier,
) -> dict[tuple, dict[str, dict[str, Any]]]:
    """Demote the wallet/trade-based portion of leads driven by arb wallets.

    Mutates ``leads`` in place: drops the ``wallet_concentration`` signal of any
    market whose dominant wallet is an arb sweeper (re-scoring + re-tiering via
    the supplied ``retier`` callable), and tags arb trades in ``flagged_trades``.
    Returns ``{event_key: {wallet: evidence}}`` for downstream annotation.
    """
    activities = activity_result.get("activity", []) or []
    act_by_market = {(a.get("platform"), a.get("market_id")): a for a in activities}

    # Group sibling markets by event, then find the arb wallets within each.
    event_acts: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for a in activities:
        event_acts[_event_key(a.get("platform"), a.get("event_id"), a.get("market_id"))].append(a)
    arb_by_event = {
        key: identify_arb_wallets(acts, config) for key, acts in event_acts.items()
    }

    for lead in leads:
        key = _event_key(lead.platform, lead.event_id, lead.market_id)
        arb_wallets = arb_by_event.get(key) or {}
        if not arb_wallets:
            continue

        # Is THIS market's dominant (top-volume) wallet an arb sweeper? If so,
        # the wallet_concentration signal is structural, not suspicious.
        act = act_by_market.get((lead.platform, lead.market_id), {})
        clusters = act.get("wallet_clusters") or []
        top_wallet = clusters[0]["cluster"] if clusters else None
        if top_wallet in arb_wallets:
            kept = [s for s in lead.signals if s.name != "wallet_concentration"]
            if len(kept) != len(lead.signals):
                lead.signals = kept
                lead.lead_score = round(sum(s.contribution for s in kept), 4)
                lead.tier = retier(lead.lead_score, tiers)
                lead.arb_adjusted = True

        # Tag the arb trades so the report labels them structural, not suspicious.
        for tr in lead.flagged_trades:
            ev = arb_wallets.get(tr.get("wallet"))
            if ev:
                tr["arb"] = True
                tr["arb_note"] = ev["reason"]

    return arb_by_event


def annotate_events(
    events: list[dict[str, Any]],
    arb_by_event: dict[tuple, dict[str, dict[str, Any]]],
) -> int:
    """Attach an ``arb`` summary to each event; return how many are arb-driven.

    An event is arb-driven when its sibling markets contain arb wallets. The
    summary lists the distinct arb actors (by public label) and why each was
    classified, so the demotion is fully transparent to a reviewer.
    """
    rank = {"low": 0, "medium": 1, "high": 2}
    n = 0
    for e in events:
        key = _event_key(e.get("platform"), e.get("event_id"), e.get("market_id", ""))
        wallets = arb_by_event.get(key) or {}
        if not wallets:
            e["arb"] = {"likely": False, "demoted": False, "wallets": []}
            continue
        # A true demotion: the arb discount lowered the event's tier. An event
        # that kept its tier on an independent price/volume signal is "likely"
        # but not "demoted" — its arb trades are tagged inline instead.
        demoted = rank.get(e.get("tier"), 0) < rank.get(e.get("pre_arb_tier"), 0)
        # Dedupe by public label; keep the richest reason per actor.
        by_label: dict[str, dict[str, Any]] = {}
        for ev in wallets.values():
            label = ev["label"]
            if label not in by_label or ev["n_markets"] > by_label[label]["n_markets"]:
                by_label[label] = ev
        e["arb"] = {
            "likely": True,
            "demoted": demoted,
            "wallets": [
                {
                    "label": ev["label"],
                    "reason": ev["reason"],
                    "n_markets": ev["n_markets"],
                    "near_certain_buys": ev["near_certain_buys"],
                }
                for ev in sorted(by_label.values(), key=lambda x: -x["n_markets"])
            ],
        }
        n += 1
    return n
