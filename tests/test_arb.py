"""Offline tests for Phase 7 arbitrage / market-maker demotion."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor.arb import (  # noqa: E402
    DEFAULT_ARB_CONFIG,
    identify_arb_wallets,
)
from predictionmonitor.anomaly import run_leads  # noqa: E402


def trade(wallet, *, outcome, price, action="bought", size=360, label=None):
    """A serialized-trade dict shaped like schema.Trade.to_dict()."""
    return {
        "t": "2026-06-04T16:07:00+00:00",
        "price": price,
        "size": size,
        "side": "buy" if action == "bought" else "sell",
        "outcome": outcome,
        "wallet": wallet,
        "wallet_address": "0x" + wallet,
        "actor_label": label or wallet,
        "action": action,
        "usd": round(size * price, 2),
    }


def market(market_id, *, event_id, trades, prices=None, top_wallet=None,
           top_share=None, title=None):
    """Build one market's activity dict with trades + (optional) signals."""
    points = [{"t": f"t{i}", "price": p} for i, p in enumerate(prices or [])]
    # wallet_clusters sorted by volume desc; first entry is the "top" wallet.
    vol = {}
    for tr in trades:
        vol[tr["wallet"]] = vol.get(tr["wallet"], 0.0) + (tr["size"] or 0)
    clusters = sorted(
        ({"cluster": w, "volume": v} for w, v in vol.items()),
        key=lambda c: c["volume"], reverse=True,
    )
    if top_wallet:  # force a specific dominant wallet
        clusters.sort(key=lambda c: 0 if c["cluster"] == top_wallet else 1)
    stats = {"suspicious_trades": list(trades)}
    if top_share is not None:
        stats["top_wallet_share"] = top_share
    return {
        "platform": "polymarket", "market_id": market_id,
        "title": title or market_id, "url": f"https://example.test/{market_id}",
        "decision": "watch", "score": 5.0,
        "event_id": event_id, "event_title": "Freddie Mac IPO Closing Market Cap",
        "price_points": points, "trades": trades,
        "wallet_clusters": clusters, "stats": stats,
    }


# A wallet ("e46m3") sweeping No across five buckets at ~99c — the canonical arb.
def sweep_markets():
    buckets = ["<150B", "150-200B", "200-250B", "250-300B", "300B+"]
    prices = [0.998, 0.998, 0.996, 0.993, 0.995]
    mkts = []
    for i, (b, p) in enumerate(zip(buckets, prices)):
        mkts.append(market(
            f"cap-{i}", event_id="FMCC-IPO-CAP", title=b, top_wallet="e46m3",
            top_share=0.8,
            trades=[trade("e46m3", outcome="No", price=p, label="Grown-Fence")],
        ))
    return mkts


class IdentifyTests(unittest.TestCase):
    def test_partition_sweep_is_flagged(self):
        arb = identify_arb_wallets(sweep_markets(), DEFAULT_ARB_CONFIG)
        self.assertIn("e46m3", arb)
        ev = arb["e46m3"]
        self.assertEqual(ev["n_markets"], 5)
        self.assertEqual(ev["label"], "Grown-Fence")
        self.assertIn("near-certain", ev["reason"])

    def test_both_sides_of_one_market_is_flagged(self):
        m = market(
            "m1", event_id="E1",
            trades=[
                trade("hedger", outcome="Yes", price=0.45),
                trade("hedger", outcome="No", price=0.52),
            ],
        )
        arb = identify_arb_wallets([m], DEFAULT_ARB_CONFIG)
        self.assertIn("hedger", arb)
        self.assertIn("both sides", arb["hedger"]["reason"])

    def test_directional_single_bet_is_not_flagged(self):
        # One cheap longshot Yes in one market — the thing we must NOT demote.
        m = market(
            "m1", event_id="E1",
            trades=[trade("insider", outcome="Yes", price=0.08, action="bought")],
        )
        arb = identify_arb_wallets([m], DEFAULT_ARB_CONFIG)
        self.assertEqual(arb, {})

    def test_near_certain_but_too_few_markets_not_flagged(self):
        # Near-certain buys, but only across 2 markets (< min_sweep_markets=3).
        mkts = [
            market(f"m{i}", event_id="E1",
                   trades=[trade("w", outcome="No", price=0.99)])
            for i in range(2)
        ]
        arb = identify_arb_wallets(mkts, DEFAULT_ARB_CONFIG)
        self.assertEqual(arb, {})


SETTINGS = {
    "anomaly": {"tiers": {"high": 3.0, "medium": 1.5}},
    "arb": DEFAULT_ARB_CONFIG,
}


class RunLeadsIntegrationTests(unittest.TestCase):
    def test_pure_wallet_concentration_arb_is_demoted_to_low(self):
        # Five sweep markets, each flagged ONLY by wallet_concentration (arb).
        result = run_leads({"activity": sweep_markets(), "window_days": 30}, SETTINGS)
        # The arb signal was discounted -> no high/medium events survive.
        self.assertEqual(result["event_counts"]["high"], 0)
        self.assertEqual(result["event_counts"]["medium"], 0)
        self.assertGreaterEqual(result["n_arb_events"], 1)
        # Every lead lost its wallet_concentration signal.
        for lead in result["leads"]:
            names = {s["name"] for s in lead["signals"]}
            self.assertNotIn("wallet_concentration", names)
            self.assertTrue(lead["arb_adjusted"])
        # The event is recorded as a true demotion (was flagged, now low).
        event = result["events"][0]
        self.assertTrue(event["arb"]["demoted"])
        self.assertEqual(event["tier"], "low")
        self.assertNotEqual(event["pre_arb_tier"], "low")

    def test_real_price_jump_survives_but_trades_tagged(self):
        # Same sweep, but one bucket ALSO has a genuine abrupt price jump that
        # the arb's near-certain trades did not cause. That lead must survive,
        # with the arb trade tagged structural.
        mkts = sweep_markets()
        mkts[0]["price_points"] = [
            {"t": f"t{i}", "price": p}
            for i, p in enumerate([0.50, 0.505, 0.50, 0.505, 0.50, 0.85])
        ]
        result = run_leads({"activity": mkts, "window_days": 30}, SETTINGS)

        # A price-driven event remains flagged...
        flagged = [e for e in result["events"] if e["tier"] != "low"]
        self.assertTrue(flagged)
        head = flagged[0]
        names = {s["name"] for s in head["top_signals"]}
        self.assertIn("price_jump", names)
        self.assertNotIn("wallet_concentration", names)
        # ...and its arb trades are tagged structural for the reviewer.
        self.assertTrue(any(t.get("arb") for t in head["flagged_trades"]))
        # ...and the event is annotated as arb-driven, but NOT demoted — it
        # survived on the independent price signal, so it isn't in the
        # auto-demoted section, only tagged inline.
        self.assertTrue(head["arb"]["likely"])
        self.assertFalse(head["arb"]["demoted"])
        self.assertEqual(head["arb"]["wallets"][0]["label"], "Grown-Fence")

    def test_directional_lead_is_untouched(self):
        # A lone wallet making one cheap, confident, directional bet that drove
        # a price jump should remain a clean high/medium lead, not arb.
        m = market(
            "solo", event_id="SOLO",
            prices=[0.10, 0.105, 0.10, 0.105, 0.10, 0.62],  # big jump
            top_share=0.7,
            trades=[trade("insider", outcome="Yes", price=0.10, action="bought")],
        )
        result = run_leads({"activity": [m], "window_days": 30}, SETTINGS)
        self.assertEqual(result["n_arb_events"], 0)
        lead = result["leads"][0]
        self.assertFalse(lead["arb_adjusted"])
        names = {s["name"] for s in lead["signals"]}
        self.assertIn("price_jump", names)

    def test_arb_can_be_disabled(self):
        settings = {"anomaly": SETTINGS["anomaly"], "arb": {"enabled": False}}
        result = run_leads({"activity": sweep_markets(), "window_days": 30}, settings)
        self.assertEqual(result["n_arb_events"], 0)
        # With arb off, the wallet_concentration signal survives -> events flag.
        self.assertGreater(
            result["event_counts"]["high"] + result["event_counts"]["medium"], 0
        )


if __name__ == "__main__":
    unittest.main()
