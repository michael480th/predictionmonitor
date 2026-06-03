"""Offline tests for Phase 4 anomaly detection + lead scoring."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor.anomaly import (  # noqa: E402
    DEFAULT_SIGNAL_THRESHOLDS,
    DEFAULT_WEIGHTS,
    anomaly_config,
    run_leads,
    score_activity,
    write_leads,
)

W = DEFAULT_WEIGHTS
T = DEFAULT_SIGNAL_THRESHOLDS
TIERS = {"high": 3.0, "medium": 1.5}


def activity(
    *, prices=None, volumes=None, top_wallet_share=None, platform="polymarket",
    market_id="m1", title="A market", decision="watch", score=3.0,
):
    points = []
    prices = prices or []
    for i, p in enumerate(prices):
        pt = {"t": f"t{i}", "price": p}
        if volumes is not None:
            pt["volume"] = volumes[i]
        points.append(pt)
    stats = {}
    if top_wallet_share is not None:
        stats["top_wallet_share"] = top_wallet_share
    return {
        "platform": platform, "market_id": market_id, "title": title,
        "url": "https://example.test/x", "decision": decision, "score": score,
        "price_points": points, "stats": stats,
    }


def score(a):
    return score_activity(a, weights=W, thresholds=T, tiers=TIERS)


class SignalTests(unittest.TestCase):
    def test_calm_market_has_no_signals(self):
        a = activity(prices=[0.50, 0.505, 0.50, 0.495, 0.50, 0.505])
        r = score(a)
        self.assertEqual(r.signals, [])
        self.assertEqual(r.lead_score, 0.0)
        self.assertEqual(r.tier, "low")

    def test_price_jump_fires(self):
        # Tiny noise then one big jump -> high σ on the max step.
        a = activity(prices=[0.50, 0.505, 0.50, 0.505, 0.50, 0.85])
        r = score(a)
        names = {s.name for s in r.signals}
        self.assertIn("price_jump", names)

    def test_abs_move_fires(self):
        a = activity(prices=[0.20, 0.25, 0.32, 0.41, 0.50, 0.60])  # +0.40 net
        r = score(a)
        names = {s.name for s in r.signals}
        self.assertIn("abs_move", names)

    def test_volume_spike_fires(self):
        a = activity(
            prices=[0.5, 0.5, 0.5, 0.5, 0.5],
            volumes=[10, 12, 11, 9, 200],  # 200 vs median ~11 -> ratio ~18
        )
        r = score(a)
        names = {s.name for s in r.signals}
        self.assertIn("volume_spike", names)

    def test_wallet_concentration_fires(self):
        a = activity(prices=[0.5, 0.5, 0.5], top_wallet_share=0.8)
        r = score(a)
        names = {s.name for s in r.signals}
        self.assertIn("wallet_concentration", names)

    def test_short_series_skipped_gracefully(self):
        a = activity(prices=[0.5])  # too few points
        r = score(a)
        self.assertEqual(r.signals, [])

    def test_kalshi_without_wallets_still_scores(self):
        # Anonymous platform: no top_wallet_share, but price signals still work.
        a = activity(
            prices=[0.20, 0.30, 0.45, 0.60, 0.75],  # big net move
            platform="kalshi", top_wallet_share=None,
        )
        r = score(a)
        self.assertNotIn("wallet_concentration", {s.name for s in r.signals})
        self.assertIn("abs_move", {s.name for s in r.signals})


class ScoringTests(unittest.TestCase):
    def test_strength_capped_and_contribution(self):
        # Wallet share of 1.0 vs threshold 0.5 -> strength 2.0, weight 1.5 -> 3.0.
        a = activity(prices=[0.5, 0.5, 0.5], top_wallet_share=1.0)
        r = score(a)
        sig = next(s for s in r.signals if s.name == "wallet_concentration")
        self.assertAlmostEqual(sig.strength, 2.0)
        self.assertAlmostEqual(sig.contribution, 3.0)

    def test_tiers(self):
        # Strong jump + big move + total wallet dominance -> high tier.
        a = activity(
            prices=[0.50, 0.505, 0.50, 0.505, 0.50, 0.95],
            top_wallet_share=1.0,
        )
        r = score(a)
        self.assertEqual(r.tier, "high")
        self.assertGreaterEqual(r.lead_score, 3.0)

    def test_reason_is_human_readable(self):
        a = activity(prices=[0.20, 0.30, 0.45, 0.60, 0.75])
        r = score(a)
        self.assertIn("Large net move", r.reason)


class RunAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.result_input = {
            "window_days": 14,
            "activity": [
                activity(prices=[0.50, 0.505, 0.50, 0.505, 0.50, 0.95],
                         top_wallet_share=1.0, market_id="hot", title="Hot market"),
                activity(prices=[0.50, 0.50, 0.50, 0.50], market_id="calm",
                         title="Calm market"),
                activity(prices=[0.20, 0.25, 0.32, 0.41, 0.50], market_id="mover",
                         title="Mover"),
            ],
        }

    def test_run_leads_sorts_and_counts(self):
        res = run_leads(self.result_input, {})
        scores = [r["lead_score"] for r in res["leads"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(res["leads"][0]["market_id"], "hot")
        self.assertEqual(sum(res["counts"].values()), 3)
        self.assertGreaterEqual(res["counts"]["high"], 1)

    def test_settings_override_thresholds(self):
        settings = {"anomaly": {"thresholds": {"abs_move": 0.01}}}
        weights, thresholds, tiers = anomaly_config(settings)
        self.assertEqual(thresholds["abs_move"], 0.01)
        # Other thresholds keep their defaults.
        self.assertEqual(thresholds["volume_spike"], DEFAULT_SIGNAL_THRESHOLDS["volume_spike"])

    def test_write_leads_files(self):
        res = run_leads(self.result_input, {})
        with tempfile.TemporaryDirectory() as d:
            json_path, md_path = write_leads(res, output_dir=d)
            self.assertTrue(os.path.exists(json_path))
            with open(md_path, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("FMCC Prediction-Market Leads", md)
        self.assertIn("Lead, not a finding", md)


if __name__ == "__main__":
    unittest.main()
