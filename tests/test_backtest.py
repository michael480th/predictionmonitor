"""Offline tests for Phase 6 calibration / backtest."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import backtest  # noqa: E402


def market(prices, *, volumes=None, top_wallet_share=None, mid="m"):
    points = []
    for i, p in enumerate(prices):
        pt = {"t": f"t{i}", "price": p}
        if volumes is not None:
            pt["volume"] = volumes[i]
        points.append(pt)
    stats = {}
    if top_wallet_share is not None:
        stats["top_wallet_share"] = top_wallet_share
    return {"platform": "polymarket", "market_id": mid,
            "title": mid, "url": "u", "decision": "watch", "score": 3.0,
            "price_points": points, "stats": stats}


class MeasureTests(unittest.TestCase):
    def test_measures_jump_and_move(self):
        m = backtest.measure_activity(
            market([0.50, 0.505, 0.50, 0.505, 0.50, 0.85])
        )
        self.assertAlmostEqual(m["price_jump_abs"], 0.35, places=2)
        self.assertGreater(m["price_jump_z"], 4)
        self.assertAlmostEqual(m["abs_move"], 0.35, places=2)

    def test_volume_and_wallet(self):
        m = backtest.measure_activity(
            market([0.5, 0.5, 0.5, 0.5, 0.5], volumes=[10, 11, 9, 10, 200],
                   top_wallet_share=0.7)
        )
        self.assertGreater(m["volume_spike"], 10)
        self.assertEqual(m["wallet_concentration"], 0.7)

    def test_flat_series_has_no_jump(self):
        m = backtest.measure_activity(market([0.5, 0.5, 0.5, 0.5]))
        self.assertIsNone(m["price_jump_abs"])
        self.assertEqual(m["abs_move"], 0.0)


class DistributionTests(unittest.TestCase):
    def test_percentiles(self):
        d = backtest.distribution([float(i) for i in range(1, 101)])
        self.assertEqual(d["n"], 100)
        self.assertEqual(d["min"], 1)
        self.assertEqual(d["max"], 100)
        self.assertTrue(89 <= d["p90"] <= 92)

    def test_drops_none_and_inf(self):
        d = backtest.distribution([None, float("inf"), 0.2, 0.4])
        self.assertEqual(d["n"], 2)

    def test_empty(self):
        self.assertEqual(backtest.distribution([None])["n"], 0)


class RunBacktestTests(unittest.TestCase):
    def setUp(self):
        # 12 calm markets + 2 jumpy ones -> p90 should sit above the calm noise.
        self.activities = [
            market([0.50, 0.50, 0.501, 0.50, 0.499, 0.50], mid=f"calm{i}")
            for i in range(12)
        ] + [
            market([0.50, 0.505, 0.50, 0.505, 0.50, 0.85], mid="jump1"),
            market([0.20, 0.22, 0.21, 0.20, 0.21, 0.62], mid="jump2"),
        ]

    def test_structure_and_current_tiers(self):
        res = backtest.run_backtest(self.activities, {}, n_inputs=1)
        self.assertEqual(res["n_markets"], 14)
        self.assertIn("price_jump_abs", res["distributions"])
        self.assertIn("price_jump_abs", res["sweeps"])
        # current high tier should flag the 2 jumpy markets
        self.assertGreaterEqual(res["current"]["tiers"]["high"], 2)

    def test_sweep_monotonic_nonincreasing(self):
        res = backtest.run_backtest(self.activities, {}, n_inputs=1)
        fires = [r["fires"] for r in res["sweeps"]["price_jump_abs"]]
        # raising the threshold can only reduce (or keep) the fire count
        self.assertEqual(fires, sorted(fires, reverse=True))

    def test_suggestions_emitted_with_enough_data(self):
        res = backtest.run_backtest(self.activities, {}, n_inputs=1)
        # >=10 markets have a measurable abs_move -> a suggestion appears
        self.assertIn("abs_move", res["suggested"]["thresholds"])

    def test_write_files(self):
        res = backtest.run_backtest(self.activities, {}, n_inputs=1)
        with tempfile.TemporaryDirectory() as d:
            jp, mp = backtest.write_backtest(res, output_dir=d)
            self.assertTrue(os.path.exists(jp))
            with open(mp, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("FMCC Lead Calibration", md)
        self.assertIn("Signal value distributions", md)


if __name__ == "__main__":
    unittest.main()
