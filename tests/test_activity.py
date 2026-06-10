"""Offline tests for Phase 3 activity collection (no network)."""

import os
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import activity  # noqa: E402
from predictionmonitor.activity import (  # noqa: E402
    MarketActivity,
    WalletCluster,
    _compute_stats,
    _wallet_clusters,
    run_activity,
    write_activity,
)
from predictionmonitor.adapters import kalshi, polymarket  # noqa: E402
from predictionmonitor.schema import Market, PricePoint, Trade, cluster_key  # noqa: E402


def mk(platform, market_id, *, meta=None, title="A market", score=3.0):
    return Market(
        platform=platform,
        market_id=market_id,
        title=title,
        url=f"https://example.test/{market_id}",
        status="open",
        platform_meta=meta or {},
    )


class PolymarketActivityTests(unittest.TestCase):
    def setUp(self):
        self.adapter = polymarket.PolymarketAdapter(base_url="https://gamma.test")
        self.market = mk(
            "polymarket",
            "1",
            meta={"condition_id": "0xabc", "clob_token_ids": ["111", "222"]},
        )

    def test_price_history_parsed_and_window_trimmed(self):
        now = time.time()
        payload = {"history": [
            {"t": now - 3 * 86400, "p": 0.53},      # inside window
            {"t": now - 1 * 86400, "p": "0.61"},    # inside window (string price)
            {"t": now - 99 * 86400, "p": 0.20},     # older than window -> trimmed
        ]}
        with mock.patch.object(polymarket, "get_json", return_value=payload):
            points = self.adapter.fetch_price_history(self.market, window_days=14)
        self.assertEqual(len(points), 2)            # stale point trimmed client-side
        self.assertAlmostEqual(points[1].price, 0.61)
        self.assertTrue(points[0].t.endswith("+00:00"))  # ISO UTC

    def test_interval_covers_window(self):
        self.assertEqual(self.adapter._interval_for_window(1), "1d")
        self.assertEqual(self.adapter._interval_for_window(14), "1m")
        self.assertEqual(self.adapter._interval_for_window(90), "max")

    def test_price_history_requires_token_ids(self):
        bare = mk("polymarket", "2", meta={"condition_id": "0x"})
        with self.assertRaises(ValueError):
            self.adapter.fetch_price_history(bare)

    def test_trades_capture_wallet_tx_and_respect_window(self):
        now = time.time()
        rows = [
            {"proxyWallet": "0xWALLET", "transactionHash": "0xTX1", "side": "BUY",
             "size": 10, "price": 0.5, "timestamp": now - 100,
             "outcome": "Yes", "pseudonym": "Brave-Honey",
             "eventSlug": "freddie-mac-ipo"},
            {"proxyWallet": "0xWALLET", "transactionHash": "0xTX2", "side": "SELL",
             "size": 4, "price": 0.5, "timestamp": now - 200},
            # older than the window -> collection should stop here
            {"proxyWallet": "0xOTHER", "side": "BUY", "size": 99, "price": 0.5,
             "timestamp": now - 999999},
        ]
        with mock.patch.object(polymarket, "get_json", return_value=rows):
            trades = self.adapter.fetch_trades(self.market, window_days=1)
        self.assertEqual(len(trades), 2)  # third is outside the window
        # Opaque cluster key is still kept for pattern detection...
        self.assertEqual(trades[0].wallet, cluster_key("0xWALLET"))
        # ...and the raw wallet + tx are now captured so the outlier can be
        # opened and investigated directly.
        self.assertEqual(trades[0].wallet_address, "0xWALLET")
        self.assertEqual(trades[0].tx_hash, "0xTX1")
        self.assertEqual(trades[0].tx_url, "https://polygonscan.com/tx/0xTX1")
        # The wallet link points at the activity feed (its trade history), not
        # the bare profile landing page.
        self.assertEqual(
            trades[0].account_url,
            "https://polymarket.com/profile/0xWALLET?tab=activity",
        )
        self.assertEqual(trades[0].side, "buy")
        # Plain-English context for the report: which side of the bet, a friendly
        # handle for the wallet, a verb, and a USD notional (size × price).
        self.assertEqual(trades[0].outcome, "Yes")
        self.assertEqual(trades[0].pseudonym, "Brave-Honey")
        self.assertEqual(trades[0].actor_label, "Brave-Honey")
        self.assertEqual(trades[0].action, "bought")
        self.assertEqual(trades[0].usd, 5.0)
        # The outcome links back to the bet's own page.
        self.assertEqual(trades[0].market_slug, "freddie-mac-ipo")
        self.assertEqual(
            trades[0].market_url, "https://polymarket.com/event/freddie-mac-ipo"
        )
        # A wallet with no handle falls back to its address.
        self.assertEqual(trades[1].actor_label, "0xWALLET")

    def test_trades_max_cap(self):
        now = time.time()
        rows = [
            {"proxyWallet": f"0x{i}", "side": "BUY", "size": 1, "price": 0.5,
             "timestamp": now - i}
            for i in range(50)
        ]
        with mock.patch.object(polymarket, "get_json", return_value=rows):
            trades = self.adapter.fetch_trades(self.market, window_days=30, max_trades=5)
        self.assertEqual(len(trades), 5)

    def test_to_market_populates_platform_meta(self):
        raw = {
            "id": "540817",
            "conditionId": "0xdeadbeef",
            "question": "Q?",
            "slug": "q-slug",
            "clobTokenIds": '["111", "222"]',
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.4", "0.6"]',
        }
        m = self.adapter._to_market(raw)
        self.assertEqual(m.platform_meta["condition_id"], "0xdeadbeef")
        self.assertEqual(m.platform_meta["clob_token_ids"], ["111", "222"])


class KalshiActivityTests(unittest.TestCase):
    def setUp(self):
        self.adapter = kalshi.KalshiAdapter(base_url="https://k.test/trade-api/v2")
        self.market = mk("kalshi", "FED-26JUN-C", meta={"series_ticker": "FED"})

    def test_candlestick_price_and_volume(self):
        payload = {"candlesticks": [
            {"end_period_ts": 1779321605, "price": {"close": 31, "mean": 30}, "volume": 1200},
            {"end_period_ts": 1779408005, "price": {"close": 42}, "volume": 800},
        ]}
        with mock.patch.object(kalshi, "get_json", return_value=payload):
            points = self.adapter.fetch_price_history(self.market, window_days=14)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].price, 0.31)  # cents -> prob
        self.assertAlmostEqual(points[1].volume, 800.0)

    def test_candlestick_requires_series(self):
        bare = mk("kalshi", "X", meta={})
        with self.assertRaises(ValueError):
            self.adapter.fetch_price_history(bare)

    def test_trades_dollars_and_cents_and_anonymous(self):
        now = time.time()
        payload = {"trades": [
            {"yes_price_dollars": "0.74", "count_fp": "21.0", "taker_side": "yes",
             "created_ts": now - 10},
            {"yes_price": 33, "count": 5, "taker_side": "no", "created_ts": now - 20},
        ], "cursor": ""}
        with mock.patch.object(kalshi, "get_json", return_value=payload):
            trades = self.adapter.fetch_trades(self.market, window_days=14)
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(trades[0].price, 0.74)   # dollars
        self.assertAlmostEqual(trades[1].price, 0.33)   # cents
        self.assertIsNone(trades[0].wallet)             # Kalshi is anonymous

    def test_to_market_derives_series_ticker(self):
        raw = {"ticker": "FEDDECISION-26JUN-C", "title": "T", "status": "open"}
        m = self.adapter._to_market(raw)
        self.assertEqual(m.platform_meta["series_ticker"], "FEDDECISION")


class AggregationTests(unittest.TestCase):
    def test_wallet_clusters_aggregate_and_sort(self):
        w = cluster_key("0xbig")
        trades = [
            Trade(t="t", price=0.5, size=10, side="buy", wallet=w),
            Trade(t="t", price=0.5, size=5, side="sell", wallet=w),
            Trade(t="t", price=0.5, size=3, side="buy", wallet=cluster_key("0xsmall")),
            Trade(t="t", price=0.5, size=1, side="buy", wallet=None),  # ignored
        ]
        clusters = _wallet_clusters(trades)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].cluster, w)        # biggest first
        self.assertEqual(clusters[0].trades, 2)
        self.assertAlmostEqual(clusters[0].volume, 15.0)
        self.assertAlmostEqual(clusters[0].buy_volume, 10.0)
        self.assertAlmostEqual(clusters[0].sell_volume, 5.0)

    def test_compute_stats(self):
        points = [
            PricePoint(t="t1", price=0.30, volume=100),
            PricePoint(t="t2", price=0.55, volume=50),
            PricePoint(t="t3", price=0.50, volume=None),
        ]
        trades = [Trade(t="t", price=0.5, size=10, side="buy", wallet=cluster_key("0xa"))]
        clusters = _wallet_clusters(trades)
        stats = _compute_stats(points, trades, clusters)
        self.assertEqual(stats["n_price_points"], 3)
        self.assertAlmostEqual(stats["price_change"], 0.20)   # 0.50 - 0.30
        self.assertAlmostEqual(stats["max_step"], 0.25)       # 0.30 -> 0.55
        self.assertAlmostEqual(stats["series_volume"], 150.0)
        self.assertAlmostEqual(stats["top_wallet_share"], 1.0)
        # Absolute-money tripwire inputs: notional = size × price = 10 × 0.5 = $5.
        self.assertAlmostEqual(stats["max_trade_usd"], 5.0)
        self.assertAlmostEqual(stats["top_wallet_usd"], 5.0)

    def test_compute_stats_material_money(self):
        # Two trades from one wallet ($1,200 + $800) and one from another ($300).
        # max_trade_usd is the single largest; top_wallet_usd sums per wallet.
        w_big = cluster_key("0xbig")
        trades = [
            Trade(t="t", price=0.60, size=2000, wallet=w_big),          # $1,200
            Trade(t="t", price=0.40, size=2000, wallet=w_big),          # $800
            Trade(t="t", price=0.30, size=1000, wallet=cluster_key("0xsmall")),  # $300
        ]
        stats = _compute_stats([], trades, _wallet_clusters(trades))
        self.assertAlmostEqual(stats["max_trade_usd"], 1200.0)
        self.assertAlmostEqual(stats["top_wallet_usd"], 2000.0)  # 1200 + 800

    def test_suspicious_trades_floor_and_ranking(self):
        from predictionmonitor.activity import _suspicious_trades

        trades = [
            Trade(t="t", price=0.50, size=400, wallet=None),   # $200
            Trade(t="t", price=0.20, size=100, wallet=None),   # $20  -> below floor
            Trade(t="t", price=0.90, size=1000, wallet=None),  # $900
            Trade(t="t", price=None, size=5000, wallet=None),  # unvalued -> skipped
        ]
        picked = _suspicious_trades(trades, min_usd=100.0)
        # Only the two trades worth >= $100, biggest dollars first.
        self.assertEqual([t["usd"] for t in picked], [900.0, 200.0])


class FakeAdapter:
    """Stand-in adapter so run_activity needs no network."""

    def __init__(self, price_points=None, trades=None):
        self._pp = price_points or []
        self._tr = trades or []

    def fetch_price_history(self, market, *, window_days, fidelity_minutes):
        return list(self._pp)

    def fetch_trades(self, market, *, window_days, max_trades):
        return list(self._tr)


class RunActivityTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [
            mk("polymarket", "1", meta={"condition_id": "0x", "clob_token_ids": ["t"]},
               title="Freddie Mac privatized?"),
            mk("kalshi", "FED-C", meta={"series_ticker": "FED"}, title="Fed cut?"),
        ]
        self.watchlist = {
            "watch": [
                {"platform": "polymarket", "market_id": "1", "title": "Freddie Mac privatized?", "score": 3.5},
                {"platform": "kalshi", "market_id": "MISSING", "title": "Gone", "score": 2.0},
            ],
            "review": [
                {"platform": "kalshi", "market_id": "FED-C", "title": "Fed cut?", "score": 1.0},
            ],
        }
        self.fake = FakeAdapter(
            price_points=[PricePoint(t="t1", price=0.3), PricePoint(t="t2", price=0.5)],
            trades=[Trade(t="t", price=0.5, size=4, side="buy", wallet=cluster_key("0xa"))],
        )

    def test_watch_only_join_and_missing(self):
        with mock.patch.object(activity, "_build_adapter", return_value=self.fake):
            res = run_activity(self.watchlist, self.catalog, {})
        self.assertEqual(res["counts"]["markets"], 1)   # only the polymarket watch entry
        self.assertEqual(res["counts"]["missing"], 1)   # the MISSING kalshi watch entry
        self.assertEqual(res["activity"][0]["platform"], "polymarket")
        self.assertEqual(res["activity"][0]["stats"]["n_price_points"], 2)

    def test_include_review_adds_market(self):
        with mock.patch.object(activity, "_build_adapter", return_value=self.fake):
            res = run_activity(self.watchlist, self.catalog, {}, include_review=True)
        platforms = {a["platform"] for a in res["activity"]}
        self.assertEqual(platforms, {"polymarket", "kalshi"})

    def test_max_markets_cap_and_sorting(self):
        with mock.patch.object(activity, "_build_adapter", return_value=self.fake):
            res = run_activity(
                self.watchlist, self.catalog, {}, include_review=True, max_markets=1
            )
        self.assertEqual(res["counts"]["markets"], 1)
        # Highest score first.
        self.assertEqual(res["activity"][0]["score"], 3.5)

    def test_write_activity_files(self):
        with mock.patch.object(activity, "_build_adapter", return_value=self.fake):
            res = run_activity(self.watchlist, self.catalog, {})
        with tempfile.TemporaryDirectory() as d:
            json_path, md_path = write_activity(res, output_dir=d)
            self.assertTrue(os.path.exists(json_path))
            with open(md_path, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("FMCC Prediction-Market Activity", md)
        self.assertIn("Lead, not a finding", md)


if __name__ == "__main__":
    unittest.main()
