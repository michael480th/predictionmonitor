"""Offline tests for Phase 2 relevance scoring + watchlist output."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor.relevance import (  # noqa: E402
    decide,
    filter_markets,
    load_taxonomy,
    score_market,
)
from predictionmonitor.schema import Market, Outcome  # noqa: E402
from predictionmonitor.watchlist import write_watchlist  # noqa: E402

TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "taxonomy.yml"
)


def mk(title, *, platform="polymarket", desc=None, category=None, tags=None):
    return Market(
        platform=platform,
        market_id="m-" + title[:8],
        title=title,
        url="https://example.test/x",
        status="open",
        outcomes=[Outcome("Yes", 0.5), Outcome("No", 0.5)],
        description=desc,
        category=category,
        tags=tags or [],
    )


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tax = load_taxonomy(TAXONOMY_PATH)

    def test_fed_market_matches_rates_bucket(self):
        r = score_market(mk("Will the Fed cut interest rates in June 2026?"), self.tax)
        bucket_keys = {b.bucket for b in r.matched_buckets}
        self.assertIn("rates_and_fed", bucket_keys)
        self.assertGreater(r.score, 0)

    def test_freddie_hits_housing_and_corporate(self):
        r = score_market(mk("Will Freddie Mac exit conservatorship in 2026?"), self.tax)
        bucket_keys = {b.bucket for b in r.matched_buckets}
        self.assertIn("housing_and_gse", bucket_keys)
        self.assertIn("fmcc_corporate", bucket_keys)
        # housing(1.5) + corporate(2.0) weighting -> high score -> watch
        self.assertEqual(decide(r), "watch")

    def test_exclusion_keyword_drops_market(self):
        r = score_market(mk("Who wins the Super Bowl? NFL final"), self.tax)
        r.decision = decide(r)
        self.assertEqual(r.decision, "excluded")
        self.assertTrue(r.excluded_by)

    def test_unrelated_market_ignored(self):
        r = score_market(mk("Will it rain in Paris tomorrow?"), self.tax)
        self.assertEqual(decide(r), "ignore")
        self.assertEqual(r.score, 0)

    def test_word_boundary_avoids_false_positive(self):
        # "fed" must not match inside "fedex" / "feedback".
        r = score_market(mk("Will FedEx and feedback volumes rise?"), self.tax)
        rate_buckets = [b for b in r.matched_buckets if b.bucket == "rates_and_fed"]
        self.assertFalse(any("fed" in b.matched_keywords for b in rate_buckets))

    def test_thresholds_respected(self):
        r = score_market(mk("Mortgage rate above 7%?"), self.tax)  # rates weight 1.0
        # One rate keyword == 1.0 -> review, not watch, with defaults.
        self.assertEqual(
            decide(r, watch_threshold=2.0, review_threshold=1.0), "review"
        )

    def test_reason_is_human_readable(self):
        r = score_market(mk("Fed rate decision and inflation in 2026"), self.tax)
        self.assertIn("Rates & Fed", r.reason)


class FilterAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.tax = load_taxonomy(TAXONOMY_PATH)
        self.markets = [
            mk("Will Freddie Mac be privatized in 2026?"),
            mk("Will the Fed cut rates in June?"),
            mk("Will the Lakers win the NBA finals?"),
            mk("Will SpaceX reach Mars in 2026?"),
        ]

    def test_filter_buckets_markets(self):
        res = filter_markets(self.markets, self.tax)
        self.assertGreaterEqual(res["counts"]["watch"], 1)
        self.assertEqual(res["counts"]["excluded"], 1)   # NBA
        self.assertGreaterEqual(res["counts"]["ignored"], 1)  # SpaceX
        # Watch list sorted by score descending.
        scores = [r["score"] for r in res["watch"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_watchlist_files_written(self):
        res = filter_markets(self.markets, self.tax)
        with tempfile.TemporaryDirectory() as d:
            json_path, md_path = write_watchlist(res, output_dir=d)
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, encoding="utf-8") as fh:
                md = fh.read()
            self.assertIn("FMCC Prediction-Market Watchlist", md)
            self.assertIn("Lead, not a finding", md)

    def test_serialized_result_includes_contribution(self):
        res = filter_markets(self.markets, self.tax)
        for item in res["watch"]:
            for bucket in item["matched_buckets"]:
                self.assertIn("contribution", bucket)


if __name__ == "__main__":
    unittest.main()
