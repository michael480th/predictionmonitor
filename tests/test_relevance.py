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

    def test_fed_market_is_out_of_scope(self):
        # General Fed / monetary-policy markets are intentionally not relevant:
        # they're huge and mostly unrelated to Freddie Mac.
        r = score_market(mk("Will the Fed cut interest rates in June 2026?"), self.tax)
        self.assertEqual(r.matched_buckets, [])
        self.assertEqual(r.score, 0)
        self.assertEqual(decide(r), "ignore")

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
        # Substrings must not match: "mortgage rate" shouldn't fire on unrelated
        # words, and a non-FMCC market should score nothing.
        r = score_market(mk("Will FedEx and feedback volumes rise?"), self.tax)
        self.assertEqual(r.matched_buckets, [])
        self.assertEqual(r.score, 0)

    def test_plural_keywords_match_singular_taxonomy_terms(self):
        # Markets say "mortgage rates"/"home prices"; the taxonomy lists the
        # singular forms. Both should still match (and two housing hits -> watch).
        r = score_market(
            mk("Will mortgage rates climb as home prices fall?"), self.tax
        )
        matched = {kw for b in r.matched_buckets for kw in b.matched_keywords}
        self.assertIn("mortgage rate", matched)
        self.assertIn("home prices", matched)
        self.assertEqual(decide(r), "watch")

    def test_thresholds_respected(self):
        r = score_market(mk("Mortgage rate above 7%?"), self.tax)  # housing weight 1.5
        # One housing keyword == 1.5 -> review, not watch, with defaults.
        self.assertEqual(
            decide(r, watch_threshold=2.0, review_threshold=1.0), "review"
        )

    def test_reason_is_human_readable(self):
        r = score_market(mk("Will the 30-year mortgage rate top 7%?"), self.tax)
        self.assertIn("Housing & GSE", r.reason)

    def test_description_only_match_is_weighted_down(self):
        # A market that only mentions a housing term in boilerplate description
        # should fall below the review threshold.
        m = mk(
            "US national Bitcoin reserve before 2027?",
            desc="Resolves YES per the latest mortgage rate data.",
        )
        r = score_market(m, self.tax)  # default description_weight = 0.5
        self.assertEqual(r.score, 0.75)              # 1.5 weight * 0.5 factor
        self.assertEqual(decide(r), "ignore")        # below review threshold (1.0)
        # And the match is recorded as weak, not strong, for transparency.
        bm = next(b for b in r.matched_buckets if b.bucket == "housing_and_gse")
        self.assertIn("mortgage rate", bm.weak_keywords)
        self.assertEqual(bm.strong_keywords, [])

    def test_title_match_unaffected_by_weighting(self):
        # Same keyword in the title keeps full weight.
        r = score_market(mk("Mortgage rate spike ahead?"), self.tax)
        bm = next(b for b in r.matched_buckets if b.bucket == "housing_and_gse")
        self.assertIn("mortgage rate", bm.strong_keywords)
        self.assertEqual(decide(r), "review")        # 1.5 -> review


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
