"""Offline tests for Phase 5 daily orchestration + digest (no network)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import pipeline  # noqa: E402

TAXONOMY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "taxonomy.yml"
)


def _jumpy_points():
    # calm then one big jump -> fires price_jump + abs_move
    return [
        {"t": "a", "price": 0.50},
        {"t": "b", "price": 0.505},
        {"t": "c", "price": 0.50},
        {"t": "d", "price": 0.505},
        {"t": "e", "price": 0.50},
        {"t": "f", "price": 0.95},
    ]


class RunDailyTests(unittest.TestCase):
    def setUp(self):
        self.fake_catalog = {
            "generated_at": "2026-06-03",
            "platforms": ["polymarket"],
            "counts": {"polymarket": 2},
            "total": 2,
            "errors": {},
            "markets": [
                {"platform": "polymarket", "market_id": "1",
                 "title": "Will Freddie Mac be privatized in 2026?",
                 "url": "https://x/1", "status": "open"},
                {"platform": "polymarket", "market_id": "2",
                 "title": "Will it rain in Paris tomorrow?",
                 "url": "https://x/2", "status": "open"},
            ],
        }
        self.fake_activity = {
            "generated_at": "2026-06-03",
            "window_days": 30,
            "counts": {"markets": 1, "missing": 0, "with_errors": 0},
            "missing": [], "errors": {},
            "activity": [
                {"platform": "polymarket", "market_id": "1",
                 "title": "Will Freddie Mac be privatized in 2026?",
                 "url": "https://x/1", "decision": "watch", "score": 3.5,
                 "event_id": "E1", "event_title": "Freddie Mac privatization",
                 "price_points": _jumpy_points(), "stats": {}},
            ],
        }

    def test_orchestration_writes_all_artifacts(self):
        with mock.patch.object(pipeline, "run_catalog", return_value=self.fake_catalog), \
             mock.patch.object(pipeline, "run_activity", return_value=self.fake_activity), \
             tempfile.TemporaryDirectory() as d:
            summary = pipeline.run_daily(
                {}, output_dir=d, taxonomy_path=TAXONOMY_PATH,
                history_path=os.path.join(d, "events.jsonl"),
            )
            arts = summary["artifacts"]
            for key in ("catalog", "watchlist_md", "activity_json", "leads_md", "digest"):
                self.assertTrue(os.path.exists(arts[key]), key)

        # Freddie Mac market is relevant -> watchlist -> a high lead.
        self.assertGreaterEqual(summary["relevance"]["watch"], 1)
        self.assertGreaterEqual(summary["leads"]["event_counts"]["high"], 1)

    def test_run_activity_called_with_filtered_watchlist(self):
        captured = {}

        def fake_activity(watchlist, markets, settings, **kwargs):
            captured["watch"] = watchlist.get("watch")
            captured["n_markets"] = len(markets)
            return self.fake_activity

        with mock.patch.object(pipeline, "run_catalog", return_value=self.fake_catalog), \
             mock.patch.object(pipeline, "run_activity", side_effect=fake_activity), \
             tempfile.TemporaryDirectory() as d:
            pipeline.run_daily(
                {}, output_dir=d, taxonomy_path=TAXONOMY_PATH,
                history_path=os.path.join(d, "events.jsonl"),
            )

        self.assertEqual(captured["n_markets"], 2)        # both catalog markets passed
        self.assertGreaterEqual(len(captured["watch"]), 1)  # Freddie Mac watched


class DigestRenderTests(unittest.TestCase):
    def _summary(self, top_events):
        return {
            "generated_at": "2026-06-03",
            "platforms": ["polymarket"],
            "catalog": {"total": 3000, "counts": {"polymarket": 3000}, "errors": {}},
            "relevance": {"watch": 48, "review": 28, "ignored": 2911, "excluded": 13},
            "activity": {"markets": 25, "missing": 0, "with_errors": 0},
            "window_days": 30,
            "leads": {
                "event_counts": {"high": len(top_events), "medium": 0, "low": 5},
                "market_counts": {"high": 9, "medium": 0, "low": 60},
                "top_events": top_events,
            },
            "artifacts": {
                "catalog": "reports/catalog-2026-06-03.json",
                "watchlist_md": "reports/watchlist-2026-06-03.md",
                "watchlist_json": "reports/watchlist-2026-06-03.json",
                "activity_md": "reports/activity-2026-06-03.md",
                "activity_json": "reports/activity-2026-06-03.json",
                "leads_md": "reports/leads-2026-06-03.md",
                "leads_json": "reports/leads-2026-06-03.json",
            },
        }

    def test_digest_with_leads(self):
        event = {
            "platform": "polymarket", "event_id": "E1",
            "event_title": "What will Fed Rate hit before 2027?",
            "url": "https://x/fed", "lead_score": 6.01, "tier": "high",
            "n_markets": 20, "n_flagged": 8,
            "headline_market": "Fed lower bound <=0.75%",
            "top_signals": [
                {"label": "Abrupt price jump", "value": 0.31, "threshold": 0.1,
                 "detail": {"sigma": 10.8}},
            ],
        }
        md = pipeline.render_digest(self._summary([event]))
        self.assertIn("FMCC Daily Scan", md)
        self.assertIn("Leads, not findings", md)
        self.assertIn("8/20 markets", md)
        self.assertIn("What will Fed Rate hit before 2027?", md)
        self.assertIn("watchlist-2026-06-03.md", md)  # basename only, no dir

    def test_digest_no_leads(self):
        md = pipeline.render_digest(self._summary([]))
        self.assertIn("No anomalous activity flagged today", md)


if __name__ == "__main__":
    unittest.main()
