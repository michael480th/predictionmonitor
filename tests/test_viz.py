"""Offline tests for the dependency-free SVG primitives."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import viz  # noqa: E402


class SparklineTests(unittest.TestCase):
    def test_draws_path_for_series(self):
        svg = viz.sparkline([0.2, 0.25, 0.5, 0.48], highlight_index=2)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("<path", svg)
        self.assertIn("<circle", svg)        # highlight marker present
        self.assertTrue(svg.rstrip().endswith("</svg>"))

    def test_handles_too_few_points(self):
        svg = viz.sparkline([None, 0.5])
        self.assertIn("no data", svg)
        self.assertNotIn("<path", svg)

    def test_skips_none_values(self):
        # Should not raise and should still draw a path.
        svg = viz.sparkline([0.1, None, 0.3, None, 0.9])
        self.assertIn("<path", svg)


class TimelineTests(unittest.TestCase):
    def test_empty_when_no_times(self):
        svg = viz.event_timeline([{"n": 1, "at": None, "tier": "high", "score": 5}])
        self.assertIn("No timed events", svg)

    def test_renders_markers(self):
        events = [
            {"n": 1, "at": "2026-05-20T00:00:00+00:00", "tier": "high", "score": 6},
            {"n": 2, "at": "2026-05-28T00:00:00+00:00", "tier": "medium", "score": 2},
        ]
        svg = viz.event_timeline(events)
        self.assertIn("<circle", svg)
        self.assertIn(viz.TIER_COLORS["high"], svg)
        self.assertIn("May", svg)            # date tick labels


class DailyBarsTests(unittest.TestCase):
    def test_empty(self):
        self.assertIn("No history", viz.daily_bars([]))

    def test_stacks_high_and_medium(self):
        days = [
            {"date": "2026-06-01", "high": 1, "medium": 2},
            {"date": "2026-06-02", "high": 3, "medium": 0},
        ]
        svg = viz.daily_bars(days)
        self.assertIn("<rect", svg)
        self.assertIn(viz.TIER_COLORS["high"], svg)
        self.assertIn(viz.TIER_COLORS["medium"], svg)
        self.assertIn("06-01", svg)          # date label MM-DD


if __name__ == "__main__":
    unittest.main()
