"""Offline tests for the visual HTML report and cross-day history."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import history  # noqa: E402
from predictionmonitor.report_html import render_report, write_html_report  # noqa: E402


def _leads_result():
    sig = {"name": "price_jump", "label": "Abrupt price jump", "value": 0.3,
           "threshold": 0.1, "contribution": 4.5,
           "detail": {"sigma": 12.0, "at": "2026-05-25T00:00:00+00:00"}}
    flagged_trade = {
        "t": "2026-05-25T14:21:00+00:00", "usd": 1610.0, "actor_label": "Brave-Honey",
        "action": "bought", "outcome": "Yes", "size": 2610, "price": 0.6168,
        "account_url": "https://polymarket.com/profile/0xabc?tab=activity",
        "market_url": "https://polymarket.com/event/freddie-mac-ipo",
        "tx_url": "https://polygonscan.com/tx/0xtx",
    }
    return {
        "generated_at": "2026-06-03",
        "source_window_days": 30,
        "event_counts": {"high": 1, "medium": 0, "low": 0},
        "counts": {"high": 1, "medium": 0, "low": 0},
        "events": [
            {"platform": "polymarket", "event_id": "E1",
             "event_title": "Freddie Mac IPO <script>",  # exercises escaping
             "url": "https://x/e1", "lead_score": 4.5, "tier": "high",
             "n_markets": 6, "n_flagged": 1, "headline_market": "Cap >= $300B",
             "top_signals": [sig], "flagged_trades": [flagged_trade],
             "members": [
                 {"market_id": "m1", "title": "Cap >= $300B", "url": "https://x/m1",
                  "lead_score": 4.5, "tier": "high", "signals": [sig]},
             ]},
        ],
    }


def _activity_result():
    return {
        "window_days": 30,
        "activity": [
            {"platform": "polymarket", "market_id": "m1", "price_points": [
                {"t": "2026-05-24T00:00:00+00:00", "price": 0.2},
                {"t": "2026-05-25T00:00:00+00:00", "price": 0.5},
                {"t": "2026-05-26T00:00:00+00:00", "price": 0.48},
            ], "trades": [
                {"t": "2026-05-24T12:00:00+00:00", "price": 0.2, "size": 500},
                {"t": "2026-05-25T14:21:00+00:00", "price": 0.6168, "size": 2610},
                {"t": "2026-05-26T09:00:00+00:00", "price": 0.48, "size": 800},
            ]},
        ],
    }


class RenderReportTests(unittest.TestCase):
    def test_renders_and_escapes(self):
        html = render_report(_leads_result(), _activity_result())
        self.assertIn("<!doctype html>", html)
        self.assertIn("Leads, not findings", html)
        self.assertIn("<svg", html)                      # charts + timeline
        self.assertIn("Freddie Mac IPO &lt;script&gt;", html)  # escaped, not raw
        self.assertNotIn("<script>", html)
        self.assertIn("last 30 days", html)              # full N-day window framing

    def test_overview_explainer_and_charts(self):
        html = render_report(_leads_result(), _activity_result())
        # Top overview of the flagged trades.
        self.assertIn("Trades we're flagging", html)
        self.assertIn("$1,610", html)                    # dollar size, formatted
        self.assertIn("Brave-Honey", html)
        self.assertIn("#lead-1", html)                   # jumps to the lead's charts
        # Plain-language explainer of the signals we scan for.
        self.assertIn("What we're scanning for", html)
        self.assertIn("Abrupt price jump", html)
        self.assertIn("Single-wallet concentration", html)
        # Per-market price+volume chart with volume bars derived from trades.
        self.assertIn('id="lead-1"', html)
        self.assertIn("<rect", html)                     # volume bars
        self.assertIn("price and volume history", html)  # chart aria-label

    def test_combined_page_includes_cross_day_timeline(self):
        hist = [
            {"date": "2026-06-02", "platform": "polymarket", "event_id": "E1",
             "event_title": "Freddie Mac IPO", "tier": "high", "lead_score": 4.5,
             "url": "https://x/e1"},
        ]
        html = render_report(_leads_result(), _activity_result(), history=hist)
        self.assertIn("Activity over time", html)        # scroll-down section
        self.assertIn("Flagged events per day", html)
        self.assertIn("Most recurring events", html)

    def test_write_html_report(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_html_report(_leads_result(), _activity_result(), output_dir=d)
            self.assertTrue(path.endswith("report-2026-06-03.html"))
            self.assertTrue(os.path.exists(path))


class HistoryTests(unittest.TestCase):
    def test_event_records(self):
        recs = history.event_records(_leads_result(), run_date="2026-06-03")
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["tier"], "high")
        self.assertEqual(recs[0]["jump_at"], "2026-05-25T00:00:00+00:00")

    def test_append_is_idempotent_per_day(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.jsonl")
            history.append_run(_leads_result(), run_date="2026-06-03", path=path)
            history.append_run(_leads_result(), run_date="2026-06-03", path=path)
            rows = history.load_history(path)
            self.assertEqual(len(rows), 1)               # same day overwrote
            history.append_run(_leads_result(), run_date="2026-06-04", path=path)
            self.assertEqual(len(history.load_history(path)), 2)

    def test_render_timeline(self):
        hist = [
            {"date": "2026-06-01", "platform": "polymarket", "event_id": "E1",
             "event_title": "Fed ladder", "tier": "high", "lead_score": 6.0,
             "url": "https://x/e1"},
            {"date": "2026-06-02", "platform": "polymarket", "event_id": "E1",
             "event_title": "Fed ladder", "tier": "medium", "lead_score": 2.0,
             "url": "https://x/e1"},
        ]
        html = history.render_timeline(hist)
        self.assertIn("FMCC Leads", html)
        self.assertIn("<rect", html)                     # daily bars
        self.assertIn("Fed ladder", html)
        self.assertIn("2 flagged events across 2 scan day(s)", html)


if __name__ == "__main__":
    unittest.main()
