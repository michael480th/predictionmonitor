"""Tests for pagination and catalog orchestration, using stubbed HTTP."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor import catalog  # noqa: E402
from predictionmonitor.adapters import kalshi, polymarket  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class PolymarketPaginationTests(unittest.TestCase):
    def test_stops_when_page_short(self):
        rows = load_fixture("polymarket_markets.json")
        # One page shorter than page_size -> single fetch, then stop.
        with mock.patch.object(polymarket, "get_json", return_value=rows) as gj:
            adapter = polymarket.PolymarketAdapter(
                base_url="https://x.test", page_size=500
            )
            markets = list(adapter.iter_markets())
        self.assertEqual(gj.call_count, 1)
        # 3 valid rows (one malformed skipped).
        self.assertEqual(len(markets), 3)

    def test_paginates_until_short_page(self):
        full = load_fixture("polymarket_markets.json")[:1]  # 1 valid row
        pages = [full, full, []]  # two full pages of size 1, then empty
        with mock.patch.object(polymarket, "get_json", side_effect=pages) as gj:
            adapter = polymarket.PolymarketAdapter(
                base_url="https://x.test", page_size=1
            )
            markets = list(adapter.iter_markets())
        self.assertEqual(gj.call_count, 3)
        self.assertEqual(len(markets), 2)


class KalshiPaginationTests(unittest.TestCase):
    def test_follows_cursor_then_stops(self):
        payload = load_fixture("kalshi_markets.json")
        page1 = {"markets": payload["markets"], "cursor": "next"}
        page2 = {"markets": payload["markets"], "cursor": ""}
        with mock.patch.object(kalshi, "get_json", side_effect=[page1, page2]) as gj:
            adapter = kalshi.KalshiAdapter(base_url="https://x.test")
            markets = list(adapter.iter_markets())
        self.assertEqual(gj.call_count, 2)
        # 2 valid rows per page (one malformed skipped) * 2 pages.
        self.assertEqual(len(markets), 4)


class RunCatalogTests(unittest.TestCase):
    SETTINGS = {
        "http": {},
        "ingestion": {"page_size": 500, "max_pages": 5, "only_open": True},
        "platforms": {
            "polymarket": {"enabled": True, "base_url": "https://pm.test"},
            "kalshi": {"enabled": True, "base_url": "https://k.test"},
        },
        "output": {"dir": "reports"},
    }

    def test_error_in_one_platform_is_isolated(self):
        pm_rows = load_fixture("polymarket_markets.json")

        def pm_get(*a, **k):
            return pm_rows

        def kalshi_boom(*a, **k):
            raise RuntimeError("network down")

        with mock.patch.object(polymarket, "get_json", side_effect=pm_get), \
             mock.patch.object(kalshi, "get_json", side_effect=kalshi_boom):
            result = catalog.run_catalog(
                ["polymarket", "kalshi"], self.SETTINGS
            )

        self.assertEqual(result["counts"]["polymarket"], 3)
        self.assertEqual(result["counts"]["kalshi"], 0)
        self.assertIn("kalshi", result["errors"])
        self.assertEqual(result["total"], 3)

    def test_max_markets_cap(self):
        pm_rows = load_fixture("polymarket_markets.json")
        with mock.patch.object(polymarket, "get_json", return_value=pm_rows):
            markets = catalog.collect_markets(
                "polymarket", self.SETTINGS, max_markets=2
            )
        self.assertEqual(len(markets), 2)

    def test_write_catalog_roundtrip(self):
        result = {
            "generated_at": "2026-06-03",
            "platforms": ["polymarket"],
            "counts": {"polymarket": 1},
            "total": 1,
            "errors": {},
            "markets": [{"platform": "polymarket", "market_id": "1"}],
        }
        with tempfile.TemporaryDirectory() as d:
            path = catalog.write_catalog(result, output_dir=d)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                reloaded = json.load(fh)
            self.assertEqual(reloaded["total"], 1)


if __name__ == "__main__":
    unittest.main()
