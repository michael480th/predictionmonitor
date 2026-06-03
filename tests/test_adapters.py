"""Offline tests for adapter normalization (no network)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predictionmonitor.adapters.kalshi import KalshiAdapter  # noqa: E402
from predictionmonitor.adapters.polymarket import PolymarketAdapter  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class PolymarketMappingTests(unittest.TestCase):
    def setUp(self):
        self.adapter = PolymarketAdapter(base_url="https://example.test")
        self.rows = load_fixture("polymarket_markets.json")

    def test_maps_core_fields(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertEqual(m.platform, "polymarket")
        self.assertEqual(m.market_id, "512724")
        self.assertEqual(m.title, "Will the Fed cut rates in June 2026?")
        self.assertEqual(m.status, "open")
        self.assertEqual(m.event_title, "Fed decision June 2026")
        self.assertEqual(m.url, "https://polymarket.com/event/fed-june-2026")
        self.assertEqual(m.volume_unit, "usd")
        self.assertAlmostEqual(m.volume, 1284567.12)
        self.assertIn("Fed", m.tags)

    def test_outcomes_zip_names_and_prices(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertEqual([o.name for o in m.outcomes], ["Yes", "No"])
        self.assertAlmostEqual(m.outcomes[0].price, 0.34)
        self.assertAlmostEqual(m.outcomes[1].price, 0.66)

    def test_string_volume_parsed(self):
        m = self.adapter._to_market(self.rows[1])
        self.assertAlmostEqual(m.volume, 44210.0)

    def test_closed_market_status(self):
        m = self.adapter._to_market(self.rows[2])
        self.assertEqual(m.status, "closed")

    def test_malformed_row_skipped(self):
        self.assertIsNone(self.adapter._to_market(self.rows[3]))

    def test_search_text_lowercased_and_joined(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertIn("fed", m.search_text)
        self.assertIn("june 2026", m.search_text)


class KalshiMappingTests(unittest.TestCase):
    def setUp(self):
        self.adapter = KalshiAdapter(base_url="https://example.test")
        self.payload = load_fixture("kalshi_markets.json")
        self.rows = self.payload["markets"]

    def test_maps_core_fields(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertEqual(m.platform, "kalshi")
        self.assertEqual(m.market_id, "FEDDECISION-26JUN-C")
        self.assertEqual(m.event_id, "FEDDECISION-26JUN")
        self.assertEqual(m.status, "open")
        self.assertEqual(m.category, "Economics")
        self.assertEqual(m.volume_unit, "contracts")
        self.assertEqual(m.url, "https://kalshi.com/markets/FEDDECISION-26JUN")

    def test_title_includes_subtitle(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertEqual(m.title, "Fed funds rate cut in June 2026 — 25+ bps cut")

    def test_last_price_to_probability(self):
        m = self.adapter._to_market(self.rows[0])
        self.assertAlmostEqual(m.outcomes[0].price, 0.31)  # Yes
        self.assertAlmostEqual(m.outcomes[1].price, 0.69)  # No

    def test_falls_back_to_bid_ask_midpoint(self):
        m = self.adapter._to_market(self.rows[1])  # no last_price
        self.assertAlmostEqual(m.outcomes[0].price, 0.50)  # (48+52)/200
        self.assertAlmostEqual(m.outcomes[1].price, 0.50)

    def test_malformed_row_skipped(self):
        self.assertIsNone(self.adapter._to_market(self.rows[2]))


if __name__ == "__main__":
    unittest.main()
