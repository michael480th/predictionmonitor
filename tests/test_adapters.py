"""Offline tests for adapter normalization (no network)."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from requests.exceptions import HTTPError  # noqa: E402

from predictionmonitor.adapters.kalshi import KalshiAdapter  # noqa: E402
from predictionmonitor.adapters.polymarket import PolymarketAdapter  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


class _FakeSession:
    """Records GET params and replays a queued list of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.urls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(params or {})
        self.urls.append(url)
        return self._responses.pop(0)


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


class PolymarketPaginationTests(unittest.TestCase):
    def _adapter(self, session):
        return PolymarketAdapter(
            base_url="https://example.test",
            session=session,
            page_size=100,
            max_pages=200,
        )

    def test_requests_highest_volume_first(self):
        session = _FakeSession([_FakeResponse([])])  # empty -> stop immediately
        list(self._adapter(session).iter_markets())
        self.assertEqual(session.calls[0].get("order"), "volumeNum")
        self.assertEqual(session.calls[0].get("ascending"), "false")

    def test_stops_gracefully_on_offset_ceiling_422(self):
        # 100 full pages walk offset 0..9900; the request at offset 10000 then
        # 422s ("offset too large") and must end iteration cleanly — no raise,
        # keeping every market gathered so far.
        rows = load_fixture("polymarket_markets.json")
        full_page = [rows[0]] * 100  # a full page forces another request
        session = _FakeSession(
            [_FakeResponse(full_page)] * 100 + [_FakeResponse("offset too large", 422)]
        )
        markets = list(self._adapter(session).iter_markets())
        self.assertEqual(len(markets), 100 * 100)
        self.assertEqual(session.calls[-1].get("offset"), 10000)

    def test_stops_gracefully_on_lowered_offset_ceiling_422(self):
        # Regression: Gamma tightened its offset ceiling to ~2k. A 422 below the
        # old 10k mark must still be treated as end-of-pagination so the
        # high-volume markets already collected are kept, not discarded.
        rows = load_fixture("polymarket_markets.json")
        full_page = [rows[0]] * 100
        session = _FakeSession(
            [_FakeResponse(full_page)] * 21 + [_FakeResponse("offset too large", 422)]
        )
        markets = list(self._adapter(session).iter_markets())
        self.assertEqual(len(markets), 21 * 100)
        self.assertEqual(session.calls[-1].get("offset"), 2100)

    def test_first_page_422_propagates(self):
        # A 422 on the very first request is a bad request, not the pagination
        # ceiling — surface it rather than silently returning an empty catalog.
        session = _FakeSession([_FakeResponse("bad request", 422)])
        with self.assertRaises(HTTPError):
            list(self._adapter(session).iter_markets())

    def test_non_ceiling_http_error_propagates(self):
        session = _FakeSession([_FakeResponse("boom", 500)])
        with self.assertRaises(HTTPError):
            list(self._adapter(session).iter_markets())


class PolymarketDiscoveryTests(unittest.TestCase):
    def _adapter(self, session):
        return PolymarketAdapter(
            base_url="https://example.test", session=session,
            page_size=100, max_pages=200,
        )

    def _event(self, markets):
        return {"id": "e1", "slug": "freddie", "title": "Freddie Mac",
                "markets": markets}

    def test_maps_nested_search_markets_with_event_context(self):
        ev = self._event([{
            "id": "100", "question": "Will Freddie Mac IPO?",
            "outcomes": '["Yes","No"]', "outcomePrices": '["0.1","0.9"]',
            "conditionId": "0xabc", "clobTokenIds": '["t1","t2"]',
            "closed": False, "active": True, "volumeNum": 1234.5, "slug": "q",
        }])
        session = _FakeSession([_FakeResponse({"events": [ev]})])
        out = list(self._adapter(session).discover_markets(["freddie mac"]))
        self.assertEqual(len(out), 1)
        m = out[0]
        self.assertEqual(m.market_id, "100")
        # Event context is attached so URL/title/grouping work.
        self.assertEqual(m.event_title, "Freddie Mac")
        self.assertEqual(m.url, "https://polymarket.com/event/freddie")
        # Identifiers Phase 3 needs survive the search shape.
        self.assertEqual(m.platform_meta.get("condition_id"), "0xabc")
        self.assertEqual(m.platform_meta.get("clob_token_ids"), ["t1", "t2"])
        # The search hits public-search, not /markets.
        self.assertIn("public-search", session.urls[-1])

    def test_skips_closed_and_dedupes_across_terms(self):
        ev = self._event([
            {"id": "1", "question": "closed q", "closed": True, "active": True},
            {"id": "2", "question": "open q", "closed": False, "active": True},
            {"id": "2", "question": "dup q", "closed": False, "active": True},
        ])
        # Two terms -> two identical responses; the open market is yielded once.
        session = _FakeSession([_FakeResponse({"events": [ev]})] * 2)
        out = list(self._adapter(session).discover_markets(["a", "b"]))
        self.assertEqual([m.market_id for m in out], ["2"])

    def test_one_bad_term_does_not_sink_discovery(self):
        ev = self._event([{"id": "9", "question": "open q",
                           "closed": False, "active": True}])
        session = _FakeSession([
            _FakeResponse("boom", 500),          # first term errors
            _FakeResponse({"events": [ev]}),      # second term succeeds
        ])
        out = list(self._adapter(session).discover_markets(["bad", "good"]))
        self.assertEqual([m.market_id for m in out], ["9"])


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
