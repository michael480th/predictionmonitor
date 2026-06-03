# PredictionMonitor

An open-source daily scanner that catalogs markets/events on public prediction
platforms, filters them down to ones relevant to Freddie Mac (FMCC), and (in
later phases) flags **anomalous trading patterns** on those markets as **leads
for Compliance to investigate**.

> **Framing & guardrails.** This tool is a *public indicator*, not an
> accusation engine. Prediction-market trades are pseudonymous (Polymarket) or
> fully anonymous (Kalshi), so the scanner can **never** identify *who* is
> trading. Its output is always "this FMCC-relevant market traded in a
> statistically unusual way around a date that matters" — a **lead, not a
> finding**. We use **public data only**, respect each platform's API Terms of
> Service and rate limits, and store no personally identifiable information.
> Polymarket wallet addresses (on-chain, public, pseudonymous) are used only as
> opaque cluster keys for pattern detection, never attributed to a person.

## Status

| Phase | Scope | State |
|------|-------|-------|
| 0 | Project scaffold, config, FMCC taxonomy, normalized schema | ✅ done |
| 1 | Catalog ingestion: Polymarket + Kalshi → normalized markets | ✅ done |
| 2 | Relevance filtering (taxonomy scoring) → watchlist | ⬜ planned |
| 3 | Activity collection (price/volume/trade/wallet time series) | ⬜ planned |
| 4 | Anomaly detection + lead scoring | ⬜ planned |
| 5 | Daily report + GitHub Actions cron | ⬜ planned |
| 6 | Backtest / threshold tuning | ⬜ planned |

## Data sources (Phase 1)

| Platform | Endpoint used | Auth | Notes |
|---|---|---|---|
| Polymarket | [Gamma API](https://gamma-api.polymarket.com) `/markets`, `/events` | none | On-chain (Polygon); richest data for later phases |
| Kalshi | [Trade API v2](https://api.elections.kalshi.com/trade-api/v2) `/markets`, `/events` | none for reads* | US, CFTC-regulated; strong rates/Fed/housing coverage |

\* Kalshi market/event reads are public. If you hit auth or rate limits, set
`KALSHI_API_KEY_ID` / `KALSHI_API_PRIVATE_KEY` (see `.env.example`) and the
adapter will sign requests.

## Quick start

```bash
pip install -r requirements.txt

# Pull the full open-market catalog from both platforms into reports/
python -m predictionmonitor catalog

# One platform, capped for a fast smoke test
python -m predictionmonitor catalog --platform polymarket --max 200

# Inspect output
ls reports/
```

`catalog` writes `reports/catalog-YYYY-MM-DD.json` (normalized markets) plus a
short summary to stdout.

> **Network note.** The two API hosts must be reachable. In a sandbox with a
> restricted egress allowlist they may be blocked; the scanner runs fully in
> GitHub Actions (Phase 5) or any host with open network. Unit tests run
> offline against captured sample payloads.

## Tests

```bash
python -m pytest -q          # or: python -m unittest discover -s tests
```

## Layout

```
src/predictionmonitor/
  schema.py            normalized Market model + helpers
  http.py              shared HTTP session (retries, timeout, UA)
  adapters/
    base.py            Adapter interface
    polymarket.py      Polymarket Gamma API adapter
    kalshi.py          Kalshi Trade API v2 adapter
  catalog.py           orchestrates ingestion across adapters
  cli.py               `python -m predictionmonitor`
config/
  taxonomy.yml         FMCC relevance taxonomy (used in Phase 2)
  settings.yml         runtime settings
tests/                 offline unit tests + fixtures
```
