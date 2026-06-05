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
| 2 | Relevance filtering (taxonomy scoring) → watchlist | ✅ done |
| 3 | Activity collection (price/volume/trade/wallet time series) | ✅ done |
| 4 | Anomaly detection + lead scoring | ✅ done |
| 5 | Daily report + GitHub Actions cron | ✅ done |
| 6 | Backtest / threshold tuning | ✅ done |
| 7 | Arbitrage / market-maker demotion (cut structural false positives) | ✅ done |

## Data sources (Phase 1)

| Platform | Endpoint used | Auth | Notes |
|---|---|---|---|
| Polymarket | [Gamma API](https://gamma-api.polymarket.com) `/markets`, `/events` | none | On-chain (Polygon); richest data for later phases |
| Kalshi | [Trade API v2](https://api.elections.kalshi.com/trade-api/v2) `/markets`, `/events` | none for reads* | US, CFTC-regulated; strong rates/Fed/housing coverage |

\* Kalshi market/event reads are public. If you hit auth or rate limits, set
`KALSHI_API_KEY_ID` / `KALSHI_API_PRIVATE_KEY` (see `.env.example`) and the
adapter will sign requests.

**Phase 3 activity** uses additional public endpoints: Polymarket price history
via the [CLOB API](https://clob.polymarket.com) `/prices-history` and trades via
the [Data API](https://data-api.polymarket.com) `/trades`; Kalshi price/volume
via `/series/{series}/markets/{ticker}/candlesticks` and `/markets/trades`. Hosts
are configurable per platform in `config/settings.yml`.

## Quick start

```bash
pip install -r requirements.txt

# Pull the full open-market catalog from both platforms into reports/
python -m predictionmonitor catalog

# One platform, capped for a fast smoke test
python -m predictionmonitor catalog --platform polymarket --max 200

# Score the latest catalog against the FMCC taxonomy -> watchlist
python -m predictionmonitor filter

# Collect recent price/volume/trade/wallet activity for the watchlisted markets
python -m predictionmonitor activity                 # watched markets, 14-day window
python -m predictionmonitor activity --include-review # also borderline markets
python -m predictionmonitor activity --window-days 30 --max-markets 5

# Score the collected activity for anomalies -> investigation leads
python -m predictionmonitor leads

# ...or run the whole pipeline in one shot and get a daily digest
python -m predictionmonitor daily --window-days 30

# Inspect output
ls reports/
```

`catalog` writes `reports/catalog-YYYY-MM-DD.json` (normalized markets) plus a
short summary to stdout. `filter` scores the newest catalog (or `--catalog
PATH`) against `config/taxonomy.yml` and writes:

- `reports/watchlist-YYYY-MM-DD.json` — machine-readable watch/review/ignore
  buckets with per-market scores and matched keywords
- `reports/watchlist-YYYY-MM-DD.md` — a reviewer-friendly report

Each market's score is `sum over taxonomy buckets of weight * (distinct
keywords matched)`, where a keyword found only in the description/tags counts
`description_weight` (default 0.5) of a title/event hit — so boilerplate that
merely name-drops "Federal Reserve" doesn't qualify a crypto market. A market is
**watched** at score ≥ `watch_threshold`,
flagged for **review** at ≥ `review_threshold`, **excluded** if it hits an
exclusion keyword, else **ignored**. Thresholds live in `config/settings.yml`.
Scoring is intentionally simple and explainable — every decision carries the
keywords that produced it.

`activity` (Phase 3) takes the newest watchlist (and the catalog it came from,
for platform identifiers) and pulls recent **public** activity for each watched
market — a price/probability time series, traded volume, and individual trades —
then writes:

- `reports/activity-YYYY-MM-DD.json` — per-market `price_points`, `trades`,
  `wallet_clusters`, and summary `stats` (last price, window change, max single
  step, trade count, distinct wallets, top-wallet volume share)
- `reports/activity-YYYY-MM-DD.md` — a reviewer-friendly summary table

These metrics are the raw inputs Phase 4 will score for anomalies. Collection is
resilient: if one feed fails for a market (e.g. an endpoint is unreachable), the
error is recorded on that market and the rest still land. Window, resolution,
and trade caps live under `activity:` in `config/settings.yml`.

> **Wallets & privacy.** Polymarket trades carry an on-chain proxy-wallet
> address (public, pseudonymous). Adapters immediately reduce each to an
> **opaque, stable cluster key** (a salt-free SHA-256 prefix) so the same wallet
> can be tracked across markets for pattern detection — **raw addresses never
> land in any report and are never attributed to a person.** Kalshi is anonymous,
> so its markets have price/volume/trade series but no wallet clusters.

`leads` (Phase 4) scores the newest activity file for **statistically unusual**
patterns and writes `reports/leads-YYYY-MM-DD.json` + `.md`, bucketed into
high/medium/low tiers. A market's `lead_score` is the sum, over the signals that
fired, of `weight * min(value / threshold, 3)` — so the score scales with how
extreme the anomaly is, capped so no single signal dominates. Signals:

| Signal | Fires when | Notes |
|---|---|---|
| `price_jump` | a single step moves ≥ `price_jump_abs` **and** is ≥ `price_jump_z` σ of the market's normal step volatility | the absolute floor stops a frozen market's tiny tick from looking like "many σ"; the σ test stops a slow steady drift from counting as a jump |
| `abs_move` | cumulative \|Δ probability\| over the window ≥ `abs_move` | the market re-priced materially |
| `volume_spike` | peak period volume ÷ median ≥ `volume_spike` | needs per-period volume (Kalshi candlesticks) |
| `wallet_concentration` | top wallet's share of trade volume ≥ `wallet_concentration` | Polymarket only (Kalshi is anonymous) |

Leads are **grouped by event**: sibling markets of one event (e.g. a Fed-rate
ladder whose rungs all re-price on the same news) collapse into a single lead,
headlined by the highest-scoring market, so one event can't flood the report.

Every flagged market carries exactly which signals fired, their measured value,
the threshold crossed, and (for `price_jump`) the σ — same explainable spirit as
Phase 2. Weights and thresholds live under `anomaly:` in `config/settings.yml`;
markets lacking a feed (anonymous, or an unreachable endpoint) simply contribute
fewer signals rather than failing. As always: a high score is a **lead to
investigate, never a finding**.

### Arbitrage / market-maker demotion (Phase 7)

A large class of leads are *structurally* explainable, not insider activity:
arbitrageurs and market-makers who **sweep every outcome of a partition at
near-certain prices** (e.g. buying "No" on all market-cap buckets of an IPO
event at ~99¢) or **hold both sides** of a market to lock a spread. That's a
near-risk-free trade, not a directional bet by someone with information — yet it
shows up as a big, concentrated, FMCC-relevant position and trips the
wallet-based signals.

`arb.py` recognizes that structure from the trade data Phase 3 already collects
(opaque wallet cluster keys + outcome + price across an event's sibling markets)
and is **surgical**:

- it drops the `wallet_concentration` signal of a market whose dominant wallet
  is an arb sweeper, then **re-scores and re-tiers** the lead — so a lead that
  was *only* "one wallet dominated volume" by an arbitrageur falls out of the
  report (a true demotion, shown transparently in an "Auto-demoted" section);
- it **tags** the arb's trades `structural arb` in the flagged-trades list so a
  reviewer isn't misled into reading them as a suspicious actor;
- it leaves **independent price/volume signals untouched** — a genuine abrupt
  price jump keeps its lead, because the arb's near-certain trades didn't cause
  it. Auto-demotion only happens on high-precision structural evidence, never
  merely because a wallet is active, so real directional leads stay intact.

Tuning lives under `arb:` in `config/settings.yml` (`near_certain_price`,
`min_sweep_markets`, `near_certain_share`), and the whole pass can be switched
off with `arb.enabled: false`.

> **Network note.** The two API hosts must be reachable. In a sandbox with a
> restricted egress allowlist they may be blocked; the scanner runs fully in
> GitHub Actions (Phase 5) or any host with open network. Unit tests run
> offline against captured sample payloads.

## Daily scan & automation (Phase 5)

`daily` runs the whole pipeline in one process — catalog → filter → activity →
leads — and writes, for each run:

- `reports/report-YYYY-MM-DD.md` — a top-level Markdown digest
- `reports/report-YYYY-MM-DD.html` — a **single scrollable visual report**
  (self-contained, no dependencies): summary cards → a *"when the jumps
  happened"* timeline across the full N-day window → a price **sparkline** for
  every flagged market with the jump dotted in red → an *"activity over time"*
  cross-day timeline at the bottom
- `reports/timeline.html` — the same cross-day timeline as a standalone page,
  built from `history/events.jsonl`: an append-only, *tracked* log that
  accumulates one record per flagged event each run

```bash
python -m predictionmonitor daily               # all enabled platforms
python -m predictionmonitor daily --platform polymarket --max 3000 --window-days 30
python -m predictionmonitor daily --include-review   # also borderline markets

# Re-render just the cross-day timeline from accumulated history
python -m predictionmonitor timeline

# Calibrate thresholds against collected activity (Phase 6)
python -m predictionmonitor backtest                       # pools all activity-*.json
python -m predictionmonitor backtest --activity reports/activity-2026-06-03.json
```

The charts are pure inline SVG generated in `viz.py` — no matplotlib/JS — so the
HTML opens in any browser and renders as a CI artifact.

`.github/workflows/daily.yml` runs `daily` on a **12:00 UTC cron** (and on
demand via *Run workflow*, with window/cap/include-review inputs). GitHub-hosted
runners have open network, so the Polymarket trades/wallet endpoints and Kalshi
activity — which a restricted dev sandbox may block — are reachable there. Each
run publishes the digest to the **job summary** and uploads every report
(`.html` + `.md` + `.json`) as a 90-day **artifact**. The bulky reports stay
gitignored; only the small `history/events.jsonl` is committed back (on the
default branch, `[skip ci]`) so the cross-day timeline accumulates. Kalshi reads
are public; set the optional `KALSHI_API_KEY_ID` / `KALSHI_API_PRIVATE_KEY` repo
secrets only if you hit rate or auth limits.

## Calibration / backtest (Phase 6)

There's no labelled "suspicious" ground truth for pseudonymous trades, so
`backtest` is a **calibration harness**, not a precision/recall backtest. It
pools the markets from one or more `activity-*.json` files and reports:

- the **distribution** (min/p50/p90/p95/max) of each signal's raw value across
  all markets — so you can see where the mass is;
- **threshold sweeps** — how many markets would fire each signal at a range of
  thresholds;
- **data-driven suggestions** — thresholds at the p90 of observed values (flag
  ≈ the top decile), with the projected high/medium/low tier mix if applied.

It re-scores stored activity only (no re-fetching), so it's fast and offline.
Pool more days for a stabler estimate, then edit `config/settings.yml →
anomaly.thresholds`. Example from a single 76-market day: `price_jump_abs` p90
≈ 0.22 vs. the default 0.10, i.e. the default is deliberately sensitive — the
backtest is how you tighten it once you've seen real volume.

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
  catalog.py           orchestrates ingestion; loads/saves catalog JSON
  relevance.py         Phase 2: taxonomy scoring -> watch/review/ignore
  watchlist.py         writes watchlist JSON + Markdown report
  activity.py          Phase 3: collect activity for watched markets -> report
  anomaly.py           Phase 4: anomaly signals + lead scoring -> leads report
  pipeline.py          Phase 5: daily orchestration + digest report
  report_html.py       visual HTML report (summary, timeline, sparklines)
  history.py           append-only event history + cross-day timeline
  viz.py               dependency-free SVG charts (sparkline, timeline, bars)
  backtest.py          Phase 6: threshold calibration (distributions, sweeps)
  cli.py               CLI: catalog|filter|activity|leads|daily|timeline|backtest
config/
  taxonomy.yml         FMCC relevance taxonomy (used in Phase 2)
  settings.yml         runtime settings
tests/                 offline unit tests + fixtures
```
