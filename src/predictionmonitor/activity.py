"""Phase 3: collect price/volume/trade/wallet activity for watchlisted markets.

Phase 2 produced a watchlist of FMCC-relevant markets. This phase pulls the
recent *activity* on those markets — a price/probability time series, traded
volume, and (where the platform exposes it) individual trades — so Phase 4 can
look for statistically unusual patterns.

Guardrails (see README). We use public data only. Polymarket trades carry an
on-chain wallet address (public, pseudonymous); adapters convert each to an
opaque, stable cluster key (:func:`schema.cluster_key`) before it reaches this
module, and we only ever aggregate those keys. Raw addresses never land in any
report, and nothing is attributed to a person. Kalshi is anonymous, so it yields
price/volume/trade series but no wallet clusters.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

from predictionmonitor.catalog import _build_adapter
from predictionmonitor.schema import Market, PricePoint, Trade

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result model
# --------------------------------------------------------------------------


@dataclass
class WalletCluster:
    """Aggregated activity for one opaque wallet cluster key on a market."""

    cluster: str                # opaque key from schema.cluster_key
    trades: int
    volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketActivity:
    """Collected activity for a single watchlisted market."""

    platform: str
    market_id: str
    title: str
    url: str
    decision: str               # carried over from the watchlist (watch|review)
    score: float
    window_days: int
    collected_at: str
    event_id: Optional[str] = None      # for grouping sibling markets in Phase 4
    event_title: Optional[str] = None
    price_points: list[PricePoint] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    wallet_clusters: list[WalletCluster] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------


def _wallet_clusters(trades: list[Trade]) -> list[WalletCluster]:
    """Aggregate trades by opaque cluster key, sorted by volume descending."""
    agg: dict[str, dict[str, float]] = {}
    for t in trades:
        if not t.wallet:
            continue
        bucket = agg.setdefault(
            t.wallet, {"trades": 0.0, "volume": 0.0, "buy": 0.0, "sell": 0.0}
        )
        size = t.size or 0.0
        bucket["trades"] += 1
        bucket["volume"] += size
        if t.side in ("buy", "yes"):
            bucket["buy"] += size
        elif t.side in ("sell", "no"):
            bucket["sell"] += size

    clusters = [
        WalletCluster(
            cluster=key,
            trades=int(v["trades"]),
            volume=round(v["volume"], 4),
            buy_volume=round(v["buy"], 4),
            sell_volume=round(v["sell"], 4),
        )
        for key, v in agg.items()
    ]
    clusters.sort(key=lambda c: c.volume, reverse=True)
    return clusters


def _compute_stats(
    price_points: list[PricePoint],
    trades: list[Trade],
    clusters: list[WalletCluster],
    *,
    min_trade_usd: float = 100.0,
) -> dict[str, Any]:
    """Explainable summary metrics — precursors for Phase 4 anomaly scoring."""
    stats: dict[str, Any] = {
        "n_price_points": len(price_points),
        "n_trades": len(trades),
        "n_wallets": len(clusters),
    }

    prices = [p.price for p in price_points if p.price is not None]
    if prices:
        stats["first_price"] = prices[0]
        stats["last_price"] = prices[-1]
        stats["price_change"] = round(prices[-1] - prices[0], 4)
        max_step = max((abs(b - a) for a, b in zip(prices, prices[1:])), default=0.0)
        stats["max_step"] = round(max_step, 4)

    series_volume = [p.volume for p in price_points if p.volume is not None]
    if series_volume:
        stats["series_volume"] = round(sum(series_volume), 4)

    if trades:
        stats["trade_volume"] = round(sum((t.size or 0.0) for t in trades), 4)
        stats["suspicious_trades"] = _suspicious_trades(trades, min_usd=min_trade_usd)
        # Absolute-money tripwire inputs (Phase 4 material_trade/material_wallet):
        # the largest single trade's USD notional, and the largest per-wallet USD
        # flow. On these thin markets a baseline is a few hundred dollars, so a
        # fixed dollar floor catches material capital the σ-based signals miss.
        valued = [t.usd for t in trades if t.usd is not None]
        if valued:
            stats["max_trade_usd"] = round(max(valued), 2)
        by_wallet: dict[str, float] = {}
        for t in trades:
            if t.wallet and t.usd is not None:
                by_wallet[t.wallet] = by_wallet.get(t.wallet, 0.0) + t.usd
        if by_wallet:
            stats["top_wallet_usd"] = round(max(by_wallet.values()), 2)

    if clusters:
        total = sum(c.volume for c in clusters)
        stats["top_wallet_share"] = (
            round(clusters[0].volume / total, 4) if total > 0 else None
        )

    return stats


def _suspicious_trades(
    trades: list[Trade], *, min_usd: float = 100.0, limit: int = 5
) -> list[dict[str, Any]]:
    """The biggest *dollar-value* trades in the window worth opening directly.

    Ranked by USD notional (size × price) and floored at ``min_usd`` so the
    report surfaces real money, not the long tail of tiny trades. Trades whose
    value can't be computed (missing price/size) are skipped. Returned as
    serialized dicts carrying the wallet + tx links (where the platform exposes
    them) so the report can link straight to the trader's positions.
    """
    valued = [t for t in trades if t.usd is not None and t.usd >= min_usd]
    ranked = sorted(valued, key=lambda t: t.usd, reverse=True)[:limit]
    return [t.to_dict() for t in ranked]


def collect_market_activity(
    adapter,
    market: Market,
    decision: str,
    score: float,
    *,
    window_days: int,
    fidelity_minutes: int,
    max_trades: int,
    store_trades: bool,
    min_trade_usd: float = 100.0,
) -> MarketActivity:
    """Collect price history + trades for one market, isolating sub-failures."""
    errors: list[str] = []

    price_points: list[PricePoint] = []
    try:
        price_points = adapter.fetch_price_history(
            market, window_days=window_days, fidelity_minutes=fidelity_minutes
        )
    except Exception as exc:  # one feed failing shouldn't drop the others
        log.warning("price history failed for %s: %s", market.market_id, exc)
        errors.append(f"price_history: {type(exc).__name__}: {exc}")

    trades: list[Trade] = []
    try:
        trades = adapter.fetch_trades(
            market, window_days=window_days, max_trades=max_trades
        )
    except Exception as exc:
        # Surface the response body on HTTP errors — it distinguishes a fixable
        # Cloudflare/UA block from a hard geo/IP block on the trades feed.
        body = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            body = f" | body: {resp.text[:200]!r}"
        log.warning("trades failed for %s: %s%s", market.market_id, exc, body)
        errors.append(f"trades: {type(exc).__name__}: {exc}")

    clusters = _wallet_clusters(trades)
    stats = _compute_stats(price_points, trades, clusters, min_trade_usd=min_trade_usd)

    return MarketActivity(
        platform=market.platform,
        market_id=market.market_id,
        title=market.title,
        url=market.url,
        decision=decision,
        score=score,
        window_days=window_days,
        collected_at=datetime.now(timezone.utc).isoformat(),
        event_id=market.event_id,
        event_title=market.event_title,
        price_points=price_points,
        trades=trades if store_trades else [],
        wallet_clusters=clusters,
        stats=stats,
        errors=errors,
    )


# --------------------------------------------------------------------------
# Orchestration: join watchlist + catalog, collect, assemble
# --------------------------------------------------------------------------


def _select_markets(
    watchlist: dict[str, Any],
    catalog_markets: list[Market],
    *,
    include_review: bool,
    platform: Optional[str],
) -> tuple[list[tuple[Market, str, float]], list[dict[str, Any]]]:
    """Resolve watchlist entries to catalog Market objects (with their meta)."""
    by_id = {(m.platform, m.market_id): m for m in catalog_markets}
    selected: list[tuple[Market, str, float]] = []
    missing: list[dict[str, Any]] = []

    decisions = ["watch"] + (["review"] if include_review else [])
    for decision in decisions:
        for entry in watchlist.get(decision, []):
            key = (entry.get("platform"), entry.get("market_id"))
            if platform and key[0] != platform:
                continue
            market = by_id.get(key)
            if market is None:
                missing.append(
                    {
                        "platform": key[0],
                        "market_id": key[1],
                        "title": entry.get("title"),
                    }
                )
                continue
            selected.append((market, decision, float(entry.get("score", 0.0))))
    return selected, missing


def run_activity(
    watchlist: dict[str, Any],
    catalog_markets: list[Market],
    settings: dict[str, Any],
    *,
    window_days: Optional[int] = None,
    include_review: bool = False,
    max_markets: Optional[int] = None,
    platform: Optional[str] = None,
) -> dict[str, Any]:
    """Collect activity for the watchlisted markets and assemble a result dict."""
    acfg = (settings or {}).get("activity", {})
    window_days = window_days if window_days is not None else acfg.get("window_days", 14)
    fidelity_minutes = acfg.get("fidelity_minutes", 60)
    max_trades = acfg.get("max_trades", 2000)
    store_trades = acfg.get("store_trades", True)
    min_trade_usd = float(acfg.get("min_trade_usd", 100))

    selected, missing = _select_markets(
        watchlist, catalog_markets, include_review=include_review, platform=platform
    )
    if max_markets is not None:
        selected = selected[:max_markets]

    adapters: dict[str, Any] = {}
    activities: list[MarketActivity] = []
    errors: dict[str, str] = {}

    for market, decision, score in selected:
        try:
            adapter = adapters.get(market.platform)
            if adapter is None:
                adapter = _build_adapter(market.platform, settings)
                adapters[market.platform] = adapter
            activities.append(
                collect_market_activity(
                    adapter,
                    market,
                    decision,
                    score,
                    window_days=window_days,
                    fidelity_minutes=fidelity_minutes,
                    max_trades=max_trades,
                    store_trades=store_trades,
                    min_trade_usd=min_trade_usd,
                )
            )
        except Exception as exc:  # adapter construction / unexpected failure
            log.exception("activity collection failed for %s", market.market_id)
            errors[f"{market.platform}:{market.market_id}"] = (
                f"{type(exc).__name__}: {exc}"
            )

    activities.sort(key=lambda a: a.score, reverse=True)

    return {
        "generated_at": date.today().isoformat(),
        "window_days": window_days,
        "include_review": include_review,
        "counts": {
            "markets": len(activities),
            "missing": len(missing),
            "with_errors": sum(1 for a in activities if a.errors),
        },
        "missing": missing,
        "errors": errors,
        "activity": [a.to_dict() for a in activities],
    }


# --------------------------------------------------------------------------
# Loading watchlists + writing the activity report
# --------------------------------------------------------------------------


def latest_watchlist_path(output_dir: str = "reports") -> Optional[str]:
    """Return the newest reports/watchlist-*.json by filename, or None."""
    matches = sorted(glob.glob(os.path.join(output_dir, "watchlist-*.json")))
    return matches[-1] if matches else None


def load_watchlist(path: str) -> dict[str, Any]:
    """Load a saved Phase 2 watchlist JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_activity(result: dict[str, Any], output_dir: str = "reports") -> tuple[str, str]:
    """Write activity-YYYY-MM-DD.json and .md. Returns (json_path, md_path)."""
    os.makedirs(output_dir, exist_ok=True)
    today = result.get("generated_at", date.today().isoformat())
    json_path = os.path.join(output_dir, f"activity-{today}.json")
    md_path = os.path.join(output_dir, f"activity-{today}.md")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(today, result))

    return json_path, md_path


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "—"
    return format(value, spec) if spec else str(value)


def _render_rows(activities: list[dict[str, Any]]) -> str:
    if not activities:
        return "_No activity collected._\n"
    lines = [
        "| Score | Platform | Market | Last | Δ window | Max step | Trades | Wallets | Top wallet |",
        "|------:|----------|--------|-----:|---------:|---------:|-------:|--------:|-----------:|",
    ]
    for a in activities:
        s = a.get("stats", {})
        title = a["title"].replace("|", "\\|")
        flag = " ⚠️" if a.get("errors") else ""
        top = s.get("top_wallet_share")
        lines.append(
            f"| {a['score']:.1f} | {a['platform']} | [{title}]({a['url']}){flag} | "
            f"{_fmt(s.get('last_price'), '.2f')} | "
            f"{_fmt(s.get('price_change'), '+.2f')} | "
            f"{_fmt(s.get('max_step'), '.2f')} | "
            f"{_fmt(s.get('n_trades'))} | "
            f"{_fmt(s.get('n_wallets'))} | "
            f"{(_fmt(round(top * 100, 1), '.1f') + '%') if top is not None else '—'} |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(today: str, result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    activities = result.get("activity", [])
    missing = result.get("missing", [])

    out = [
        f"# FMCC Prediction-Market Activity — {today}\n",
        "> **Lead, not a finding.** This report summarizes *public* trading "
        "activity on FMCC-relevant markets. Unusual numbers here are inputs to "
        "later anomaly scoring, not evidence of wrongdoing. Wallet figures are "
        "opaque cluster keys derived from public on-chain addresses and are "
        "never attributed to a person.\n",
        f"**Window:** last {result.get('window_days')} days · "
        f"**Markets:** {counts.get('markets', 0)} · "
        f"**With partial errors:** {counts.get('with_errors', 0)} · "
        f"**Not found in catalog:** {counts.get('missing', 0)}\n",
        "## Activity\n",
        _render_rows(activities),
    ]
    if missing:
        out.append(
            "\n_Note: "
            + str(len(missing))
            + " watchlisted market(s) were not present in the catalog used and "
            "were skipped. Re-run `catalog` and `filter` so dates align._\n"
        )
    out.append(
        "\n_Kalshi is anonymous, so its rows have no wallet clusters by design._\n"
    )
    return "\n".join(out)
