"""Command-line entrypoint: `python -m predictionmonitor`."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from predictionmonitor.catalog import (
    latest_catalog_path,
    load_catalog_markets,
    load_settings,
    run_catalog,
    write_catalog,
)
from predictionmonitor.relevance import (
    filter_markets,
    load_taxonomy,
    relevance_thresholds,
)
from predictionmonitor.watchlist import write_watchlist
from predictionmonitor.activity import (
    latest_watchlist_path,
    load_watchlist,
    run_activity,
    write_activity,
)
from predictionmonitor.anomaly import (
    latest_activity_path,
    load_activity,
    run_leads,
    write_leads,
)


def _enabled_platforms(settings: dict) -> list[str]:
    plats = settings.get("platforms", {})
    enabled = [name for name, cfg in plats.items() if cfg.get("enabled", True)]
    return enabled or ["polymarket", "kalshi"]


def _cmd_catalog(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)

    if args.platform:
        platforms = [args.platform]
    else:
        platforms = _enabled_platforms(settings)

    print(f"Cataloging platforms: {', '.join(platforms)}", file=sys.stderr)
    result = run_catalog(platforms, settings, max_markets=args.max)

    output_dir = settings.get("output", {}).get("dir", "reports")
    path = write_catalog(result, output_dir=output_dir)

    # Human summary.
    print(f"\nWrote {result['total']} markets -> {path}")
    for platform, count in result["counts"].items():
        print(f"  {platform:<12} {count} markets")
    if result["errors"]:
        print("\nErrors:", file=sys.stderr)
        for platform, err in result["errors"].items():
            print(f"  {platform}: {err}", file=sys.stderr)
        # Non-zero exit if every requested platform failed.
        if result["total"] == 0:
            return 1
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    output_dir = settings.get("output", {}).get("dir", "reports")

    catalog_path = args.catalog or latest_catalog_path(output_dir)
    if not catalog_path:
        print(
            "No catalog found. Run `predictionmonitor catalog` first, "
            "or pass --catalog PATH.",
            file=sys.stderr,
        )
        return 2

    taxonomy = load_taxonomy(args.taxonomy)
    markets = load_catalog_markets(catalog_path)
    watch_th, review_th = relevance_thresholds(settings)

    print(f"Scoring {len(markets)} markets from {catalog_path}", file=sys.stderr)
    result = filter_markets(
        markets, taxonomy, watch_threshold=watch_th, review_threshold=review_th
    )
    json_path, md_path = write_watchlist(result, output_dir=output_dir)

    counts = result["counts"]
    print(
        f"\nWatch: {counts['watch']}  Review: {counts['review']}  "
        f"Ignored: {counts['ignored']}  Excluded: {counts['excluded']}"
    )
    print(f"Wrote {json_path}\n      {md_path}")
    return 0


def _cmd_activity(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    output_dir = settings.get("output", {}).get("dir", "reports")

    watchlist_path = args.watchlist or latest_watchlist_path(output_dir)
    if not watchlist_path:
        print(
            "No watchlist found. Run `predictionmonitor filter` first, "
            "or pass --watchlist PATH.",
            file=sys.stderr,
        )
        return 2

    catalog_path = args.catalog or latest_catalog_path(output_dir)
    if not catalog_path:
        print(
            "No catalog found (needed for market identifiers). Run "
            "`predictionmonitor catalog` first, or pass --catalog PATH.",
            file=sys.stderr,
        )
        return 2

    watchlist = load_watchlist(watchlist_path)
    catalog_markets = load_catalog_markets(catalog_path)

    print(
        f"Collecting activity for {watchlist_path} "
        f"(identifiers via {catalog_path})",
        file=sys.stderr,
    )
    result = run_activity(
        watchlist,
        catalog_markets,
        settings,
        window_days=args.window_days,
        include_review=args.include_review,
        max_markets=args.max_markets,
        platform=args.platform,
    )
    json_path, md_path = write_activity(result, output_dir=output_dir)

    counts = result["counts"]
    print(
        f"\nCollected activity for {counts['markets']} markets "
        f"({counts['with_errors']} with partial errors, "
        f"{counts['missing']} not found in catalog)"
    )
    print(f"Wrote {json_path}\n      {md_path}")
    return 0


def _cmd_leads(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    output_dir = settings.get("output", {}).get("dir", "reports")

    activity_path = args.activity or latest_activity_path(output_dir)
    if not activity_path:
        print(
            "No activity file found. Run `predictionmonitor activity` first, "
            "or pass --activity PATH.",
            file=sys.stderr,
        )
        return 2

    activity_result = load_activity(activity_path)
    print(f"Scoring activity from {activity_path}", file=sys.stderr)
    result = run_leads(activity_result, settings)
    json_path, md_path = write_leads(result, output_dir=output_dir)

    counts = result["counts"]
    print(
        f"\nHigh: {counts['high']}  Medium: {counts['medium']}  "
        f"Low/none: {counts['low']}"
    )
    print(f"Wrote {json_path}\n      {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictionmonitor",
        description="FMCC-relevant prediction-market scanner.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable info logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cat = sub.add_parser("catalog", help="ingest the open-market catalog")
    cat.add_argument(
        "--platform",
        choices=["polymarket", "kalshi"],
        help="restrict to a single platform (default: all enabled)",
    )
    cat.add_argument(
        "--max",
        type=int,
        default=None,
        help="cap markets per platform (smoke testing)",
    )
    cat.add_argument(
        "--settings",
        default="config/settings.yml",
        help="path to settings.yml",
    )
    cat.set_defaults(func=_cmd_catalog)

    filt = sub.add_parser(
        "filter", help="score a catalog against the FMCC taxonomy -> watchlist"
    )
    filt.add_argument(
        "--catalog",
        help="catalog JSON to score (default: newest in the reports dir)",
    )
    filt.add_argument(
        "--taxonomy", default="config/taxonomy.yml", help="path to taxonomy.yml"
    )
    filt.add_argument(
        "--settings", default="config/settings.yml", help="path to settings.yml"
    )
    filt.set_defaults(func=_cmd_filter)

    act = sub.add_parser(
        "activity",
        help="collect price/volume/trade/wallet activity for watchlisted markets",
    )
    act.add_argument(
        "--watchlist",
        help="watchlist JSON to read (default: newest in the reports dir)",
    )
    act.add_argument(
        "--catalog",
        help="catalog JSON for market identifiers (default: newest)",
    )
    act.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="days of history to pull (default: settings activity.window_days)",
    )
    act.add_argument(
        "--include-review",
        action="store_true",
        help="also collect activity for 'review' (borderline) markets",
    )
    act.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="cap markets collected (smoke testing)",
    )
    act.add_argument(
        "--platform",
        choices=["polymarket", "kalshi"],
        help="restrict to a single platform",
    )
    act.add_argument(
        "--settings", default="config/settings.yml", help="path to settings.yml"
    )
    act.set_defaults(func=_cmd_activity)

    leads = sub.add_parser(
        "leads",
        help="score collected activity for anomalies -> investigation leads",
    )
    leads.add_argument(
        "--activity",
        help="activity JSON to score (default: newest in the reports dir)",
    )
    leads.add_argument(
        "--settings", default="config/settings.yml", help="path to settings.yml"
    )
    leads.set_defaults(func=_cmd_leads)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
