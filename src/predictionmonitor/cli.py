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
