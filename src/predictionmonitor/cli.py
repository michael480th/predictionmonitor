"""Command-line entrypoint: `python -m predictionmonitor`."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from predictionmonitor.catalog import (
    load_settings,
    run_catalog,
    write_catalog,
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
