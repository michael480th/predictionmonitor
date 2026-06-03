"""Phase 2: score catalogued markets against the FMCC taxonomy.

The scorer is deliberately simple and **explainable** — a compliance reviewer
must be able to see exactly *why* a market was put on the watchlist. Each market
gets a score that is the sum, over taxonomy buckets, of
``bucket.weight * (number of distinct bucket keywords matched)``. Markets that
hit an exclusion keyword are dropped regardless of score.

Decisions:
    score >= watch_threshold   -> "watch"     (monitor in later phases)
    score >= review_threshold  -> "review"    (borderline; human triage)
    otherwise                  -> "ignore"
    any exclusion keyword hit  -> "excluded"
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Optional

import yaml

from predictionmonitor.schema import Market

_DEFAULT_TAXONOMY_PATH = os.path.join("config", "taxonomy.yml")

# Decision thresholds (overridable via settings.yml -> relevance:).
DEFAULT_WATCH_THRESHOLD = 2.0
DEFAULT_REVIEW_THRESHOLD = 1.0

# How much a keyword that appears ONLY in a market's description/tags counts
# relative to one in the title/event. Boilerplate descriptions often mention
# "Federal Reserve" etc. in passing, so weighting them down sharpens precision.
DEFAULT_DESCRIPTION_WEIGHT = 0.5


def load_taxonomy(path: str = _DEFAULT_TAXONOMY_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "buckets" not in data:
        raise ValueError(f"taxonomy at {path} has no 'buckets' section")
    return data


@lru_cache(maxsize=4096)
def _compile(keyword: str) -> re.Pattern:
    """Word-boundary matcher for a keyword/phrase (cached)."""
    return re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")


def _matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    return [kw for kw in keywords if _compile(kw).search(text)]


@dataclass
class BucketMatch:
    bucket: str
    label: str
    weight: float
    matched_keywords: list[str]                     # union (title + description-only)
    strong_keywords: list[str] = field(default_factory=list)  # in title/event
    weak_keywords: list[str] = field(default_factory=list)    # description/tags only
    description_weight: float = DEFAULT_DESCRIPTION_WEIGHT

    @property
    def contribution(self) -> float:
        effective = len(self.strong_keywords) + self.description_weight * len(
            self.weak_keywords
        )
        return round(self.weight * effective, 4)


@dataclass
class RelevanceResult:
    platform: str
    market_id: str
    title: str
    url: str
    score: float
    decision: str                                   # watch|review|ignore|excluded
    matched_buckets: list[BucketMatch] = field(default_factory=list)
    excluded_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict drops the computed contribution; add it back for transparency.
        for bm, raw in zip(self.matched_buckets, d["matched_buckets"]):
            raw["contribution"] = bm.contribution
        return d

    @property
    def reason(self) -> str:
        """One-line human explanation of the decision."""
        if self.decision == "excluded":
            return f"excluded by: {', '.join(self.excluded_by)}"
        if not self.matched_buckets:
            return "no FMCC keywords matched"
        parts = [
            f"{bm.label} ({', '.join(bm.matched_keywords)})"
            for bm in self.matched_buckets
        ]
        return "; ".join(parts)


def score_market(
    market: Market,
    taxonomy: dict[str, Any],
    *,
    description_weight: float = DEFAULT_DESCRIPTION_WEIGHT,
) -> RelevanceResult:
    strong_text = market.strong_text
    weak_text = market.weak_text
    exclude_kw = [k.lower() for k in taxonomy.get("exclude_keywords", [])]
    excluded_by = _matched_keywords(market.search_text, exclude_kw)

    matches: list[BucketMatch] = []
    score = 0.0
    for key, bucket in taxonomy.get("buckets", {}).items():
        weight = float(bucket.get("weight", 1.0))
        keywords = bucket.get("keywords", [])
        strong_hits = _matched_keywords(strong_text, keywords)
        weak_only = [
            kw
            for kw in _matched_keywords(weak_text, keywords)
            if kw not in strong_hits
        ]
        if strong_hits or weak_only:
            bm = BucketMatch(
                bucket=key,
                label=bucket.get("label", key),
                weight=weight,
                matched_keywords=strong_hits + weak_only,
                strong_keywords=strong_hits,
                weak_keywords=weak_only,
                description_weight=description_weight,
            )
            matches.append(bm)
            score += bm.contribution

    base = RelevanceResult(
        platform=market.platform,
        market_id=market.market_id,
        title=market.title,
        url=market.url,
        score=round(score, 4),
        decision="ignore",
        matched_buckets=matches,
        excluded_by=excluded_by,
    )
    return base


def decide(
    result: RelevanceResult,
    *,
    watch_threshold: float = DEFAULT_WATCH_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> str:
    if result.excluded_by:
        return "excluded"
    if result.score >= watch_threshold:
        return "watch"
    if result.score >= review_threshold:
        return "review"
    return "ignore"


def filter_markets(
    markets: Iterable[Market],
    taxonomy: dict[str, Any],
    *,
    watch_threshold: float = DEFAULT_WATCH_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    description_weight: float = DEFAULT_DESCRIPTION_WEIGHT,
) -> dict[str, Any]:
    """Score a set of markets and bucket them into watch/review/ignore/excluded.

    Returns a result dict with sorted lists (highest score first) and counts.
    """
    watch: list[RelevanceResult] = []
    review: list[RelevanceResult] = []
    ignored = 0
    excluded = 0

    for market in markets:
        result = score_market(market, taxonomy, description_weight=description_weight)
        result.decision = decide(
            result,
            watch_threshold=watch_threshold,
            review_threshold=review_threshold,
        )
        if result.decision == "watch":
            watch.append(result)
        elif result.decision == "review":
            review.append(result)
        elif result.decision == "excluded":
            excluded += 1
        else:
            ignored += 1

    watch.sort(key=lambda r: r.score, reverse=True)
    review.sort(key=lambda r: r.score, reverse=True)

    return {
        "thresholds": {
            "watch": watch_threshold,
            "review": review_threshold,
        },
        "counts": {
            "watch": len(watch),
            "review": len(review),
            "ignored": ignored,
            "excluded": excluded,
        },
        "watch": [r.to_dict() for r in watch],
        "review": [r.to_dict() for r in review],
    }


def relevance_thresholds(settings: dict[str, Any]) -> tuple[float, float]:
    rel = (settings or {}).get("relevance", {})
    return (
        float(rel.get("watch_threshold", DEFAULT_WATCH_THRESHOLD)),
        float(rel.get("review_threshold", DEFAULT_REVIEW_THRESHOLD)),
    )


def description_weight(settings: dict[str, Any]) -> float:
    rel = (settings or {}).get("relevance", {})
    return float(rel.get("description_weight", DEFAULT_DESCRIPTION_WEIGHT))
