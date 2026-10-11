"""Diversity controls: Maximal Marginal Relevance (MMR) and Calibrated Popularity."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def _item_similarity(f_i: Any, f_j: Any) -> float:
    """Calculate similarity between two items in [0, 1]."""
    if f_i is None or f_j is None:
        return 0.0

    if isinstance(f_i, np.ndarray) and isinstance(f_j, np.ndarray):
        denom = (np.linalg.norm(f_i) * np.linalg.norm(f_j)) + 1e-9
        cosine = float(np.dot(f_i, f_j) / denom)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0 if cosine < 0 else cosine))

    # Set-based (e.g. genres) Jaccard similarity
    s_i = set(f_i) if not isinstance(f_i, set) else f_i
    s_j = set(f_j) if not isinstance(f_j, set) else f_j
    union = s_i | s_j
    if not union:
        return 0.0
    return float(len(s_i & s_j) / len(union))


def maximal_marginal_relevance(
    candidate_ids: list[int],
    item_features: Mapping[int, Any],
    scores: dict[int, float] | None = None,
    diversity: float = 0.5,
    k: int = 10,
) -> list[int]:
    """Re-ranks candidates using Maximal Marginal Relevance (MMR).

    diversity in [0.0, 1.0]:
      - diversity = 0.0: standard greedy relevance ranking (lambda = 1.0).
      - diversity = 1.0: maximum intra-list diversity (lambda = 0.0).
    """
    if not candidate_ids or k <= 0:
        return []

    if diversity <= 0.0:
        # Standard relevance ranking
        if scores:
            return sorted(candidate_ids, key=lambda iid: scores.get(iid, 0.0), reverse=True)[:k]
        return candidate_ids[:k]

    # Pre-normalize relevance scores to [0, 1]
    raw_scores = [scores.get(iid, 1.0) if scores else 1.0 for iid in candidate_ids]
    min_sc = min(raw_scores)
    max_sc = max(raw_scores)
    span = max_sc - min_sc

    norm_scores: dict[int, float] = {}
    for iid, raw in zip(candidate_ids, raw_scores):
        norm_scores[iid] = (raw - min_sc) / span if span > 1e-9 else 1.0

    lambda_relevance = 1.0 - float(diversity)
    selected: list[int] = []
    remaining = set(candidate_ids)

    # Greedily pick first item with highest relevance score
    first_item = max(remaining, key=lambda iid: norm_scores.get(iid, 0.0))
    selected.append(first_item)
    remaining.remove(first_item)

    while len(selected) < k and remaining:
        best_item: int | None = None
        best_mmr_score = -float("inf")

        for cand in remaining:
            rel = norm_scores.get(cand, 0.0)
            f_cand = item_features.get(cand)
            max_sim = max(
                (_item_similarity(f_cand, item_features.get(sel)) for sel in selected),
                default=0.0,
            )
            mmr_score = (lambda_relevance * rel) - ((1.0 - lambda_relevance) * max_sim)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_item = cand

        if best_item is None:
            break

        selected.append(best_item)
        remaining.remove(best_item)

    return selected


def calibrated_popularity_rerank(
    candidate_ids: list[int],
    item_popularity: dict[int, int],
    scores: dict[int, float] | None = None,
    popularity_discount: float = 0.5,
    k: int = 10,
) -> list[int]:
    """Re-ranks candidates by penalizing over-concentrated popular blockbusters.

    Uses inverse propensity / logarithmic popularity dampening:
      adjusted_score = score / ((log2(popularity + 2)) ^ popularity_discount)
    """
    if not candidate_ids or k <= 0:
        return []

    def adjusted_score(iid: int) -> float:
        base_sc = scores.get(iid, 1.0) if scores else 1.0
        pop = max(0, item_popularity.get(iid, 0))
        # Logarithmic dampening factor >= 1.0
        dampener = math.pow(math.log2(pop + 2.0), popularity_discount)
        return float(base_sc / max(dampener, 1e-9))

    sorted_candidates = sorted(candidate_ids, key=adjusted_score, reverse=True)
    return sorted_candidates[:k]
