"""Batch pre-computation of RAG recommendations into the SQLite cache.

Usage:
    python -m pipeline.cli --phase 8            # all users
    python -m pipeline.cli --phase 8 --limit 10 # first 10 users only

This avoids the ~30-60 s per-request CPU inference cost by generating
all rationale texts ahead of time.
"""

from __future__ import annotations

import logging
import pickle
import time

import pandas as pd
from tqdm import tqdm

from pipeline.cache import RecommendationCache
from pipeline.config import MOVIES_CSV, USER_SEQUENCES_PKL
from pipeline.models.embedding import load_artefacts
from pipeline.models.rag import generate_recommendations

logger = logging.getLogger(__name__)


def run_batch(limit: int | None = None) -> None:
    """Generate and cache RAG rationale for every user (or first *limit*)."""

    # 1. Load data
    movies = pd.read_csv(MOVIES_CSV)
    with open(USER_SEQUENCES_PKL, "rb") as f:
        sequences: dict[int, list[int]] = pickle.load(f)

    index, item_ids, embeddings = load_artefacts()

    user_ids = list(sequences.keys())
    if limit is not None:
        user_ids = user_ids[:limit]

    logger.info("Batch RAG: %d users to process.", len(user_ids))

    cache = RecommendationCache()
    skipped = 0
    t0 = time.perf_counter()

    for uid in tqdm(user_ids, desc="Batch RAG inference"):
        if cache.has(uid):
            skipped += 1
            continue

        rationale = generate_recommendations(
            user_history_ids=sequences[uid],
            item_ids=item_ids,
            embeddings=embeddings,
            index=index,
            movies_df=movies,
        )
        cache.put(uid, rationale)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Batch complete — %d generated, %d skipped (cached), %.1f s elapsed. Total cached: %d.",
        len(user_ids) - skipped,
        skipped,
        elapsed,
        cache.count(),
    )
    cache.close()
