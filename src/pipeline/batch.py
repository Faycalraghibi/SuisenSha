from __future__ import annotations

import logging
import time

from tqdm import tqdm

from pipeline.cache import RecommendationCache
from pipeline.data.loader import build_movie_lookup, load_processed_movies, load_sequences
from pipeline.models.embedding import load_artefacts
from pipeline.models.rag import generate_recommendations

logger = logging.getLogger(__name__)


def run_batch(limit: int | None = None) -> None:
    movie_lookup = build_movie_lookup(load_processed_movies())
    sequences = load_sequences()
    item_index = load_artefacts()

    user_ids = list(sequences)
    if limit is not None:
        user_ids = user_ids[:limit]

    logger.info("Batch RAG: %d users to process.", len(user_ids))

    skipped = 0
    t0 = time.perf_counter()

    with RecommendationCache() as cache:
        for uid in tqdm(user_ids, desc="Batch RAG inference"):
            if cache.has(uid):
                skipped += 1
                continue
            cache.put(uid, generate_recommendations(sequences[uid], item_index, movie_lookup))

        logger.info(
            "Batch complete — %d generated, %d skipped (cached), %.1f s elapsed. Total cached: %d.",
            len(user_ids) - skipped,
            skipped,
            time.perf_counter() - t0,
            cache.count(),
        )
