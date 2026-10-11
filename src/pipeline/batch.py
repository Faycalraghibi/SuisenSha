from __future__ import annotations

import json
import logging
import time

from tqdm import tqdm

from pipeline.api.main import TOP_K
from pipeline.cache import RecommendationCache
from pipeline.data.loader import build_movie_lookup, load_processed_movies, load_sequences
from pipeline.models.rag import (
    format_explanations,
    generate_recommendations,
    retrieve_candidates,
)
from pipeline.models.sequential import load_sasrec_model

logger = logging.getLogger(__name__)


def run_batch(
    limit: int | None = None,
    recommender: str = "sasrec",
    top_k: int = TOP_K,
    lang: str = "en",
) -> None:
    """Precompute and cache grounded recommendations matching the API retrieval logic."""
    movie_lookup = build_movie_lookup(load_processed_movies())
    sequences = load_sequences()

    rec_model = None
    if recommender == "sasrec":
        try:
            rec_model, _ = load_sasrec_model()
        except Exception as exc:
            logger.warning("Could not load SASRec model (%s); falling back to embedding.", exc)
            recommender = "embedding"

    user_ids = list(sequences)
    if limit is not None:
        user_ids = user_ids[:limit]

    logger.info(
        "Batch RAG: %d users to process (recommender=%s, top_k=%d, lang=%s).",
        len(user_ids),
        recommender,
        top_k,
        lang,
    )

    skipped = 0
    t0 = time.perf_counter()

    with RecommendationCache() as cache:
        for uid in tqdm(user_ids, desc="Batch RAG inference"):
            history = sequences[uid]
            rec_engine = (
                rec_model if (recommender == "sasrec" and rec_model is not None) else recommender
            )
            candidates = retrieve_candidates(
                history,
                recommender=rec_engine,
                movie_lookup=movie_lookup,
                top_k=top_k,
            )
            candidate_ids = [c["item_id"] for c in candidates]

            if cache.has(uid, candidate_ids=candidate_ids, lang=lang):
                skipped += 1
                continue

            items = generate_recommendations(
                history,
                recommender=rec_engine,
                movie_lookup=movie_lookup,
                candidates=candidates,
                top_k=top_k,
                lang=lang,
            )
            rationale = format_explanations(items)
            to_cache = json.dumps(
                {
                    "rationale": rationale,
                    "recommendations": [item.model_dump() for item in items],
                }
            )
            cache.put(uid, to_cache, candidate_ids=candidate_ids, lang=lang)

        logger.info(
            "Batch complete — %d generated, %d skipped (cached), %.1f s elapsed. Total cached: %d.",
            len(user_ids) - skipped,
            skipped,
            time.perf_counter() - t0,
            cache.count(),
        )
