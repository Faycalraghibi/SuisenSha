from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from pipeline.api.dependencies import (
    get_cache,
    get_item_index,
    get_movie_lookup,
    get_movies_df,
    get_sasrec_model,
    get_sequences,
    get_user_history,
)
from pipeline.api.models import (
    ErrorResponse,
    ExplainedItem,
    HistoryResponse,
    MovieItem,
    RAGResponse,
    RecommendationItem,
    RecommendationResponse,
)
from pipeline.cache import RecommendationCache

logger = logging.getLogger(__name__)

TOP_K = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_movies_df()
    get_sequences()
    get_movie_lookup()

    logger.info("Loading ML Artifacts into memory...")
    from pipeline.models.embedding import load_artefacts
    from pipeline.models.sequential import load_sasrec_model

    app.state.item_index = load_artefacts()
    app.state.sasrec_model, _ = load_sasrec_model()
    # The LLM is loaded lazily on the first uncached RAG request

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down. Releasing ML models.")
    app.state.item_index = None
    app.state.sasrec_model = None


app = FastAPI(
    title="SuisenSha API",
    description="Recommendation Engine Backend serving Embedding, Transformer, and RAG models.",
    version="0.1.0",
    lifespan=lifespan,
)


def _movie_fields(item_id: int, lookup: dict[int, dict]) -> dict:
    info = lookup.get(item_id, {})
    return {
        "item_id": item_id,
        "title": info.get("title", f"Unknown {item_id}"),
        "genres": info.get("genres", "Unknown"),
    }


@app.get(
    "/users/{user_id}/history",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Users"],
)
def user_history(
    user_id: int,
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
):
    recent = [MovieItem(**_movie_fields(iid, lookup)) for iid in reversed(history[-10:])]
    return HistoryResponse(user_id=user_id, recent_history=recent)


@app.get(
    "/recommend/embedding/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_embedding(
    user_id: int,
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    item_index=Depends(get_item_index),
):
    items = [
        RecommendationItem(**_movie_fields(iid, lookup), score=score)
        for iid, score in item_index.search(history, TOP_K)
    ]
    return RecommendationResponse(
        user_id=user_id, model_name="Embedding + FAISS", recommendations=items
    )


@app.get(
    "/recommend/sasrec/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_sasrec(
    user_id: int,
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    model=Depends(get_sasrec_model),
):
    from pipeline.models.sequential import predict_next_items

    rec_ids = predict_next_items(model, history, top_k=TOP_K, exclude=set(history))
    items = [RecommendationItem(**_movie_fields(iid, lookup)) for iid in rec_ids]
    return RecommendationResponse(
        user_id=user_id, model_name="SASRec (Transformer)", recommendations=items
    )


@app.get(
    "/recommend/rag/{user_id}",
    response_model=RAGResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_rag(
    user_id: int,
    recommender: str = "sasrec",
    lang: str = "en",
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    item_index=Depends(get_item_index),
    sasrec_model=Depends(get_sasrec_model),
    cache: RecommendationCache = Depends(get_cache),
):
    from pipeline.models.rag import (
        format_explanations,
        generate_recommendations,
        retrieve_candidates,
    )

    rec_engine: Any = (
        sasrec_model
        if (recommender == "sasrec" and sasrec_model is not None)
        else (item_index if recommender == "embedding" else recommender)
    )
    candidates = retrieve_candidates(
        history, recommender=rec_engine, movie_lookup=lookup, top_k=TOP_K
    )
    candidate_ids = [c["item_id"] for c in candidates]

    # Check the SQLite cache first for an instant response
    cached = cache.get(user_id, candidate_ids=candidate_ids, lang=lang)
    if cached is not None:
        import json

        try:
            cached_data = json.loads(cached)
            if isinstance(cached_data, dict) and "recommendations" in cached_data:
                return RAGResponse(
                    user_id=user_id,
                    rationale=cached_data.get("rationale", ""),
                    recommendations=[
                        ExplainedItem(**item) for item in cached_data["recommendations"]
                    ],
                )
        except Exception:
            pass
        return RAGResponse(user_id=user_id, rationale=cached, recommendations=[])

    import json

    items = generate_recommendations(
        history,
        recommender=rec_engine,
        movie_lookup=lookup,
        candidates=candidates,
        lang=lang,
    )
    rationale = format_explanations(items)
    to_cache = json.dumps(
        {
            "rationale": rationale,
            "recommendations": [item.model_dump() for item in items],
        }
    )
    cache.put(user_id, to_cache, candidate_ids=candidate_ids, lang=lang)
    return RAGResponse(user_id=user_id, rationale=rationale, recommendations=items)
