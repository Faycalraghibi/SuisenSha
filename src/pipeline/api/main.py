from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from pipeline.api.dependencies import (
    get_faiss_artifacts,
    get_movie_lookup,
    get_movies_df,
    get_sasrec_model,
    get_sequences,
    init_data,
)
from pipeline.api.models import (
    ErrorResponse,
    HistoryResponse,
    MovieItem,
    RAGResponse,
    RecommendationItem,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_data()

    logger.info("Loading ML Artifacts into memory...")
    from pipeline.models.embedding import load_artefacts
    from pipeline.models.rag import _get_generator
    from pipeline.models.sequential import load_sasrec_model

    index, item_ids, embeddings = load_artefacts()
    app.state.faiss_artifacts = (index, item_ids, embeddings)

    sasrec_model, _ = load_sasrec_model()
    app.state.sasrec_model = sasrec_model

    app.state.llm_generator = _get_generator()

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down. Releasing ML models.")
    app.state.faiss_artifacts = None
    app.state.sasrec_model = None
    app.state.llm_generator = None


app = FastAPI(
    title="SuisenSha API",
    description="Recommendation Engine Backend serving Embedding, Transformer, and RAG models.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/users/{user_id}/history",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Users"],
)
def get_user_history(
    user_id: int,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
):
    if user_id not in sequences:
        raise HTTPException(status_code=404, detail="User not found in evaluation set.")

    history_ids = sequences[user_id]
    recent = []
    for item_id in reversed(history_ids[-10:]):
        info = lookup.get(item_id, {})
        recent.append(
            MovieItem(
                item_id=item_id,
                title=info.get("title", f"Unknown {item_id}"),
                genres=info.get("genres", "Unknown"),
            )
        )
    return HistoryResponse(user_id=user_id, recent_history=recent)


@app.get(
    "/recommend/embedding/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_embedding(
    user_id: int,
    request: Request,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
):
    if user_id not in sequences:
        raise HTTPException(status_code=404, detail="User not found.")

    from pipeline.models.embedding import recommend_for_user

    index, item_ids, embeddings = get_faiss_artifacts(request)
    rec_ids = recommend_for_user(sequences[user_id], item_ids, embeddings, index, top_k=10)

    items = []
    for iid in rec_ids:
        info = lookup.get(iid, {})
        items.append(
            RecommendationItem(
                item_id=iid,
                title=info.get("title", f"Unknown {iid}"),
                genres=info.get("genres", "Unknown"),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        model_name="Embedding + FAISS",
        recommendations=items,
    )


@app.get(
    "/recommend/sasrec/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_sasrec(
    user_id: int,
    request: Request,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
):
    if user_id not in sequences:
        raise HTTPException(status_code=404, detail="User not found.")

    from pipeline.models.sequential import predict_next_items

    model = get_sasrec_model(request)
    history = sequences[user_id]
    rec_ids = predict_next_items(model, history, top_k=10, exclude=set(history))

    items = []
    for iid in rec_ids:
        info = lookup.get(iid, {})
        items.append(
            RecommendationItem(
                item_id=iid,
                title=info.get("title", f"Unknown {iid}"),
                genres=info.get("genres", "Unknown"),
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        model_name="SASRec (Transformer)",
        recommendations=items,
    )


@app.get(
    "/recommend/rag/{user_id}",
    response_model=RAGResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_rag(
    user_id: int,
    request: Request,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    movies_df=Depends(get_movies_df),
):
    if user_id not in sequences:
        raise HTTPException(status_code=404, detail="User not found.")

    # Check the SQLite cache first for an instant response
    from pipeline.cache import RecommendationCache

    cache = RecommendationCache()
    cached = cache.get(user_id)
    if cached is not None:
        cache.close()
        return RAGResponse(user_id=user_id, rationale=cached)

    from pipeline.models.rag import generate_recommendations

    index, item_ids, embeddings = get_faiss_artifacts(request)

    response_text = generate_recommendations(
        user_history_ids=sequences[user_id],
        item_ids=item_ids,
        embeddings=embeddings,
        index=index,
        movies_df=movies_df,
    )

    # Store for future instant retrieval
    cache.put(user_id, response_text)
    cache.close()

    return RAGResponse(user_id=user_id, rationale=response_text)
