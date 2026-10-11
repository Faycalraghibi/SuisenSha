from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from pipeline.api.dependencies import (
    get_cache,
    get_item_index,
    get_movie_lookup,
    get_movies_df,
    get_popular_items,
    get_sasrec_model,
    get_sequences,
    get_user_history,
    parse_exclude_ids,
    parse_genre_filter,
)
from pipeline.api.models import (
    ErrorResponse,
    ExplainedItem,
    HistoryResponse,
    MovieItem,
    RAGResponse,
    RecommendationItem,
    RecommendationResponse,
    SimilarItemsResponse,
    TransparencyResponse,
)
from pipeline.cache import RecommendationCache

logger = logging.getLogger(__name__)

TOP_K = 10

TRANSPARENCY_INFO: dict[str, Any] = {
    "title": "SuisenSha Recommender System Transparency Notice",
    "regulation": "Regulation (EU) 2022/2065 (Digital Services Act), Articles 27 & 38",
    "main_parameters": [
        {
            "name": "Sequential User Interaction History",
            "importance": "Primary (Highest)",
            "description": (
                "The chronological sequence of movies watched by the user informs "
                "temporal next-item prediction via self-attention (SASRec)."
            ),
        },
        {
            "name": "Content Plot & Genre Embeddings",
            "importance": "Secondary (High)",
            "description": (
                "Sentence-transformer embeddings of movie descriptions and genres "
                "provide semantic candidate similarity matching in FAISS."
            ),
        },
        {
            "name": "Explicit User Filtering",
            "importance": "Strict Constraint (Overrides model)",
            "description": (
                "User-supplied genre preferences and excluded item IDs strictly "
                "filter algorithmic recommendation candidates."
            ),
        },
        {
            "name": "Global Popularity (Non-Profiling Baseline)",
            "importance": "Baseline / Non-Profiling Alternative",
            "description": (
                "Unbiased popularity distribution used when algorithmic profiling "
                "is disabled under DSA Art. 38, or as cold-start fallback."
            ),
        },
    ],
    "models_summary": {
        "SASRec": "Transformer-based causal self-attention over item sequences",
        "Embedding + FAISS": "MiniLM sentence embeddings indexed with FAISS inner-product search",
        "RAG + Template": "Retrieval-augmented explanations with verified template grounding",
        "Popularity Baseline": "Non-profiling item ranking based on overall catalog frequency",
    },
    "user_controls": [
        "Pass `personalised=false` to opt out of algorithmic profiling (DSA Art. 38).",
        "Pass `genres` parameter (e.g. `genres=Sci-Fi,Action`) to filter recommendations.",
        "Pass `exclude` parameter (e.g. `exclude=10,20`) to remove specific items.",
        "Pass `k` parameter (e.g. `k=5`) to adjust recommendation list size.",
    ],
    "dsa_compliance": (
        "Fully compliant with Regulation (EU) 2022/2065 Article 27 by explaining "
        "main ranking parameters and Article 38 by providing a non-profiling option."
    ),
}


def _render_transparency_html() -> str:
    params_html = "".join(
        f"""<div class="param-card">
            <div class="param-title">{p["name"]} <span class="badge">{p["importance"]}</span></div>
            <p>{p["description"]}</p>
        </div>"""
        for p in TRANSPARENCY_INFO["main_parameters"]
    )
    models_html = "".join(
        f"<li><strong>{name}:</strong> {desc}</li>"
        for name, desc in TRANSPARENCY_INFO["models_summary"].items()
    )
    controls_html = "".join(f"<li>{c}</li>" for c in TRANSPARENCY_INFO["user_controls"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{TRANSPARENCY_INFO["title"]}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 820px;
      margin: 40px auto;
      padding: 0 20px;
      line-height: 1.6;
      color: #1f2937;
    }}
    h1 {{ color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 12px; }}
    h2 {{ color: #374151; margin-top: 32px; }}
    .badge {{
      background: #e0e7ff;
      color: #3730a3;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.82em;
      font-weight: 600;
    }}
    .param-card {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .param-title {{ font-weight: 600; font-size: 1.05em; margin-bottom: 6px; }}
    ul {{ padding-left: 22px; }}
  </style>
</head>
<body>
  <h1>{TRANSPARENCY_INFO["title"]}</h1>
  <p><span class="badge">{TRANSPARENCY_INFO["regulation"]}</span></p>
  <p>{TRANSPARENCY_INFO["dsa_compliance"]}</p>

  <h2>Main Ranking Parameters & Criteria (DSA Art. 27)</h2>
  {params_html}

  <h2>Recommender Models Overview</h2>
  <ul>{models_html}</ul>

  <h2>User Options & Controls (DSA Art. 38)</h2>
  <ul>{controls_html}</ul>
</body>
</html>"""


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

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down. Releasing ML models.")
    app.state.item_index = None
    app.state.sasrec_model = None


app = FastAPI(
    title="SuisenSha API",
    description="Recommendation Engine Backend serving Embedding, Transformer, and RAG models.",
    version="1.0.0",
    lifespan=lifespan,
)

v1_router = APIRouter(prefix="/v1")


def _movie_fields(item_id: int, lookup: dict[int, dict]) -> dict[str, Any]:
    info = lookup.get(item_id, {})
    return {
        "item_id": item_id,
        "title": info.get("title", f"Unknown {item_id}"),
        "genres": info.get("genres", "Unknown"),
    }


def _filter_and_build_recs(
    candidate_ids: list[int],
    lookup: dict[int, dict],
    exclude_set: set[int],
    genre_filters: set[str] | None,
    k: int,
    scores: dict[int, float] | None = None,
) -> list[RecommendationItem]:
    items: list[RecommendationItem] = []
    for iid in candidate_ids:
        if iid in exclude_set:
            continue
        info = lookup.get(iid, {})
        if genre_filters:
            item_genres = {
                g.strip().lower() for g in str(info.get("genres", "")).split(",") if g.strip()
            }
            if not (item_genres & genre_filters):
                continue
        score = scores.get(iid) if scores else None
        items.append(RecommendationItem(**_movie_fields(iid, lookup), score=score))
        if len(items) >= k:
            break
    return items


# ─────────────────────────────────────────────────────────────────────────────
# V1 Handlers
# ─────────────────────────────────────────────────────────────────────────────


@v1_router.get(
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


@v1_router.get(
    "/recommend/embedding/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_embedding(
    user_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
):
    exclude_set = parse_exclude_ids(exclude)
    genre_filters = parse_genre_filter(genres)

    if not personalised:
        items = _filter_and_build_recs(popular_items, lookup, exclude_set, genre_filters, k=k)
        return RecommendationResponse(
            user_id=user_id,
            model_name="Popularity (Non-Personalised, DSA Art. 38)",
            recommendations=items,
        )

    exclude_set |= set(history)
    scores_dict: dict[int, float] = {}
    if item_index is not None:
        fetch_k = max(k * 5, 50)
        search_res = item_index.search(history, fetch_k, exclude=exclude_set)
        cand_ids = [iid for iid, _ in search_res]
        scores_dict = {iid: sc for iid, sc in search_res}
    else:
        cand_ids = [i for i in popular_items if i not in exclude_set]

    items = _filter_and_build_recs(
        cand_ids, lookup, exclude_set, genre_filters, k=k, scores=scores_dict
    )
    return RecommendationResponse(
        user_id=user_id, model_name="Embedding + FAISS", recommendations=items
    )


@v1_router.get(
    "/recommend/sasrec/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_sasrec(
    user_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    model=Depends(get_sasrec_model),
):
    exclude_set = parse_exclude_ids(exclude)
    genre_filters = parse_genre_filter(genres)

    if not personalised:
        items = _filter_and_build_recs(popular_items, lookup, exclude_set, genre_filters, k=k)
        return RecommendationResponse(
            user_id=user_id,
            model_name="Popularity (Non-Personalised, DSA Art. 38)",
            recommendations=items,
        )

    from pipeline.models.sequential import predict_next_items

    exclude_set |= set(history)
    fetch_k = max(k * 5, 50)
    cand_ids = predict_next_items(model, history, top_k=fetch_k, exclude=exclude_set)
    items = _filter_and_build_recs(cand_ids, lookup, exclude_set, genre_filters, k=k)
    return RecommendationResponse(
        user_id=user_id, model_name="SASRec (Transformer)", recommendations=items
    )


@v1_router.get(
    "/recommend/rag/{user_id}",
    response_model=RAGResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_rag(
    user_id: int,
    recommender: str = "sasrec",
    lang: str = "en",
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
    sasrec_model=Depends(get_sasrec_model),
    cache: RecommendationCache = Depends(get_cache),
):
    exclude_set = parse_exclude_ids(exclude)
    genre_filters = parse_genre_filter(genres)

    if not personalised:
        cand_ids = [
            item.item_id
            for item in _filter_and_build_recs(
                popular_items, lookup, exclude_set, genre_filters, k=k
            )
        ]
        non_prof_items = [
            ExplainedItem(
                item_id=iid,
                title=lookup.get(iid, {}).get("title", f"Unknown {iid}"),
                genres=lookup.get(iid, {}).get("genres", "Unknown"),
                reason=(
                    "Recommandé en fonction de la popularité globale du catalogue "
                    "(option sans profilage, art. 38 du règlement sur les services numériques)."
                    if lang == "fr"
                    else (
                        "Recommended based on overall catalog popularity "
                        "(non-profiling option under EU Digital Services Act, Art. 38)."
                    )
                ),
            )
            for iid in cand_ids
        ]
        rationale = (
            "Recommandations sans profilage algorithmique (art. 38 DSA)."
            if lang == "fr"
            else "Non-profiling recommendations based on global popularity (EU DSA Art. 38)."
        )
        return RAGResponse(user_id=user_id, rationale=rationale, recommendations=non_prof_items)

    from pipeline.models.rag import (
        format_explanations,
        generate_recommendations,
        retrieve_candidates,
    )

    exclude_set |= set(history)
    rec_engine: Any = (
        sasrec_model
        if (recommender == "sasrec" and sasrec_model is not None)
        else (item_index if recommender == "embedding" else recommender)
    )

    fetch_k = max(k * 4, 30)
    raw_candidates = retrieve_candidates(
        history, recommender=rec_engine, movie_lookup=lookup, top_k=fetch_k
    )

    # Apply exclude and genre filters
    filtered_candidates: list[dict[str, Any]] = []
    for cand in raw_candidates:
        cid = cand["item_id"]
        if cid in exclude_set:
            continue
        if genre_filters:
            cand_genres = {
                g.strip().lower() for g in str(cand.get("genres", "")).split(",") if g.strip()
            }
            if not (cand_genres & genre_filters):
                continue
        filtered_candidates.append(cand)
        if len(filtered_candidates) >= k:
            break

    candidate_ids = [c["item_id"] for c in filtered_candidates]

    # Check cache
    cached = cache.get(user_id, candidate_ids=candidate_ids, lang=lang)
    if cached is not None:
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

    items = generate_recommendations(
        history,
        recommender=rec_engine,
        movie_lookup=lookup,
        candidates=filtered_candidates,
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


@v1_router.get(
    "/items/{item_id}/similar",
    response_model=SimilarItemsResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Items"],
)
def get_similar_items(
    item_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
):
    if item_id not in lookup:
        raise HTTPException(status_code=404, detail="Item not found.")

    exclude_set = parse_exclude_ids(exclude) | {item_id}
    genre_filters = parse_genre_filter(genres)
    scores_dict: dict[int, float] = {}

    cand_ids: list[int] = []
    if item_index is not None and hasattr(item_index, "similar_items"):
        fetch_k = max(k * 5, 50)
        sim_res = item_index.similar_items(item_id, top_k=fetch_k, exclude=exclude_set)
        cand_ids = [iid for iid, _ in sim_res]
        scores_dict = {iid: sc for iid, sc in sim_res}

    if not cand_ids:
        # Fallback to genre overlap in catalogue
        target_info = lookup[item_id]
        target_genres = {
            g.strip().lower() for g in str(target_info.get("genres", "")).split(",") if g.strip()
        }
        scored_candidates = []
        for other_id, other_info in lookup.items():
            if other_id in exclude_set:
                continue
            other_genres = {
                g.strip().lower() for g in str(other_info.get("genres", "")).split(",") if g.strip()
            }
            overlap = len(target_genres & other_genres)
            union = len(target_genres | other_genres) or 1
            jaccard = overlap / union
            scored_candidates.append((other_id, jaccard))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        cand_ids = [iid for iid, _ in scored_candidates]
        scores_dict = {iid: sc for iid, sc in scored_candidates}

    items = _filter_and_build_recs(
        cand_ids, lookup, exclude_set, genre_filters, k=k, scores=scores_dict
    )
    return SimilarItemsResponse(
        item_id=item_id,
        source_item=MovieItem(**_movie_fields(item_id, lookup)),
        similar_items=items,
    )


@v1_router.get(
    "/transparency",
    response_model=TransparencyResponse,
    tags=["Transparency"],
)
def get_transparency_page(
    request: Request,
    format: str | None = Query(None, description="Output format ('json' or 'html')"),
):
    accept_html = "text/html" in request.headers.get("accept", "").lower()
    if format == "html" or (format is None and accept_html):
        return HTMLResponse(_render_transparency_html())
    return TransparencyResponse(**TRANSPARENCY_INFO)


# Include v1 router
app.include_router(v1_router)

# ─────────────────────────────────────────────────────────────────────────────
# Deprecated Legacy Routes (Backward-Compatibility)
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/users/{user_id}/history",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Users (Deprecated)"],
    deprecated=True,
)
def legacy_user_history(
    user_id: int,
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
):
    return user_history(user_id=user_id, history=history, lookup=lookup)


@app.get(
    "/recommend/embedding/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations (Deprecated)"],
    deprecated=True,
)
def legacy_recommend_embedding(
    user_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
):
    return recommend_embedding(
        user_id=user_id,
        k=k,
        exclude=exclude,
        genres=genres,
        personalised=personalised,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        item_index=item_index,
    )


@app.get(
    "/recommend/sasrec/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations (Deprecated)"],
    deprecated=True,
)
def legacy_recommend_sasrec(
    user_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    model=Depends(get_sasrec_model),
):
    return recommend_sasrec(
        user_id=user_id,
        k=k,
        exclude=exclude,
        genres=genres,
        personalised=personalised,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        model=model,
    )


@app.get(
    "/recommend/rag/{user_id}",
    response_model=RAGResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations (Deprecated)"],
    deprecated=True,
)
def legacy_recommend_rag(
    user_id: int,
    recommender: str = "sasrec",
    lang: str = "en",
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    personalised: bool = Query(default=True),
    history: list[int] = Depends(get_user_history),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
    sasrec_model=Depends(get_sasrec_model),
    cache: RecommendationCache = Depends(get_cache),
):
    return recommend_rag(
        user_id=user_id,
        recommender=recommender,
        lang=lang,
        k=k,
        exclude=exclude,
        genres=genres,
        personalised=personalised,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        item_index=item_index,
        sasrec_model=sasrec_model,
        cache=cache,
    )


@app.get(
    "/transparency",
    response_model=TransparencyResponse,
    tags=["Transparency (Deprecated)"],
    deprecated=True,
)
def legacy_transparency(
    request: Request,
    format: str | None = Query(None),
):
    return get_transparency_page(request=request, format=format)
