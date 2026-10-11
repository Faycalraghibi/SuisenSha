from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from pipeline.api.dependencies import (
    get_cache,
    get_event_store,
    get_experiment_manager,
    get_item_index,
    get_movie_lookup,
    get_movies_df,
    get_popular_items,
    get_redis_cache,
    get_rule_engine,
    get_sasrec_model,
    get_sequences,
    get_session_manager,
    get_user_history,
    get_user_history_or_cold,
    parse_exclude_ids,
    parse_genre_filter,
)
from pipeline.api.middleware import TracingAndMetricsMiddleware, get_metrics_response
from pipeline.api.models import (
    BusinessRuleCreate,
    BusinessRuleResponse,
    ConflictResponse,
    ErasureResponse,
    ErrorResponse,
    EventBatchRequest,
    EventBatchResponse,
    ExperimentCreate,
    ExperimentResponse,
    ExplainedItem,
    HistoryResponse,
    MovieItem,
    RAGResponse,
    RecommendationItem,
    RecommendationResponse,
    RulesListResponse,
    SampleSizeResponse,
    SessionEventRequest,
    SessionRecommendRequest,
    SessionResponse,
    SimilarItemsResponse,
    TransparencyResponse,
    UpliftResponse,
    VariantMetricsModel,
    VariantModel,
)
from pipeline.api.security import (
    SecurityManager,
    get_api_key,
    get_security_manager,
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
            "name": "Cold-Start & Onboarding Handling",
            "importance": "High for New Users / Items",
            "description": (
                "Users with fewer than 3 interactions receive popular items or onboarding-genre "
                "recommendations. Unseen catalog items with 0 interactions are surfaced via "
                "semantic content embeddings."
            ),
        },
        {
            "name": "Business Rules & Policy Overrides",
            "importance": "Strict Post-Ranking Constraints",
            "description": (
                "Commercial promotions (boosts), catalogue licensing exclusions, and editorial "
                "pinned positions applied post-ranking with deterministic conflict resolution."
            ),
        },
        {
            "name": "Diversity & Calibration Controls (MMR)",
            "importance": "Re-Ranking Trade-Off",
            "description": (
                "Maximal Marginal Relevance (MMR) re-ranking balances relevance against intra-list "
                "semantic and genre diversity, preventing redundant recommendations."
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
            "name": "User Feedback & Real-Time Exclusions",
            "importance": "Interactive User Control",
            "description": (
                "Explicit negative feedback ('not interested') immediately removes items "
                "from subsequent recommendations, while interaction signals (clicks, purchases, "
                "ratings, thumbs up) are captured for model retraining."
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
        "Hybrid Recommender": (
            "Joint collaborative sequence modeling and content embeddings enabling zero-click "
            "cold-start item discovery"
        ),
        "RAG + Template": "Retrieval-augmented explanations with verified template grounding",
        "Popularity Baseline": "Non-profiling item ranking based on overall catalog frequency",
    },
    "user_controls": [
        "Pass `personalised=false` to opt out of algorithmic profiling (DSA Art. 38).",
        "Pass `diversity` parameter (0.0 to 1.0) to control list diversity using MMR.",
        "Pass `onboarding_genres` parameter to seed recommendations for cold-start users.",
        "Pass `cold_start=true` to enable graceful cold-start fallback for new users.",
        (
            "Pass `alpha` parameter (0.0 to 1.0) on hybrid recommendations "
            "to balance collaborative vs content scores."
        ),
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
    from pipeline.mlops.registry import ModelRegistry
    from pipeline.models.embedding import load_artefacts
    from pipeline.models.sequential import load_sasrec_model

    app.state.item_index = load_artefacts()

    registry = ModelRegistry()
    prod_ver = registry.get_production_version("sasrec")
    if prod_ver and Path(prod_ver.artifact_path).exists():
        logger.info(
            "Loading Production SASRec model from registry (v%d, path: %s)",
            prod_ver.version,
            prod_ver.artifact_path,
        )
        app.state.sasrec_model, _ = load_sasrec_model(weights_path=prod_ver.artifact_path)
        app.state.sasrec_version = prod_ver.version
    else:
        app.state.sasrec_model, _ = load_sasrec_model()
        app.state.sasrec_version = "default"

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down. Releasing ML models.")
    app.state.item_index = None
    app.state.sasrec_model = None
    app.state.sasrec_version = None


app = FastAPI(
    title="SuisenSha API",
    description="Recommendation Engine Backend serving Embedding, Transformer, and RAG models.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(TracingAndMetricsMiddleware)


@app.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    return get_metrics_response()


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
    diversity: float = 0.0,
    item_features: Any | None = None,
    rule_engine: Any | None = None,
) -> list[RecommendationItem]:
    filtered_cands: list[int] = []
    for iid in candidate_ids:
        if iid in exclude_set:
            continue
        info = lookup.get(iid, {})
        if genre_filters:
            item_genres = {
                g.strip().lower()
                for g in str(info.get("genres", "")).replace("|", ",").split(",")
                if g.strip()
            }
            if not (item_genres & genre_filters):
                continue
        filtered_cands.append(iid)

    try:
        div_val = float(diversity)
    except (TypeError, ValueError):
        div_val = 0.0

    if div_val > 0.0 and len(filtered_cands) > 1:
        from pipeline.ranking.diversity import maximal_marginal_relevance

        features = item_features
        if features is None:
            features = {
                iid: {
                    g.strip().lower()
                    for g in str(lookup.get(iid, {}).get("genres", "")).replace("|", ",").split(",")
                    if g.strip()
                }
                for iid in filtered_cands
            }
        filtered_cands = maximal_marginal_relevance(
            filtered_cands,
            item_features=features,
            scores=scores,
            diversity=div_val,
            k=len(filtered_cands),
        )

    items: list[RecommendationItem] = []
    for iid in filtered_cands:
        score = scores.get(iid) if scores else None
        items.append(RecommendationItem(**_movie_fields(iid, lookup), score=score))

    if rule_engine is not None and hasattr(rule_engine, "apply"):
        items = rule_engine.apply(items, item_lookup=lookup, top_k=k)

    return items[:k]


# ─────────────────────────────────────────────────────────────────────────────
# V1 Handlers
# ─────────────────────────────────────────────────────────────────────────────


@v1_router.get(
    "/users/{user_id}",
    response_model=HistoryResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Users"],
)
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
    auth: dict[str, Any] | None = Depends(get_api_key),
):
    recent = [MovieItem(**_movie_fields(iid, lookup)) for iid in reversed(history[-10:])]
    return HistoryResponse(user_id=user_id, recent_history=recent)


@v1_router.delete(
    "/users/{user_id}",
    response_model=ErasureResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Users", "Privacy"],
)
def erase_user_data(
    user_id: int,
    auth: dict[str, Any] = Depends(get_api_key),
    sec_mgr: SecurityManager = Depends(get_security_manager),
    cache: RecommendationCache = Depends(get_cache),
    sequences: dict[int, list[int]] = Depends(get_sequences),
    event_store=Depends(get_event_store),
) -> ErasureResponse:
    if user_id not in sequences and not cache.has(user_id) and not sec_mgr.is_user_erased(user_id):
        raise HTTPException(status_code=404, detail="User not found.")

    sequences.pop(user_id, None)
    erased_cache = cache.delete_user(user_id)
    if hasattr(event_store, "erase_user_events"):
        event_store.erase_user_events(user_id)
    sec_mgr.erase_user(user_id=user_id, client_id=str(auth.get("client_id", "unknown")))

    return ErasureResponse(
        user_id=user_id,
        status="erased",
        erased_at=datetime.now(timezone.utc).isoformat(),
        erased_cache_entries=erased_cache,
        message=(
            f"All interactions, cached explanations, and profile data for "
            f"user {user_id} have been permanently erased (GDPR Art. 17)."
        ),
    )


@v1_router.get(
    "/rules",
    response_model=RulesListResponse,
    tags=["Business Rules"],
)
def list_rules(
    rule_engine=Depends(get_rule_engine),
    auth: dict[str, Any] = Depends(get_api_key),
):
    active_ids = {r.rule_id for r in rule_engine.get_active_rules()}
    rule_resps = [
        BusinessRuleResponse(
            rule_id=r.rule_id,
            rule_type=r.rule_type.value if hasattr(r.rule_type, "value") else str(r.rule_type),
            item_ids=sorted(r.item_ids),
            genres=sorted(r.genres),
            value=r.value,
            position=r.position,
            start_date=str(r.start_date) if r.start_date else None,
            end_date=str(r.end_date) if r.end_date else None,
            priority=r.priority,
            is_active=(r.rule_id in active_ids),
        )
        for r in rule_engine.rules
    ]
    conflicts = [
        ConflictResponse(
            rule_ids=c.rule_ids,
            conflict_type=c.conflict_type,
            description=c.description,
        )
        for c in rule_engine.detect_conflicts()
    ]
    return RulesListResponse(rules=rule_resps, conflicts=conflicts)


@v1_router.post(
    "/rules",
    response_model=BusinessRuleResponse,
    tags=["Business Rules"],
)
def create_rule(
    body: BusinessRuleCreate,
    rule_engine=Depends(get_rule_engine),
    auth: dict[str, Any] = Depends(get_api_key),
):
    from pipeline.ranking.rules import BusinessRule, RuleType

    rule = BusinessRule(
        rule_id=body.rule_id,
        rule_type=RuleType(body.rule_type.lower()),
        item_ids=set(body.item_ids),
        genres=set(body.genres),
        value=body.value,
        position=body.position,
        start_date=body.start_date,
        end_date=body.end_date,
        priority=body.priority,
    )
    rule_engine.add_rule(rule)
    return BusinessRuleResponse(
        rule_id=rule.rule_id,
        rule_type=rule.rule_type.value if hasattr(rule.rule_type, "value") else str(rule.rule_type),
        item_ids=sorted(rule.item_ids),
        genres=sorted(rule.genres),
        value=rule.value,
        position=rule.position,
        start_date=str(rule.start_date) if rule.start_date else None,
        end_date=str(rule.end_date) if rule.end_date else None,
        priority=rule.priority,
        is_active=rule_engine.is_rule_active(rule),
    )


@v1_router.post(
    "/events",
    response_model=EventBatchResponse,
    tags=["Events & Feedback"],
)
def ingest_events(
    batch: EventBatchRequest,
    idempotency_key_header: str | None = Header(None, alias="Idempotency-Key"),
    event_store=Depends(get_event_store),
    redis_cache=Depends(get_redis_cache),
    session_manager=Depends(get_session_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> EventBatchResponse:
    from pipeline.data.events import EventType, UserEvent

    effective_key = idempotency_key_header or batch.idempotency_key
    user_events: list[UserEvent] = []
    for item in batch.events:
        try:
            etype = EventType(item.event_type.lower())
        except ValueError:
            valid_types = [e.value for e in EventType]
            raise HTTPException(
                status_code=422,
                detail=f"Invalid event_type '{item.event_type}'. Must be one of {valid_types}.",
            )
        user_events.append(
            UserEvent(
                user_id=item.user_id,
                session_id=item.session_id,
                item_id=item.item_id,
                event_type=etype,
                timestamp=item.timestamp or datetime.now(timezone.utc),
                rating=item.rating,
                metadata=item.metadata,
            )
        )
        if item.session_id:
            session_manager.record_event(
                session_id=item.session_id,
                item_id=item.item_id,
                event_type=etype.value,
                metadata=item.metadata,
                timestamp=item.timestamp,
            )

    ingested, is_replay = event_store.record_batch(user_events, idempotency_key=effective_key)

    # Invalidate cached recommendations for affected users immediately
    if hasattr(redis_cache, "delete_user"):
        for ev in user_events:
            if ev.user_id:
                redis_cache.delete_user(ev.user_id)

    return EventBatchResponse(
        status="success",
        ingested_count=ingested,
        is_idempotent_replay=is_replay,
        idempotency_key=effective_key,
    )


@v1_router.get(
    "/recommend/popular",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
def recommend_popular(
    user_id: int | None = Query(default=None),
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    event_store=Depends(get_event_store),
    rule_engine=Depends(get_rule_engine),
    auth: dict[str, Any] | None = Depends(get_api_key),
) -> RecommendationResponse:
    exclude_set = parse_exclude_ids(exclude)
    if user_id is not None and hasattr(event_store, "get_user_excluded_items"):
        exclude_set |= event_store.get_user_excluded_items(user_id)
    genre_filters = parse_genre_filter(genres)
    items = _filter_and_build_recs(
        popular_items,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        diversity=diversity,
        rule_engine=rule_engine,
    )
    return RecommendationResponse(
        user_id=user_id or 0,
        model_name="Popularity (Global Non-Profiling)",
        recommendations=items,
    )


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
    onboarding_genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    cold_start: bool = Query(default=False),
    personalised: bool = Query(default=True),
    history: list[int] | None = Depends(get_user_history_or_cold),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
    rule_engine=Depends(get_rule_engine),
    event_store=Depends(get_event_store),
    redis_cache=Depends(get_redis_cache),
    auth: dict[str, Any] | None = Depends(get_api_key),
):
    if history is None:
        if not cold_start and not onboarding_genres:
            raise HTTPException(status_code=404, detail="User not found.")
        history = []

    exclude_set = parse_exclude_ids(exclude)
    if hasattr(event_store, "get_user_excluded_items"):
        exclude_set |= event_store.get_user_excluded_items(user_id)
    genre_filters = parse_genre_filter(genres)
    onboard_filters = parse_genre_filter(onboarding_genres)
    effective_genres = onboard_filters or genre_filters

    if not personalised or len(history) < 3:
        items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            effective_genres,
            k=k,
            diversity=diversity,
            rule_engine=rule_engine,
        )
        model_name = (
            "Popularity (Cold Start Onboarding)"
            if onboarding_genres
            else (
                "Popularity (Cold Start)"
                if len(history) < 3 and personalised
                else "Popularity (Non-Personalised, DSA Art. 38)"
            )
        )
        return RecommendationResponse(user_id=user_id, model_name=model_name, recommendations=items)

    if hasattr(redis_cache, "get_top_n") and not exclude and not genres and not onboarding_genres:
        cached_recs = redis_cache.get_top_n(user_id=user_id, model="embedding")
        if cached_recs is not None:
            filtered_cached = [it for it in cached_recs if it.get("item_id") not in exclude_set]
            if len(filtered_cached) >= k:
                from pipeline.api.middleware import CACHE_HITS

                CACHE_HITS.labels(cache_type="redis_top_n").inc()
                items = [RecommendationItem(**it) for it in filtered_cached[:k]]
                return RecommendationResponse(
                    user_id=user_id,
                    model_name="Embedding (Precomputed Cache)",
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

    item_feats = None
    if (
        item_index is not None
        and hasattr(item_index, "id_to_idx")
        and hasattr(item_index, "embeddings")
    ):
        item_feats = {
            iid: item_index.embeddings[item_index.id_to_idx[iid]]
            for iid in cand_ids
            if iid in item_index.id_to_idx
        }

    items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        scores=scores_dict,
        diversity=diversity,
        item_features=item_feats,
        rule_engine=rule_engine,
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
    onboarding_genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    cold_start: bool = Query(default=False),
    personalised: bool = Query(default=True),
    history: list[int] | None = Depends(get_user_history_or_cold),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    model=Depends(get_sasrec_model),
    item_index=Depends(get_item_index),
    rule_engine=Depends(get_rule_engine),
    event_store=Depends(get_event_store),
    redis_cache=Depends(get_redis_cache),
    auth: dict[str, Any] | None = Depends(get_api_key),
):
    if history is None:
        if not cold_start and not onboarding_genres:
            raise HTTPException(status_code=404, detail="User not found.")
        history = []

    exclude_set = parse_exclude_ids(exclude)
    if hasattr(event_store, "get_user_excluded_items"):
        exclude_set |= event_store.get_user_excluded_items(user_id)
    genre_filters = parse_genre_filter(genres)
    onboard_filters = parse_genre_filter(onboarding_genres)
    effective_genres = onboard_filters or genre_filters

    if not personalised or len(history) < 3:
        items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            effective_genres,
            k=k,
            diversity=diversity,
            rule_engine=rule_engine,
        )
        model_name = (
            "Popularity (Cold Start Onboarding)"
            if onboarding_genres
            else (
                "Popularity (Cold Start)"
                if len(history) < 3 and personalised
                else "Popularity (Non-Personalised, DSA Art. 38)"
            )
        )
        return RecommendationResponse(user_id=user_id, model_name=model_name, recommendations=items)

    if hasattr(redis_cache, "get_top_n") and not exclude and not genres and not onboarding_genres:
        cached_recs = redis_cache.get_top_n(user_id=user_id, model="sasrec")
        if cached_recs is not None:
            filtered_cached = [it for it in cached_recs if it.get("item_id") not in exclude_set]
            if len(filtered_cached) >= k:
                from pipeline.api.middleware import CACHE_HITS

                CACHE_HITS.labels(cache_type="redis_top_n").inc()
                items = [RecommendationItem(**it) for it in filtered_cached[:k]]
                return RecommendationResponse(
                    user_id=user_id,
                    model_name="SASRec (Precomputed Cache)",
                    recommendations=items,
                )

    from pipeline.models.sequential import predict_next_items

    exclude_set |= set(history)
    fetch_k = max(k * 5, 50)
    cand_ids = predict_next_items(model, history, top_k=fetch_k, exclude=exclude_set)

    item_feats = None
    if (
        item_index is not None
        and hasattr(item_index, "id_to_idx")
        and hasattr(item_index, "embeddings")
    ):
        item_feats = {
            iid: item_index.embeddings[item_index.id_to_idx[iid]]
            for iid in cand_ids
            if iid in item_index.id_to_idx
        }

    items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        diversity=diversity,
        item_features=item_feats,
        rule_engine=rule_engine,
    )
    return RecommendationResponse(
        user_id=user_id, model_name="SASRec (Transformer)", recommendations=items
    )


@v1_router.get(
    "/recommend/hybrid/{user_id}",
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Recommendations"],
)
def recommend_hybrid(
    user_id: int,
    k: int = Query(default=TOP_K, ge=1, le=100),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    onboarding_genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    alpha: float = Query(default=0.6, ge=0.0, le=1.0),
    cold_start: bool = Query(default=False),
    personalised: bool = Query(default=True),
    history: list[int] | None = Depends(get_user_history_or_cold),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
    model=Depends(get_sasrec_model),
    rule_engine=Depends(get_rule_engine),
    event_store=Depends(get_event_store),
    auth: dict[str, Any] | None = Depends(get_api_key),
):
    if history is None:
        if not cold_start and not onboarding_genres:
            raise HTTPException(status_code=404, detail="User not found.")
        history = []

    exclude_set = parse_exclude_ids(exclude)
    if hasattr(event_store, "get_user_excluded_items"):
        exclude_set |= event_store.get_user_excluded_items(user_id)
    genre_filters = parse_genre_filter(genres)
    onboard_filters = parse_genre_filter(onboarding_genres)
    effective_genres = onboard_filters or genre_filters

    if not personalised:
        items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            effective_genres,
            k=k,
            diversity=diversity,
            rule_engine=rule_engine,
        )
        return RecommendationResponse(
            user_id=user_id,
            model_name="Popularity (Non-Personalised, DSA Art. 38)",
            recommendations=items,
        )

    from pipeline.models.hybrid import HybridRecommender

    onboard_list = (
        [g.strip() for g in onboarding_genres.split(",") if g.strip()]
        if onboarding_genres
        else None
    )

    hybrid_rec = HybridRecommender(
        collaborative_model=model,
        content_index=item_index,
        popular_items=popular_items,
        movie_lookup=lookup,
        alpha=alpha,
        cold_start_threshold=3,
    )

    fetch_k = max(k * 4, 40)
    scored_candidates = hybrid_rec.recommend_with_scores(
        history=history,
        k=fetch_k,
        onboarding_genres=onboard_list,
        diversity=0.0,
        exclude=exclude_set,
    )

    cand_ids = [iid for iid, _ in scored_candidates]
    scores_dict = {iid: sc for iid, sc in scored_candidates}

    item_feats = None
    if (
        item_index is not None
        and hasattr(item_index, "id_to_idx")
        and hasattr(item_index, "embeddings")
    ):
        item_feats = {
            iid: item_index.embeddings[item_index.id_to_idx[iid]]
            for iid in cand_ids
            if iid in item_index.id_to_idx
        }

    items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        scores=scores_dict,
        diversity=diversity,
        item_features=item_feats,
        rule_engine=rule_engine,
    )
    return RecommendationResponse(
        user_id=user_id,
        model_name="Hybrid (SASRec + Content Embeddings)",
        recommendations=items,
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
    onboarding_genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    cold_start: bool = Query(default=False),
    personalised: bool = Query(default=True),
    history: list[int] | None = Depends(get_user_history_or_cold),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    item_index=Depends(get_item_index),
    sasrec_model=Depends(get_sasrec_model),
    cache: RecommendationCache = Depends(get_cache),
    rule_engine=Depends(get_rule_engine),
    event_store=Depends(get_event_store),
    auth: dict[str, Any] | None = Depends(get_api_key),
):
    if history is None:
        if not cold_start and not onboarding_genres:
            raise HTTPException(status_code=404, detail="User not found.")
        history = []

    exclude_set = parse_exclude_ids(exclude)
    if hasattr(event_store, "get_user_excluded_items"):
        exclude_set |= event_store.get_user_excluded_items(user_id)
    genre_filters = parse_genre_filter(genres)
    onboard_filters = parse_genre_filter(onboarding_genres)
    effective_genres = onboard_filters or genre_filters

    if not personalised or len(history) < 3:
        cand_items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            effective_genres,
            k=k,
            diversity=diversity,
            rule_engine=rule_engine,
        )
        non_prof_items = [
            ExplainedItem(
                item_id=item.item_id,
                title=item.title,
                genres=item.genres,
                reason=(
                    "Recommandé en fonction de la popularité globale du catalogue "
                    "(option sans profilage ou accueil nouvel utilisateur, art. 38 DSA)."
                    if lang == "fr"
                    else (
                        "Recommended based on overall catalog popularity "
                        "(cold start onboarding or non-profiling option under EU DSA, Art. 38)."
                    )
                ),
            )
            for item in cand_items
        ]
        rationale = (
            "Recommandations d'accueil ou sans profilage algorithmique (art. 38 DSA)."
            if lang == "fr"
            else (
                "Onboarding or non-profiling recommendations based on global popularity "
                "(EU DSA Art. 38)."
            )
        )
        return RAGResponse(user_id=user_id, rationale=rationale, recommendations=non_prof_items)

    from pipeline.models.rag import (
        format_explanations,
        generate_recommendations,
        retrieve_candidates,
    )

    exclude_set |= set(history)
    rec_engine: Any
    if recommender == "hybrid":
        from pipeline.models.hybrid import HybridRecommender

        rec_engine = HybridRecommender(
            collaborative_model=sasrec_model,
            content_index=item_index,
            popular_items=popular_items,
            movie_lookup=lookup,
        )
    elif recommender == "sasrec" and sasrec_model is not None:
        rec_engine = sasrec_model
    elif recommender == "embedding" and item_index is not None:
        rec_engine = item_index
    else:
        rec_engine = recommender

    fetch_k = max(k * 4, 30)
    raw_candidates = retrieve_candidates(
        history, recommender=rec_engine, movie_lookup=lookup, top_k=fetch_k
    )

    cand_ids = [c["item_id"] for c in raw_candidates]
    rec_items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        diversity=diversity,
        rule_engine=rule_engine,
    )
    final_cand_ids = [it.item_id for it in rec_items]
    filtered_candidates = [
        {"item_id": it.item_id, "title": it.title, "genres": it.genres} for it in rec_items
    ]

    # Check cache
    cached = cache.get(user_id, candidate_ids=final_cand_ids, lang=lang)
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
    cache.put(user_id, to_cache, candidate_ids=final_cand_ids, lang=lang)
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
    auth: dict[str, Any] = Depends(get_api_key),
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
        diversity=0.0,
        cold_start=False,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        item_index=item_index,
        auth=None,
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
        diversity=0.0,
        cold_start=False,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        model=model,
        auth=None,
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
        diversity=0.0,
        cold_start=False,
        history=history,
        lookup=lookup,
        popular_items=popular_items,
        item_index=item_index,
        sasrec_model=sasrec_model,
        cache=cache,
        auth=None,
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


# ---------------------------------------------------------------------------
# Experiments Routes (/v1/experiments)
# ---------------------------------------------------------------------------
@v1_router.get("/experiments", response_model=list[ExperimentResponse], tags=["Experiments"])
def list_experiments(
    exp_mgr=Depends(get_experiment_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> list[ExperimentResponse]:
    """List all online A/B experiments."""
    experiments = exp_mgr.list_experiments()
    return [
        ExperimentResponse(
            id=e.id,
            name=e.name,
            variants=[
                VariantModel(
                    name=v.name,
                    weight=v.weight,
                    model_type=v.model_type,
                    params=v.params,
                )
                for v in e.variants
            ],
            status=e.status,
            created_at=e.created_at,
            description=e.description,
        )
        for e in experiments
    ]


@v1_router.post(
    "/experiments", response_model=ExperimentResponse, status_code=201, tags=["Experiments"]
)
def create_experiment(
    req: ExperimentCreate,
    exp_mgr=Depends(get_experiment_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> ExperimentResponse:
    """Create a new A/B experiment."""
    from pipeline.experiments.manager import Variant

    variants = [
        Variant(
            name=v.name,
            weight=v.weight,
            model_type=v.model_type,
            params=v.params,
        )
        for v in req.variants
    ]
    try:
        exp = exp_mgr.create_experiment(
            exp_id=req.id,
            name=req.name,
            variants=variants,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        variants=[
            VariantModel(
                name=v.name,
                weight=v.weight,
                model_type=v.model_type,
                params=v.params,
            )
            for v in exp.variants
        ],
        status=exp.status,
        created_at=exp.created_at,
        description=exp.description,
    )


@v1_router.get("/experiments/{exp_id}/uplift", response_model=UpliftResponse, tags=["Experiments"])
def get_experiment_uplift(
    exp_id: str,
    control: str = Query("control", description="Control variant name"),
    treatment: str | None = Query(
        None, description="Treatment variant name (defaults to first non-control)"
    ),
    exp_mgr=Depends(get_experiment_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> UpliftResponse:
    """Compute live conversion uplift and statistical confidence intervals for an experiment."""
    try:
        rep = exp_mgr.get_uplift_report(
            exp_id=exp_id, control_name=control, treatment_name=treatment
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return UpliftResponse(
        experiment_id=rep.experiment_id,
        control=VariantMetricsModel(
            variant_name=rep.control.variant_name,
            exposures=rep.control.exposures,
            conversions=rep.control.conversions,
            conversion_rate=rep.control.conversion_rate,
        ),
        treatment=VariantMetricsModel(
            variant_name=rep.treatment.variant_name,
            exposures=rep.treatment.exposures,
            conversions=rep.treatment.conversions,
            conversion_rate=rep.treatment.conversion_rate,
        ),
        absolute_uplift=rep.absolute_uplift,
        relative_uplift_pct=rep.relative_uplift_pct,
        ci_lower=rep.ci_lower,
        ci_upper=rep.ci_upper,
        p_value=rep.p_value,
        is_significant=rep.is_significant,
        confidence_level=rep.confidence_level,
    )


@v1_router.get("/experiments/sample-size", response_model=SampleSizeResponse, tags=["Experiments"])
def calculate_experiment_sample_size(
    baseline_conversion: float = Query(
        ..., gt=0.0, lt=1.0, description="Current baseline conversion rate (e.g. 0.05)"
    ),
    mde: float = Query(
        ..., gt=0.0, lt=1.0, description="Minimum detectable effect relative uplift (e.g. 0.10)"
    ),
    alpha: float = Query(0.05, gt=0.0, lt=0.5, description="Significance level alpha"),
    power: float = Query(0.80, gt=0.5, lt=1.0, description="Statistical power (1 - beta)"),
    auth: dict[str, Any] = Depends(get_api_key),
) -> SampleSizeResponse:
    """Calculate required sample size per variant for two-proportion A/B test."""
    from pipeline.experiments.manager import calculate_sample_size

    n_per_variant = calculate_sample_size(
        baseline_conversion=baseline_conversion,
        minimum_detectable_effect=mde,
        alpha=alpha,
        power=power,
    )
    return SampleSizeResponse(
        baseline_conversion=baseline_conversion,
        minimum_detectable_effect=mde,
        required_sample_size_per_variant=n_per_variant,
        total_required_sample_size=n_per_variant * 2,
    )


# ---------------------------------------------------------------------------
# Anonymous Sessions & Real-Time Session Recommendations
# ---------------------------------------------------------------------------
@v1_router.post(
    "/sessions/{session_id}/events",
    response_model=SessionResponse,
    tags=["Sessions"],
)
def record_session_event(
    session_id: str,
    event: SessionEventRequest,
    session_manager=Depends(get_session_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> SessionResponse:
    """Record an interaction event into an anonymous session sequence."""
    sess = session_manager.record_event(
        session_id=session_id,
        item_id=event.item_id,
        event_type=event.event_type,
        metadata=event.metadata,
        timestamp=event.timestamp,
    )
    return SessionResponse(**sess.to_dict())


@v1_router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    tags=["Sessions"],
)
def get_session_info(
    session_id: str,
    session_manager=Depends(get_session_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> SessionResponse:
    """Retrieve current state and items for an anonymous session."""
    sess = session_manager.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionResponse(**sess.to_dict())


@v1_router.delete(
    "/sessions/{session_id}",
    tags=["Sessions"],
)
def delete_session(
    session_id: str,
    session_manager=Depends(get_session_manager),
    auth: dict[str, Any] = Depends(get_api_key),
) -> dict[str, Any]:
    """Clear all interactions and state for an anonymous session."""
    cleared = session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id, "found": cleared}


@v1_router.get(
    "/recommend/session/{session_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations", "Sessions"],
)
def recommend_session(
    session_id: str,
    k: int = Query(default=TOP_K, ge=1, le=100),
    item_ids: str | None = Query(
        default=None, description="Optional comma-separated explicit item sequence override"
    ),
    exclude: str | None = Query(default=None),
    genres: str | None = Query(default=None),
    diversity: float = Query(default=0.0, ge=0.0, le=1.0),
    session_manager=Depends(get_session_manager),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    model=Depends(get_sasrec_model),
    item_index=Depends(get_item_index),
    rule_engine=Depends(get_rule_engine),
    auth: dict[str, Any] | None = Depends(get_api_key),
) -> RecommendationResponse:
    """Generate real-time next-item recommendations from anonymous session items."""
    exclude_set = parse_exclude_ids(exclude)
    genre_filters = parse_genre_filter(genres)

    seq_items: list[int] = []
    if item_ids:
        for x in item_ids.split(","):
            x = x.strip()
            if x.isdigit():
                seq_items.append(int(x))
    elif session_id:
        seq_items = session_manager.get_session_items(session_id)

    if not seq_items:
        items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            genre_filters,
            k=k,
            diversity=diversity,
            rule_engine=rule_engine,
        )
        return RecommendationResponse(
            user_id=0,
            model_name="Popularity (Empty Session Fallback)",
            recommendations=items,
        )

    from pipeline.models.sequential import predict_next_items

    exclude_set |= set(seq_items)
    fetch_k = max(k * 5, 50)
    cand_ids = (
        predict_next_items(model, seq_items, top_k=fetch_k, exclude=exclude_set)
        if model is not None
        else []
    )

    if not cand_ids:
        cand_ids = [iid for iid in popular_items if iid not in exclude_set]

    item_feats = None
    if (
        item_index is not None
        and hasattr(item_index, "id_to_idx")
        and hasattr(item_index, "embeddings")
    ):
        item_feats = {
            iid: item_index.embeddings[item_index.id_to_idx[iid]]
            for iid in cand_ids
            if iid in item_index.id_to_idx
        }

    items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=k,
        diversity=diversity,
        item_features=item_feats,
        rule_engine=rule_engine,
    )
    return RecommendationResponse(
        user_id=0,
        model_name="SASRec (Session-Based)",
        recommendations=items,
    )


@v1_router.post(
    "/recommend/session",
    response_model=RecommendationResponse,
    tags=["Recommendations", "Sessions"],
)
def recommend_session_post(
    req: SessionRecommendRequest,
    session_manager=Depends(get_session_manager),
    lookup: dict[int, dict] = Depends(get_movie_lookup),
    popular_items: list[int] = Depends(get_popular_items),
    model=Depends(get_sasrec_model),
    item_index=Depends(get_item_index),
    rule_engine=Depends(get_rule_engine),
    auth: dict[str, Any] | None = Depends(get_api_key),
) -> RecommendationResponse:
    """Generate real-time next-item recommendations via POST with session ID or item IDs."""
    exclude_set = parse_exclude_ids(req.exclude)
    genre_filters = parse_genre_filter(req.genres)

    seq_items: list[int] = []
    if req.item_ids:
        seq_items = list(req.item_ids)
    elif req.session_id:
        seq_items = session_manager.get_session_items(req.session_id)

    if not seq_items:
        items = _filter_and_build_recs(
            popular_items,
            lookup,
            exclude_set,
            genre_filters,
            k=req.k,
            diversity=req.diversity,
            rule_engine=rule_engine,
        )
        return RecommendationResponse(
            user_id=0,
            model_name="Popularity (Empty Session Fallback)",
            recommendations=items,
        )

    from pipeline.models.sequential import predict_next_items

    exclude_set |= set(seq_items)
    fetch_k = max(req.k * 5, 50)
    cand_ids = (
        predict_next_items(model, seq_items, top_k=fetch_k, exclude=exclude_set)
        if model is not None
        else []
    )

    if not cand_ids:
        cand_ids = [iid for iid in popular_items if iid not in exclude_set]

    item_feats = None
    if (
        item_index is not None
        and hasattr(item_index, "id_to_idx")
        and hasattr(item_index, "embeddings")
    ):
        item_feats = {
            iid: item_index.embeddings[item_index.id_to_idx[iid]]
            for iid in cand_ids
            if iid in item_index.id_to_idx
        }

    items = _filter_and_build_recs(
        cand_ids,
        lookup,
        exclude_set,
        genre_filters,
        k=req.k,
        diversity=req.diversity,
        item_features=item_feats,
        rule_engine=rule_engine,
    )
    return RecommendationResponse(
        user_id=0,
        model_name="SASRec (Session-Based)",
        recommendations=items,
    )
