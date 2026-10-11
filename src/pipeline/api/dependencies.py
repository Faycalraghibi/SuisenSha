from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastapi import Depends, HTTPException, Request

from pipeline.api.security import get_security_manager
from pipeline.cache import RecommendationCache
from pipeline.data.loader import build_movie_lookup, load_processed_movies, load_sequences

if TYPE_CHECKING:
    from pipeline.models.embedding import ItemIndex
    from pipeline.models.sequential import SASRec


# Datasets are read once on first use and shared across requests
@lru_cache(maxsize=1)
def get_movies_df() -> pd.DataFrame:
    return load_processed_movies()


@lru_cache(maxsize=1)
def get_sequences() -> dict[int, list[int]]:
    return load_sequences()


@lru_cache(maxsize=1)
def get_movie_lookup() -> dict[int, dict]:
    return build_movie_lookup(get_movies_df())


@lru_cache(maxsize=1)
def get_event_store() -> Any:
    from pipeline.data.events import EventStore

    return EventStore()


def _augment_history_with_events(
    user_id: int,
    base_history: list[int],
    event_store: Any,
) -> list[int]:
    hist = list(base_history)
    if hasattr(event_store, "get_user_events"):
        from pipeline.data.events import EventType

        recent_events = event_store.get_user_events(
            user_id,
            event_types=[
                EventType.CLICK,
                EventType.PURCHASE,
                EventType.THUMBS_UP,
                EventType.RATING,
                EventType.VIEW,
            ],
        )
        for ev in recent_events:
            if ev.event_type == EventType.RATING and ev.rating is not None and ev.rating < 3.0:
                continue
            hist.append(ev.item_id)
    return hist


def get_user_history(
    user_id: int,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    sec_mgr: Any = Depends(get_security_manager),
    event_store: Any = Depends(get_event_store),
) -> list[int]:
    if sec_mgr.is_user_erased(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    base = list(sequences.get(user_id, []))
    hist = _augment_history_with_events(user_id, base, event_store)
    if not hist:
        raise HTTPException(status_code=404, detail="User not found.")
    return hist


def get_user_history_or_cold(
    user_id: int,
    sequences: dict[int, list[int]] = Depends(get_sequences),
    sec_mgr: Any = Depends(get_security_manager),
    event_store: Any = Depends(get_event_store),
) -> list[int] | None:
    if sec_mgr.is_user_erased(user_id):
        raise HTTPException(status_code=404, detail="User not found.")
    base = list(sequences.get(user_id, []))
    hist = _augment_history_with_events(user_id, base, event_store)
    if not hist and user_id not in sequences:
        return None
    return hist


@lru_cache(maxsize=1)
def get_rule_engine() -> Any:
    from pipeline.ranking.rules import RuleEngine

    return RuleEngine()


@lru_cache(maxsize=1)
def get_popular_items() -> list[int]:
    from collections import Counter

    sequences = get_sequences()
    counter: Counter[int] = Counter()
    for seq in sequences.values():
        counter.update(seq)
    return [item for item, _ in counter.most_common()]


def parse_exclude_ids(exclude: Any) -> set[int]:
    if not isinstance(exclude, str) or not exclude:
        return set()
    res: set[int] = set()
    for part in exclude.split(","):
        part = part.strip()
        if part.isdigit():
            res.add(int(part))
    return res


def parse_genre_filter(genres: Any) -> set[str] | None:
    if not isinstance(genres, str) or not genres:
        return None
    res = {g.strip().lower() for g in genres.split(",") if g.strip()}
    return res if res else None


def get_cache() -> Iterator[RecommendationCache]:
    with RecommendationCache() as cache:
        yield cache


# ML artifacts are loaded into app state by the lifespan handler
def get_item_index(request: Request) -> ItemIndex | None:
    return getattr(request.app.state, "item_index", None)


def get_sasrec_model(request: Request) -> SASRec | None:
    return getattr(request.app.state, "sasrec_model", None)


@lru_cache(maxsize=1)
def get_redis_cache() -> Any:
    from pipeline.cache import RedisCache

    return RedisCache()


@lru_cache(maxsize=1)
def get_model_registry() -> Any:
    from pipeline.mlops.registry import ModelRegistry

    return ModelRegistry()


@lru_cache(maxsize=1)
def get_experiment_manager() -> Any:
    from pipeline.experiments.manager import ExperimentManager

    return ExperimentManager()


@lru_cache(maxsize=1)
def get_session_manager() -> Any:
    from pipeline.sessions.manager import SessionManager

    return SessionManager()
