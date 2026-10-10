from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING

import pandas as pd
from fastapi import Depends, HTTPException, Request

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


def get_user_history(
    user_id: int,
    sequences: dict[int, list[int]] = Depends(get_sequences),
) -> list[int]:
    if user_id not in sequences:
        raise HTTPException(status_code=404, detail="User not found.")
    return sequences[user_id]


def get_cache() -> Iterator[RecommendationCache]:
    with RecommendationCache() as cache:
        yield cache


# ML artifacts are loaded into app state by the lifespan handler
def get_item_index(request: Request) -> ItemIndex:
    return request.app.state.item_index


def get_sasrec_model(request: Request) -> SASRec:
    return request.app.state.sasrec_model
