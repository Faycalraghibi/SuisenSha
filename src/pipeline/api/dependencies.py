from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from fastapi import Request

from pipeline.config import MOVIES_CSV, USER_SEQUENCES_PKL

logger = logging.getLogger(__name__)


# Global objects to hold into memory.
_movies_df: pd.DataFrame | None = None
_sequences: dict[int, list[int]] | None = None
_movie_lookup: dict[int, dict] | None = None


def init_data() -> None:
    """Load datasets into memory once on startup."""
    global _movies_df, _sequences, _movie_lookup
    logger.info("Initializing datasets for API...")

    _movies_df = pd.read_csv(MOVIES_CSV)
    with open(USER_SEQUENCES_PKL, "rb") as f:
        _sequences = pickle.load(f)

    _movie_lookup = _movies_df.set_index("item_id").to_dict("index")
    logger.info("Datasets loaded successfully.")


def get_movies_df() -> pd.DataFrame:
    if _movies_df is None:
        init_data()
    return _movies_df  # type: ignore


def get_sequences() -> dict[int, list[int]]:
    if _sequences is None:
        init_data()
    return _sequences  # type: ignore


def get_movie_lookup() -> dict[int, dict]:
    if _movie_lookup is None:
        init_data()
    return _movie_lookup  # type: ignore


# ML Artifacts (loaded via App State to avoid circular imports during startup)
def get_faiss_artifacts(request: Request):
    return request.app.state.faiss_artifacts


def get_sasrec_model(request: Request):
    return request.app.state.sasrec_model


def get_llm_generator(request: Request):
    return request.app.state.llm_generator
