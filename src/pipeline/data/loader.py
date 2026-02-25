from __future__ import annotations

import io
import logging
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import requests

# Agg backend: avoids requiring a display server on headless machines
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import (
    DATA_DIR,
    INTERACTIONS_CSV,
    MOVIES_CSV,
    MOVIELENS_DIR,
    MOVIELENS_URL,
    OUTPUTS_DIR,
    USER_SEQUENCES_PKL,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def download_movielens() -> Path:
    ensure_dirs()
    if MOVIELENS_DIR.is_dir():
        logger.info("MovieLens-100K already present at %s", MOVIELENS_DIR)
        return MOVIELENS_DIR

    logger.info("Downloading MovieLens-100K from %s …", MOVIELENS_URL)
    resp = requests.get(MOVIELENS_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_DIR)

    logger.info("Extracted to %s", MOVIELENS_DIR)
    return MOVIELENS_DIR


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    path = (data_dir or MOVIELENS_DIR) / "u.data"
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


_GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    path = (data_dir or MOVIELENS_DIR) / "u.item"
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + _GENRE_COLS

    df = pd.read_csv(path, sep="|", names=columns, encoding="latin-1", engine="python")

    df["genres"] = df[_GENRE_COLS].apply(
        lambda row: ", ".join(g for g, v in zip(_GENRE_COLS, row) if v == 1),
        axis=1,
    )
    # Description combines title + genres for embedding — gives the encoder richer signal than title alone
    df["description"] = df.apply(
        lambda r: f"{r['title']} — Genres: {r['genres']}" if r["genres"] else r["title"],
        axis=1,
    )
    return df[["item_id", "title", "release_date", "genres", "description"]]


def run_eda(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    n_interactions = len(ratings)
    sparsity = 1 - n_interactions / (n_users * n_items)

    summary = (
        f"\n{'═' * 55}\n"
        f"            📊  MovieLens-100K — EDA\n"
        f"{'═' * 55}\n"
        f"  Users            : {n_users:,}\n"
        f"  Items (movies)   : {n_items:,}\n"
        f"  Interactions     : {n_interactions:,}\n"
        f"  Sparsity         : {sparsity:.2%}\n"
        f"  Avg rating       : {ratings['rating'].mean():.2f}\n"
        f"  Rating range     : [{ratings['rating'].min()}, {ratings['rating'].max()}]\n"
        f"{'═' * 55}"
    )
    logger.info(summary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(ratings["rating"], bins=5, edgecolor="black", color="#4e79a7")
    axes[0].set_title("Rating Distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Count")

    user_counts = ratings.groupby("user_id").size()
    axes[1].hist(user_counts, bins=50, edgecolor="black", color="#f28e2b")
    axes[1].set_title("Ratings per User")
    axes[1].set_xlabel("# Ratings")
    axes[1].set_ylabel("# Users")

    item_counts = ratings.groupby("item_id").size()
    axes[2].hist(item_counts, bins=50, edgecolor="black", color="#59a14f")
    axes[2].set_title("Ratings per Item")
    axes[2].set_xlabel("# Ratings")
    axes[2].set_ylabel("# Items")

    plt.tight_layout()
    plot_path = OUTPUTS_DIR / "eda_distributions.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("EDA plot saved → %s", plot_path)


def build_user_sequences(
    ratings: pd.DataFrame,
    min_rating: float = 3.5,
    min_interactions: int = 5,
) -> dict[int, list[int]]:
    """Only keeps ratings >= min_rating as an implicit positive signal."""
    positive = ratings.loc[ratings["rating"] >= min_rating].copy()
    positive.sort_values(["user_id", "timestamp"], inplace=True)

    sequences: dict[int, list[int]] = defaultdict(list)
    for _, row in positive.iterrows():
        sequences[int(row["user_id"])].append(int(row["item_id"]))

    sequences = {u: seq for u, seq in sequences.items() if len(seq) >= min_interactions}
    logger.info(
        "Built sequences for %d users (min %d positive interactions, rating ≥ %.1f)",
        len(sequences),
        min_interactions,
        min_rating,
    )
    return sequences


def save_processed(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    sequences: dict[int, list[int]],
) -> None:
    ensure_dirs()
    ratings.to_csv(INTERACTIONS_CSV, index=False)
    movies.to_csv(MOVIES_CSV, index=False)
    with open(USER_SEQUENCES_PKL, "wb") as fh:
        pickle.dump(sequences, fh)
    logger.info("Saved interactions → %s", INTERACTIONS_CSV)
    logger.info("Saved movies       → %s", MOVIES_CSV)
    logger.info("Saved sequences    → %s", USER_SEQUENCES_PKL)


def run_phase1() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[int]]]:
    data_dir = download_movielens()
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    run_eda(ratings, movies)
    sequences = build_user_sequences(ratings)
    save_processed(ratings, movies, sequences)
    return ratings, movies, sequences
