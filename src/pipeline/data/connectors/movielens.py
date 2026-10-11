"""MovieLens data connector supporting MovieLens-100K and MovieLens-1M datasets."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import DATA_DIR
from pipeline.data.connectors.base import BaseDataConnector

logger = logging.getLogger(__name__)

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

_ML_100K_GENRE_COLS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class MovieLensConnector(BaseDataConnector):
    """Connector for MovieLens datasets (100K and 1M variants)."""

    def __init__(self, data_dir: Path | str, variant: str = "100k") -> None:
        self.data_dir = Path(data_dir)
        self.variant = variant.lower().strip()
        if self.variant not in ("100k", "1m"):
            raise ValueError(f"Unsupported MovieLens variant: {variant}. Expected '100k' or '1m'.")

    @classmethod
    def download_1m(cls, target_dir: Path | str | None = None) -> Path:
        """Download and extract MovieLens-1M dataset."""
        dest_dir = Path(target_dir) if target_dir else DATA_DIR / "ml-1m"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if (dest_dir / "ratings.dat").is_file() and (dest_dir / "movies.dat").is_file():
            logger.info("MovieLens-1M already exists at %s", dest_dir)
            return dest_dir

        logger.info("Downloading MovieLens-1M from %s...", ML_1M_URL)
        resp = requests.get(ML_1M_URL, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(DATA_DIR)

        logger.info("Extracted MovieLens-1M to %s", dest_dir)
        return dest_dir

    def load_items(self) -> pd.DataFrame:
        if self.variant == "100k":
            return self._load_items_100k()
        return self._load_items_1m()

    def load_interactions(self) -> pd.DataFrame:
        if self.variant == "100k":
            return self._load_interactions_100k()
        return self._load_interactions_1m()

    def _load_items_100k(self) -> pd.DataFrame:
        path = self.data_dir / "u.item"
        columns = [
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
        ] + _ML_100K_GENRE_COLS
        df = pd.read_csv(path, sep="|", names=columns, encoding="latin-1", engine="python")
        df["genres"] = df[_ML_100K_GENRE_COLS].apply(
            lambda row: ", ".join(g for g, v in zip(_ML_100K_GENRE_COLS, row) if v == 1),
            axis=1,
        )
        df["description"] = df.apply(
            lambda r: f"{r['title']} — Genres: {r['genres']}" if r["genres"] else str(r["title"]),
            axis=1,
        )
        df["attributes"] = df.apply(
            lambda r: json.dumps(
                {
                    "release_date": str(r["release_date"]),
                    "imdb_url": str(r["imdb_url"]),
                }
            ),
            axis=1,
        )
        df["release_date"] = df["release_date"].fillna("")
        return df[["item_id", "title", "release_date", "genres", "description", "attributes"]]

    def _load_interactions_100k(self) -> pd.DataFrame:
        path = self.data_dir / "u.data"
        df = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        df["event_type"] = "rating"
        df["value"] = df["rating"].astype(float)
        return df[["user_id", "item_id", "event_type", "timestamp", "value"]]

    def _load_items_1m(self) -> pd.DataFrame:
        path = self.data_dir / "movies.dat"
        # Format: MovieID::Title::Genres
        df = pd.read_csv(
            path,
            sep="::",
            names=["item_id", "title", "genres_raw"],
            encoding="latin-1",
            engine="python",
        )
        df["genres"] = df["genres_raw"].str.replace("|", ", ", regex=False)
        df["release_date"] = ""
        df["description"] = df.apply(
            lambda r: f"{r['title']} — Genres: {r['genres']}" if r["genres"] else str(r["title"]),
            axis=1,
        )
        df["attributes"] = df.apply(
            lambda r: json.dumps({"source": "ml-1m"}),
            axis=1,
        )
        return df[["item_id", "title", "release_date", "genres", "description", "attributes"]]

    def _load_interactions_1m(self) -> pd.DataFrame:
        path = self.data_dir / "ratings.dat"
        # Format: UserID::MovieID::Rating::Timestamp
        df = pd.read_csv(
            path,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
            engine="python",
        )
        df["event_type"] = "rating"
        df["value"] = df["rating"].astype(float)
        return df[["user_id", "item_id", "event_type", "timestamp", "value"]]
