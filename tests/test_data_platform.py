"""Tests for data platform: contracts, schemas, connectors, and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.data.connectors.amazon import AmazonReviewsConnector
from pipeline.data.connectors.file import CSVConnector, ParquetConnector
from pipeline.data.connectors.movielens import MovieLensConnector
from pipeline.data.connectors.sql import SQLConnector
from pipeline.data.connectors.steam import SteamConnector
from pipeline.data.contracts import (
    DataValidationError,
    validate_interactions,
    validate_items,
)


@pytest.fixture
def valid_items_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["Toy Story (1995)", "GoldenEye (1995)", "Four Rooms (1995)"],
            "genres": ["Animation, Children, Comedy", "Action, Adventure", "Thriller"],
            "description": [
                "Toy Story (1995) — Genres: Animation, Children, Comedy",
                "GoldenEye (1995) — Genres: Action, Adventure",
                "Four Rooms (1995) — Genres: Thriller",
            ],
            "attributes": [
                json.dumps({"year": 1995}),
                json.dumps({"year": 1995}),
                json.dumps({"year": 1995}),
            ],
        }
    )


@pytest.fixture
def valid_interactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 1, 3],
            "event_type": ["rating", "rating", "rating", "click", "rating"],
            "timestamp": [1000, 2000, 3000, 1500, 2500],
            "value": [5.0, 4.0, 2.0, 1.0, 4.5],
        }
    )


class TestContractsAndValidation:
    def test_valid_items_passes(self, valid_items_df: pd.DataFrame):
        df = validate_items(valid_items_df)
        assert len(df) == 3
        assert "item_id" in df.columns
        assert "title" in df.columns

    def test_missing_items_column_raises_readable_error(self, valid_items_df: pd.DataFrame):
        invalid_df = valid_items_df.drop(columns=["title"])
        with pytest.raises(DataValidationError) as excinfo:
            validate_items(invalid_df)
        assert "title" in str(excinfo.value)
        assert "missing" in str(excinfo.value).lower()

    def test_negative_item_id_raises_readable_error(self, valid_items_df: pd.DataFrame):
        invalid_df = valid_items_df.copy()
        invalid_df.loc[0, "item_id"] = -5
        with pytest.raises(DataValidationError) as excinfo:
            validate_items(invalid_df)
        assert "item_id" in str(excinfo.value)
        assert (
            "greater than or equal to 0" in str(excinfo.value)
            or "negative" in str(excinfo.value).lower()
        )

    def test_valid_interactions_passes(self, valid_interactions_df: pd.DataFrame):
        df = validate_interactions(valid_interactions_df)
        assert len(df) == 5
        assert "user_id" in df.columns
        assert "item_id" in df.columns
        assert "event_type" in df.columns

    def test_missing_interactions_column_raises_readable_error(
        self, valid_interactions_df: pd.DataFrame
    ):
        invalid_df = valid_interactions_df.drop(columns=["user_id"])
        with pytest.raises(DataValidationError) as excinfo:
            validate_interactions(invalid_df)
        assert "user_id" in str(excinfo.value)
        assert "missing" in str(excinfo.value).lower()

    def test_invalid_event_type_raises_readable_error(self, valid_interactions_df: pd.DataFrame):
        invalid_df = valid_interactions_df.copy()
        invalid_df.loc[0, "event_type"] = ""
        with pytest.raises(DataValidationError) as excinfo:
            validate_interactions(invalid_df)
        assert "event_type" in str(excinfo.value)


class TestFileConnectors:
    def test_csv_connector_roundtrip(
        self, tmp_path: Path, valid_items_df: pd.DataFrame, valid_interactions_df: pd.DataFrame
    ):
        items_csv = tmp_path / "items.csv"
        interactions_csv = tmp_path / "interactions.csv"

        valid_items_df.to_csv(items_csv, index=False)
        valid_interactions_df.to_csv(interactions_csv, index=False)

        connector = CSVConnector(items_path=items_csv, interactions_path=interactions_csv)
        items, interactions = connector.validate()

        assert len(items) == 3
        assert len(interactions) == 5

        seqs = connector.build_sequences(min_interactions=1)
        assert 1 in seqs
        assert seqs[1] == [1, 2, 3]

        # Test export
        out_items = tmp_path / "out_items.csv"
        out_interactions = tmp_path / "out_interactions.csv"
        connector.export_csv(out_items, out_interactions)
        assert out_items.exists()
        assert out_interactions.exists()

    def test_parquet_connector_roundtrip(
        self, tmp_path: Path, valid_items_df: pd.DataFrame, valid_interactions_df: pd.DataFrame
    ):
        items_pq = tmp_path / "items.parquet"
        interactions_pq = tmp_path / "interactions.parquet"

        valid_items_df.to_parquet(items_pq, index=False)
        valid_interactions_df.to_parquet(interactions_pq, index=False)

        connector = ParquetConnector(items_path=items_pq, interactions_path=interactions_pq)
        items, interactions = connector.validate()

        assert len(items) == 3
        assert len(interactions) == 5

        out_items = tmp_path / "out_items.parquet"
        out_interactions = tmp_path / "out_interactions.parquet"
        connector.export_parquet(out_items, out_interactions)
        assert out_items.exists()
        assert out_interactions.exists()


class TestSQLConnector:
    def test_sql_connector_sqlite(
        self, tmp_path: Path, valid_items_df: pd.DataFrame, valid_interactions_df: pd.DataFrame
    ):
        db_path = tmp_path / "test.db"
        uri = f"sqlite:///{db_path}"

        import sqlite3

        with sqlite3.connect(db_path) as conn:
            valid_items_df.to_sql("items", conn, index=False)
            valid_interactions_df.to_sql("interactions", conn, index=False)

        connector = SQLConnector(
            connection_uri=uri, items_table="items", interactions_table="interactions"
        )
        items, interactions = connector.validate()
        assert len(items) == 3
        assert len(interactions) == 5

        seqs = connector.build_sequences(min_interactions=1)
        assert 1 in seqs
        assert seqs[1] == [1, 2, 3]


class TestDatasetConnectors:
    def test_movielens_connector_100k(self, tmp_path: Path):
        # Create a tiny mock MovieLens-100K directory
        ml_dir = tmp_path / "ml-100k"
        ml_dir.mkdir()

        # u.data
        u_data = ml_dir / "u.data"
        u_data.write_text("1\t10\t5\t100\n1\t20\t4\t200\n2\t10\t4\t150\n", encoding="utf-8")

        # u.item (24 pipe-separated fields)
        pipe_row = "10|Movie Ten|01-Jan-1995||http://imdb|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        pipe_row_2 = (
            "20|Movie Twenty|01-Jan-1995||http://imdb|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
        )
        (ml_dir / "u.item").write_text(pipe_row + pipe_row_2, encoding="latin-1")

        connector = MovieLensConnector(data_dir=ml_dir, variant="100k")
        items, interactions = connector.validate()

        assert len(items) == 2
        assert len(interactions) == 3
        assert 10 in items["item_id"].values
        assert 20 in items["item_id"].values

        seqs = connector.build_sequences(min_rating=3.0, min_interactions=1)
        assert seqs[1] == [10, 20]
        assert seqs[2] == [10]

    def test_movielens_connector_1m(self, tmp_path: Path):
        # Create a tiny mock MovieLens-1M directory (:: separated)
        ml_dir = tmp_path / "ml-1m"
        ml_dir.mkdir()

        # ratings.dat: UserID::MovieID::Rating::Timestamp
        ratings_dat = ml_dir / "ratings.dat"
        ratings_dat.write_text(
            "1::101::5::1000\n1::102::4::2000\n2::101::4::1500\n", encoding="utf-8"
        )

        # movies.dat: MovieID::Title::Genres
        movies_dat = ml_dir / "movies.dat"
        ml_movies_text = (
            "101::Toy Story (1995)::Animation|Children's|Comedy\n"
            "102::Jumanji (1995)::Adventure|Children's|Fantasy\n"
        )
        movies_dat.write_text(ml_movies_text, encoding="latin-1")

        connector = MovieLensConnector(data_dir=ml_dir, variant="1m")
        items, interactions = connector.validate()

        assert len(items) == 2
        assert len(interactions) == 3
        assert 101 in items["item_id"].values

        seqs = connector.build_sequences(min_rating=3.0, min_interactions=1)
        assert seqs[1] == [101, 102]

    def test_amazon_connector(self, tmp_path: Path):
        # Create a sample Amazon reviews JSON lines file
        amz_dir = tmp_path / "amazon"
        amz_dir.mkdir()
        reviews_file = amz_dir / "reviews.json"
        reviews_data = [
            {
                "reviewerID": "A101",
                "asin": "B001",
                "overall": 5.0,
                "unixReviewTime": 100,
                "reviewText": "Great book",
                "title": "Book Alpha",
            },
            {
                "reviewerID": "A101",
                "asin": "B002",
                "overall": 4.0,
                "unixReviewTime": 200,
                "reviewText": "Nice read",
                "title": "Book Beta",
            },
            {
                "reviewerID": "A102",
                "asin": "B001",
                "overall": 3.0,
                "unixReviewTime": 150,
                "reviewText": "Okay",
                "title": "Book Alpha",
            },
        ]
        with open(reviews_file, "w", encoding="utf-8") as f:
            for item in reviews_data:
                f.write(json.dumps(item) + "\n")

        connector = AmazonReviewsConnector(data_path=reviews_file)
        items, interactions = connector.validate()

        assert len(items) == 2
        assert len(interactions) == 3
        seqs = connector.build_sequences(min_interactions=1)
        assert len(seqs) >= 1

    def test_steam_connector(self, tmp_path: Path):
        # Create a sample Steam interactions JSON/CSV file
        steam_dir = tmp_path / "steam"
        steam_dir.mkdir()
        steam_csv = steam_dir / "steam_interactions.csv"
        steam_data = pd.DataFrame(
            {
                "user_id": [10, 10, 20],
                "game_id": [730, 570, 730],
                "game_title": ["Counter-Strike", "Dota 2", "Counter-Strike"],
                "behavior": ["play", "play", "purchase"],
                "hours": [120.5, 45.0, 1.0],
                "timestamp": [1000, 2000, 1500],
            }
        )
        steam_data.to_csv(steam_csv, index=False)

        connector = SteamConnector(data_path=steam_csv)
        items, interactions = connector.validate()

        assert len(items) == 2
        assert len(interactions) == 3
        seqs = connector.build_sequences(min_interactions=1)
        assert 10 in seqs


class TestMovieLensReproductionAndBenchmark:
    def test_movielens_csv_export_reproduces_exact_sequences(self, tmp_path: Path):
        """Criterion: A CSV export of MovieLens reproduces current metrics / sequences."""
        from pipeline.config import MOVIELENS_DIR
        from pipeline.data.loader import load_sequences

        orig_sequences = load_sequences()

        ml_connector = MovieLensConnector(data_dir=MOVIELENS_DIR, variant="100k")
        items_csv = tmp_path / "items.csv"
        interactions_csv = tmp_path / "interactions.csv"

        ml_connector.export_csv(items_path=items_csv, interactions_path=interactions_csv)

        # Load back via generic CSV connector
        csv_connector = CSVConnector(items_path=items_csv, interactions_path=interactions_csv)
        csv_items, csv_interactions = csv_connector.validate()

        assert len(csv_items) == 1682
        assert len(csv_interactions) == 100000

        reproduced_sequences = csv_connector.build_sequences(min_rating=3.5, min_interactions=5)

        # Must reproduce exact user count and exact item lists
        assert len(reproduced_sequences) == len(orig_sequences)
        for user_id, items in orig_sequences.items():
            assert reproduced_sequences[user_id] == items

    def test_benchmark_runs_with_ml1m(self, tmp_path: Path):
        """Criterion: ML-1M benchmark runs."""
        from pipeline.benchmark import run_benchmark

        # Build mock ML-1M
        ml1m_dir = tmp_path / "ml-1m"
        ml1m_dir.mkdir()

        ratings_file = ml1m_dir / "ratings.dat"
        # 10 users with multiple ratings
        lines = []
        for u in range(1, 15):
            for i in range(1, 10):
                lines.append(f"{u}::{i}::4::{u * 100 + i}")
        ratings_file.write_text("\n".join(lines), encoding="utf-8")

        movies_file = ml1m_dir / "movies.dat"
        m_lines = [f"{i}::Movie {i} (1995)::Action|Drama" for i in range(1, 10)]
        movies_file.write_text("\n".join(m_lines), encoding="latin-1")

        connector = MovieLensConnector(data_dir=ml1m_dir, variant="1m")
        items, _ = connector.validate()
        seqs = connector.build_sequences(min_rating=3.0, min_interactions=3)

        out_dir = tmp_path / "reports"
        res = run_benchmark(
            seeds=1,
            output_dir=out_dir,
            include_models=["Popularity"],
            sequences=seqs,
            movies_df=items,
            dataset="ml-1m",
        )

        assert Path(res["md_path"]).exists()
        assert Path(res["csv_path"]).exists()
        md_content = Path(res["md_path"]).read_text(encoding="utf-8")
        assert "MovieLens-1M" in md_content
