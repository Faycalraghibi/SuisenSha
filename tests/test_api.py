"""API tests with in-memory data and fakes — no dataset files or models required."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api import dependencies as deps
from pipeline.api.main import app
from pipeline.cache import RecommendationCache

SEQUENCES = {1: [10, 20, 30]}
LOOKUP = {
    10: {"title": "Alpha", "genres": "Drama"},
    20: {"title": "Beta", "genres": "Comedy"},
    30: {"title": "Gamma", "genres": "Action"},
    40: {"title": "Delta", "genres": "Horror"},
}


class FakeItemIndex:
    def search(self, history_ids: list[int], top_k: int) -> list[tuple[int, float]]:
        return [(40, 0.9)][:top_k]


@pytest.fixture
def client(tmp_path: Path):
    def cache_override():
        with RecommendationCache(db_path=tmp_path / "cache.db") as cache:
            yield cache

    app.dependency_overrides = {
        deps.get_sequences: lambda: SEQUENCES,
        deps.get_movie_lookup: lambda: LOOKUP,
        deps.get_item_index: lambda: FakeItemIndex(),
        deps.get_cache: cache_override,
    }
    # No `with` block: skip the lifespan so real artifacts are never loaded
    yield TestClient(app)
    app.dependency_overrides = {}


def test_get_user_history_missing_user(client: TestClient):
    response = client.get("/users/99999/history")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found."}


def test_get_user_history_most_recent_first(client: TestClient):
    response = client.get("/users/1/history")
    assert response.status_code == 200
    titles = [m["title"] for m in response.json()["recent_history"]]
    assert titles == ["Gamma", "Beta", "Alpha"]


def test_recommend_embedding(client: TestClient):
    response = client.get("/recommend/embedding/1")
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    assert recs == [{"item_id": 40, "title": "Delta", "genres": "Horror", "score": 0.9}]


@pytest.mark.parametrize("model", ["embedding", "sasrec", "rag"])
def test_recommend_missing_user(client: TestClient, model: str):
    response = client.get(f"/recommend/{model}/99999")
    assert response.status_code == 404


def test_recommend_rag_served_from_cache(client: TestClient, tmp_path: Path):
    with RecommendationCache(db_path=tmp_path / "cache.db") as cache:
        cache.put(1, "cached rationale", candidate_ids=[40], lang="en")
    response = client.get("/recommend/rag/1?recommender=embedding")
    assert response.status_code == 200
    assert response.json()["user_id"] == 1
    assert response.json()["rationale"] == "cached rationale"
    assert response.json()["recommendations"] == []


def test_recommend_rag_uncached_returns_one_reason_per_item(client: TestClient):
    response = client.get("/recommend/rag/1?recommender=embedding")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) > 0
    first_item = data["recommendations"][0]
    assert first_item["item_id"] == 40
    assert first_item["title"] == "Delta"
    assert "reason" in first_item and len(first_item["reason"]) > 0


def test_recommend_rag_default_sasrec(client: TestClient):
    response = client.get("/recommend/rag/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) > 0
    assert "reason" in data["recommendations"][0]


def test_api_docs_available(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200
