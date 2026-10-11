"""Unit tests for versioned API (v1), query parameters, similar items, and DSA transparency."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api import dependencies as deps
from pipeline.api.main import app
from pipeline.cache import RecommendationCache

SEQUENCES = {
    1: [10, 20, 30],
    2: [20, 40],
}
LOOKUP = {
    10: {"title": "Alpha Drama", "genres": "Drama"},
    20: {"title": "Beta Comedy", "genres": "Comedy"},
    30: {"title": "Gamma Action", "genres": "Action"},
    40: {"title": "Delta Horror", "genres": "Horror"},
    50: {"title": "Epsilon Action", "genres": "Action"},
    60: {"title": "Zeta Drama", "genres": "Drama"},
}


class FakeItemIndex:
    def search(
        self,
        history_ids: list[int],
        top_k: int,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        cands = [(40, 0.9), (50, 0.8), (60, 0.7)]
        ex = exclude or set()
        return [(i, s) for i, s in cands if i not in ex][:top_k]

    def similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        cands = [(30, 0.95), (50, 0.85), (60, 0.75)]
        ex = exclude or set()
        return [(i, s) for i, s in cands if i not in ex and i != item_id][:top_k]


@pytest.fixture
def client(tmp_path: Path):
    def cache_override():
        with RecommendationCache(db_path=tmp_path / "cache.db") as cache:
            yield cache

    app.dependency_overrides = {
        deps.get_sequences: lambda: SEQUENCES,
        deps.get_movie_lookup: lambda: LOOKUP,
        deps.get_item_index: lambda: FakeItemIndex(),
        deps.get_popular_items: lambda: [20, 10, 30, 40, 50, 60],
        deps.get_cache: cache_override,
    }
    yield TestClient(app)
    app.dependency_overrides = {}


def test_v1_history(client: TestClient):
    resp = client.get("/v1/users/1/history")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == 1
    titles = [m["title"] for m in resp.json()["recent_history"]]
    assert titles == ["Gamma Action", "Beta Comedy", "Alpha Drama"]


def test_legacy_routes_marked_deprecated(client: TestClient):
    # Verify legacy route still works
    resp = client.get("/users/1/history")
    assert resp.status_code == 200

    # Verify OpenAPI documentation marks legacy route as deprecated
    openapi = client.get("/openapi.json").json()
    paths = openapi["paths"]
    assert paths["/users/{user_id}/history"]["get"]["deprecated"] is True
    assert paths["/recommend/embedding/{user_id}"]["get"]["deprecated"] is True
    assert paths["/recommend/sasrec/{user_id}"]["get"]["deprecated"] is True
    assert paths["/recommend/rag/{user_id}"]["get"]["deprecated"] is True
    # v1 route is not deprecated
    assert paths["/v1/users/{user_id}/history"]["get"].get("deprecated", False) is False


def test_recommend_with_k_parameter(client: TestClient):
    resp = client.get("/v1/recommend/embedding/1?k=2")
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert len(recs) == 2


def test_recommend_with_exclude_parameter(client: TestClient):
    # Delta (40) is top recommendation; exclude it
    resp = client.get("/v1/recommend/embedding/1?exclude=40")
    assert resp.status_code == 200
    rec_ids = [r["item_id"] for r in resp.json()["recommendations"]]
    assert 40 not in rec_ids


def test_recommend_with_genre_filter(client: TestClient):
    # Filter by Action: only Gamma (30) or Epsilon (50)
    resp = client.get("/v1/recommend/embedding/1?genres=Action")
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    for r in recs:
        assert "Action" in r["genres"]


def test_non_personalised_mode_returns_identical_for_all_users(client: TestClient):
    # EU DSA Art. 38 non-profiling mode: user 1 and user 2 must receive the exact same items
    resp1 = client.get("/v1/recommend/embedding/1?personalised=false&k=3")
    resp2 = client.get("/v1/recommend/embedding/2?personalised=false&k=3")

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    recs1 = [r["item_id"] for r in resp1.json()["recommendations"]]
    recs2 = [r["item_id"] for r in resp2.json()["recommendations"]]
    assert recs1 == recs2
    assert "Non-Personalised" in resp1.json()["model_name"]


def test_non_personalised_rag(client: TestClient):
    resp = client.get("/v1/recommend/rag/1?personalised=false&k=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["recommendations"]) == 2
    assert "DSA" in data["rationale"] or "non-profiling" in data["rationale"].lower()


def test_similar_items_success(client: TestClient):
    resp = client.get("/v1/items/10/similar?k=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["item_id"] == 10
    assert data["source_item"]["title"] == "Alpha Drama"
    similar_ids = [item["item_id"] for item in data["similar_items"]]
    assert 10 not in similar_ids  # Cannot be similar to itself
    assert len(similar_ids) <= 2


def test_similar_items_not_found(client: TestClient):
    resp = client.get("/v1/items/99999/similar")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Item not found."


def test_similar_items_with_filter(client: TestClient):
    resp = client.get("/v1/items/10/similar?genres=Action")
    assert resp.status_code == 200
    for item in resp.json()["similar_items"]:
        assert "Action" in item["genres"]


def test_transparency_page_json(client: TestClient):
    resp = client.get("/v1/transparency")
    assert resp.status_code == 200
    data = resp.json()
    assert "DSA" in data["title"] or "Transparency" in data["title"]
    assert "main_parameters" in data
    assert len(data["main_parameters"]) >= 3
    assert "user_controls" in data


def test_transparency_page_html(client: TestClient):
    resp = client.get("/v1/transparency?format=html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>" in resp.text
    assert "Transparency" in resp.text
