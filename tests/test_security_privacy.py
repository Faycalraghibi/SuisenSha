"""Tests for API security, rate limiting, right to erasure, and audit logging (PR-07)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api import dependencies as deps
from pipeline.api.main import app
from pipeline.api.security import (
    SecurityManager,
    get_security_manager,
)
from pipeline.cache import RecommendationCache

SEQUENCES = {1: [10, 20, 30], 2: [10, 40]}
LOOKUP = {
    10: {"title": "Alpha", "genres": "Drama"},
    20: {"title": "Beta", "genres": "Comedy"},
    30: {"title": "Gamma", "genres": "Action"},
    40: {"title": "Delta", "genres": "Horror"},
}


class FakeItemIndex:
    def search(
        self, history_ids: list[int], top_k: int, exclude: set[int] | None = None
    ) -> list[tuple[int, float]]:
        res = [(40, 0.9)]
        if exclude:
            res = [(iid, sc) for iid, sc in res if iid not in exclude]
        return res[:top_k]

    def similar_items(
        self, item_id: int, top_k: int = 10, exclude: set[int] | None = None
    ) -> list[tuple[int, float]]:
        res = [(40, 0.9)]
        exclude_set = {item_id} | (exclude or set())
        res = [(iid, sc) for iid, sc in res if iid not in exclude_set]
        return res[:top_k]


@pytest.fixture
def sec_db(tmp_path: Path) -> Path:
    return tmp_path / "security_test.db"


@pytest.fixture
def sec_manager(sec_db: Path) -> SecurityManager:
    return SecurityManager(db_path=sec_db)


@pytest.fixture
def client_with_security(tmp_path: Path, sec_manager: SecurityManager):
    cache_path = tmp_path / "cache_test.db"
    cache = RecommendationCache(db_path=cache_path)

    # In-memory sequences copy that can be modified during tests
    test_seqs = dict(SEQUENCES)

    app.dependency_overrides = {
        deps.get_sequences: lambda: test_seqs,
        deps.get_movie_lookup: lambda: LOOKUP,
        deps.get_item_index: lambda: FakeItemIndex(),
        deps.get_cache: lambda: cache,
        get_security_manager: lambda: sec_manager,
    }

    client = TestClient(app)
    yield client, sec_manager, cache, test_seqs
    app.dependency_overrides = {}
    cache.close()


class TestAPIKeyAuth:
    def test_request_without_key_returns_401(self, client_with_security):
        client, _, _, _ = client_with_security
        response = client.get("/v1/recommend/embedding/1")
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_request_with_invalid_key_returns_403(self, client_with_security):
        client, _, _, _ = client_with_security
        response = client.get("/v1/recommend/embedding/1", headers={"X-API-Key": "invalid_key_123"})
        assert response.status_code == 403
        assert "Invalid or revoked API key" in response.json()["detail"]

    def test_request_with_valid_key_returns_200(self, client_with_security):
        client, sec_mgr, _, _ = client_with_security
        plain_key, _ = sec_mgr.create_key(client_id="test-client", rate_limit=100)

        response = client.get("/v1/recommend/embedding/1", headers={"X-API-Key": plain_key})
        assert response.status_code == 200
        assert response.json()["user_id"] == 1


class TestRateLimiting:
    def test_exceeding_rate_limit_returns_429_with_retry_after(self, client_with_security):
        client, sec_mgr, _, _ = client_with_security
        plain_key, _ = sec_mgr.create_key(client_id="limited-client", rate_limit=3)
        headers = {"X-API-Key": plain_key}

        # First 3 requests succeed
        for _ in range(3):
            resp = client.get("/v1/recommend/embedding/1", headers=headers)
            assert resp.status_code == 200

        # 4th request must fail with 429
        resp = client.get("/v1/recommend/embedding/1", headers=headers)
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]
        assert "Retry-After" in resp.headers
        retry_after = int(resp.headers["Retry-After"])
        assert retry_after >= 1


class TestRightToErasure:
    def test_delete_user_removes_data_and_cache(self, client_with_security):
        client, sec_mgr, cache, sequences = client_with_security
        plain_key, _ = sec_mgr.create_key(client_id="admin-client", scopes=["admin"])
        headers = {"X-API-Key": plain_key}

        # Populate cache for user 1
        cache.put(
            user_id=1,
            rationale="Recommended because of Alpha",
            candidate_ids=[40],
            lang="en",
        )
        assert cache.user_entry_count(1) == 1
        assert 1 in sequences

        # Execute DELETE /v1/users/1
        del_resp = client.delete("/v1/users/1", headers=headers)
        assert del_resp.status_code == 200
        data = del_resp.json()
        assert data["user_id"] == 1
        assert data["status"] == "erased"

        # Verify no cache entries remain for user 1
        assert cache.user_entry_count(1) == 0

        # Verify user is removed from active sequences
        assert 1 not in sequences

        # Subsequent history request returns 404
        hist_resp = client.get("/v1/users/1/history", headers=headers)
        assert hist_resp.status_code == 404

        # Subsequent recommendation request returns 404
        rec_resp = client.get("/v1/recommend/embedding/1", headers=headers)
        assert rec_resp.status_code == 404

        # Retraining exclusion check: user 1 is in erased registry
        assert sec_mgr.is_user_erased(1) is True


class TestAuditLog:
    def test_administrative_erasure_creates_audit_entry(self, client_with_security):
        client, sec_mgr, _, _ = client_with_security
        plain_key, _ = sec_mgr.create_key(client_id="audit-test-client", scopes=["admin"])
        headers = {"X-API-Key": plain_key}

        client.delete("/v1/users/2", headers=headers)

        logs = sec_mgr.get_audit_logs(resource_id="2")
        assert len(logs) >= 1
        log = logs[0]
        assert log["action"] == "erasure"
        assert log["resource_id"] == "2"
        assert log["status"] == "success"
        assert log["client_id"] == "audit-test-client"
