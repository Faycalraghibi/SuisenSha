from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api.dependencies import get_event_store, get_redis_cache, get_session_manager
from pipeline.api.main import app
from pipeline.api.security import get_api_key
from pipeline.cache import RedisCache
from pipeline.data.events import EventStore
from pipeline.sessions.manager import SessionStore


class TestSessionStore:
    """Unit tests for SQLite SessionStore."""

    def test_record_and_get_session(self, tmp_path: Path):
        db_path = tmp_path / "test_sessions.db"
        store = SessionStore(db_path=db_path)

        sess = store.record_event("sess_100", 12, "click")
        assert sess.session_id == "sess_100"
        assert sess.item_ids == [12]

        store.record_event("sess_100", 45, "view")
        store.record_event("sess_100", 99, "click")

        retrieved = store.get_session("sess_100")
        assert retrieved is not None
        assert retrieved.item_ids == [12, 45, 99]
        assert len(retrieved.item_ids) == 3

    def test_clear_session(self, tmp_path: Path):
        db_path = tmp_path / "test_sessions.db"
        store = SessionStore(db_path=db_path)

        store.record_event("sess_clear", 50, "click")
        assert store.get_session("sess_clear") is not None

        cleared = store.clear_session("sess_clear")
        assert cleared is True
        assert store.get_session("sess_clear") is None
        assert store.get_session_items("sess_clear") == []

    def test_get_nonexistent_session(self, tmp_path: Path):
        db_path = tmp_path / "test_sessions.db"
        store = SessionStore(db_path=db_path)
        assert store.get_session("nonexistent") is None
        assert store.get_session_items("nonexistent") == []


class TestRealtimeEventsAndSessionsAPI:
    """Integration tests for real-time cache invalidation and session recommendations."""

    @pytest.fixture
    def client(self, tmp_path: Path):
        test_events_db = tmp_path / "realtime_events.db"
        test_sessions_db = tmp_path / "realtime_sessions.db"

        event_store = EventStore(db_path=test_events_db)
        session_store = SessionStore(db_path=test_sessions_db)
        redis_cache = RedisCache(fallback_to_memory=True)

        app.dependency_overrides[get_api_key] = lambda: {
            "key_hash": "dummy",
            "client_name": "test_admin",
            "tier": "admin",
            "is_active": True,
        }
        app.dependency_overrides[get_event_store] = lambda: event_store
        app.dependency_overrides[get_session_manager] = lambda: session_store
        app.dependency_overrides[get_redis_cache] = lambda: redis_cache

        with TestClient(app) as test_client:
            yield test_client, redis_cache

        app.dependency_overrides.clear()

    def test_event_ingestion_invalidates_top_n_cache(self, client):
        test_client, cache = client

        # 1. Populate top-N cache for user 1
        dummy_items = [{"item_id": 1, "title": "Toy Story (1995)", "genres": "Animation"}]
        cache.set_top_n(user_id=1, items=dummy_items, model="sasrec")
        assert cache.get_top_n(user_id=1, model="sasrec") is not None

        # 2. Ingest an interaction event for user 1
        resp = test_client.post(
            "/v1/events",
            json={
                "events": [
                    {
                        "user_id": 1,
                        "item_id": 100,
                        "event_type": "click",
                    }
                ]
            },
        )
        assert resp.status_code == 200

        # 3. Verify top-N cache for user 1 was immediately invalidated
        assert cache.get_top_n(user_id=1, model="sasrec") is None

    def test_realtime_click_alters_sasrec_history_and_recommendations(self, client):
        test_client, _ = client

        # 1. Fetch initial user 1 history and recommendations
        hist_resp = test_client.get("/v1/users/1/history")
        assert hist_resp.status_code == 200
        initial_first_item = hist_resp.json()["recent_history"][0]["item_id"]

        rec_resp1 = test_client.get("/v1/recommend/sasrec/1?k=5")
        assert rec_resp1.status_code == 200
        initial_cands = [r["item_id"] for r in rec_resp1.json()["recommendations"]]

        # 2. Ingest a distinctive real-time click event on item 50 (Star Wars)
        click_resp = test_client.post(
            "/v1/events",
            json={
                "events": [
                    {
                        "user_id": 1,
                        "item_id": 50,
                        "event_type": "click",
                    }
                ]
            },
        )
        assert click_resp.status_code == 200

        # 3. Verify that user 1 history now features item 50 as most recent
        hist_resp_after = test_client.get("/v1/users/1/history")
        assert hist_resp_after.status_code == 200
        assert hist_resp_after.json()["recent_history"][0]["item_id"] == 50
        assert hist_resp_after.json()["recent_history"][1]["item_id"] == initial_first_item

        # 4. Verify that recommendations immediately update without retraining
        rec_resp2 = test_client.get("/v1/recommend/sasrec/1?k=5")
        assert rec_resp2.status_code == 200
        updated_cands = [r["item_id"] for r in rec_resp2.json()["recommendations"]]

        # Item 50 was clicked so it must not be recommended again
        assert 50 not in updated_cands
        assert len(initial_cands) == 5
        assert len(updated_cands) == 5

    def test_anonymous_session_workflow(self, client):
        test_client, _ = client
        session_id = "sess_sci_fi_fan"

        # 1. Record session events for Star Wars franchise movies
        ev_resp = test_client.post(
            f"/v1/sessions/{session_id}/events",
            json={"item_id": 50, "event_type": "click"},
        )
        assert ev_resp.status_code == 200
        assert ev_resp.json()["item_ids"] == [50]

        ev_resp2 = test_client.post(
            f"/v1/sessions/{session_id}/events",
            json={"item_id": 172, "event_type": "click"},
        )
        assert ev_resp2.status_code == 200
        assert ev_resp2.json()["item_ids"] == [50, 172]

        # 2. Get session details
        sess_info = test_client.get(f"/v1/sessions/{session_id}")
        assert sess_info.status_code == 200
        assert sess_info.json()["event_count"] == 2

        # 3. Request session-based recommendations
        rec_resp = test_client.get(f"/v1/recommend/session/{session_id}?k=5")
        assert rec_resp.status_code == 200
        data = rec_resp.json()
        assert data["model_name"] == "SASRec (Session-Based)"
        assert len(data["recommendations"]) == 5
        cand_ids = [r["item_id"] for r in data["recommendations"]]
        # Session items must be excluded from recommendations
        assert 50 not in cand_ids
        assert 172 not in cand_ids

    def test_session_recommend_post_workflow(self, client):
        test_client, _ = client

        # Direct recommendation with explicit item sequence
        payload = {
            "item_ids": [50, 172],
            "k": 5,
            "diversity": 0.2,
        }
        resp = test_client.post("/v1/recommend/session", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "SASRec (Session-Based)"
        assert len(data["recommendations"]) == 5

    def test_empty_session_fallback(self, client):
        test_client, _ = client
        resp = test_client.get("/v1/recommend/session/brand_new_empty_session?k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "Popularity (Empty Session Fallback)"
        assert len(data["recommendations"]) == 5

    def test_session_deletion(self, client):
        test_client, _ = client
        session_id = "sess_to_delete"

        test_client.post(
            f"/v1/sessions/{session_id}/events",
            json={"item_id": 1, "event_type": "view"},
        )
        del_resp = test_client.delete(f"/v1/sessions/{session_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["found"] is True

        get_resp = test_client.get(f"/v1/sessions/{session_id}")
        assert get_resp.status_code == 404

    def test_batch_events_with_session_id(self, client):
        test_client, _ = client
        session_id = "sess_batch_123"

        batch_payload = {
            "events": [
                {
                    "session_id": session_id,
                    "item_id": 10,
                    "event_type": "view",
                },
                {
                    "session_id": session_id,
                    "item_id": 20,
                    "event_type": "click",
                },
            ]
        }
        resp = test_client.post("/v1/events", json=batch_payload)
        assert resp.status_code == 200

        sess_info = test_client.get(f"/v1/sessions/{session_id}")
        assert sess_info.status_code == 200
        assert sess_info.json()["item_ids"] == [10, 20]
