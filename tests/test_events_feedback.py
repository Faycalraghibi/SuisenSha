from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pipeline.api.main import app
from pipeline.data.events import EventStore, EventType, UserEvent
from pipeline.data.loader import build_user_sequences


@pytest.fixture
def temp_event_store(tmp_path):
    db_file = tmp_path / "test_events.db"
    return EventStore(db_path=db_file)


class TestEventValidation:
    def test_valid_user_events(self):
        event = UserEvent(
            user_id=1,
            item_id=10,
            event_type=EventType.CLICK,
            metadata={"source": "recommendation"},
        )
        assert event.user_id == 1
        assert event.item_id == 10
        assert event.event_type == EventType.CLICK
        assert isinstance(event.timestamp, datetime)

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            UserEvent(user_id=1, item_id=10, event_type="invalid_event")  # type: ignore[arg-type]

    def test_invalid_user_or_item_id(self):
        with pytest.raises(ValidationError):
            UserEvent(user_id=-1, item_id=10, event_type=EventType.VIEW)
        with pytest.raises(ValidationError):
            UserEvent(user_id=1, item_id=0, event_type=EventType.VIEW)

    def test_rating_bounds_validation(self):
        valid = UserEvent(user_id=1, item_id=10, event_type=EventType.RATING, rating=4.5)
        assert valid.rating == 4.5
        with pytest.raises(ValidationError):
            UserEvent(user_id=1, item_id=10, event_type=EventType.RATING, rating=6.0)
        with pytest.raises(ValidationError):
            UserEvent(user_id=1, item_id=10, event_type=EventType.RATING, rating=0.0)


class TestEventStoreAndIdempotency:
    def test_batch_ingestion_and_idempotency(self, temp_event_store: EventStore):
        events = [
            UserEvent(user_id=1, item_id=101, event_type=EventType.VIEW),
            UserEvent(user_id=1, item_id=102, event_type=EventType.CLICK),
            UserEvent(user_id=1, item_id=103, event_type=EventType.PURCHASE),
        ]
        key = "idemp-batch-001"
        count, is_replay = temp_event_store.record_batch(events, idempotency_key=key)
        assert count == 3
        assert is_replay is False
        assert temp_event_store.count_events() == 3

        # Replay same batch with same idempotency key
        count2, is_replay2 = temp_event_store.record_batch(events, idempotency_key=key)
        assert count2 == 3
        assert is_replay2 is True
        # Must not duplicate records
        assert temp_event_store.count_events() == 3

    def test_not_interested_immediate_exclusion(self, temp_event_store: EventStore):
        assert temp_event_store.get_user_excluded_items(user_id=42) == set()

        events = [
            UserEvent(user_id=42, item_id=999, event_type=EventType.NOT_INTERESTED),
            UserEvent(user_id=42, item_id=888, event_type=EventType.NOT_INTERESTED),
        ]
        temp_event_store.record_batch(events)

        excluded = temp_event_store.get_user_excluded_items(user_id=42)
        assert 999 in excluded
        assert 888 in excluded
        assert 777 not in excluded
        # Other users unaffected
        assert temp_event_store.get_user_excluded_items(user_id=43) == set()

    def test_feedback_thumbs_signals(self, temp_event_store: EventStore):
        events = [
            UserEvent(user_id=10, item_id=200, event_type=EventType.THUMBS_UP),
            UserEvent(user_id=10, item_id=201, event_type=EventType.THUMBS_DOWN),
        ]
        temp_event_store.record_batch(events)
        user_events = temp_event_store.get_user_events(user_id=10)
        types = [e.event_type for e in user_events]
        assert EventType.THUMBS_UP in types
        assert EventType.THUMBS_DOWN in types

    def test_erase_user_events(self, temp_event_store: EventStore):
        events = [
            UserEvent(user_id=7, item_id=101, event_type=EventType.CLICK),
            UserEvent(user_id=7, item_id=102, event_type=EventType.NOT_INTERESTED),
            UserEvent(user_id=8, item_id=103, event_type=EventType.CLICK),
        ]
        temp_event_store.record_batch(events)
        assert len(temp_event_store.get_user_events(user_id=7)) == 2
        assert 102 in temp_event_store.get_user_excluded_items(user_id=7)

        erased = temp_event_store.erase_user_events(user_id=7)
        assert erased == 2
        assert len(temp_event_store.get_user_events(user_id=7)) == 0
        assert temp_event_store.get_user_excluded_items(user_id=7) == set()
        # Other user unaffected
        assert len(temp_event_store.get_user_events(user_id=8)) == 1


class TestRetrainingIntegration:
    def test_merge_events_into_interactions(self, temp_event_store: EventStore):
        base_ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1],
                "item_id": [10, 20, 30, 40],
                "rating": [4.0, 4.0, 4.0, 4.0],
                "timestamp": [1000, 2000, 3000, 4000],
            }
        )
        # User 1 has only 4 interactions (< min_interactions=5)
        initial_seqs = build_user_sequences(base_ratings, min_interactions=5)
        assert 1 not in initial_seqs

        # New events: click on item 50, purchase of item 60
        now_ts = datetime.fromtimestamp(5000, tz=timezone.utc)
        events = [
            UserEvent(user_id=1, item_id=50, event_type=EventType.CLICK, timestamp=now_ts),
            UserEvent(user_id=1, item_id=60, event_type=EventType.PURCHASE, timestamp=now_ts),
        ]
        temp_event_store.record_batch(events)

        merged = temp_event_store.merge_events_into_ratings(base_ratings)
        assert len(merged) == 6
        # Retraining builds sequences with new events included
        updated_seqs = build_user_sequences(merged, min_interactions=5)
        assert 1 in updated_seqs
        assert updated_seqs[1] == [10, 20, 30, 40, 50, 60]


class TestApiEventsEndpoint:
    @pytest.fixture
    def client(self, temp_event_store: EventStore):
        from pipeline.api import dependencies as deps
        from pipeline.api.security import get_api_key

        app.dependency_overrides[deps.get_event_store] = lambda: temp_event_store
        app.dependency_overrides[get_api_key] = lambda: {
            "client_id": "test_client",
            "is_active": True,
            "rate_limit": 1000,
            "scopes": ["read", "write", "admin"],
        }
        client = TestClient(app)
        yield client
        app.dependency_overrides = {}

    def test_post_events_batch_and_idempotency(self, client: TestClient):
        headers = {"X-API-Key": "test-admin-key", "Idempotency-Key": "api-idemp-001"}
        payload = {
            "events": [
                {"user_id": 1, "item_id": 50, "event_type": "view"},
                {"user_id": 1, "item_id": 51, "event_type": "click"},
                {"user_id": 1, "item_id": 52, "event_type": "rating", "rating": 4.5},
            ]
        }
        resp = client.post("/v1/events", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["ingested_count"] == 3
        assert data["is_idempotent_replay"] is False

        # Replay same batch with same Idempotency-Key
        resp2 = client.post("/v1/events", json=payload, headers=headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "success"
        assert data2["ingested_count"] == 3
        assert data2["is_idempotent_replay"] is True

    def test_post_events_invalid_payload(self, client: TestClient):
        headers = {"X-API-Key": "test-admin-key"}
        # Empty events
        resp = client.post("/v1/events", json={"events": []}, headers=headers)
        assert resp.status_code == 422

        # Invalid event type
        resp = client.post(
            "/v1/events",
            json={"events": [{"user_id": 1, "item_id": 10, "event_type": "dance"}]},
            headers=headers,
        )
        assert resp.status_code == 422

    def test_immediate_not_interested_exclusion_in_recommendations(self, client: TestClient):
        headers = {"X-API-Key": "test-admin-key"}
        # 1. Get initial recommendations for user 1
        rec_resp = client.get("/v1/recommend/popular?user_id=1&k=10", headers=headers)
        assert rec_resp.status_code == 200
        initial_items = [r["item_id"] for r in rec_resp.json()["recommendations"]]
        target_item = initial_items[0]

        # 2. User marks target_item as 'not_interested'
        event_payload = {
            "events": [{"user_id": 1, "item_id": target_item, "event_type": "not_interested"}]
        }
        evt_resp = client.post("/v1/events", json=event_payload, headers=headers)
        assert evt_resp.status_code == 200

        # 3. Next recommendation call for user 1 MUST immediately exclude target_item
        rec_resp_after = client.get("/v1/recommend/popular?user_id=1&k=10", headers=headers)
        assert rec_resp_after.status_code == 200
        new_items = [r["item_id"] for r in rec_resp_after.json()["recommendations"]]
        assert target_item not in new_items

    def test_immediate_not_interested_in_embedding_and_hybrid(self, client: TestClient):
        headers = {"X-API-Key": "test-admin-key"}
        # Cold start user 999 gets recommendations
        rec_resp = client.get("/v1/recommend/embedding/999?cold_start=true&k=10", headers=headers)
        assert rec_resp.status_code == 200
        target = rec_resp.json()["recommendations"][0]["item_id"]

        # User excludes item via not_interested
        client.post(
            "/v1/events",
            json={"events": [{"user_id": 999, "item_id": target, "event_type": "not_interested"}]},
            headers=headers,
        )

        # Check embedding recommendation again
        rec_after = client.get("/v1/recommend/embedding/999?cold_start=true&k=10", headers=headers)
        assert rec_after.status_code == 200
        after_ids = [r["item_id"] for r in rec_after.json()["recommendations"]]
        assert target not in after_ids

        # Check hybrid recommendation
        hyb_after = client.get("/v1/recommend/hybrid/999?cold_start=true&k=10", headers=headers)
        assert hyb_after.status_code == 200
        hyb_ids = [r["item_id"] for r in hyb_after.json()["recommendations"]]
        assert target not in hyb_ids

    def test_user_erasure_endpoint_erases_events(
        self, client: TestClient, temp_event_store: EventStore
    ):
        headers = {"X-API-Key": "test-admin-key"}
        # Create events for user 10
        client.post(
            "/v1/events",
            json={
                "events": [
                    {"user_id": 10, "item_id": 50, "event_type": "click"},
                    {"user_id": 10, "item_id": 60, "event_type": "not_interested"},
                ]
            },
            headers=headers,
        )
        assert len(temp_event_store.get_user_events(10)) == 2
        assert 60 in temp_event_store.get_user_excluded_items(10)

        # Erase user 10
        del_resp = client.delete("/v1/users/10", headers=headers)
        assert del_resp.status_code == 200

        # Events and exclusions must be empty now
        assert len(temp_event_store.get_user_events(10)) == 0
        assert temp_event_store.get_user_excluded_items(10) == set()
