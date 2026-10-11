"""Tests for PR-08: Cold start, hybrid model, business rules, and diversity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np
import pytest

from pipeline.models.embedding import ItemIndex
from pipeline.models.hybrid import HybridRecommender
from pipeline.ranking.diversity import (
    calibrated_popularity_rerank,
    maximal_marginal_relevance,
)
from pipeline.ranking.rules import BusinessRule, RuleEngine, RuleType


class TestBusinessRules:
    """Tests for business rules: exclude, boost, pin, time ranges, and conflicts."""

    def test_exclude_rule_filters_items(self):
        engine = RuleEngine()
        engine.add_rule(BusinessRule(rule_id="r1", rule_type=RuleType.EXCLUDE, item_ids={2, 4}))

        candidates = [
            {"item_id": 1, "score": 0.9, "title": "A"},
            {"item_id": 2, "score": 0.8, "title": "B"},
            {"item_id": 3, "score": 0.7, "title": "C"},
            {"item_id": 4, "score": 0.6, "title": "D"},
        ]
        filtered = engine.apply(candidates)
        assert [c["item_id"] for c in filtered] == [1, 3]

    def test_exclude_rule_by_genre(self):
        engine = RuleEngine()
        engine.add_rule(
            BusinessRule(rule_id="r_genre", rule_type=RuleType.EXCLUDE, genres={"Horror"})
        )

        candidates = [
            {"item_id": 1, "score": 0.9, "genres": "Action|Horror"},
            {"item_id": 2, "score": 0.8, "genres": "Comedy"},
            {"item_id": 3, "score": 0.7, "genres": "Drama"},
        ]
        filtered = engine.apply(candidates)
        assert [c["item_id"] for c in filtered] == [2, 3]

    def test_boost_rule_multiplies_score_and_reorders(self):
        engine = RuleEngine()
        # Item 3 gets a 2.0x boost, moving from 0.5 to 1.0 (ahead of item 1 at 0.9)
        engine.add_rule(
            BusinessRule(rule_id="r_boost", rule_type=RuleType.BOOST, item_ids={3}, value=2.0)
        )

        candidates = [
            {"item_id": 1, "score": 0.9},
            {"item_id": 2, "score": 0.7},
            {"item_id": 3, "score": 0.5},
        ]
        ranked = engine.apply(candidates)
        assert [c["item_id"] for c in ranked] == [3, 1, 2]
        assert ranked[0]["score"] == pytest.approx(1.0)

    def test_pin_rule_pins_to_exact_position(self):
        engine = RuleEngine()
        # Pin item 42 to position 1 (top slot)
        engine.add_rule(
            BusinessRule(rule_id="r_pin", rule_type=RuleType.PIN, item_ids={42}, position=1)
        )

        candidates = [
            {"item_id": 1, "score": 0.9},
            {"item_id": 2, "score": 0.8},
            {"item_id": 42, "score": 0.1},
        ]
        pinned = engine.apply(candidates)
        assert pinned[0]["item_id"] == 42
        assert [c["item_id"] for c in pinned] == [42, 1, 2]

    def test_pin_rule_injects_unranked_item_if_lookup_provided(self):
        engine = RuleEngine()
        engine.add_rule(
            BusinessRule(rule_id="r_pin_new", rule_type=RuleType.PIN, item_ids={99}, position=2)
        )

        candidates = [
            {"item_id": 1, "score": 0.9},
            {"item_id": 2, "score": 0.8},
            {"item_id": 3, "score": 0.7},
        ]
        lookup = {99: {"title": "Special Promotion", "genres": "Sci-Fi"}}
        pinned = engine.apply(candidates, item_lookup=lookup, top_k=3)
        assert [c["item_id"] for c in pinned] == [1, 99, 2]
        assert pinned[1]["title"] == "Special Promotion"

    def test_rule_active_time_window(self):
        engine = RuleEngine()
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Active rule
        engine.add_rule(
            BusinessRule(
                rule_id="active",
                rule_type=RuleType.EXCLUDE,
                item_ids={1},
                start_date=yesterday,
                end_date=tomorrow,
            )
        )
        # Expired rule
        engine.add_rule(
            BusinessRule(
                rule_id="expired",
                rule_type=RuleType.EXCLUDE,
                item_ids={2},
                start_date=yesterday - timedelta(days=10),
                end_date=yesterday,
            )
        )
        # Future rule
        engine.add_rule(
            BusinessRule(
                rule_id="future",
                rule_type=RuleType.EXCLUDE,
                item_ids={3},
                start_date=tomorrow,
                end_date=tomorrow + timedelta(days=2),
            )
        )

        candidates = [{"item_id": 1}, {"item_id": 2}, {"item_id": 3}, {"item_id": 4}]
        result = engine.apply(candidates, now=now)
        # Only item 1 should be excluded by active rule
        assert [c["item_id"] for c in result] == [2, 3, 4]

    def test_conflicting_rules_exclusion_overrides_pin(self):
        engine = RuleEngine()
        # Item 5 is both pinned to pos 1 and excluded
        engine.add_rule(
            BusinessRule(rule_id="r_pin", rule_type=RuleType.PIN, item_ids={5}, position=1)
        )
        engine.add_rule(BusinessRule(rule_id="r_excl", rule_type=RuleType.EXCLUDE, item_ids={5}))

        conflicts = engine.detect_conflicts()
        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == "exclude_overrides_pin"

        candidates = [{"item_id": 5, "score": 0.9}, {"item_id": 6, "score": 0.8}]
        result = engine.apply(candidates)
        # Exclusion must win! Item 5 must NOT appear
        assert [c["item_id"] for c in result] == [6]

    def test_conflicting_pins_priority_resolution(self):
        engine = RuleEngine()
        # Two items pinned to position 1; rule 2 has higher priority
        engine.add_rule(
            BusinessRule(
                rule_id="r_pin1", rule_type=RuleType.PIN, item_ids={10}, position=1, priority=1
            )
        )
        engine.add_rule(
            BusinessRule(
                rule_id="r_pin2", rule_type=RuleType.PIN, item_ids={20}, position=1, priority=2
            )
        )

        conflicts = engine.detect_conflicts()
        assert any(c.conflict_type == "pin_collision" for c in conflicts)

        candidates = [{"item_id": 10}, {"item_id": 20}, {"item_id": 30}]
        result = engine.apply(candidates)
        # Higher priority item (20) gets position 1, item (10) displaced to next slot
        assert result[0]["item_id"] == 20
        assert result[1]["item_id"] == 10
        assert result[2]["item_id"] == 30


class TestDiversityAndMMR:
    """Tests for MMR re-ranking and calibrated popularity."""

    def test_mmr_zero_diversity_preserves_relevance_order(self):
        candidates = [1, 2, 3, 4]
        scores = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
        features = {
            1: np.array([1.0, 0.0]),
            2: np.array([1.0, 0.0]),  # identical to 1
            3: np.array([0.0, 1.0]),  # orthogonal
            4: np.array([0.0, 1.0]),
        }
        res = maximal_marginal_relevance(candidates, features, scores=scores, diversity=0.0, k=3)
        assert res == [1, 2, 3]

    def test_mmr_high_diversity_promotes_dissimilar_items(self):
        candidates = [1, 2, 3, 4]
        scores = {1: 1.0, 2: 0.95, 3: 0.7, 4: 0.65}
        # Items 1 and 2 are almost identical feature-wise
        features = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.99, 0.01]),
            3: np.array([0.0, 1.0]),  # completely different
            4: np.array([0.0, 0.99]),
        }
        # With high diversity (diversity=0.8), item 3 should be selected before item 2
        res = maximal_marginal_relevance(candidates, features, scores=scores, diversity=0.8, k=3)
        assert res[0] == 1
        assert res[1] == 3  # diverse item prioritized over near-duplicate 2

    def test_mmr_with_genre_sets(self):
        candidates = [1, 2, 3]
        scores = {1: 1.0, 2: 0.9, 3: 0.8}
        genres = {
            1: {"Action", "Sci-Fi"},
            2: {"Action", "Sci-Fi"},  # identical genres
            3: {"Comedy", "Romance"},  # distinct genres
        }
        res = maximal_marginal_relevance(candidates, genres, scores=scores, diversity=0.8, k=2)
        assert res == [1, 3]

    def test_calibrated_popularity_rerank(self):
        candidates = [1, 2, 3, 4]
        # Item 1 and 2 are blockbusters; 3 and 4 are niche items
        item_popularity = {1: 10000, 2: 8000, 3: 20, 4: 15}
        scores = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80}

        reranked = calibrated_popularity_rerank(
            candidates,
            item_popularity=item_popularity,
            scores=scores,
            popularity_discount=0.8,
            k=4,
        )
        assert len(reranked) == 4
        # Niche high-relevance items get elevated compared to uncalibrated ranking
        assert reranked != candidates or set(reranked) == set(candidates)


class TestColdStartAndHybridModel:
    """Tests for cold start (new users & new items) and HybridRecommender."""

    @pytest.fixture
    def mock_content_index(self):
        # 4 items with 2D embeddings: 1, 2, 3 (known), and 999 (cold item with 0 interactions)
        item_ids = np.array([1, 2, 3, 999])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [0.7, 0.7],  # Cold item 999 has strong content embedding
            ],
            dtype="float32",
        )
        index = faiss.IndexFlatIP(2)
        index.add(embeddings)
        return ItemIndex(index=index, item_ids=item_ids, embeddings=embeddings)

    def test_cold_start_user_gets_popular_or_onboarding_genres(self, mock_content_index):
        # Mock collaborative recommender that returns [] for empty history
        class MockCollab:
            def recommend(self, history: list[int], k: int = 10) -> list[int]:
                return [] if not history else [1, 2]

        popular_items = [10, 20, 30, 40]
        movie_lookup = {
            10: {"genres": "Action"},
            20: {"genres": "Comedy"},
            30: {"genres": "Action|Thriller"},
            40: {"genres": "Drama"},
        }
        hybrid = HybridRecommender(
            collaborative_model=MockCollab(),
            content_index=mock_content_index,
            popular_items=popular_items,
            movie_lookup=movie_lookup,
            cold_start_threshold=3,
        )

        # 1. Unknown / empty user without genres -> gets popular items
        recs_empty = hybrid.recommend([], k=2)
        assert recs_empty == [10, 20]

        # 2. Cold start user with onboarding genres -> gets popular items matching genre
        recs_genres = hybrid.recommend([], k=2, onboarding_genres=["Action"])
        assert recs_genres == [10, 30]

    def test_cold_item_recommended_via_content_embeddings(self, mock_content_index):
        # Collaborative model only knows items 1, 2, 3; item 999 is completely unknown to it
        class MockCollab:
            num_items = 3

            def recommend(self, history: list[int], k: int = 10) -> list[int]:
                return [1, 2, 3]

        hybrid = HybridRecommender(
            collaborative_model=MockCollab(),
            content_index=mock_content_index,
            popular_items=[1, 2, 3],
            alpha=0.5,
            cold_start_threshold=1,
        )

        # User watched item 2 ([0.8, 0.6]); cold item 999 ([0.7, 0.7]) has high cosine similarity
        recs = hybrid.recommend(history=[2], k=4)
        assert 999 in recs, "Cold item 999 should be recommended via content embeddings"


class TestAPIRankingControls:
    """Tests for API endpoints serving hybrid recommendations, cold start, rules, and diversity."""

    @pytest.fixture
    def client(self, tmp_path: Path):
        from fastapi.testclient import TestClient

        from pipeline.api import dependencies as deps
        from pipeline.api.main import app
        from pipeline.api.security import get_api_key
        from pipeline.cache import RecommendationCache
        from pipeline.ranking.rules import RuleEngine

        sequences = {
            1: [10, 20, 30],
            2: [20],  # Cold user: 1 interaction < 3
        }
        lookup = {
            10: {"title": "Drama 10", "genres": "Drama"},
            20: {"title": "Comedy 20", "genres": "Comedy"},
            30: {"title": "Action 30", "genres": "Action"},
            40: {"title": "Action 40", "genres": "Action"},
            50: {"title": "Horror 50", "genres": "Horror"},
            60: {"title": "SciFi 60", "genres": "Sci-Fi"},
        }
        item_ids = np.array([10, 20, 30, 40, 50, 60])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.7, 0.7],
                [0.8, 0.6],
                [-0.5, 0.5],
                [0.2, 0.8],
            ],
            dtype="float32",
        )
        index = faiss.IndexFlatIP(2)
        index.add(embeddings)
        fake_item_index = ItemIndex(index=index, item_ids=item_ids, embeddings=embeddings)

        rule_engine = RuleEngine()

        def cache_override():
            with RecommendationCache(db_path=tmp_path / "cache.db") as cache:
                yield cache

        app.dependency_overrides = {
            deps.get_sequences: lambda: sequences,
            deps.get_movie_lookup: lambda: lookup,
            deps.get_item_index: lambda: fake_item_index,
            deps.get_popular_items: lambda: [40, 50, 60, 10, 20, 30],
            deps.get_cache: cache_override,
            deps.get_rule_engine: lambda: rule_engine,
            get_api_key: lambda: {"key_id": "test", "client_id": "test_client", "rate_limit": 100},
        }
        yield TestClient(app)
        app.dependency_overrides = {}

    def test_api_hybrid_endpoint(self, client):
        resp = client.get("/v1/recommend/hybrid/1?k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert "Hybrid" in data["model_name"]
        assert len(data["recommendations"]) <= 3

    def test_api_cold_start_unknown_user_with_flag(self, client):
        # Unknown user without cold_start flag returns 404
        resp_404 = client.get("/v1/recommend/hybrid/99999")
        assert resp_404.status_code == 404

        # Unknown user with cold_start=true returns 200 with popular items
        resp_200 = client.get("/v1/recommend/hybrid/99999?cold_start=true&k=2")
        assert resp_200.status_code == 200
        recs = resp_200.json()["recommendations"]
        assert len(recs) == 2
        assert recs[0]["item_id"] == 40  # Most popular item

    def test_api_cold_start_onboarding_genres(self, client):
        resp = client.get("/v1/recommend/sasrec/99999?onboarding_genres=Horror&k=2")
        assert resp.status_code == 200
        recs = resp.json()["recommendations"]
        assert len(recs) >= 1
        assert recs[0]["item_id"] == 50
        assert "Horror" in recs[0]["genres"]

    def test_api_diversity_parameter(self, client):
        resp = client.get("/v1/recommend/embedding/1?diversity=0.7&k=3")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) == 3

    def test_api_rules_post_and_get(self, client):
        # Post a PIN rule pinning item 60 to position 1
        rule_data = {
            "rule_id": "pin_scifi",
            "rule_type": "pin",
            "item_ids": [60],
            "position": 1,
            "priority": 10,
        }
        post_resp = client.post("/v1/rules", json=rule_data)
        assert post_resp.status_code == 200
        assert post_resp.json()["rule_id"] == "pin_scifi"

        # Check /v1/rules listing
        get_resp = client.get("/v1/rules")
        assert get_resp.status_code == 200
        rules = get_resp.json()["rules"]
        assert any(r["rule_id"] == "pin_scifi" for r in rules)

        # Recommendations should now have item 60 pinned at position 1
        rec_resp = client.get("/v1/recommend/embedding/1?k=3")
        assert rec_resp.status_code == 200
        recs = rec_resp.json()["recommendations"]
        assert recs[0]["item_id"] == 60

    def test_transparency_includes_ranking_controls(self, client):
        resp = client.get("/v1/transparency")
        assert resp.status_code == 200
        content = resp.text
        assert "Cold-Start" in content or "Cold start" in content or "cold-start" in content.lower()
        assert "Business Rules" in content or "rules" in content.lower()
        assert "Diversity" in content or "MMR" in content
