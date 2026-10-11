from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline.api.main import app
from pipeline.cache import RedisCache
from pipeline.evaluation.cost import CostBenchmark, calculate_serving_costs
from pipeline.models.embedding import (
    ItemIndex,
    evaluate_hnsw_recall,
)


class TestHNSWIndexAndRecall:
    """Tests for FAISS HNSW ANN indexing and recall evaluation against exact flat index."""

    def test_build_hnsw_and_evaluate_recall(self):
        # Generate 500 normalized random vectors in 64 dims
        rng = np.random.default_rng(42)
        n_items = 500
        dim = 64
        vectors = rng.standard_normal((n_items, dim)).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        item_ids = np.arange(1, n_items + 1)

        # Build exact flat index
        flat_item_index = ItemIndex.build(vectors, item_ids, index_type="flat")

        # Build HNSW index with reasonable M and ef
        hnsw_item_index = ItemIndex.build(
            vectors, item_ids, index_type="hnsw", m=32, ef_construction=128, ef_search=64
        )

        assert hnsw_item_index.index_type == "hnsw"
        assert hnsw_item_index.index.ntotal == n_items

        # Generate query vectors (e.g. from existing items or unseen queries)
        queries = vectors[:50]
        recall_at_10 = evaluate_hnsw_recall(
            flat_index=flat_item_index.index,
            hnsw_index=hnsw_item_index.index,
            query_vectors=queries,
            k=10,
        )

        # PR-10 Done when criterion: HNSW recall >= 0.95 vs flat
        assert recall_at_10 >= 0.95, f"Expected HNSW recall@10 >= 0.95, got {recall_at_10:.4f}"

    def test_hnsw_search_with_exclusion(self):
        vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )
        item_ids = np.array([101, 102, 103, 104])
        hnsw_index = ItemIndex.build(vectors, item_ids, index_type="hnsw", m=16)

        # Search from item 101 ([1, 0, 0]), top nearest is 101, then 102
        results = hnsw_index.search([101], top_k=2, exclude={101})
        top_ids = [r[0] for r in results]
        assert 101 not in top_ids
        assert 102 in top_ids


class TestRedisCacheAndPrecomputation:
    """Tests for Redis cache backend, connection fallback, and nightly batch precomputation."""

    def test_redis_cache_in_memory_fallback(self):
        # RedisCache falls back gracefully to in-memory dict when Redis is unreachable
        cache = RedisCache(redis_url="redis://nonexistent-host:9999/0", fallback_to_memory=True)
        assert cache.is_fallback_mode is True

        # Set and get top-N
        items = [{"item_id": 10, "title": "Inception", "score": 0.95}]
        cache.set_top_n(user_id=1, model="sasrec", items=items, ttl_seconds=3600)

        retrieved = cache.get_top_n(user_id=1, model="sasrec")
        assert retrieved is not None
        assert retrieved[0]["item_id"] == 10
        assert retrieved[0]["title"] == "Inception"

        # Nonexistent user misses
        assert cache.get_top_n(user_id=999, model="sasrec") is None

    def test_nightly_batch_precomputation(self, tmp_path: Path):
        from pipeline.batch import precompute_batch_top_n

        cache = RedisCache(fallback_to_memory=True)
        lookup = {
            10: {"title": "Movie 10", "genres": "Action"},
            20: {"title": "Movie 20", "genres": "Comedy"},
            30: {"title": "Movie 30", "genres": "Drama"},
        }

        class MockRecommender:
            def recommend(self, history, k=10, exclude=None):
                return [10, 20][:k]

        user_histories = {
            1: [30],
            2: [10],
        }

        count = precompute_batch_top_n(
            user_histories=user_histories,
            recommender=MockRecommender(),
            lookup=lookup,
            cache=cache,
            model="sasrec",
            k=2,
        )
        assert count == 2
        user1_recs = cache.get_top_n(user_id=1, model="sasrec")
        assert user1_recs is not None
        assert [r["item_id"] for r in user1_recs] == [10, 20]


class TestPrometheusMetricsAndTracing:
    """Tests for Prometheus metrics, request counters, latency, and X-Request-ID."""

    @pytest.fixture
    def client(self):
        from pipeline.api.security import get_api_key

        app.dependency_overrides[get_api_key] = lambda: {
            "client_id": "metrics_client",
            "is_active": True,
            "rate_limit": 1000,
            "scopes": ["read", "write", "admin"],
        }
        client = TestClient(app)
        yield client
        app.dependency_overrides = {}

    def test_request_id_tracing_middleware(self, client: TestClient):
        # 1. Custom Request-ID preserved
        resp = client.get("/v1/recommend/popular", headers={"X-Request-ID": "trace-uuid-123"})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == "trace-uuid-123"

        # 2. Generated Request-ID when missing
        resp2 = client.get("/v1/recommend/popular")
        assert resp2.status_code == 200
        assert "X-Request-ID" in resp2.headers
        assert len(resp2.headers["X-Request-ID"]) > 0

    def test_metrics_endpoint_exposes_prometheus(self, client: TestClient):
        # Make a recommendation request
        client.get("/v1/recommend/popular?k=2")

        # Scrape /metrics
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200
        content = metrics_resp.text

        # Verify key Prometheus metric descriptors are present
        assert "recommendation_requests_total" in content
        assert "recommendation_latency_seconds" in content


class TestCostBenchmark:
    """Tests for cost benchmark calculations comparing SuisenSha to cloud services."""

    def test_calculate_serving_costs(self):
        costs = calculate_serving_costs(
            monthly_recommendations=1_000_000,
            monthly_explanations=1_000,
        )
        assert isinstance(costs, CostBenchmark)
        # SuisenSha self-hosted cost must be well below managed cloud costs (€ per 1M)
        assert costs.suisensha_monthly_cost_eur > 0
        assert costs.aws_personalize_monthly_cost_eur > costs.suisensha_monthly_cost_eur
        assert costs.gcp_retail_monthly_cost_eur > costs.suisensha_monthly_cost_eur
        assert costs.suisensha_cost_per_million_eur < costs.aws_cost_per_million_eur

        # Generate markdown report
        report_md = costs.to_markdown()
        assert "# Cost Benchmark" in report_md
        assert "AWS Personalize" in report_md
        assert "SuisenSha (Self-Hosted)" in report_md
