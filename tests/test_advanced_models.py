"""Tests for PR-15 Advanced Models: GRU4Rec, BERT4Rec, LightGCN, Two-Tower,
and Two-Stage Pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from pipeline.api import dependencies as deps
from pipeline.api.main import app
from pipeline.models.bert4rec import BERT4Rec, train_bert4rec
from pipeline.models.gru4rec import GRU4Rec, train_gru4rec
from pipeline.models.lightgcn import LightGCN, train_lightgcn
from pipeline.models.two_stage import TwoStageRecommender
from pipeline.models.two_tower import TwoTowerModel, train_two_tower


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_sequences() -> dict[int, list[int]]:
    return {
        1: [1, 2, 3, 4, 5],
        2: [2, 3, 4, 5, 6],
        3: [1, 3, 5, 7],
        4: [2, 4, 6, 8],
    }


@pytest.fixture
def mock_lookup() -> dict[int, dict]:
    return {
        i: {"title": f"Movie {i}", "genres": "Action" if i % 2 == 0 else "Drama"}
        for i in range(1, 20)
    }


# ---------------------------------------------------------------------------
# 1. GRU4Rec Tests
# ---------------------------------------------------------------------------
def test_gru4rec_forward_and_recommend(sample_sequences: dict[int, list[int]]):
    model = GRU4Rec(num_items=10, embedding_dim=16, hidden_dim=16, max_seq_len=10)
    seqs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    logits = model(seqs, lengths)
    assert logits.shape == (2, 11)  # 10 items + padding token

    recs = model.recommend([1, 2, 3], k=5)
    assert len(recs) <= 5
    assert all(r not in [1, 2, 3] for r in recs)
    assert all(0 < r <= 10 for r in recs)


def test_gru4rec_training(sample_sequences: dict[int, list[int]]):
    model = train_gru4rec(sample_sequences, num_items=10, epochs=2, batch_size=4, hidden_dim=16)
    assert isinstance(model, GRU4Rec)
    recs = model.recommend([1, 2], k=3)
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# 2. BERT4Rec Tests
# ---------------------------------------------------------------------------
def test_bert4rec_forward_and_recommend(sample_sequences: dict[int, list[int]]):
    model = BERT4Rec(num_items=10, hidden_dim=16, num_heads=2, num_layers=1, max_seq_len=10)
    seqs = torch.tensor([[1, 2, 11, 0]], dtype=torch.long)  # 11 is mask token

    logits = model(seqs)
    assert logits.shape == (1, 4, 11)

    recs = model.recommend([1, 2], k=4)

    assert len(recs) <= 4
    assert 1 not in recs and 2 not in recs
    assert all(0 < r <= 10 for r in recs)


def test_bert4rec_training(sample_sequences: dict[int, list[int]]):
    model = train_bert4rec(sample_sequences, num_items=10, epochs=2, batch_size=4, hidden_dim=16)
    assert isinstance(model, BERT4Rec)
    recs = model.recommend([2, 3], k=3)
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# 3. LightGCN Tests
# ---------------------------------------------------------------------------
def test_lightgcn_graph_and_propagation(sample_sequences: dict[int, list[int]]):
    model = LightGCN(num_users=4, num_items=10, embedding_dim=16, num_layers=2)
    model.set_graph(sample_sequences)

    u_embed, i_embed = model.compute_layer_embeddings()
    assert u_embed.shape == (5, 16)
    assert i_embed.shape == (11, 16)

    # Loss computation
    users = torch.tensor([1, 2], dtype=torch.long)
    pos = torch.tensor([2, 3], dtype=torch.long)
    neg = torch.tensor([9, 10], dtype=torch.long)
    bpr_loss, reg_loss = model(users, pos, neg)
    assert bpr_loss.item() > 0
    assert reg_loss.item() > 0

    # User recommendation
    user_recs = model.recommend_for_user(1, k=3)
    assert len(user_recs) == 3
    assert all(r not in sample_sequences[1] for r in user_recs)

    # History pseudo-user recommendation
    hist_recs = model.recommend([1, 2], k=3)
    assert len(hist_recs) == 3


def test_lightgcn_training(sample_sequences: dict[int, list[int]]):
    model = train_lightgcn(
        sample_sequences, num_users=4, num_items=10, epochs=2, batch_size=4, embedding_dim=16
    )
    assert isinstance(model, LightGCN)
    recs = model.recommend_for_user(2, k=3)
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# 4. Two-Tower Retrieval Tests
# ---------------------------------------------------------------------------
def test_two_tower_faiss_retrieval(sample_sequences: dict[int, list[int]]):
    model = TwoTowerModel(num_items=15, embedding_dim=16, output_dim=16)
    model.build_faiss_index()

    cands = model.retrieve_candidates([1, 2], top_k=5)
    assert len(cands) == 5
    assert all(cid not in [1, 2] for cid, _ in cands)

    recs = model.recommend([1, 2], k=5)
    assert len(recs) == 5


def test_two_tower_training(sample_sequences: dict[int, list[int]]):
    model = train_two_tower(
        sample_sequences, num_items=10, epochs=2, batch_size=4, embedding_dim=16, output_dim=16
    )
    assert isinstance(model, TwoTowerModel)
    assert model._faiss_index is not None
    recs = model.recommend([1, 3], k=3)
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# 5. Two-Stage Re-Ranking Pipeline Tests
# ---------------------------------------------------------------------------
def test_two_stage_recommender_pipeline(sample_sequences: dict[int, list[int]]):
    tower = TwoTowerModel(num_items=10, embedding_dim=16, output_dim=16)
    tower.build_faiss_index()
    gru = GRU4Rec(num_items=10, embedding_dim=16, hidden_dim=16)

    two_stage = TwoStageRecommender(
        retrieval_model=tower,
        reranker_model=gru,
        candidate_pool_size=8,
    )

    recs = two_stage.recommend([1, 2], k=4)
    assert len(recs) <= 4
    assert all(0 < r <= 10 for r in recs)


# ---------------------------------------------------------------------------
# 6. API Endpoints Integration Tests
# ---------------------------------------------------------------------------
def test_api_advanced_models_endpoints(mock_lookup: dict[int, dict]):
    tower = TwoTowerModel(num_items=15, embedding_dim=16, output_dim=16)
    tower.build_faiss_index()
    gru = GRU4Rec(num_items=15, embedding_dim=16, hidden_dim=16)
    bert = BERT4Rec(num_items=15, hidden_dim=16, num_heads=2, num_layers=1)
    gcn = LightGCN(num_users=10, num_items=15, embedding_dim=16)
    gcn.set_graph({1: [1, 2, 3]})

    from pipeline.api.security import get_api_key, get_current_tenant

    app.dependency_overrides = {
        deps.get_sequences: lambda: {1: [1, 2, 3]},
        deps.get_movie_lookup: lambda: mock_lookup,
        deps.get_gru4rec_model: lambda: gru,
        deps.get_bert4rec_model: lambda: bert,
        deps.get_lightgcn_model: lambda: gcn,
        deps.get_two_tower_model: lambda: tower,
        get_api_key: lambda: {"client_id": "test_client", "scopes": ["read", "write"]},
        get_current_tenant: lambda: "default",
    }

    client = TestClient(app)

    # 1. GRU4Rec endpoint
    r1 = client.get("/v1/recommend/gru4rec/1?k=3")
    assert r1.status_code == 200
    assert r1.json()["model_name"] == "GRU4Rec"
    assert len(r1.json()["recommendations"]) == 3

    # 2. BERT4Rec endpoint
    r2 = client.get("/v1/recommend/bert4rec/1?k=3")
    assert r2.status_code == 200
    assert r2.json()["model_name"] == "BERT4Rec"
    assert len(r2.json()["recommendations"]) == 3

    # 3. LightGCN endpoint
    r3 = client.get("/v1/recommend/lightgcn/1?k=3")
    assert r3.status_code == 200
    assert r3.json()["model_name"] == "LightGCN"
    assert len(r3.json()["recommendations"]) == 3

    # 4. Two-Tower endpoint
    r4 = client.get("/v1/recommend/two-tower/1?k=3")
    assert r4.status_code == 200
    assert r4.json()["model_name"] == "Two-Tower Retrieval"
    assert len(r4.json()["recommendations"]) == 3

    app.dependency_overrides = {}


# ---------------------------------------------------------------------------
# 7. Benchmark Suite Integration Tests
# ---------------------------------------------------------------------------
def test_advanced_models_in_benchmark_runner(
    sample_sequences: dict[int, list[int]], tmp_path: Path
):
    import pandas as pd

    from pipeline.benchmark import run_benchmark

    movies_df = pd.DataFrame(
        [{"item_id": i, "title": f"Movie {i}", "genres": "Action"} for i in range(1, 15)]
    )

    results = run_benchmark(
        sequences=sample_sequences,
        movies_df=movies_df,
        seeds=[42],
        k_values=(5,),
        include_models=["GRU4Rec", "BERT4Rec", "LightGCN", "TwoTower"],
        output_dir=tmp_path,
    )

    agg = results["aggregated_results"]
    assert "GRU4Rec" in agg
    assert "BERT4Rec" in agg
    assert "LightGCN" in agg
    assert "TwoTower" in agg

    for model_name in ["GRU4Rec", "BERT4Rec", "LightGCN", "TwoTower"]:
        assert "HitRate@5" in agg[model_name]
        assert "NDCG@5" in agg[model_name]
        assert "train_time_sec" in agg[model_name]
