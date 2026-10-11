from __future__ import annotations

import pickle
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import pipeline.config as cfg
from pipeline.api.main import app
from pipeline.data.loader import build_user_sequences
from pipeline.evaluation.metrics import run_phase5
from pipeline.models.embedding import run_phase2
from pipeline.models.rag import run_phase4
from pipeline.models.sequential import (
    SASRec,
    pad_right,
    predict_next_items,
    run_phase3,
)


@pytest.fixture
def fixture_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up an isolated fixture environment with paths redirected to tmp_path."""
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    outputs_dir = tmp_path / "outputs"
    for d in (data_dir, artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    import pipeline.api.dependencies as api_deps
    import pipeline.api.main as api_main
    import pipeline.cache as cache_mod
    import pipeline.data.loader as loader_mod
    import pipeline.evaluation.metrics as metrics_mod
    import pipeline.models.embedding as emb_mod
    import pipeline.models.sequential as seq_mod

    modules = [cfg, api_deps, api_main, cache_mod, loader_mod, metrics_mod, emb_mod, seq_mod]
    path_mappings = {
        "DATA_DIR": data_dir,
        "ARTIFACTS_DIR": artifacts_dir,
        "OUTPUTS_DIR": outputs_dir,
        "INTERACTIONS_CSV": data_dir / "interactions.csv",
        "MOVIES_CSV": data_dir / "movies.csv",
        "USER_SEQUENCES_PKL": data_dir / "user_sequences.pkl",
        "FAISS_INDEX_PATH": artifacts_dir / "faiss_index.bin",
        "EMBEDDINGS_NPY": artifacts_dir / "item_embeddings.npy",
        "ITEM_IDS_NPY": artifacts_dir / "item_ids.npy",
        "SEQ_MODEL_PATH": artifacts_dir / "sasrec_model.pt",
        "RAG_CACHE_DB": artifacts_dir / "rag_cache.db",
    }
    for m in modules:
        for attr, val in path_mappings.items():
            if hasattr(m, attr):
                monkeypatch.setattr(m, attr, val)

    # Fast 2-epoch SASRec configuration for e2e test
    from pipeline.config import SASRecConfig

    monkeypatch.setattr(cfg, "sasrec_cfg", SASRecConfig(epochs=2, batch_size=16, hidden_dim=16))

    # Create 50 users and 60 movies
    num_users = 50
    num_items = 60

    genres_list = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]
    movie_rows = []
    for iid in range(1, num_items + 1):
        g = f"{genres_list[iid % len(genres_list)]}|{genres_list[(iid + 1) % len(genres_list)]}"
        title = f"Movie {iid} (2020)"
        movie_rows.append(
            {
                "item_id": iid,
                "title": title,
                "genres": g,
                "description": f"{title} | {g}",
            }
        )
    movies_df = pd.DataFrame(movie_rows)
    movies_df.to_csv(data_dir / "movies.csv", index=False)

    rng = np.random.default_rng(42)
    interaction_rows = []
    for uid in range(1, num_users + 1):
        # 8 to 15 interactions per user
        n_items = int(rng.integers(8, 16))
        chosen_items = rng.choice(range(1, num_items + 1), size=n_items, replace=False)
        for step, iid in enumerate(chosen_items):
            interaction_rows.append(
                {
                    "user_id": uid,
                    "item_id": int(iid),
                    "rating": float(rng.choice([3.5, 4.0, 4.5, 5.0])),
                    "timestamp": 1600000000 + step * 100,
                }
            )
    interactions_df = pd.DataFrame(interaction_rows)
    interactions_df.to_csv(data_dir / "interactions.csv", index=False)

    sequences = build_user_sequences(interactions_df, min_interactions=5)
    with open(data_dir / "user_sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    return movies_df, sequences


def test_e2e_fixture_pipeline_under_one_minute(fixture_env):
    """Run full phases 1-5 on a 50-user fixture in < 60s, then test real API lifespan."""
    movies_df, sequences = fixture_env

    t_start = time.perf_counter()

    # Phase 2: Embedding + FAISS
    emb_results = run_phase2(movies_df, sequences)
    assert "HitRate@10" in emb_results

    # Phase 3: SASRec training
    model = run_phase3(sequences)
    assert model is not None
    assert cfg.SEQ_MODEL_PATH.exists()

    # Phase 4: RAG generation
    with patch("pipeline.models.rag.get_llm_backend", return_value=None):
        run_phase4(movies_df, sequences)

    # Phase 5: Evaluation
    run_phase5()
    assert (cfg.OUTPUTS_DIR / "comparison.png").exists()

    elapsed = time.perf_counter() - t_start
    # Verify the entire pipeline runs in under 60 seconds
    assert elapsed < 60.0, f"Pipeline took {elapsed:.1f}s, expected < 60s"

    # Now verify the API with the real lifespan against the generated artifacts
    app.dependency_overrides = {}
    with TestClient(app) as client:
        # History
        resp = client.get("/users/1/history")
        assert resp.status_code == 200
        assert len(resp.json()["recent_history"]) > 0

        # Embedding
        resp = client.get("/recommend/embedding/1")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) > 0

        # SASRec
        resp = client.get("/recommend/sasrec/1")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) > 0

        # RAG
        resp = client.get("/recommend/rag/1")
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) > 0
        assert "reason" in resp.json()["recommendations"][0]


def test_reverting_padding_fix_fails():
    """Verify that reproducing the old defect (left-padding and reading hidden state at len-1)
    differs from the correct last_logits result and that predictions change with recent items.
    """
    model = SASRec(num_items=50, hidden_dim=16, maxlen=20, num_heads=2, num_layers=2)
    model.eval()

    short_seq = [10, 20, 30]

    # Correct right-padding and last_logits evaluation
    padded_right = pad_right(short_seq, 20)
    tensor_right = torch.tensor([padded_right], dtype=torch.long)
    logits_correct = model.last_logits(tensor_right)[0]

    # Old buggy behaviour: left-padding a short sequence and indexing at index len-1
    left_padded = [0] * (20 - len(short_seq)) + short_seq
    tensor_left = torch.tensor([left_padded], dtype=torch.long)
    hidden_left = model.encode(tensor_left)
    # Reading index len - 1 or taking position corrupted by padding mask
    buggy_logits = model.output_proj(hidden_left[0, len(short_seq) - 1])

    # Assert that the buggy approach produces different representations than correct last_logits
    assert not torch.allclose(logits_correct, buggy_logits, atol=1e-3)
    assert torch.isfinite(logits_correct).all()

    # Assert that predict_next_items output changes when the most recent item changes
    seq_a = [10, 20, 30]
    seq_b = [10, 20, 35]
    preds_a = predict_next_items(model, seq_a, top_k=5)
    preds_b = predict_next_items(model, seq_b, top_k=5)
    assert preds_a != preds_b
