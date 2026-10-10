"""Tests for the FAISS-backed ItemIndex (tiny hand-made vectors, no model download)."""

from __future__ import annotations

import numpy as np

from pipeline.evaluation.metrics import evaluate_leave_one_out
from pipeline.models.embedding import build_item_index


def _index():
    # Items 1/2 point along x, items 3/4 along y
    vecs = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype="float32",
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return build_item_index(np.array([1, 2, 3, 4]), vecs)


def test_search_excludes_history_and_ranks_by_similarity():
    results = _index().search([1], top_k=3)
    assert [iid for iid, _ in results] == [2, 4, 3]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_unknown_history_returns_empty():
    assert _index().search([999], top_k=3) == []


def test_search_caps_at_catalogue_size():
    assert len(_index().search([1], top_k=100)) == 3


def test_evaluate_leave_one_out_scores_all_k():
    sequences = {1: [1, 2], 2: [3, 1]}
    result = evaluate_leave_one_out(sequences, _index().recommend, k_values=(1, 3))
    # User 1: target 2 ranked 1st. User 2: target 1 ranked 3rd (after 4, 2).
    assert result["HitRate@1"] == 0.5
    assert result["HitRate@3"] == 1.0
    assert abs(result["NDCG@3"] - (1.0 + 0.5) / 2) < 1e-9
