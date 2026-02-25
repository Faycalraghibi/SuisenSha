"""Unit tests for pipeline.evaluation.metrics."""

from pipeline.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k([1, 2, 3], target=2, k=3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k([1, 2, 3], target=99, k=3) == 0.0

    def test_beyond_k(self):
        assert hit_rate_at_k([1, 2, 3, 4, 5], target=5, k=3) == 0.0


class TestNDCG:
    def test_first_position(self):
        assert ndcg_at_k([42, 1, 2], target=42, k=3) == 1.0

    def test_second_position(self):
        expected = 1.0 / 1.5849625007211563  # log2(3)
        assert abs(ndcg_at_k([1, 42, 2], target=42, k=3) - expected) < 1e-6

    def test_miss(self):
        assert ndcg_at_k([1, 2, 3], target=99, k=3) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k([1, 2, 3], relevant={1, 2, 3}, k=3) == 1.0

    def test_half_relevant(self):
        assert precision_at_k([1, 2, 3, 4], relevant={1, 3}, k=4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k([1, 2], relevant={99}, k=2) == 0.0


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k([1, 2, 3], relevant={1, 2}, k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k([1, 2, 3], relevant={1, 2, 4, 5}, k=3) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k([1, 2], relevant=set(), k=2) == 0.0
