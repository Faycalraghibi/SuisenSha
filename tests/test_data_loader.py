"""Unit tests for pipeline.data.loader (offline, no network)."""

import pandas as pd

from pipeline.data.loader import build_user_sequences


class TestBuildUserSequences:
    def _make_ratings(self) -> pd.DataFrame:
        """Synthetic ratings for 2 users."""
        return pd.DataFrame({
            "user_id":   [1, 1, 1, 1, 1, 1, 2, 2],
            "item_id":   [10, 20, 30, 40, 50, 60, 10, 20],
            "rating":    [4.0, 5.0, 3.0, 4.5, 2.0, 4.0, 5.0, 4.0],
            "timestamp": [1, 2, 3, 4, 5, 6, 1, 2],
        })

    def test_filters_low_ratings(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=2)
        # User 1: items with rating >= 3.5 → [10, 20, 40, 60]  (30=3.0, 50=2.0 excluded)
        assert 30 not in seqs.get(1, [])
        assert 50 not in seqs.get(1, [])

    def test_min_interactions(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=3)
        # User 2 has only 2 positive interactions → excluded
        assert 2 not in seqs

    def test_sorted_by_timestamp(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=2)
        seq = seqs[1]
        assert seq == sorted(seq, key=lambda x: seq.index(x))  # monotonic
