"""Unit tests for pipeline.data.loader (offline, no network)."""

import pandas as pd

from pipeline.data.loader import build_user_sequences


class TestBuildUserSequences:
    def _make_ratings(self) -> pd.DataFrame:
        """Synthetic ratings for 2 users."""
        return pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 1, 1, 2, 2],
                "item_id": [10, 20, 30, 40, 50, 60, 10, 20],
                "rating": [4.0, 5.0, 3.0, 4.5, 2.0, 4.0, 5.0, 4.0],
                "timestamp": [1, 2, 3, 4, 5, 6, 1, 2],
            }
        )

    def test_filters_low_ratings(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=2)
        assert 30 not in seqs.get(1, [])
        assert 50 not in seqs.get(1, [])

    def test_min_interactions(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=3)
        assert 2 not in seqs

    def test_sorted_by_timestamp(self):
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 1],
                "item_id": [10, 20, 30, 40],
                "rating": [5.0, 5.0, 5.0, 5.0],
                "timestamp": [30, 10, 40, 20],
            }
        )
        seqs = build_user_sequences(ratings, min_rating=3.5, min_interactions=1)
        assert seqs[1] == [20, 40, 10, 30]

    def test_returns_plain_ints(self):
        seqs = build_user_sequences(self._make_ratings(), min_rating=3.5, min_interactions=2)
        assert all(type(u) is int for u in seqs)
        assert all(type(i) is int for seq in seqs.values() for i in seq)
