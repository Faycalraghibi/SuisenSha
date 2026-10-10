from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pipeline.cache import RecommendationCache, make_cache_key
from pipeline.models.rag import generate_recommendations, retrieve_candidates


def test_cache_composite_key(tmp_path: Path):
    db_path = tmp_path / "test_comp_cache.db"
    with RecommendationCache(db_path=db_path) as cache:
        # User 1 with candidates [10, 20] in English
        cache.put(1, "En reason 1", candidate_ids=[10, 20], lang="en")
        # User 1 with candidates [10, 20] in French
        cache.put(1, "Fr reason 1", candidate_ids=[10, 20], lang="fr")
        # User 1 with different candidates [30, 40] in English
        cache.put(1, "En reason 2", candidate_ids=[30, 40], lang="en")

        assert cache.get(1, candidate_ids=[10, 20], lang="en") == "En reason 1"
        assert cache.get(1, candidate_ids=[10, 20], lang="fr") == "Fr reason 1"
        assert cache.get(1, candidate_ids=[30, 40], lang="en") == "En reason 2"
        assert cache.get(1, candidate_ids=[50, 60], lang="en") is None


def test_make_cache_key_deterministic_sorting():
    key1 = make_cache_key(5, [20, 10], "en")
    key2 = make_cache_key(5, [10, 20], "EN")
    assert key1 == key2
    assert key1 == "5:en:10,20"


def test_retrieve_candidates_supports_different_recommenders():
    movie_lookup = {
        10: {"title": "Movie 10", "genres": "Drama"},
        20: {"title": "Movie 20", "genres": "Action"},
    }

    # Fake SASRec model that returns item IDs
    fake_sasrec = MagicMock()
    fake_sasrec.return_value = [10, 20]

    cands_callable = retrieve_candidates(
        user_history_ids=[1],
        recommender=fake_sasrec,
        movie_lookup=movie_lookup,
        top_k=2,
    )
    assert len(cands_callable) == 2
    assert cands_callable[0]["item_id"] == 10
    assert cands_callable[1]["item_id"] == 20


def test_generate_recommendations_defaults_to_sasrec():
    # If fake SASRec function is provided, verify it is used
    movie_lookup = {
        10: {"title": "Matrix", "genres": "Sci-Fi"},
    }
    fake_recommender = lambda hist, k: [10]  # noqa: E731
    recs = generate_recommendations(
        user_history_ids=[1],
        recommender=fake_recommender,
        movie_lookup=movie_lookup,
        top_k=1,
    )
    assert len(recs) == 1
    assert recs[0].item_id == 10
    assert recs[0].title == "Matrix"
