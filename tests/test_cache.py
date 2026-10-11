"""Tests for the RecommendationCache (SQLite-backed RAG cache)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.cache import RecommendationCache


@pytest.fixture
def cache(tmp_path: Path) -> RecommendationCache:
    db = tmp_path / "test_cache.db"
    c = RecommendationCache(db_path=db)
    yield c
    c.close()


def test_put_and_get(cache: RecommendationCache) -> None:
    cache.put(42, "Great movies for you!")
    assert cache.get(42) == "Great movies for you!"


def test_get_missing_returns_none(cache: RecommendationCache) -> None:
    assert cache.get(999) is None


def test_has(cache: RecommendationCache) -> None:
    assert not cache.has(1)
    cache.put(1, "rationale")
    assert cache.has(1)


def test_count(cache: RecommendationCache) -> None:
    assert cache.count() == 0
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.count() == 2


def test_put_replaces_existing(cache: RecommendationCache) -> None:
    cache.put(1, "old text")
    cache.put(1, "new text")
    assert cache.get(1) == "new text"
    assert cache.count() == 1


def test_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with RecommendationCache(db_path=db) as c:
        c.put(10, "hello")
        assert c.get(10) == "hello"


def test_cache_misses_on_mismatched_candidates(cache: RecommendationCache) -> None:
    cache.put(1, "English rationale for 10 and 20", candidate_ids=[10, 20], lang="en")
    # Mismatched candidate sets must miss
    assert cache.get(1, candidate_ids=[10, 30], lang="en") is None
    assert not cache.has(1, candidate_ids=[10, 30], lang="en")
    # Exact candidate match hits
    assert cache.get(1, candidate_ids=[20, 10], lang="en") == "English rationale for 10 and 20"
    assert cache.has(1, candidate_ids=[20, 10], lang="en")


def test_cache_misses_on_mismatched_language(cache: RecommendationCache) -> None:
    cache.put(1, "English rationale", candidate_ids=[10, 20], lang="en")
    assert cache.get(1, candidate_ids=[10, 20], lang="fr") is None
    assert not cache.has(1, candidate_ids=[10, 20], lang="fr")


def test_cache_no_wildcard_fallback_on_specific_query(cache: RecommendationCache) -> None:
    # A generic put without candidates (or with diff candidates) must never match a specific query
    cache.put(1, "Generic explanation")
    assert cache.get(1, candidate_ids=[272, 483], lang="fr") is None
    assert not cache.has(1, candidate_ids=[272, 483], lang="fr")
    assert cache.get(1, candidate_ids=[272, 483], lang="en") is None
    assert not cache.has(1, candidate_ids=[272, 483], lang="en")


def test_batch_entry_hit_by_api_path(cache: RecommendationCache) -> None:
    candidates = [272, 483, 100]
    payload = '{"rationale": "batch generated", "recommendations": []}'
    cache.put(1, payload, candidate_ids=candidates, lang="en")

    # API query path with same user, candidate list (different permutation), and language
    hit = cache.get(1, candidate_ids=[100, 272, 483], lang="en")
    assert hit == payload
    assert cache.has(1, candidate_ids=[100, 272, 483], lang="en")
