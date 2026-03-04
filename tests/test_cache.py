"""Tests for the RecommendationCache (SQLite-backed RAG cache)."""

from __future__ import annotations

import tempfile
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
