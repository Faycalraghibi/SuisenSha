"""Stores generated rationale text per user and candidate set so the API can serve them
instantly instead of running the LLM on every request.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import RAG_CACHE_DB

logger = logging.getLogger(__name__)

_CREATE_TABLE_LEGACY = """
CREATE TABLE IF NOT EXISTS rag_cache (
    user_id    INTEGER PRIMARY KEY,
    rationale  TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
"""

_CREATE_TABLE_V2 = """
CREATE TABLE IF NOT EXISTS rag_cache_v2 (
    cache_key      TEXT PRIMARY KEY,
    user_id        INTEGER NOT NULL,
    candidate_ids  TEXT NOT NULL,
    lang           TEXT NOT NULL,
    rationale      TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
"""


def make_cache_key(user_id: int, candidate_ids: list[int] | None = None, lang: str = "en") -> str:
    """Build a deterministic cache key from user ID, candidates, and language."""
    if candidate_ids is None:
        cands_str = "*"
    else:
        cands_str = ",".join(str(cid) for cid in sorted(candidate_ids))
    return f"{user_id}:{lang.lower()}:{cands_str}"


class RecommendationCache:
    def __init__(self, db_path: Path | str = RAG_CACHE_DB) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI may open and use one connection on different worker threads;
        # each connection is still used by only one request at a time.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_LEGACY)
        self._conn.execute(_CREATE_TABLE_V2)
        self._conn.commit()

    def get(
        self,
        user_id: int,
        candidate_ids: list[int] | None = None,
        lang: str = "en",
    ) -> str | None:
        key = make_cache_key(user_id, candidate_ids, lang)
        row = self._conn.execute(
            "SELECT rationale FROM rag_cache_v2 WHERE cache_key = ?", (key,)
        ).fetchone()
        if row:
            return row[0]

        # Fallback to wildcard key (e.g. cached without specific candidates)
        if candidate_ids is not None:
            wildcard_key = make_cache_key(user_id, None, lang)
            wildcard_row = self._conn.execute(
                "SELECT rationale FROM rag_cache_v2 WHERE cache_key = ?", (wildcard_key,)
            ).fetchone()
            if wildcard_row:
                return wildcard_row[0]

        # Fallback to legacy single user_id table
        legacy_row = self._conn.execute(
            "SELECT rationale FROM rag_cache WHERE user_id = ?", (user_id,)
        ).fetchone()
        if legacy_row:
            return legacy_row[0]

        return None

    def has(
        self,
        user_id: int,
        candidate_ids: list[int] | None = None,
        lang: str = "en",
    ) -> bool:
        key = make_cache_key(user_id, candidate_ids, lang)
        row = self._conn.execute(
            "SELECT 1 FROM rag_cache_v2 WHERE cache_key = ?", (key,)
        ).fetchone()
        if row:
            return True
        if candidate_ids is not None:
            wildcard_key = make_cache_key(user_id, None, lang)
            wildcard_row = self._conn.execute(
                "SELECT 1 FROM rag_cache_v2 WHERE cache_key = ?", (wildcard_key,)
            ).fetchone()
            if wildcard_row:
                return True
        legacy_row = self._conn.execute(
            "SELECT 1 FROM rag_cache WHERE user_id = ?", (user_id,)
        ).fetchone()
        return legacy_row is not None

    def put(
        self,
        user_id: int,
        rationale: str,
        candidate_ids: list[int] | None = None,
        lang: str = "en",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        key = make_cache_key(user_id, candidate_ids, lang)
        cands_str = ",".join(str(cid) for cid in sorted(candidate_ids)) if candidate_ids else "*"

        self._conn.execute(
            "INSERT OR REPLACE INTO rag_cache_v2 "
            "(cache_key, user_id, candidate_ids, lang, rationale, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, user_id, cands_str, lang.lower(), rationale, now),
        )
        if candidate_ids is None:
            self._conn.execute(
                "INSERT OR REPLACE INTO rag_cache "
                "(user_id, rationale, created_at) VALUES (?, ?, ?)",
                (user_id, rationale, now),
            )
        self._conn.commit()

    def count(self) -> int:
        row_v2 = self._conn.execute("SELECT COUNT(*) FROM rag_cache_v2").fetchone()
        count_v2 = row_v2[0] if row_v2 else 0
        row_legacy = self._conn.execute("SELECT COUNT(*) FROM rag_cache").fetchone()
        count_legacy = row_legacy[0] if row_legacy else 0
        return max(count_v2, count_legacy)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RecommendationCache:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()
