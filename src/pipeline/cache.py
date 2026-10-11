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
        self._conn.execute(_CREATE_TABLE_V2)
        self._conn.commit()

    def get(
        self,
        user_id: int,
        candidate_ids: list[int] | None = None,
        lang: str = "en",
    ) -> str | None:
        """Get cached rationale. Exact match on user_id, candidate set, and language required."""
        key = make_cache_key(user_id, candidate_ids, lang)
        row = self._conn.execute(
            "SELECT rationale FROM rag_cache_v2 WHERE cache_key = ?", (key,)
        ).fetchone()
        if row:
            return row[0]
        return None

    def has(
        self,
        user_id: int,
        candidate_ids: list[int] | None = None,
        lang: str = "en",
    ) -> bool:
        """Check if cached rationale exists for exact user_id, candidates, and language."""
        key = make_cache_key(user_id, candidate_ids, lang)
        row = self._conn.execute(
            "SELECT 1 FROM rag_cache_v2 WHERE cache_key = ?", (key,)
        ).fetchone()
        return row is not None

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
        self._conn.commit()

    def count(self) -> int:
        row_v2 = self._conn.execute("SELECT COUNT(*) FROM rag_cache_v2").fetchone()
        return row_v2[0] if row_v2 else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RecommendationCache:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()
