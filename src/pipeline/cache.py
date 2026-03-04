"""SQLite-backed cache for pre-computed RAG recommendations.

Stores generated rationale text per user so the API can serve them
instantly instead of running DistilGPT-2 on every request.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import RAG_CACHE_DB

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS rag_cache (
    user_id    INTEGER PRIMARY KEY,
    rationale  TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
"""


class RecommendationCache:
    """Thin wrapper around a SQLite database for RAG rationale caching."""

    def __init__(self, db_path: Path | str = RAG_CACHE_DB) -> None:
        self._db_path = str(db_path)
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, user_id: int) -> str | None:
        """Return cached rationale for *user_id*, or ``None`` on miss."""
        row = self._conn.execute(
            "SELECT rationale FROM rag_cache WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else None

    def has(self, user_id: int) -> bool:
        """Return ``True`` if a cached entry exists for *user_id*."""
        return self.get(user_id) is not None

    def put(self, user_id: int, rationale: str) -> None:
        """Insert or replace the rationale for *user_id*."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO rag_cache (user_id, rationale, created_at) VALUES (?, ?, ?)",
            (user_id, rationale, now),
        )
        self._conn.commit()

    def count(self) -> int:
        """Return the total number of cached entries."""
        row = self._conn.execute("SELECT COUNT(*) FROM rag_cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RecommendationCache:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()
