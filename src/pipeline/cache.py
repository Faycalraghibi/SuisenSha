"""Stores generated rationale text per user and candidate set so the API can serve them
instantly instead of running the LLM on every request.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def delete_user(self, user_id: int) -> int:
        """Remove all cached entries for user_id (GDPR right to erasure)."""
        cursor = self._conn.execute("DELETE FROM rag_cache_v2 WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cursor.rowcount

    def user_entry_count(self, user_id: int) -> int:
        """Count cached entries for a specific user."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM rag_cache_v2 WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> RecommendationCache:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()


class RedisCache:
    """Redis cache for serving precomputed top-N recommendations with in-memory fallback."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "rec:",
        fallback_to_memory: bool = True,
    ) -> None:
        self.prefix = prefix
        self.is_fallback_mode = False
        self._memory_store: dict[str, tuple[str, float | None]] = {}
        self._redis_client: Any = None

        try:
            import redis

            client = redis.from_url(redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
            client.ping()
            self._redis_client = client
        except Exception:
            if fallback_to_memory:
                self.is_fallback_mode = True
                self._redis_client = None
            else:
                raise

    def _make_key(self, user_id: int, model: str, tenant_id: str = "default") -> str:
        return f"{self.prefix}{tenant_id}:{model.lower()}:{user_id}"

    def get_top_n(
        self,
        user_id: int,
        model: str = "sasrec",
        tenant_id: str = "default",
    ) -> list[dict[str, Any]] | None:
        key = self._make_key(user_id, model, tenant_id=tenant_id)
        if self.is_fallback_mode or self._redis_client is None:
            entry = self._memory_store.get(key)
            if entry is None:
                return None
            val_str, exp_time = entry
            if exp_time is not None and datetime.now(timezone.utc).timestamp() > exp_time:
                self._memory_store.pop(key, None)
                return None
            return json.loads(val_str)

        try:
            val = self._redis_client.get(key)
            if val is None:
                return None
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            return json.loads(val)
        except Exception as exc:
            logger.warning("Redis get error, falling back: %s", exc)
            return None

    def set_top_n(
        self,
        user_id: int,
        items: list[dict[str, Any]],
        model: str = "sasrec",
        ttl_seconds: int = 86400,
        tenant_id: str = "default",
    ) -> None:
        key = self._make_key(user_id, model, tenant_id=tenant_id)
        val_str = json.dumps(items)
        if self.is_fallback_mode or self._redis_client is None:
            exp_time = (
                datetime.now(timezone.utc).timestamp() + ttl_seconds if ttl_seconds > 0 else None
            )
            self._memory_store[key] = (val_str, exp_time)
            return

        try:
            if ttl_seconds > 0:
                self._redis_client.setex(key, ttl_seconds, val_str)
            else:
                self._redis_client.set(key, val_str)
        except Exception as exc:
            logger.warning("Redis set error: %s", exc)
            exp_time = (
                datetime.now(timezone.utc).timestamp() + ttl_seconds if ttl_seconds > 0 else None
            )
            self._memory_store[key] = (val_str, exp_time)

    def delete_user(self, user_id: int, tenant_id: str | None = None) -> int:
        count = 0
        suffix = f":{user_id}"
        prefix_filter = f"{self.prefix}{tenant_id}:" if tenant_id else self.prefix

        if self.is_fallback_mode or self._redis_client is None:
            keys_to_del = [
                k for k in self._memory_store if k.startswith(prefix_filter) and k.endswith(suffix)
            ]
            for k in keys_to_del:
                self._memory_store.pop(k, None)
                count += 1
            return count

        try:
            pattern = (
                f"{self.prefix}{tenant_id}:*:{user_id}"
                if tenant_id
                else f"{self.prefix}*:{user_id}"
            )
            keys = self._redis_client.keys(pattern)

            if keys:
                count = self._redis_client.delete(*keys)
        except Exception as exc:
            logger.warning("Redis delete error: %s", exc)
        return count
