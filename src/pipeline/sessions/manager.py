from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import DATA_DIR, ensure_dirs


@dataclass
class Session:
    """Represents an anonymous user browsing session and its interaction sequence."""

    session_id: str
    item_ids: list[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "item_ids": self.item_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "event_count": len(self.item_ids),
            "metadata": self.metadata,
        }


class SessionStore:
    """Thread-safe SQLite-backed session store for anonymous real-time interaction sequences."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            ensure_dirs()
            self.db_path = str(DATA_DIR / "sessions.db")
        else:
            self.db_path = str(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_events_sid ON session_events(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_events_ts ON session_events(timestamp)"
            )
            conn.commit()

    def record_event(
        self,
        session_id: str,
        item_id: int,
        event_type: str = "click",
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> Session:
        """Record an interaction event for an anonymous session and update session timestamp."""
        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat()
        meta_str = json.dumps(metadata or {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO session_events (session_id, item_id, event_type, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, item_id, event_type, ts_str, meta_str),
            )
            cur = conn.execute(
                "SELECT created_at, metadata FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, ts_str, ts_str, meta_str),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (ts_str, session_id),
                )
            conn.commit()

        return self.get_session(session_id) or Session(
            session_id=session_id, item_ids=[item_id], created_at=ts, updated_at=ts
        )

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve session information and chronologically ordered interaction items."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT session_id, created_at, updated_at, metadata "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None

            c_ts = datetime.fromisoformat(row["created_at"])
            u_ts = datetime.fromisoformat(row["updated_at"])
            meta = json.loads(row["metadata"]) if row["metadata"] else {}

            cur_ev = conn.execute(
                """
                SELECT item_id FROM session_events
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            )
            item_ids = [int(r["item_id"]) for r in cur_ev.fetchall()]

            return Session(
                session_id=session_id,
                item_ids=item_ids,
                created_at=c_ts,
                updated_at=u_ts,
                metadata=meta,
            )

    def get_session_items(self, session_id: str) -> list[int]:
        """Convenience method to retrieve item IDs in chronological order."""
        sess = self.get_session(session_id)
        return sess.item_ids if sess else []

    def clear_session(self, session_id: str) -> bool:
        """Clear all events and metadata for an anonymous session."""
        with self._get_connection() as conn:
            cur1 = conn.execute("DELETE FROM session_events WHERE session_id = ?", (session_id,))
            cur2 = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return (cur1.rowcount + cur2.rowcount) > 0

    def count_sessions(self) -> int:
        """Count total unique sessions tracked."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM sessions")
            row = cur.fetchone()
            return int(row[0]) if row else 0


# Alias SessionManager to SessionStore for ergonomic imports
SessionManager = SessionStore
