from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from pipeline.config import DATA_DIR, ensure_dirs


class EventType(str, Enum):
    VIEW = "view"
    CLICK = "click"
    PURCHASE = "purchase"
    RATING = "rating"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    NOT_INTERESTED = "not_interested"


class UserEvent(BaseModel):
    event_id: str | None = None
    user_id: int = Field(gt=0, description="Positive user identifier")
    item_id: int = Field(gt=0, description="Positive item identifier")
    event_type: EventType = Field(description="Interaction or feedback type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event UTC timestamp",
    )
    rating: float | None = Field(
        default=None,
        ge=0.5,
        le=5.0,
        description="Rating value between 0.5 and 5.0 (only for rating events)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured metadata",
    )

    @field_validator("event_type", mode="before")
    @classmethod
    def parse_event_type(cls, v: Any) -> EventType:
        if isinstance(v, str):
            try:
                return EventType(v.lower())
            except ValueError:
                valid = [e.value for e in EventType]
                raise ValueError(f"Invalid event type: {v}. Must be one of {valid}")
        return v


class EventStore:
    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            ensure_dirs()
            self.db_path = str(DATA_DIR / "events.db")
        else:
            self.db_path = str(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Any:
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
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    rating REAL,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    ingested_count INTEGER NOT NULL,
                    event_ids TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_exclusions (
                    user_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, item_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_exclusions_user ON user_exclusions(user_id)"
            )
            conn.commit()

    def record_batch(
        self,
        events: list[UserEvent],
        idempotency_key: str | None = None,
    ) -> tuple[int, bool]:
        """Record a batch of events with optional idempotency checking.

        Returns (ingested_count, is_idempotent_replay).
        """
        if not events:
            return 0, False

        with self._get_connection() as conn:
            if idempotency_key:
                cur = conn.execute(
                    "SELECT ingested_count FROM idempotency_keys WHERE idempotency_key = ?",
                    (idempotency_key,),
                )
                row = cur.fetchone()
                if row is not None:
                    return int(row["ingested_count"]), True

            event_ids: list[str] = []
            for ev in events:
                eid = ev.event_id or uuid.uuid4().hex
                event_ids.append(eid)
                ts_str = ev.timestamp.isoformat()
                meta_str = json.dumps(ev.metadata)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO events
                    (event_id, user_id, item_id, event_type, timestamp, rating, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        ev.user_id,
                        ev.item_id,
                        ev.event_type.value,
                        ts_str,
                        ev.rating,
                        meta_str,
                    ),
                )

                if ev.event_type == EventType.NOT_INTERESTED:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO user_exclusions
                        (user_id, item_id, reason, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ev.user_id, ev.item_id, "not_interested", ts_str),
                    )

            if idempotency_key:
                conn.execute(
                    """
                    INSERT INTO idempotency_keys
                    (idempotency_key, created_at, ingested_count, event_ids)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        datetime.now(timezone.utc).isoformat(),
                        len(events),
                        json.dumps(event_ids),
                    ),
                )
            conn.commit()

        return len(events), False

    def get_user_excluded_items(self, user_id: int) -> set[int]:
        """Get set of item IDs immediately excluded for this user."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT item_id FROM user_exclusions WHERE user_id = ?",
                (user_id,),
            )
            return {int(row["item_id"]) for row in cur.fetchall()}

    def get_user_events(
        self,
        user_id: int,
        event_types: list[EventType] | None = None,
    ) -> list[UserEvent]:
        query = (
            "SELECT event_id, user_id, item_id, event_type, timestamp, rating, metadata "
            "FROM events WHERE user_id = ?"
        )
        params: list[Any] = [user_id]
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(e.value for e in event_types)
        query += " ORDER BY timestamp ASC"

        with self._get_connection() as conn:
            cur = conn.execute(query, tuple(params))
            results: list[UserEvent] = []
            for row in cur.fetchall():
                ts = datetime.fromisoformat(row["timestamp"])
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                results.append(
                    UserEvent(
                        event_id=row["event_id"],
                        user_id=row["user_id"],
                        item_id=row["item_id"],
                        event_type=EventType(row["event_type"]),
                        timestamp=ts,
                        rating=row["rating"],
                        metadata=meta,
                    )
                )
            return results

    def count_events(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM events")
            return int(cur.fetchone()[0])

    def erase_user_events(self, user_id: int) -> int:
        """Permanently erase all events and exclusions for a user (GDPR Art. 17)."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM events WHERE user_id = ?", (user_id,))
            count = int(cur.fetchone()[0])
            conn.execute("DELETE FROM events WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_exclusions WHERE user_id = ?", (user_id,))
            conn.commit()
            return count

    def merge_events_into_ratings(
        self,
        ratings_df: pd.DataFrame,
        min_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Merge positive interaction events (clicks, purchases, positive ratings, thumbs-up)

        into an interactions/ratings DataFrame for model retraining.
        """
        with self._get_connection() as conn:
            query = """
                SELECT user_id, item_id, event_type, timestamp, rating
                FROM events
                WHERE event_type IN ('click', 'purchase', 'thumbs_up', 'rating')
            """
            params: list[Any] = []
            if min_timestamp:
                query += " AND timestamp >= ?"
                params.append(min_timestamp.isoformat())

            cur = conn.execute(query, tuple(params))
            new_rows: list[dict[str, Any]] = []
            for row in cur.fetchall():
                etype = row["event_type"]
                rtg: float = 4.0
                if etype == "purchase" or etype == "thumbs_up":
                    rtg = 5.0
                elif etype == "rating" and row["rating"] is not None:
                    rtg = float(row["rating"])
                elif etype == "click":
                    rtg = 4.0

                ts = datetime.fromisoformat(row["timestamp"])
                # Match ratings_df timestamp format (usually integer unix epoch)
                first_ts = ratings_df["timestamp"].iloc[0] if len(ratings_df) > 0 else 0
                is_int_ts = isinstance(first_ts, int | float)
                ts_val: Any = int(ts.timestamp()) if is_int_ts else ts.isoformat()

                new_rows.append(
                    {
                        "user_id": int(row["user_id"]),
                        "item_id": int(row["item_id"]),
                        "rating": rtg,
                        "timestamp": ts_val,
                    }
                )

        if not new_rows:
            return ratings_df.copy()

        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([ratings_df, new_df], ignore_index=True)
        # Stable sort on user_id and timestamp
        combined = combined.sort_values(["user_id", "timestamp"], kind="stable").reset_index(
            drop=True
        )
        return combined
