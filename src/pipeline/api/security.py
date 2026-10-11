"""Security, API key authentication, rate limiting, and right-to-erasure registry."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, Header, HTTPException, Request

from pipeline.config import DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_SECURITY_DB = DATA_DIR / "security.db"


def hash_key(plain_key: str) -> str:
    """Return SHA-256 hex digest of plain API key."""
    return hashlib.sha256(plain_key.encode("utf-8")).hexdigest()


class SecurityManager:
    """Manages API keys, rate limiting windows, user erasure registry, and audit logging."""

    def __init__(self, db_path: Path | str = DEFAULT_SECURITY_DB) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

        # In-memory rate limiting: key_hash -> list of monotonic timestamps
        self._rate_limits: dict[str, list[float]] = defaultdict(list)

    def _init_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id      TEXT PRIMARY KEY,
                    key_hash    TEXT UNIQUE NOT NULL,
                    client_id   TEXT NOT NULL,
                    tier        TEXT NOT NULL,
                    rate_limit  INTEGER NOT NULL,
                    scopes      TEXT NOT NULL,
                    is_active   INTEGER NOT NULL,
                    created_at  TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS erased_users (
                    user_id     INTEGER PRIMARY KEY,
                    client_id   TEXT NOT NULL,
                    erased_at   TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    client_id   TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    details     TEXT
                );
                """
            )

    def create_key(
        self,
        client_id: str,
        tier: str = "standard",
        rate_limit: int = 60,
        scopes: list[str] | None = None,
    ) -> tuple[str, str]:
        """Generate and store a new API key.

        Returns:
            Tuple of (plain_key, key_id). Plain key is only returned once upon creation.
        """
        plain_key = f"sk_{secrets.token_urlsafe(32)}"
        key_id = f"key_{secrets.token_hex(8)}"
        k_hash = hash_key(plain_key)
        now = datetime.now(timezone.utc).isoformat()
        scopes_str = ",".join(scopes or ["read"])

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, key_hash, client_id, tier, rate_limit, scopes, is_active, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (key_id, k_hash, client_id, tier, rate_limit, scopes_str, now),
            )
        self.log_audit(
            client_id=client_id,
            action="create_key",
            resource_id=key_id,
            status="success",
            details=f"tier={tier}, limit={rate_limit}",
        )
        return plain_key, key_id

    def get_key_info(self, key_hash: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if not row:
            return None
        return {
            "key_id": row["key_id"],
            "client_id": row["client_id"],
            "tier": row["tier"],
            "rate_limit": row["rate_limit"],
            "scopes": [s.strip() for s in row["scopes"].split(",") if s.strip()],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
        }

    def revoke_key(self, key_id: str, admin_client: str = "system") -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,)
            )
            success = cursor.rowcount > 0
        self.log_audit(
            client_id=admin_client,
            action="revoke_key",
            resource_id=key_id,
            status="success" if success else "not_found",
        )
        return success

    def check_rate_limit(
        self, key_hash: str, limit: int, window_seconds: int = 60
    ) -> tuple[bool, int]:
        """Sliding-window rate limiter.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        now = time.monotonic()
        cutoff = now - window_seconds
        timestamps = self._rate_limits[key_hash]

        # Prune timestamps older than window
        self._rate_limits[key_hash] = [ts for ts in timestamps if ts > cutoff]
        timestamps = self._rate_limits[key_hash]

        if len(timestamps) >= limit:
            oldest = timestamps[0]
            retry_after = max(1, int(oldest + window_seconds - now))
            return False, retry_after

        timestamps.append(now)
        return True, 0

    def is_user_erased(self, user_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM erased_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row is not None

    def erase_user(self, user_id: int, client_id: str) -> dict[str, Any]:
        """Record user erasure in registry for GDPR compliance and retraining exclusion."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO erased_users (user_id, client_id, erased_at)
                VALUES (?, ?, ?)
                """,
                (user_id, client_id, now),
            )
        self.log_audit(
            client_id=client_id,
            action="erasure",
            resource_id=str(user_id),
            status="success",
        )
        return {"status": "erased", "user_id": user_id, "erased_at": now}

    def log_audit(
        self,
        client_id: str,
        action: str,
        resource_id: str,
        status: str = "success",
        details: Any = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        details_str: str | None
        if isinstance(details, dict | list):
            details_str = json.dumps(details)
        else:
            details_str = str(details) if details else None
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO audit_log (timestamp, client_id, action, resource_id, status, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now, client_id, action, resource_id, status, details_str),
            )

    def get_audit_logs(
        self, action: str | None = None, resource_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM audit_log"
        params: list[Any] = []
        conditions: list[str] = []

        if action:
            conditions.append("action = ?")
            params.append(action)
        if resource_id:
            conditions.append("resource_id = ?")
            params.append(resource_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


_security_manager_instance: SecurityManager | None = None


def get_security_manager() -> SecurityManager:
    global _security_manager_instance
    if _security_manager_instance is None:
        _security_manager_instance = SecurityManager()
    return _security_manager_instance


def get_api_key(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
    sec_mgr: SecurityManager = Depends(get_security_manager),
) -> dict[str, Any]:
    """Verify API key authentication and enforce per-key rate limiting."""
    key = x_api_key
    if not key and authorization and authorization.startswith("Bearer "):
        key = authorization[7:].strip()

    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    k_hash = hash_key(key)
    key_info = sec_mgr.get_key_info(k_hash)
    if not key_info or not key_info["is_active"]:
        raise HTTPException(status_code=403, detail="Invalid or revoked API key.")

    # Enforce rate limit
    allowed, retry_after = sec_mgr.check_rate_limit(k_hash, limit=key_info["rate_limit"])
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )

    return key_info


def create_api_key(
    client_id: str,
    tier: str = "standard",
    rate_limit: int = 60,
    scopes: list[str] | None = None,
    sec_mgr: SecurityManager | None = None,
) -> tuple[str, str]:
    """Generate and store an API key, returning (plain_key, key_id)."""
    mgr = sec_mgr or get_security_manager()
    return mgr.create_key(client_id=client_id, tier=tier, rate_limit=rate_limit, scopes=scopes)


def create_default_test_key(sec_mgr: SecurityManager | None = None) -> str:
    """Helper for test environments to generate a valid key."""
    mgr = sec_mgr or get_security_manager()
    plain, _ = mgr.create_key(
        "test_suite_client", rate_limit=1000, scopes=["read", "write", "admin"]
    )
    return plain
