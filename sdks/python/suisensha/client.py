"""SuisenSha Python Client SDK."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class SuisenShaError(Exception):
    """Base exception for SuisenSha API errors."""

    def __init__(self, message: str, status_code: int = 500, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class SuisenShaClient:
    """Client for the SuisenSha Recommendation API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        tenant_id: str = "default",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None}
            if clean_params:
                url += "?" + urllib.parse.urlencode(clean_params)

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            req_headers["X-API-Key"] = self.api_key
        if self.tenant_id:
            req_headers["X-Tenant-ID"] = self.tenant_id
        if headers:
            req_headers.update(headers)

        body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
        req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                detail = err_json.get("detail", err_body)
            except Exception:
                detail = err_body
            raise SuisenShaError(
                f"API Error ({exc.code}): {detail}",
                status_code=exc.code,
                detail=detail,
            ) from exc
        except urllib.error.URLError as exc:
            raise SuisenShaError(f"Connection error: {exc.reason}", status_code=503) from exc

    # Recommendations
    def recommend_popular(
        self,
        user_id: int | None = None,
        k: int = 10,
        exclude: str | None = None,
        genres: str | None = None,
        diversity: float = 0.0,
    ) -> dict[str, Any]:
        """Fetch global popular recommendations."""
        return self._request(
            "GET",
            "/v1/recommend/popular",
            params={
                "user_id": user_id,
                "k": k,
                "exclude": exclude,
                "genres": genres,
                "diversity": diversity,
            },
        )

    def recommend_sasrec(
        self,
        user_id: int,
        k: int = 10,
        exclude: str | None = None,
        genres: str | None = None,
        diversity: float = 0.0,
        cold_start: bool = False,
    ) -> dict[str, Any]:
        """Fetch transformer sequential recommendations."""
        return self._request(
            "GET",
            f"/v1/recommend/sasrec/{user_id}",
            params={
                "k": k,
                "exclude": exclude,
                "genres": genres,
                "diversity": diversity,
                "cold_start": cold_start,
            },
        )

    def recommend_hybrid(
        self,
        user_id: int,
        k: int = 10,
        alpha: float = 0.6,
        exclude: str | None = None,
        genres: str | None = None,
        diversity: float = 0.0,
        cold_start: bool = False,
    ) -> dict[str, Any]:
        """Fetch hybrid recommendations balancing sequential and content signals."""
        return self._request(
            "GET",
            f"/v1/recommend/hybrid/{user_id}",
            params={
                "k": k,
                "alpha": alpha,
                "exclude": exclude,
                "genres": genres,
                "diversity": diversity,
                "cold_start": cold_start,
            },
        )

    def recommend_embedding(
        self,
        user_id: int,
        k: int = 10,
        exclude: str | None = None,
        genres: str | None = None,
        diversity: float = 0.0,
        cold_start: bool = False,
    ) -> dict[str, Any]:
        """Fetch content embedding recommendations via FAISS."""
        return self._request(
            "GET",
            f"/v1/recommend/embedding/{user_id}",
            params={
                "k": k,
                "exclude": exclude,
                "genres": genres,
                "diversity": diversity,
                "cold_start": cold_start,
            },
        )

    def recommend_session(
        self,
        session_id: str,
        k: int = 10,
        exclude: str | None = None,
        genres: str | None = None,
        diversity: float = 0.0,
    ) -> dict[str, Any]:
        """Fetch real-time recommendations for an anonymous browsing session."""
        return self._request(
            "GET",
            f"/v1/recommend/session/{session_id}",
            params={
                "k": k,
                "exclude": exclude,
                "genres": genres,
                "diversity": diversity,
            },
        )

    # Event telemetry
    def record_events(
        self,
        events: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Ingest interaction events."""
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return self._request(
            "POST",
            "/v1/events",
            data={"events": events, "idempotency_key": idempotency_key},
            headers=headers,
        )

    # Sessions
    def record_session_event(
        self,
        session_id: str,
        item_id: int,
        event_type: str = "click",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an event for an anonymous session."""
        return self._request(
            "POST",
            f"/v1/sessions/{session_id}/events",
            data={
                "item_id": item_id,
                "event_type": event_type,
                "metadata": metadata or {},
            },
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Retrieve anonymous session metadata and history."""
        return self._request("GET", f"/v1/sessions/{session_id}")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Clear anonymous session history."""
        return self._request("DELETE", f"/v1/sessions/{session_id}")

    # Privacy & GDPR
    def erase_user(self, user_id: int) -> dict[str, Any]:
        """Execute GDPR Article 17 right to erasure for a user."""
        return self._request("DELETE", f"/v1/users/{user_id}")

    # Business Rules
    def list_rules(self) -> dict[str, Any]:
        """List active business rules and detect conflicts."""
        return self._request("GET", "/v1/rules")

    def create_rule(
        self,
        rule_id: str,
        rule_type: str,
        item_ids: list[int] | None = None,
        genres: list[str] | None = None,
        value: float = 1.0,
        position: int | None = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        """Create a new post-ranking business rule."""
        return self._request(
            "POST",
            "/v1/rules",
            data={
                "rule_id": rule_id,
                "rule_type": rule_type,
                "item_ids": item_ids or [],
                "genres": genres or [],
                "value": value,
                "position": position,
                "priority": priority,
            },
        )

    def delete_rule(self, rule_id: str) -> dict[str, Any]:
        """Delete a business rule."""
        return self._request("DELETE", f"/v1/rules/{rule_id}")

    # Experiments
    def list_experiments(self) -> list[dict[str, Any]]:
        """List online A/B testing experiments."""
        return self._request("GET", "/v1/experiments")

    def create_experiment(
        self,
        exp_id: str,
        name: str,
        variants: list[dict[str, Any]],
        description: str = "",
    ) -> dict[str, Any]:
        """Create a new online experiment."""
        return self._request(
            "POST",
            "/v1/experiments",
            data={
                "id": exp_id,
                "name": name,
                "variants": variants,
                "description": description,
            },
        )

    def get_uplift(
        self,
        exp_id: str,
        control: str = "control",
        treatment: str | None = None,
    ) -> dict[str, Any]:
        """Compute statistical uplift report for an experiment."""
        return self._request(
            "GET",
            f"/v1/experiments/{exp_id}/uplift",
            params={"control": control, "treatment": treatment},
        )

    def assign_variant(self, exp_id: str, user_id: int) -> dict[str, Any]:
        """Deterministically assign user to experiment variant."""
        return self._request(
            "POST",
            f"/v1/experiments/{exp_id}/assignment",
            params={"user_id": user_id},
        )

    def health(self) -> dict[str, Any]:
        """Check API service health status."""
        return self._request("GET", "/health")
