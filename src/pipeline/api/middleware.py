from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("pipeline.api.access")

REQUEST_COUNT = Counter(
    "recommendation_requests_total",
    "Total number of recommendation and API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Latency of recommendation requests in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Total recommendation and rationale cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total recommendation and rationale cache misses",
    ["cache_type"],
)


class TracingAndMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        t0 = time.perf_counter()

        response: Response
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - t0
            REQUEST_COUNT.labels(
                endpoint=request.url.path,
                method=request.method,
                status="500",
            ).inc()
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
            raise

        duration = time.perf_counter() - t0
        status_code = str(response.status_code)

        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            method=request.method,
            status=status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

        response.headers["X-Request-ID"] = request_id

        # Structured JSON access logging
        log_entry = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(duration * 1000, 2),
        }
        logger.info(json.dumps(log_entry))

        return response


def get_metrics_response() -> Response:
    """Return latest Prometheus metrics formatted for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
