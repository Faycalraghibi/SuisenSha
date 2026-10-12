# API Reference

The SuisenSha REST API is built on FastAPI and documented via OpenAPI 3.1.0.

## Authentication Headers

All protected endpoints require an API key passed via one of:
- `Authorization: Bearer <api-key>`
- `X-API-Key: <api-key>`
- Optional Tenant header: `X-Tenant-ID: <tenant-id>` (requires admin privilege to switch away from the key's native tenant).

## Core Endpoints

### Recommendations

| Endpoint | Method | Description |
|---|---|---|
| `/v1/recommend/popular` | `GET` | Retrieve global or segment-specific top-rated items |
| `/v1/recommend/embedding` | `GET` | Dense vector nearest-neighbor recommendations for a user |
| `/v1/recommend/sasrec` | `GET` | Transformer-based sequential recommendations |
| `/v1/recommend/hybrid` | `GET` | Blended ensemble ranking combining multiple models |

#### Common Query Parameters
- `user_id` (string): Target user identifier.
- `n` (integer, default 10): Number of recommendations to return.
- `enable_rules` (boolean, default true): Apply business rules (boost, pin, filter).
- `explain` (boolean, default false): Include DSA explainability metadata and feature importances.

### Events & Sessions

| Endpoint | Method | Description |
|---|---|---|
| `/v1/events` | `POST` | Batch record user interaction events (`view`, `click`, `purchase`, `dislike`) |
| `/v1/sessions/{session_id}` | `GET` | Get current session state and recently interacted items |
| `/v1/sessions/{session_id}/events` | `POST` | Append interaction to session |

### Business Rules & Merchandising

| Endpoint | Method | Description |
|---|---|---|
| `/v1/rules` | `GET` | List active business rules for current tenant |
| `/v1/rules` | `POST` | Create a new business rule (`pin`, `filter`, `boost`, `diversity`) |
| `/v1/rules/{rule_id}` | `DELETE` | Delete or deactivate a rule |

### A/B Experiments

| Endpoint | Method | Description |
|---|---|---|
| `/v1/experiments` | `GET` | List active experiments |
| `/v1/experiments` | `POST` | Create a new experiment with traffic split |
| `/v1/experiments/{id}/assignment` | `POST` | Evaluate variant assignment for a user |
| `/v1/experiments/{id}/report` | `GET` | Compute statistical uplift and p-value |

### Privacy & Compliance

| Endpoint | Method | Description |
|---|---|---|
| `/v1/users/{user_id}/data` | `DELETE` | Delete all events and cached records for a user (Right to Erasure) |
| `/health` | `GET` | Liveness and readiness health check |
| `/metrics` | `GET` | Prometheus telemetry metrics |
