# Multi-Tenancy & Data Isolation

SuisenSha provides end-to-end multi-tenant isolation across all components.

## Tenant Isolation Strategy

### 1. Database & Table-Level Segregation
Every transactional and analytical table includes a non-nullable `tenant_id` column:
- `events`: Tracks user interactions (`user_id`, `item_id`, `tenant_id`, `timestamp`).
- `user_exclusions`: User-driven dislikes and DSA opt-outs isolated per tenant.
- `sessions`: Session identifiers and temporal states partitioned by tenant.
- `business_rules`: Tenant-specific merchandising and filtering policies.
- `experiments`: A/B testing definitions and traffic splits isolated per tenant.
- `model_versions`: MLflow model registry version metadata tagged by tenant.

All internal queries and repositories enforce `WHERE tenant_id = ?` filters.

### 2. Cache Key Isolation
The Redis cache layer partitions recommendation keys by tenant:
```
rec:{tenant_id}:{model_name}:{user_id}
```
Flushing or invalidating user recommendations within one tenant never affects other tenants.

### 3. Tenant Authentication & Header Matching
API keys in SuisenSha are bound to a specific tenant upon creation:
- Each key record has a `tenant_id` attribute.
- Incoming requests can optionally pass `X-Tenant-ID`.
- If the authenticated API key belongs to `tenant_a`, attempting to pass `X-Tenant-ID: tenant_b` results in an immediate **HTTP 403 Forbidden**.
- Keys with `admin` scope can switch tenants by providing the `X-Tenant-ID` header.

### 4. Quotas and Rate Limiting
- **Burst Rate Limiting**: Per-minute token bucket rate limiter enforced per API key.
- **Monthly Quotas**: In-memory and persisted usage counters track total recommendation and event requests per tenant per calendar month. Requests exceeding quota are rejected with **HTTP 429 Too Many Requests**.
