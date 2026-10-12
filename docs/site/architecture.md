# Architecture Overview

SuisenSha is architected around a layered, decoupled service topology optimized for sub-10ms P95 latency and horizontal scaling.

```
+-----------------------------------------------------------+
|                       Clients                             |
|  (JavaScript Widget / Python SDK / TypeScript SDK / Web)  |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                     FastAPI Gateway                       |
|   - API Key Auth & Scopes (read/write/admin)              |
|   - Multi-tenant Quota Limiter & Burst Rate Limiter        |
|   - Prometheus Metrics & DSA Audit Logging                |
+-----------------------------------------------------------+
           |                         |              |
           v                         v              v
+--------------------+      +----------------+   +-------------------+
| Multi-Model Ranker |      | Business Rules |   | Redis / LRU Cache |
| - Popularity       |      | - Boost        |   | - Key segregated  |
| - FAISS Embedding  |      | - Filter       |   |   by tenant_id    |
| - SASRec Attention |      | - Pin          |   | - 5-min TTL       |
| - Hybrid Fusion    |      | - Diversity    |   +-------------------+
+--------------------+      +----------------+
           |                         |
           v                         v
+--------------------+      +----------------+
| Model Registry     |      | Event Ingestion|
| - Version metadata |      | - Session mgmt |
| - Staging/Prod     |      | - Exclusions   |
| - Tenant isolated  |      | - User events  |
+--------------------+      +----------------+
```

## Layer Descriptions

1. **API Gateway (`src/pipeline/api`)**:
   FastAPI-based async web service providing REST endpoints, OpenAPI validation, token security, rate-limiting, and tenant routing.
2. **Recommendation Engines (`src/pipeline/models`)**:
   - `PopularityModel`: Fast baseline with segment support.
   - `EmbeddingModel`: Dense vector index powered by FAISS for sub-millisecond approximate nearest neighbor search.
   - `SASRecRecommender`: Transformer-based self-attention sequential encoder for session next-item prediction.
   - `HybridModel`: Weighted rank combination blending candidate pools with min-max score normalization.
3. **Business Rules Engine (`src/pipeline/ranking/rules.py`)**:
   Applies configurable promotion, pinning, category filtering, and diversity re-ranking after model scoring.
4. **Caching & Invalidation (`src/pipeline/cache.py`)**:
   Two-tier caching strategy. Cache keys incorporate `tenant_id`, model name, and user ID (`rec:{tenant_id}:{model}:{user_id}`) to ensure zero cross-tenant contamination.
