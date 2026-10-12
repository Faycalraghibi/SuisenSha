# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-tenancy architecture with tenant ID isolation across database tables, Redis cache keys (`rec:{tenant_id}:{model}:{user_id}`), business rules, sessions, experiments, and model registries.
- Per-tenant monthly request quota tracking and enforcement with administrative reset capabilities.
- Per-tenant API key authentication and header enforcement with admin tenant switching.
- Embeddable vanilla JavaScript recommendation carousel widget (`widget.js`, `widget.css`) with DSA explainability modal, interactive feedback, and demo page (`/widget/demo`).
- Typed Python client SDK (`sdks/python/`) with zero external runtime dependencies.
- Typed TypeScript client SDK (`sdks/typescript/`) for modern JavaScript and Node.js environments.
- OpenAPI 3.1.0 specification export script (`scripts/export_openapi.py`) and schema artifact (`docs/openapi.json`).
- Production Kubernetes Helm chart (`deploy/helm/suisensha/`) with Horizontal Pod Autoscaler (HPA) configs for API and UI, Redis deployment, Ingress, and ConfigMaps/Secrets.
- Documentation site powered by MkDocs Material (`mkdocs.yml`, `docs/site/`) covering architecture, multi-tenancy, API reference, client SDKs, DSA compliance, and Kubernetes deployment.
- Comprehensive platform test suite (`tests/test_platform.py`) verifying tenant isolation, quota management, and Helm chart manifests.
- Multi-stage Dockerfile and docker-compose service configuration for API and static UI.

- Sub-minute end-to-end fixture pipeline test running phases 1-5 against real FastAPI lifespan.
- CI security audit check via `pip-audit`.
- Minimum code coverage threshold (75%) enforced in test suite.
- Container vulnerability scanning with Trivy in CI.
- Automated release workflow publishing Docker images and GitHub releases on semantic tags.
- Versioned `/v1` API routes with query parameters (`k`, `exclude`, `genres`, `personalised`).
- Product-page similar items endpoint `GET /v1/items/{item_id}/similar`.
- EU DSA Art. 27 ranking transparency endpoint `GET /v1/transparency` in JSON and HTML.
- Data platform contracts for `items` and `interactions` with Pandera schema validation and readable error messages.
- Extensible data connectors for CSV, Parquet, SQL (PostgreSQL/SQLite), MovieLens (100K/1M), Amazon Reviews, and Steam.
- Multi-dataset benchmark evaluation support with MovieLens-1M integration.
- Hashed API key authentication (SHA-256) scoped per client with 401/403 status handling.
- Sliding-window per-key rate limiting returning HTTP 429 with `Retry-After` header.
- GDPR Art. 17 right-to-erasure endpoint `DELETE /v1/users/{user_id}` clearing interactions, cached explanations, and registry exclusion.
- Audit logging table recording administrative and data erasure actions.
- Cold-start handling for new users via popularity and onboarding-genre matching.
- Hybrid recommender (`HybridRecommender`) combining SASRec collaborative filtering with sentence-transformer embeddings for zero-click cold-start item discovery.
- Business rules engine (`RuleEngine`) supporting item/genre exclusion, score boosting, and positional pinning with active datetime windows and conflict resolution.
- Maximal Marginal Relevance (MMR) re-ranking with tunable `diversity` parameter and calibrated popularity dampening.
- Hybrid recommendation endpoint `GET /v1/recommend/hybrid/{user_id}` and business rules endpoints `GET /v1/rules` & `POST /v1/rules`.
- Accuracy vs. diversity trade-off evaluation in benchmark reports and CSV.
- Batched event ingestion endpoint `POST /v1/events` with idempotency keys and validation for views, clicks, purchases, ratings, and feedback signals.
- User feedback system supporting thumbs up/down and "not interested", with instantaneous exclusion from subsequent recommendations across all model endpoints.
- Integration of newly ingested events into user interaction datasets and retraining pipelines (`load_interactions_with_events`).
- Approximate Nearest Neighbor (ANN) indexing with FAISS `IndexHNSWFlat` and recall evaluation verifying >= 0.95 recall against exact flat search.
- Redis cache layer (`RedisCache`) with automatic in-memory fallback, supporting precomputed nightly batch top-N recommendations.
- Nightly batch precomputation pipeline (`precompute_batch_top_n`) caching top-N items per user before real-time inference fallback.
- Prometheus metrics exposition (`GET /metrics`) tracking `recommendation_requests_total`, `recommendation_latency_seconds`, and cache hit/miss rates.
- Structured JSON access logging with `X-Request-ID` propagation and correlation across requests.
- Provisioned Grafana dashboard configuration (`recommender_overview.json`) and Docker compose monitoring services (Redis, Prometheus, Grafana).
- Locust load test scenarios (`tests/load/locustfile.py`) measuring p95 serving latency under concurrent traffic.
- Serving cost benchmark report (`reports/cost_benchmark.md`) demonstrating 28% savings vs. AWS Personalize and 94% vs. GCP Vertex AI Retail.
- File-backed and MLflow-compatible Model Registry (`ModelRegistry`) tracking versioned model artifacts, staging/production stages, and promotion history.
- Automated retraining orchestration (`RetrainingPipeline`) with gated validation checks preventing promotion of degraded models.
- Statistical drift detection (`DriftDetector`, `calculate_psi`) using Kolmogorov-Smirnov two-sample tests and Population Stability Index (PSI) over interaction volumes and recommendation score distributions.
- Instant single-command model rollback (`suisensha rollback` / `ModelRegistry.rollback`) restoring previous production models.
- Dynamic production model resolution during FastAPI startup lifespan from the Model Registry.
- A/B testing experimentation framework (`ExperimentManager`) with deterministic salt-hashed user assignment across variants.
- Real-time exposure and conversion event tracking per experiment variant.
- Statistical conversion uplift reporting with two-proportion z-tests, p-values, and 95% confidence intervals.
- Minimum sample size calculator (`calculate_sample_size`) based on baseline conversion, MDE, statistical power, and significance levels.
- Online experiment management API endpoints (`/v1/experiments`, `/v1/experiments/{id}/uplift`, `/v1/experiments/sample-size`).
- Comprehensive Client Admin Console (`src/pipeline/ui/admin.py`) for managing API keys, business rules, live A/B experiments, uplift metrics, and model inspectability.
- Real-time event ingestion immediate cache invalidation across Redis and in-memory caches upon interaction event arrival (`POST /v1/events`).
- Dynamic real-time sequence aggregation augmenting static historical interaction sequences with chronologically ordered live events for instant next-item prediction without retraining.
- Thread-safe SQLite-backed session store and manager (`SessionStore`, `SessionManager`) for tracking anonymous user interaction sequences and lifecycle.
- Real-time anonymous session recommendation endpoints (`GET /v1/recommend/session/{session_id}` and `POST /v1/recommend/session`) serving SASRec next-item predictions with popularity fallback for empty sessions.
- Anonymous session lifecycle endpoints (`POST /v1/sessions/{session_id}/events`, `GET /v1/sessions/{session_id}`, `DELETE /v1/sessions/{session_id}`).




### Changed
- RAG cache strictly enforces matching user ID, candidate set, and language; wildcard and legacy-table fallbacks removed from lookups. Legacy table writes removed.

## [0.1.0] - 2026-10-10

### Added
- Grounded RAG explanation module rejecting candidate hallucinations.
- `LLMBackend` pluggable interface supporting local instruction-tuned, hosted API, and fake test backends.
- Structured JSON output format for per-item recommendation reasons with template fallbacks.
- Multi-recommender candidate retrieval (SASRec and embedding index) with content and language-keyed caching.
- Initial MovieLens sequential and content embedding recommendation pipeline.
