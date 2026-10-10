# PR Plan

The [Roadmap](ROADMAP.md) split into individual pull requests, ordered by urgency.
Each PR is one branch, one purpose, and leaves `main` working and tested.

## How the order was decided

1. **Things that hurt every demo today** come first (the explanations are unusable text).
2. **Foundations other PRs depend on** come before the PRs that need them
   (configs and a validation split before tuning and benchmarks).
3. **Credibility before features**: baselines and honest benchmarks before selling.
4. **What a pilot client checks before signing** (data import, security, privacy).
5. **What turns a pilot into a product** (A/B testing, console, multi-tenancy).
6. **Scale and polish** last.

Effort for one developer: **S** ≤ 3 days, **M** ≈ 1–2 weeks, **L** ≈ 3–6 weeks.

## Overview

| # | Priority | Branch | Title | Effort | Depends on |
|---|----------|--------|-------|--------|------------|
| 01 | P0 | `feat/rag-grounding-fallback` | Grounding check and template fallback for explanations | S | — |
| 02 | P0 | `feat/rag-instruct-llm` | Pluggable LLM backend with an instruction-tuned model | S | 01 |
| 03 | P0 | `feat/rag-structured-output` | Structured per-item explanations | S | 02 |
| 04 | P0 | `build/docker` | Dockerfile and docker-compose for API + UI | S | — |
| 05 | P0 | `test/e2e-smoke` | End-to-end smoke test in CI with a tiny trained model | S | — |
| 06 | P1 | `ci/security-coverage` | Coverage threshold, `pip-audit` and container scan in CI | S | 04 |
| 07 | P1 | `refactor/config-files` | Experiment configs in TOML, overridable from the CLI | S | — |
| 08 | P1 | `feat/eval-splits` | Validation split and temporal split | S | 07 |
| 09 | P1 | `feat/baselines-pop-itemknn` | Random, Popularity and ItemKNN baselines | S | 08 |
| 10 | P1 | `feat/baseline-bpr` | BPR-MF baseline | S | 08 |
| 11 | P1 | `feat/sasrec-early-stopping` | Early stopping on validation and sliding windows | S | 08 |
| 12 | P1 | `feat/eval-beyond-accuracy` | Coverage, diversity, novelty, popularity bias, segments | S | 08 |
| 13 | P1 | `feat/eval-multi-seed` | Multi-seed runs with confidence intervals | S | 08 |
| 14 | P1 | `feat/benchmark-report` | One-command benchmark report | M | 09–13 |
| 15 | P1 | `feat/experiment-tracking` | MLflow experiment tracking | S | 07 |
| 16 | P1 | `feat/api-v1-params` | `/v1` API, `k` and filter parameters, similar-items endpoint | S | — |
| 17 | P1 | `feat/rag-sasrec-candidates` | Explain SASRec recommendations, not only content ones | S | 03 |
| 18 | P2 | `feat/data-connectors` | Data connectors and schema validation | M | 07 |
| 19 | P2 | `feat/datasets-generality` | MovieLens-1M, Amazon Books and Steam loaders | M | 18 |
| 20 | P2 | `feat/api-auth-ratelimit` | API keys and rate limiting | M | 16 |
| 21 | P2 | `feat/privacy-erasure` | Right to erasure and pseudonymised IDs | S | 20 |
| 22 | P2 | `feat/cold-start-hybrid` | Cold start for new users and items, hybrid model | M | 09 |
| 23 | P2 | `feat/business-rules` | Business rules: exclude, boost, pin | M | 16 |
| 24 | P2 | `feat/rerank-diversity` | Diversity re-ranking and popularity-bias correction | S | 12 |
| 25 | P2 | `feat/event-ingestion` | Event ingestion endpoint | M | 18, 20 |
| 26 | P2 | `feat/feedback-endpoint` | User feedback endpoint | S | 25 |
| 27 | P2 | `perf/ann-batch-cache` | HNSW index, batch top-N and Redis cache | M | 04 |
| 28 | P2 | `test/load-tests` | Load tests with latency budgets | S | 27 |
| 29 | P2 | `feat/observability` | Prometheus metrics, structured logs, Grafana | M | 04 |
| 30 | P2 | `feat/scheduled-retraining` | Scheduled retraining, quality gate, model registry | M | 14, 15, 25 |
| 31 | P3 | `feat/bench-explanations` | Explanation quality benchmark | M | 03 |
| 32 | P3 | `feat/bench-cost` | Cost benchmark | S | 27, 28 |
| 33 | P3 | `feat/ab-testing` | A/B experiment service and uplift report | L | 25, 29 |
| 34 | P3 | `feat/admin-console` | Client-facing admin console | L | 20, 23, 33 |
| 35 | P3 | `feat/compliance-dsa` | Transparency page and non-personalised mode | S | 16 |
| 36 | P3 | `feat/session-recs` | Session-based recommendations for anonymous users | M | 25 |
| 37 | P3 | `feat/realtime-updates` | Real-time updates without retraining | L | 25, 27 |
| 38 | P3 | `feat/js-widget-sdk` | JavaScript widget and Python/JS clients | M | 16, 20 |
| 39 | P3 | `feat/drift-rollback` | Drift monitoring and model rollback | M | 29, 30 |
| 40 | P3 | `feat/multi-tenancy` | Multi-tenancy | L | 20, 30 |
| 41 | P4 | `feat/models-sequential-graph` | GRU4Rec, BERT4Rec and LightGCN | M | 14 |
| 42 | P4 | `feat/two-tower-reranker` | Two-tower retrieval with a re-ranker | L | 14, 27 |
| 43 | P4 | `feat/i18n-explanations` | Explanations in the user's language | S | 03 |
| 44 | P4 | `deploy/helm` | Helm chart for Kubernetes | M | 04, 29 |
| 45 | P4 | `docs/site-releases` | Docs site, changelog and versioned releases | M | — |

**Priorities.** P0: fixes what hurts every demo. P1: makes the results credible.
P2: makes a pilot possible. P3: turns a pilot into a product. P4: scale and polish.

---

## P0 — Fix what hurts every demo

### PR-01 · Grounding check and template fallback for explanations
`feat/rag-grounding-fallback` · S

The explanation feature currently outputs repeated, invented text. Before changing the
model, make sure nothing ungrounded ever reaches the user.

- Add a deterministic template explanation, e.g. "Because you liked *X* and *Y*
  (both Sci-Fi)", built from shared genres between history and candidate.
- Reject generated text that names titles outside the retrieved candidates, or is mostly
  repeated sentences; fall back to the template.
- Tests for both the rejection rules and the fallback.

**Done when** no output can contain a title the retriever did not return.

### PR-02 · Pluggable LLM backend with an instruction-tuned model
`feat/rag-instruct-llm` · S · after PR-01

- Define an `LLMBackend` interface; keep the Hugging Face backend, add a hosted-API backend
  selected by config.
- Default to a small instruction-tuned model that runs on CPU.
- Keep the LLM out of tests with a fake backend.

**Done when** the demo shows readable, relevant explanations for sample users.

### PR-03 · Structured per-item explanations
`feat/rag-structured-output` · S · after PR-02

- Ask the model for JSON `[{item_id, reason}]`, validate it with Pydantic, fall back to
  the template per item on failure.
- API returns a list of explained items instead of one text block; UI shows each reason
  next to its movie.
- Cache key includes the candidate set, so stale explanations are not served.

### PR-04 · Dockerfile and docker-compose for API + UI
`build/docker` · S

- Multi-stage Dockerfile (CPU torch), non-root user, health check.
- `docker-compose.yml` starting API and UI, with artefacts mounted as a volume.
- README: "run the demo in one command".

**Done when** `docker compose up` gives a working UI on a clean machine.

### PR-05 · End-to-end smoke test in CI
`test/e2e-smoke` · S

- Fixture dataset (about 50 users) that runs phases 1–5 in under a minute.
- API test with the lifespan enabled, against those artefacts.
- Catches breakages that unit tests with fakes cannot (like the old padding bug).

---

## P1 — Make the results credible

### PR-06 · Coverage threshold and security scans in CI
`ci/security-coverage` · S · after PR-04

- `pytest --cov` with a minimum threshold (start at the current value).
- `pip-audit` for dependency vulnerabilities, Trivy for the Docker image.

### PR-07 · Experiment configs in TOML
`refactor/config-files` · S

- Move hyperparameters from `config.py` constants to `configs/*.toml`, keep dataclasses as
  the typed schema.
- CLI: `--config configs/sasrec.toml --set sasrec.epochs=40`.
- Every benchmark run records the exact config it used.

### PR-08 · Validation split and temporal split
`feat/eval-splits` · S · after PR-07

- Leave-two-out: second-to-last item for validation, last for test.
- Temporal split option: global cut-off timestamp.
- Evaluator takes the split as a parameter; test split is never used for tuning.

### PR-09 · Random, Popularity and ItemKNN baselines
`feat/baselines-pop-itemknn` · S · after PR-08

- Common `Recommender` interface (`fit`, `recommend(history, k)`) that existing models
  also implement.
- Three baselines registered in phase 5.

**Why urgent:** "5.5× better than a content model" is weak evidence; beating Popularity
and ItemKNN is the minimum a technical buyer expects.

### PR-10 · BPR-MF baseline
`feat/baseline-bpr` · S · after PR-08

- Implicit matrix factorisation with BPR loss in PyTorch, using the same interface.

### PR-11 · SASRec early stopping and sliding windows
`feat/sasrec-early-stopping` · S · after PR-08

- Evaluate validation NDCG@10 every epoch, keep the best checkpoint, stop on patience.
- Sequences longer than `max_seq_len` produce several training windows.

### PR-12 · Beyond-accuracy metrics and segments
`feat/eval-beyond-accuracy` · S · after PR-08

- Recall@k, MRR, catalogue coverage, intra-list diversity, novelty, average popularity,
  Gini index of exposure.
- Results broken down by user activity (cold / medium / heavy) and item popularity
  (head / tail).

### PR-13 · Multi-seed runs with confidence intervals
`feat/eval-multi-seed` · S · after PR-08

- `--seeds 5` runs training and evaluation per seed; reports mean ± 95% CI.

### PR-14 · One-command benchmark report
`feat/benchmark-report` · M · after PR-09 to PR-13

- `suisensha benchmark --config configs/bench.toml` trains every model, evaluates every
  metric, writes `reports/benchmark.md` and a CSV.
- Includes training time and peak memory per model.
- Replaces the hand-written results table in `GUIDE.md`.

### PR-15 · MLflow experiment tracking
`feat/experiment-tracking` · S · after PR-07

- Log config, metrics, artefacts and git commit for every training run.

### PR-16 · `/v1` API with parameters and a similar-items endpoint
`feat/api-v1-params` · S

- Prefix routes with `/v1`; keep old routes as deprecated aliases for one release.
- Query parameters: `k`, `exclude`, `genres`.
- `GET /v1/items/{id}/similar` for product-page widgets.

### PR-17 · Explain SASRec recommendations
`feat/rag-sasrec-candidates` · S · after PR-03

- RAG can take candidates from any `Recommender`; default to the best model from the
  benchmark, so explanations describe the recommendations users actually see.

---

## P2 — Make a pilot possible

### PR-18 · Data connectors and schema validation
`feat/data-connectors` · M · after PR-07

- Input contracts: `items` (id, title, attributes) and `interactions`
  (user, item, event type, timestamp, optional value).
- Connectors: CSV, Parquet, PostgreSQL. MovieLens becomes just one connector.
- Pandera validation with clear error messages.

**Why urgent:** a client cannot evaluate the product without loading their own data.

### PR-19 · More datasets
`feat/datasets-generality` · M · after PR-18

- MovieLens-1M, Amazon Reviews (Books), Steam through the connector layer.
- Benchmark report covers all of them.

### PR-20 · API keys and rate limiting
`feat/api-auth-ratelimit` · M · after PR-16

- API keys hashed in the database, scoped per client; `401`/`403` handling.
- Per-key rate limit; `429` with `Retry-After`.

### PR-21 · Right to erasure and pseudonymised IDs
`feat/privacy-erasure` · S · after PR-20

- `DELETE /v1/users/{id}` removes interactions and cached explanations, and flags the
  user for exclusion at the next retraining.
- Document that only pseudonymous IDs are stored.

### PR-22 · Cold start and hybrid model
`feat/cold-start-hybrid` · M · after PR-09

- New users: popularity or onboarding-genre fallback, switching to personal results after
  N interactions.
- New items: content embeddings mapped into the collaborative space so they can be
  recommended before anyone has clicked them.

### PR-23 · Business rules
`feat/business-rules` · M · after PR-16

- Rules stored per client: exclude (out of stock, unlicensed), boost, pin to a position,
  start and end dates.
- Applied after ranking, with tests for rule conflicts.

### PR-24 · Diversity re-ranking and popularity-bias correction
`feat/rerank-diversity` · S · after PR-12

- MMR re-ranking with a `diversity` parameter; calibrated popularity option.
- Benchmark shows the accuracy/diversity trade-off curve.

### PR-25 · Event ingestion endpoint
`feat/event-ingestion` · M · after PR-18, PR-20

- `POST /v1/events` (batch), validated and stored; idempotency key.
- New interactions are used at the next retraining.

### PR-26 · User feedback endpoint
`feat/feedback-endpoint` · S · after PR-25

- "Not interested" and thumbs up/down stored as events; "not interested" is excluded
  immediately.

### PR-27 · HNSW index, batch top-N and Redis cache
`perf/ann-batch-cache` · M · after PR-04

- FAISS HNSW option with a recall check against the flat index.
- Nightly batch top-N per user stored in Redis; API reads cache first.
- Redis added to docker-compose.

### PR-28 · Load tests with latency budgets
`test/load-tests` · S · after PR-27

- Locust scenarios per endpoint; CI job fails if p95 exceeds the budget.

### PR-29 · Observability
`feat/observability` · M · after PR-04

- Prometheus metrics (latency, errors, cache hit rate, recommendations served).
- JSON structured logs with request IDs.
- Grafana dashboard provisioned in docker-compose.

### PR-30 · Scheduled retraining, quality gate, model registry
`feat/scheduled-retraining` · M · after PR-14, PR-15, PR-25

- Prefect flow: ingest → train → benchmark → promote only if validation metrics do not
  drop more than a set tolerance.
- MLflow model registry; API loads the "production" version.

---

## P3 — Turn a pilot into a product

### PR-31 · Explanation quality benchmark
`feat/bench-explanations` · M · after PR-03

- Grounding rate, hallucination rate (genre/year facts checked against the catalogue),
  LLM-as-judge relevance, length, latency, tokens.
- Template for a small human evaluation.

### PR-32 · Cost benchmark
`feat/bench-cost` · S · after PR-27, PR-28

- € per 1M recommendations and per 1k explanations from load-test throughput and instance
  prices; comparison with managed recommendation services' list prices.

### PR-33 · A/B experiment service
`feat/ab-testing` · L · after PR-25, PR-29

- Deterministic user-to-variant assignment, exposure logging, conversion events.
- Uplift report with confidence intervals and a sample-size calculator.

**Why it matters:** a measured uplift is the strongest sales argument the product can have.

### PR-34 · Admin console
`feat/admin-console` · L · after PR-20, PR-23, PR-33

- Replace the Streamlit demo for clients: manage keys, business rules, experiments; view
  metrics and sample explanations.

### PR-35 · DSA transparency and non-personalised mode
`feat/compliance-dsa` · S · after PR-16

- `personalised=false` parameter returning non-profiling results (popularity/trending).
- Generated plain-language page describing ranking factors.

### PR-36 · Session-based recommendations
`feat/session-recs` · M · after PR-25

- Recommend from the current session's items for anonymous users, using SASRec on the
  session sequence.

### PR-37 · Real-time updates
`feat/realtime-updates` · L · after PR-25, PR-27

- New events update the user's SASRec input immediately; cached top-N invalidated for
  that user.

### PR-38 · JavaScript widget and client SDKs
`feat/js-widget-sdk` · M · after PR-16, PR-20

- Embeddable carousel with explanations; generated Python and TypeScript clients from
  the OpenAPI spec.

### PR-39 · Drift monitoring and rollback
`feat/drift-rollback` · M · after PR-29, PR-30

- Evidently reports on interaction and score distributions; alerts on drift.
- One command to roll back to the previous registered model.

### PR-40 · Multi-tenancy
`feat/multi-tenancy` · L · after PR-20, PR-30

- Tenant ID on every table, model and cache key; per-tenant training and quotas.

---

## P4 — Scale and polish

### PR-41 · GRU4Rec, BERT4Rec and LightGCN
`feat/models-sequential-graph` · M · after PR-14

- Three more models in the benchmark; cross-check numbers with RecBole.

### PR-42 · Two-tower retrieval with a re-ranker
`feat/two-tower-reranker` · L · after PR-14, PR-27

- Two-tower model for candidate retrieval over large catalogues, SASRec or a gradient
  boosting model as the re-ranker.

### PR-43 · Explanations in the user's language
`feat/i18n-explanations` · S · after PR-03

- `lang` parameter passed to the LLM prompt and template fallback; cache key includes it.

### PR-44 · Helm chart
`deploy/helm` · M · after PR-04, PR-29

- Deployments for API, UI, Redis; autoscaling on CPU and request rate.

### PR-45 · Docs site and versioned releases
`docs/site-releases` · M

- MkDocs site with API reference and tutorials; semantic versioning, changelog, tagged
  Docker images on release.

---

## Milestones

| Milestone | PRs | Approx. effort | Outcome |
|-----------|-----|----------------|---------|
| **Demo fixed** | 01–05 | 2–3 weeks | Readable explanations, one-command run, CI catches regressions |
| **Credible demo** | 06–17 | 4–6 weeks | Baselines, honest benchmark report, clean API |
| **Pilot-ready** | 18–30 | 3–4 months | Client data, security, privacy, rules, monitoring, retraining |
| **Product** | 31–40 | 4–6 months | A/B testing, console, compliance, real-time, multi-tenant |
| **Scale** | 41–45 | ongoing | More models, large catalogues, Kubernetes, docs site |

PRs at the same priority with no dependency between them can run in parallel; the
"Depends on" column is the only hard ordering.
