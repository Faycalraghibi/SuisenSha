# PR Plan

The [Roadmap](ROADMAP.md) split into 15 pull requests, ordered by urgency.
Each PR is one branch with one theme and leaves `main` working and tested.

## How the order was decided

1. **Things that hurt every demo today** come first (the explanations are unusable text).
2. **Foundations other PRs depend on** come before the PRs that need them
   (configs and a validation split before tuning and benchmarks).
3. **Credibility before features**: baselines and honest benchmarks before selling.
4. **What a pilot client checks before signing** (data import, security, privacy).
5. **What turns a pilot into a product** (A/B testing, console, multi-tenancy).
6. **Scale** last.

Related work is grouped into one PR when it touches the same code and ships the same
outcome. Inside a PR, keep **one commit per bullet group** so the PR stays reviewable
commit by commit.

Effort for one developer: **S** ≤ 3 days, **M** ≈ 1–2 weeks, **L** ≈ 3–6 weeks,
**XL** > 6 weeks.

## Overview

| # | Priority | Branch | Title | Effort | Depends on |
|---|----------|--------|-------|--------|------------|
| 01 | P0 | `feat/rag-explanations` | Reliable, grounded explanations | M | — |
| 02 | P0 | `build/docker-ci` | Docker, end-to-end test, CI hardening, releases | M | — |
| 03 | P1 | `feat/experiment-foundation` | Configs, data splits, tracking, baselines, SASRec training | L | — |
| 04 | P1 | `feat/benchmark-suite` | Full metrics, multi-seed benchmark report | M | 01, 03 |
| 05 | P1 | `feat/api-v1` | Versioned API, parameters, similar items, non-personalised mode | S | — |
| 06 | P2 | `feat/data-platform` | Client data connectors, validation, more datasets | L | 03 |
| 07 | P2 | `feat/api-security-privacy` | API keys, rate limits, right to erasure | M | 05 |
| 08 | P2 | `feat/ranking-controls` | Cold start, hybrid model, business rules, diversity | L | 03, 04, 05 |
| 09 | P2 | `feat/events-feedback` | Event ingestion and user feedback | M | 06, 07 |
| 10 | P2 | `perf/serving-observability` | ANN index, Redis cache, load tests, monitoring, cost | L | 02 |
| 11 | P2 | `feat/mlops` | Scheduled retraining, model registry, drift, rollback | L | 04, 09, 10 |
| 12 | P3 | `feat/experiments-console` | A/B testing and client admin console | XL | 07, 08, 09, 10 |
| 13 | P3 | `feat/realtime-sessions` | Real-time updates and session recommendations | L | 09, 10 |
| 14 | P3 | `feat/platform` | Multi-tenancy, SDKs and widget, Helm, docs site | XL | 02, 07, 11 |
| 15 | P4 | `feat/advanced-models` | GRU4Rec, BERT4Rec, LightGCN, two-tower retrieval | L | 04, 10 |

**Priorities.** P0: fixes what hurts every demo. P1: makes the results credible.
P2: makes a pilot possible. P3: turns a pilot into a product. P4: scale.

PRs with no dependency between them can be built in parallel; the "Depends on" column is
the only hard ordering.

---

## P0 — Fix what hurts every demo

### PR-01 · Reliable, grounded explanations
`feat/rag-explanations` · M

The explanation feature currently outputs repeated, invented text. This PR makes it the
strongest part of the demo instead of the weakest.

- **Safety net:** deterministic template explanation ("Because you liked *X* and *Y*,
  both Sci-Fi"); reject generated text that names titles outside the retrieved
  candidates or is mostly repetition, and fall back to the template.
- **Better model:** `LLMBackend` interface with a local instruction-tuned model by default
  and an optional hosted-API backend; a fake backend keeps the LLM out of unit tests.
- **Structured output:** model returns JSON `[{item_id, reason}]` validated with
  Pydantic; the API returns a list of explained items and the UI shows each reason next to
  its movie; per-item template fallback.
- **Right candidates:** candidates can come from any recommender, defaulting to SASRec;
  cache key includes the candidate set and a `lang` parameter for localised text.

**Done when** no output can contain a title the retriever did not return, and sample users
get readable, relevant reasons.

### PR-02 · Docker, end-to-end test, CI hardening, releases
`build/docker-ci` · M

- **Docker:** multi-stage Dockerfile (CPU torch), non-root user, health check;
  `docker-compose.yml` for API + UI with artefacts as a volume.
- **End-to-end test:** a fixture dataset of about 50 users runs phases 1–5 in under a
  minute in CI, then the API is tested with the real lifespan against those artefacts.
- **CI hardening:** coverage threshold, `pip-audit`, Trivy image scan.
- **Releases:** semantic versioning, `CHANGELOG.md`, tagged Docker images on release.

**Done when** `docker compose up` gives a working UI on a clean machine, and a bug like
the old SASRec padding issue would fail CI.

---

## P1 — Make the results credible

### PR-03 · Configs, data splits, tracking, baselines, SASRec training
`feat/experiment-foundation` · L

- **Configs:** hyperparameters move to `configs/*.toml`, with dataclasses as the typed
  schema and CLI overrides (`--set sasrec.epochs=40`).
- **Splits:** leave-two-out (validation and test) and a temporal split; the test split is
  never used for tuning.
- **Tracking:** MLflow logs config, metrics, artefacts and git commit for every run.
- **Recommender interface:** `fit` / `recommend(history, k)` implemented by all models.
- **Baselines:** Random, Popularity, ItemKNN, BPR-MF.
- **SASRec training:** early stopping on validation NDCG@10, best checkpoint kept,
  sliding windows for sequences longer than `max_seq_len`.

**Why urgent:** "5.5× better than a content model" is weak evidence. Beating Popularity,
ItemKNN and BPR-MF on a proper validation protocol is the minimum a technical buyer
expects.

### PR-04 · Full metrics and multi-seed benchmark report
`feat/benchmark-suite` · M · after PR-01, PR-03

- **Metrics:** Recall@k, MRR, coverage, intra-list diversity, novelty, average
  popularity, Gini index of exposure, all broken down by user activity and item
  popularity.
- **Seeds:** `--seeds 5` with mean ± 95% confidence intervals.
- **Explanation quality:** grounding rate, hallucination rate (genre and year checked
  against the catalogue), LLM-as-judge relevance, length, latency.
- **Report:** `suisensha benchmark` trains and evaluates every model and writes
  `reports/benchmark.md` and a CSV, including training time and peak memory. It replaces
  the hand-written results table in `GUIDE.md`.

### PR-05 · Versioned API, parameters, similar items, non-personalised mode
`feat/api-v1` · S

- Routes under `/v1`, old routes kept as deprecated aliases for one release.
- Query parameters: `k`, `exclude`, `genres`, `personalised=false` (non-profiling results
  for the EU Digital Services Act, Art. 38).
- `GET /v1/items/{id}/similar` for product-page widgets.
- Generated transparency page describing ranking factors (DSA Art. 27).

---

## P2 — Make a pilot possible

### PR-06 · Client data connectors, validation, more datasets
`feat/data-platform` · L · after PR-03

- **Contracts:** `items` (id, title, attributes) and `interactions` (user, item, event
  type, timestamp, optional value).
- **Connectors:** CSV, Parquet, PostgreSQL; MovieLens becomes one connector among others.
- **Validation:** Pandera schemas with clear error messages.
- **Datasets:** MovieLens-1M, Amazon Reviews (Books) and Steam, all included in the
  benchmark report.

**Why urgent:** a client cannot evaluate the product without loading their own data.

### PR-07 · API keys, rate limits, right to erasure
`feat/api-security-privacy` · M · after PR-05

- API keys hashed in the database, scoped per client; `401`/`403` handling.
- Per-key rate limiting with `429` and `Retry-After`.
- `DELETE /v1/users/{id}` removes interactions and cached explanations and excludes the
  user at the next retraining; only pseudonymous IDs are stored.
- Audit log of administrative actions.

### PR-08 · Cold start, hybrid model, business rules, diversity
`feat/ranking-controls` · L · after PR-03, PR-04, PR-05

- **Cold start:** new users get popularity or onboarding-genre results until they have
  enough interactions; new items are placed via content embeddings so they can be
  recommended before anyone clicks them (hybrid model).
- **Business rules:** exclude, boost, pin to a position, with start and end dates,
  applied after ranking; tests for conflicting rules.
- **Diversity:** MMR re-ranking with a `diversity` parameter and calibrated popularity;
  the benchmark shows the accuracy/diversity trade-off.

### PR-09 · Event ingestion and user feedback
`feat/events-feedback` · M · after PR-06, PR-07

- `POST /v1/events` (batched, validated, idempotency key) for views, clicks, purchases
  and ratings.
- Feedback events: thumbs up/down and "not interested", which is excluded immediately.
- New events are used at the next retraining.

### PR-10 · ANN index, Redis cache, load tests, monitoring, cost
`perf/serving-observability` · L · after PR-02

- **Speed:** FAISS HNSW option with a recall check against the flat index; nightly batch
  top-N per user in Redis, read first by the API.
- **Load tests:** Locust scenarios per endpoint; CI fails if p95 latency exceeds the
  budget.
- **Monitoring:** Prometheus metrics (latency, errors, cache hit rate), JSON logs with
  request IDs, a provisioned Grafana dashboard.
- **Cost benchmark:** € per 1M recommendations and per 1k explanations, compared with
  managed recommendation services' list prices.

### PR-11 · Scheduled retraining, model registry, drift, rollback
`feat/mlops` · L · after PR-04, PR-09, PR-10

- Prefect flow: ingest → train → benchmark → promote only if validation metrics stay
  within tolerance.
- MLflow model registry; the API loads the "production" version.
- Evidently drift reports on interactions and scores, with alerts.
- One command to roll back to the previous registered model.

---

## P3 — Turn a pilot into a product

### PR-12 · A/B testing and client admin console
`feat/experiments-console` · XL · after PR-07 to PR-10

- **A/B testing:** deterministic user-to-variant assignment, exposure and conversion
  logging, uplift report with confidence intervals, sample-size calculator.
- **Admin console:** replaces the Streamlit demo for clients — manage API keys, business
  rules and experiments; view metrics and sample explanations.

**Why it matters:** a measured uplift is the strongest sales argument the product can have.

### PR-13 · Real-time updates and session recommendations
`feat/realtime-sessions` · L · after PR-09, PR-10

- New events update the user's SASRec input immediately and invalidate their cached
  top-N.
- Anonymous users get recommendations from the current session's items.

### PR-14 · Multi-tenancy, SDKs and widget, Helm, docs site
`feat/platform` · XL · after PR-02, PR-07, PR-11

- **Multi-tenancy:** tenant ID on every table, model and cache key; per-tenant training
  and quotas.
- **Integration:** embeddable JavaScript carousel with explanations; Python and TypeScript
  clients generated from the OpenAPI spec.
- **Deployment:** Helm chart with autoscaling for API, UI and Redis.
- **Docs site:** MkDocs with API reference and integration tutorials.

---

## P4 — Scale

### PR-15 · GRU4Rec, BERT4Rec, LightGCN, two-tower retrieval
`feat/advanced-models` · L · after PR-04, PR-10

- Three more models in the benchmark, cross-checked against RecBole.
- Two-tower retrieval for large catalogues, with SASRec or gradient boosting as the
  re-ranker.

---

## Milestones

| Milestone | PRs | Approx. effort | Outcome |
|-----------|-----|----------------|---------|
| **Demo fixed** | 01–02 | 2–3 weeks | Readable explanations, one-command run, CI catches regressions |
| **Credible demo** | 03–05 | 5–7 weeks | Baselines, honest benchmark report, clean API |
| **Pilot-ready** | 06–11 | 3.5–4.5 months | Client data, security, privacy, rules, monitoring, retraining |
| **Product** | 12–14 | 4–6 months | A/B testing, console, real-time, multi-tenant platform |
| **Scale** | 15 | 1–2 months | More models, large catalogues |
