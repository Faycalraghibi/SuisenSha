# Product Roadmap & Benchmark Plan

This document lists what SuisenSha needs to become a product a paying client would adopt:
who the clients are, what they would ask for, what money and effort the tool saves them,
which enhancements to build (with effort estimates), and how to benchmark the system so
claims about quality and performance are defensible.

Effort is estimated for one developer: **S** ≤ 3 days, **M** ≈ 1–2 weeks, **L** ≈ 3–6 weeks,
**XL** > 6 weeks.

---

## Contents

1. [Where the project stands](#1-where-the-project-stands)
2. [Who would buy this](#2-who-would-buy-this)
3. [What the tool saves a client](#3-what-the-tool-saves-a-client)
4. [What clients will demand](#4-what-clients-will-demand)
5. [Enhancements by area](#5-enhancements-by-area)
6. [Benchmark plan](#6-benchmark-plan)
7. [Phased roadmap](#7-phased-roadmap)
8. [Commercial packaging](#8-commercial-packaging)
9. [References](#9-references)

---

## 1. Where the project stands

**Strengths today**

- End-to-end pipeline: data → two recommenders → LLM explanations → API → UI.
- SASRec is correct after the padding fix: HitRate@10 = 0.0885, 5.5× the content baseline.
- Tested (41 tests), linted, type-checked, CI on Linux and Windows.
- Rationale cache in SQLite so the LLM is not on the hot path.

**Gaps a client would notice immediately**

| Gap | Why it blocks a sale |
|-----|----------------------|
| Only MovieLens-100K, a frozen research dataset | Client cannot load their own catalogue or users |
| No baselines beyond one content model | "5.5× better" than a weak baseline does not prove much |
| Users and items are fixed at training time | New users, new items and new clicks are ignored until retrained by hand |
| DistilGPT-2 rationales are unusable text | The "explainable" selling point currently hurts the demo |
| No auth, no rate limits, no multi-tenancy | Cannot be exposed to the internet or shared between clients |
| No business rules (exclude, boost, availability) | Every real catalogue has items that must never or must always be shown |
| No monitoring, no A/B testing | Client cannot prove the recommender makes money |
| No Docker image or deployment story | Client's ops team cannot run it |

---

## 2. Who would buy this

| Client type | Example | What they recommend | Main pain |
|-------------|---------|---------------------|-----------|
| Niche streaming / VOD | Regional film platform, anime or documentary service | Movies, series, episodes | Cannot afford a Netflix-size ML team; churn after the first month |
| E-commerce (SMB to mid-market) | Fashion, books, electronics shops | Products, bundles | Low average basket; "customers who bought" widgets are generic |
| Media & publishing | News sites, magazines, podcast apps | Articles, episodes | Readers leave after one article; editors hand-pick "related" links |
| E-learning | Course marketplaces, corporate LMS | Courses, lessons | Learners drop out; catalogue is too big to browse |
| Marketplaces & classifieds | Jobs, real estate, second-hand | Listings, jobs | Items expire fast, so cold start is constant |
| Gaming | Indie game stores, in-game shops | Games, items, DLC | Discovery of long-tail titles |
| Cinemas & ticketing | Cinema chains, event platforms | Showings, events | Time-limited inventory, location constraints |
| Libraries & cultural institutions | Public libraries, museums | Books, collections | Zero ML budget, privacy-sensitive users |

The common profile: **a catalogue too large to browse, interaction data they already
collect, and no in-house recommendation team.** This is exactly the gap between doing
nothing and paying for a large SaaS (Algolia Recommend, AWS Personalize, Google
Recommendations AI) or hiring ML engineers.

---

## 3. What the tool saves a client

### 3.1 Revenue the client gains

| Lever | How a recommender moves it | How to measure it |
|-------|---------------------------|-------------------|
| Conversion / click-through | Relevant items shown instead of generic best-sellers | CTR on recommendation slots, A/B test |
| Basket size / watch time | "Next item" and "complete the set" suggestions | Average order value, minutes watched per session |
| Retention / churn | Users keep finding something they like | 30/60/90-day retention, subscription churn |
| Long-tail sales | Surfaces catalogue items that never reach the home page | Catalogue coverage, share of sales outside top 10% |
| Re-engagement | Personalised emails and push notifications | Email CTR, reactivation rate |

Published reference points (use as orders of magnitude, not promises):

- Netflix estimated its recommender and personalisation save it **over $1B per year** in
  avoided churn (Gomez-Uribe & Hunt, 2015).
- About **35% of Amazon purchases** and **75% of Netflix viewing** were attributed to
  recommendations (McKinsey, 2013).

### 3.2 Costs and effort the client stops paying

| Today, without the tool | With SuisenSha | Saving |
|-------------------------|----------------|--------|
| Editors hand-curate "related" and "you may also like" lists | Generated automatically, editors only override | Hours of editorial work per week; scales with catalogue size |
| Copywriters write "why we recommend this" blurbs | RAG generates grounded explanations, cached in bulk | Copy cost per item → near zero |
| Hiring an ML engineer to build recommendations (€60–120k/year loaded cost) | Deploy a ready pipeline, retrain on a schedule | Most of one headcount for a small client |
| Paying a usage-priced SaaS that grows with traffic | Self-hosted, flat infrastructure cost | Large at high traffic; also no vendor lock-in |
| Support tickets asking "why am I seeing this?" | Explanation shown next to each recommendation | Fewer tickets, higher trust |
| Compliance work explaining the recommender to regulators | Built-in explanation and a non-personalised mode | Reduces legal effort under the EU Digital Services Act |
| Guessing whether a homepage change helped | Built-in A/B testing and dashboards | Decisions based on measured uplift |

### 3.3 ROI template for a sales conversation

```
Monthly uplift   = sessions × rec_slot_CTR_gain × conversion_rate × avg_order_value
Monthly savings  = editor_hours_saved × hourly_rate + copy_items × cost_per_blurb
                 + saas_fee_avoided
Monthly cost     = hosting + licence/support fee
ROI              = (uplift + savings − cost) / cost
```

Fill it in with the client's own numbers during the pilot. A pilot that measures the CTR
gain with an A/B test is the strongest sales asset the product can have.

---

## 4. What clients will demand

Grouped roughly by how early in a sales process each request appears.

### 4.1 Before signing (deal-breakers)

| Demand | Notes | Effort |
|--------|-------|--------|
| **Use my data** | Import catalogue + interactions from CSV, Parquet, a database, or an event stream | M |
| **Proof it works on my data** | Offline benchmark report on their history before going live | M |
| **Runs in my infrastructure** | Docker image, Helm chart, documented hardware needs | M |
| **Data privacy** | GDPR: delete a user and their history, data stays in the client's region | M |
| **Security** | API keys or OAuth2, HTTPS, rate limits, audit log | M |
| **Latency guarantee** | e.g. p95 < 50 ms for recommendations | M |

### 4.2 During the pilot

| Demand | Notes | Effort |
|--------|-------|--------|
| **Business rules** | Exclude out-of-stock or unlicensed items, boost promotions, pin editorial picks, region or age restrictions | M |
| **Cold start** | Sensible results for new users (popular, onboarding quiz) and new items (content embeddings) | M |
| **Real-time updates** | A click now changes the next recommendation, without retraining | L |
| **Several widget types** | "Because you watched X", "Similar items", "Trending", "Continue watching", "Complete the bundle" | M |
| **A/B testing** | Split traffic between models, report uplift with confidence intervals | L |
| **Dashboard** | CTR, coverage, top items, model health, explanation samples | M |

### 4.3 After go-live (retention of the client)

| Demand | Notes | Effort |
|--------|-------|--------|
| **Automatic retraining** | Nightly or weekly, with a quality gate before promotion | M |
| **Monitoring & alerts** | Latency, error rate, data drift, metric drops | M |
| **Diversity and fairness controls** | Avoid showing ten near-identical items; give new creators exposure | M |
| **Multi-language explanations** | Explanations in the user's language | S–M |
| **SDKs and widgets** | JavaScript widget, Python/JS client, webhook on new items | M |
| **Multi-tenant SaaS** | One deployment, many isolated clients | L |
| **Regulatory transparency** | DSA Art. 27: explain main ranking parameters; Art. 38: offer a non-profiling option for very large platforms | S–M |

---

## 5. Enhancements by area

### 5.1 Recommendation quality

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Popularity and ItemKNN baselines** | Any claim of quality needs these; they are often surprisingly strong | S |
| **BPR-MF / ALS (implicit)** | Classic collaborative filtering, cheap, strong on dense data | S |
| **GRU4Rec, BERT4Rec** | Standard sequential baselines to compare SASRec against | M |
| **LightGCN** | Graph-based collaborative filtering, strong in recent literature | M |
| **Two-tower retrieval + re-ranker** | The production standard: fast candidate retrieval, then a precise ranker | L |
| **Hybrid model** | SASRec item embeddings + content embeddings, so new items are not invisible | M |
| **Side features** | Genres, release year, user demographics, time of day, device | M |
| **Re-ranking for diversity / novelty** | MMR or calibration so lists are not ten copies of one genre | S |
| **Popularity-bias correction** | Inverse propensity or calibrated popularity to surface the long tail | M |
| **Hyperparameter search** | Optuna sweeps with a validation split, not the test set | S |
| **Early stopping on a validation split** | Loss was still falling at epoch 20; stop on validation NDCG | S |
| **Sliding windows for long sequences** | Users with > 50 items currently train only on their last window | S |

### 5.2 Explanations (RAG)

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Instruction-tuned LLM** | DistilGPT-2 cannot follow the prompt; a small instruct model (local) or a hosted API fixes the demo | S |
| **Structured output** | Return JSON `{item_id, reason}` so the UI can attach each reason to its item | S |
| **Grounding check** | Reject any recommended title not in the retrieved candidate list | S |
| **Use SASRec candidates** | Explain the better recommender, not only the content one | S |
| **Richer item context** | Plot summaries, tags, reviews in the retrieval corpus | M |
| **Template fallback** | Deterministic "Because you liked X and Y (both Sci-Fi)" when the LLM is unavailable | S |
| **Explanation quality evaluation** | See [benchmark 6.4](#64-explanation-quality) | M |
| **Cost control** | Batch generation, caching by (user, candidate set) hash, token budgets | S |

### 5.3 Data platform

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Pluggable data connectors** | CSV/Parquet, PostgreSQL, BigQuery, S3 | M |
| **Event ingestion endpoint** | `POST /events` for views, clicks, purchases, ratings | M |
| **Schema validation** | Pandera or Pydantic checks on incoming data | S |
| **Temporal train/validation/test split** | More realistic than leave-one-out; prevents future leakage | S |
| **More datasets** | MovieLens-1M/20M, Amazon Reviews, Steam, Yelp, MIND (news) to show generality | M |
| **Feature store** | Consistent features between training and serving (e.g. Feast) | L |

### 5.4 Serving and operations

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Docker + docker-compose** | One-command demo: API + UI + cache | S |
| **Approximate nearest-neighbour index** | FAISS HNSW or IVF for catalogues of millions of items | S |
| **Batch inference + Redis cache** | Pre-compute top-N per user; serve in ~1 ms | M |
| **Model registry and versioning** | MLflow: which model served which request, with rollback | M |
| **Scheduled retraining** | Prefect or Airflow pipeline with quality gate | M |
| **Observability** | Prometheus metrics, Grafana dashboards, structured JSON logs, OpenTelemetry traces | M |
| **Drift monitoring** | Alert when the interaction distribution or metrics shift (Evidently) | M |
| **Load testing** | Locust or k6 scripts in CI to protect latency targets | S |
| **Kubernetes / Helm chart** | Autoscaling for clients with real traffic | L |
| **GPU-optional training** | Already supported by PyTorch; document and test it | S |

### 5.5 API and product features

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **API versioning (`/v1/`)** | Clients integrate once and do not break on updates | S |
| **Auth** | API keys per client, OAuth2 for the dashboard | M |
| **Rate limiting and quotas** | Protect the service and enable usage-based pricing | S |
| **Request parameters** | `k`, filters (genre, price, region), exclude list, diversity level | S |
| **Item-to-item endpoint** | "Similar to this item" for product pages | S |
| **Session-based recommendations** | Anonymous users recommended from the current session only | M |
| **Feedback endpoint** | Thumbs up/down, "not interested", fed back into training | M |
| **Business-rules engine** | Boost, bury, pin, exclude, schedule — editable in the dashboard | M |
| **Admin dashboard** | Replace the Streamlit demo with a client-facing console | L |
| **A/B experiment service** | Assign users to variants, log exposures, compute uplift | L |
| **Multi-tenancy** | Isolated data, models and keys per client | L |
| **JavaScript widget** | Drop-in carousel so small clients integrate without developers | M |

### 5.6 Security, privacy and compliance

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Right to erasure** | `DELETE /users/{id}` removes interactions, cache entries and triggers retraining | S |
| **Pseudonymised user IDs** | Never store emails or names in the recommender | S |
| **Encryption at rest and in transit** | TLS everywhere; encrypted volumes | S |
| **Audit log** | Who changed which business rule and when | S |
| **Dependency and container scanning** | `pip-audit`, Trivy in CI | S |
| **Non-personalised mode** | Required for very large platforms under DSA Art. 38 | S |
| **Transparency page** | Plain-language description of ranking factors (DSA Art. 27) | S |
| **Bias and fairness report** | Exposure by item group and provider; useful for marketplaces | M |

### 5.7 Engineering quality

| Enhancement | Why | Effort |
|-------------|-----|--------|
| **Coverage target in CI** | `pytest --cov` with a minimum threshold | S |
| **Integration test with a tiny trained model** | Exercise the full pipeline in CI in under a minute | S |
| **Config files instead of code constants** | YAML/TOML experiment configs, overridable from the CLI | S |
| **Experiment tracking** | MLflow or Weights & Biases for every training run | S |
| **Release process** | Semantic versioning, changelog, tagged Docker images | S |
| **Docs site** | MkDocs with API reference and tutorials | M |

---

## 6. Benchmark plan

Benchmarks are what turn "it works" into "it is worth paying for". Each benchmark below
should be a script that writes a Markdown/CSV report so results are reproducible and can be
put in front of a client.

### 6.1 Offline accuracy

**Datasets.** MovieLens-100K (current), MovieLens-1M, Amazon Reviews (one category such as
Books), Steam. Different sizes and densities show whether conclusions generalise.

**Protocol.**

- Leave-one-out *and* temporal split (train on the past, test on the future).
- Full-catalogue ranking (current) plus sampled-negative ranking (100 negatives) for
  comparison with published papers.
- Validation split for tuning; the test split is touched once.
- 5 random seeds; report mean ± 95% confidence interval.

**Models.** Random, Popularity, ItemKNN, BPR-MF, Embedding + FAISS, GRU4Rec, SASRec,
BERT4Rec, LightGCN, Hybrid.

**Metrics.**

| Family | Metrics |
|--------|---------|
| Accuracy | HitRate@k, NDCG@k, Recall@k, MRR, MAP (k = 5, 10, 20) |
| Beyond accuracy | Catalogue coverage, intra-list diversity, novelty, serendipity |
| Bias | Popularity bias (average popularity of recommended items), Gini index of exposure |
| Segments | Results split by user activity (cold, medium, heavy) and item popularity (head, tail) |

**Result table template.**

| Model | NDCG@10 | HR@10 | MRR | Coverage | Diversity | Novelty | Train time |
|-------|---------|-------|-----|----------|-----------|---------|------------|
| Popularity | | | | | | | |
| ItemKNN | | | | | | | |
| SASRec | | | | | | | |

Consider cross-checking numbers against RecBole, a standard library that implements most
of these models, to confirm the implementation is not off.

### 6.2 System performance

| Benchmark | Measure | Target to aim for |
|-----------|---------|-------------------|
| API latency | p50 / p95 / p99 per endpoint under load | p95 < 50 ms (embedding, SASRec); cached RAG < 20 ms |
| Throughput | Requests per second per CPU core | Report the curve, find the saturation point |
| Training time | Minutes per epoch vs catalogue/user count | Linear scaling up to 1M interactions |
| Memory | Peak RAM for training and serving | Fits a 2 GB container for small clients |
| Index scaling | FAISS flat vs HNSW: recall and latency at 10k, 100k, 1M items | HNSW recall > 0.95 at 10× lower latency |
| Cold start of the service | Seconds from container start to first response | < 10 s without LLM preload |

Tools: Locust or k6 for load, `pytest-benchmark` for micro-benchmarks, `memray` for memory.

### 6.3 Cost

| Benchmark | Measure |
|-----------|---------|
| Infrastructure | € per 1M recommendations served (CPU instance prices) |
| Training | € per retraining run on each dataset size |
| LLM explanations | € per 1k explanations: local model vs hosted API; cache hit rate |
| Comparison | Same traffic priced on AWS Personalize / Google Recommendations AI list prices |

This table is the direct answer to "how much money do I save".

### 6.4 Explanation quality

| Metric | How |
|--------|-----|
| Grounding rate | Share of recommended titles that exist in the candidate list (target: 100%) |
| Hallucination rate | Explanations that state false facts about an item (genre, year) |
| Relevance | LLM-as-judge score: does the reason connect history to the item? |
| Readability and length | Automatic readability score, word count |
| Human evaluation | 50 users × 5 explanations rated 1–5 for helpfulness and trust |
| Latency and cost | Seconds and tokens per explanation |

### 6.5 Online evaluation (with a pilot client)

- A/B test: current client solution vs SuisenSha, primary metric agreed in advance
  (CTR, conversion or watch time), sample size computed before the test.
- Interleaving for faster, lower-traffic comparisons between two models.
- Guardrail metrics: latency, error rate, diversity, complaint rate.

---

## 7. Phased roadmap

The same work split into 45 ordered pull requests, with dependencies and acceptance
criteria, is in the [PR Plan](PR_PLAN.md).

### Phase A — Credible demo (≈ 6–9 weeks)

Goal: a demo that survives a technical buyer's questions.

- [ ] Instruction-tuned LLM, structured output, grounding check, template fallback
- [ ] Popularity, ItemKNN and BPR-MF baselines
- [ ] Validation split, early stopping, 5-seed confidence intervals
- [ ] Beyond-accuracy metrics (coverage, diversity, novelty)
- [ ] Docker + docker-compose
- [ ] API: `/v1/`, `k` and filter parameters, item-to-item endpoint
- [ ] Benchmark report generated by one command

### Phase B — Pilot-ready (≈ 3–4 months)

Goal: run on one client's real data.

- [ ] Data connectors and schema validation
- [ ] Event ingestion endpoint and incremental updates
- [ ] Business-rules engine (exclude, boost, pin)
- [ ] Cold-start strategy for users and items; hybrid model
- [ ] Auth, rate limiting, right to erasure
- [ ] Redis cache, HNSW index, load-test targets in CI
- [ ] Monitoring: Prometheus + Grafana, structured logs
- [ ] Scheduled retraining with a quality gate

### Phase C — Product (≈ 4–6 months)

Goal: sell to several clients without custom work for each.

- [ ] A/B testing service and uplift dashboard
- [ ] Client-facing admin console
- [ ] Multi-tenancy
- [ ] JavaScript widget and client SDKs
- [ ] Two-tower retrieval + re-ranker
- [ ] Drift monitoring, model registry, rollback
- [ ] DSA transparency page and non-personalised mode
- [ ] Docs site, versioned releases

---

## 8. Commercial packaging

| Tier | For | Contents |
|------|-----|----------|
| **Open core** | Developers, evaluation | Pipeline, models, API, Docker, benchmarks |
| **Pro (self-hosted licence)** | SMB with an ops person | Business rules, connectors, dashboard, retraining, support |
| **Managed (SaaS)** | Clients without ops | Hosted, multi-tenant, A/B testing, SLA, priced per 1k requests or per active user |
| **Services** | Larger clients | Integration, custom models, pilot with A/B test and ROI report |

The pilot with a measured A/B uplift and an ROI report (section 3.3) is the main sales tool
for every tier.

---

## 9. References

- C. Gomez-Uribe, N. Hunt. *The Netflix Recommender System: Algorithms, Business Value,
  and Innovation.* ACM TMIS, 2015.
- McKinsey & Company. *How retailers can keep up with consumers.* 2013.
- W.-C. Kang, J. McAuley. *Self-Attentive Sequential Recommendation.* ICDM, 2018.
- F. Sun et al. *BERT4Rec.* CIKM, 2019.
- X. He et al. *LightGCN.* SIGIR, 2020.
- S. Rendle et al. *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI, 2009.
- W. X. Zhao et al. *RecBole: A Unified Recommendation Library.* CIKM, 2021.
- W. Krichene, S. Rendle. *On Sampled Metrics for Item Recommendation.* KDD, 2020.
- Regulation (EU) 2022/2065, *Digital Services Act*, Articles 27 and 38.
