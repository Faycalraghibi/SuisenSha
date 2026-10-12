# DSA Compliance & Explainability

Under the **European Union Digital Services Act (DSA)** (Article 27), online platforms must disclose the main parameters used in their algorithmic recommender systems and provide end-users with meaningful options to influence the system.

## Compliance Architecture

SuisenSha implements built-in compliance hooks across the recommendation life-cycle:

### 1. Parameter Transparency (DSA Article 27.1)
- Every recommendation endpoint accepts `?explain=true`.
- Responses return:
  - `explanation_type`: Algorithm paradigm (`popularity`, `similarity`, `sequential`, `hybrid`).
  - `feature_importances`: Weights assigned to recency, rating, similarity, and interaction frequency.
  - Per-item `explanation` string summarizing why the specific item was prioritized.

### 2. User Modification of Parameters (DSA Article 27.2)
- End-users can exclude items or themes by emitting `dislike` events or setting negative preferences.
- SuisenSha maintains a `user_exclusions` table per tenant.
- Excluded items are filtered out in candidate generation before ranking.

### 3. GDPR / DSA Right to Erasure
- The `DELETE /v1/users/{user_id}/data` endpoint allows instant purging of:
  - Historical interactions in SQLite / PostgreSQL.
  - Cached recommendation vectors and ranking lists in Redis.
  - Session histories.

### 4. Audit Logging & Non-Profiling Modes
- SuisenSha allows platforms to query fallback recommendations (`/v1/recommend/popular`) for unprofiled users.
- Every algorithmic inference is logged with tenant identity, model version, and timestamp.
