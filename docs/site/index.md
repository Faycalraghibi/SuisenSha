# SuisenSha Recommendation Platform

Welcome to **SuisenSha** (推薦者), an enterprise-ready, low-latency recommendation engine built for real-time personalization, retrieval-augmented recommendations, sequential deep learning, and multi-tenant SaaS workloads.

## Core Capabilities

- **State-of-the-Art Recommendation Models**:
  - Global & segment popularity baselines
  - Vector similarity search with FAISS and Sentence Transformers
  - SASRec self-attention sequential recommendations (PyTorch)
  - Configurable hybrid ranker combining popularity, vector embeddings, and sequential signals
- **Multi-Tenant Architecture**:
  - Strict tenant data isolation across SQLite/PostgreSQL storage, Redis caching, and MLflow model registries
  - Tenant-level monthly request quotas and burst rate limiting
  - Admin RBAC and secure API key authentication
- **EU Digital Services Act (DSA) Compliance**:
  - Transparent item explainability badges & reason breakdowns
  - User-driven exclusions ("Not interested" / dislike actions)
  - GDPR/DSA Right to Erasure endpoints (`DELETE /v1/users/{user_id}/data`)
- **Developer Integration**:
  - Client SDKs in typed Python and TypeScript
  - Embeddable vanilla JavaScript carousel widget with zero build dependencies
  - OpenAPI 3.1.0 specifications
- **Production-Ready Deployment**:
  - Kubernetes Helm chart with Horizontal Pod Autoscaling (HPA) for API and UI
  - Prometheus metrics instrumentation and structured JSON audit logging
  - Dual cache layer (in-memory LRU + Redis cluster support)
