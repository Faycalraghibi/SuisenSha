# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-stage Dockerfile and docker-compose service configuration for API and static UI.
- Sub-minute end-to-end fixture pipeline test running phases 1-5 against real FastAPI lifespan.
- CI security audit check via `pip-audit`.
- Minimum code coverage threshold (75%) enforced in test suite.
- Container vulnerability scanning with Trivy in CI.
- Automated release workflow publishing Docker images and GitHub releases on semantic tags.
- Versioned `/v1` API routes with query parameters (`k`, `exclude`, `genres`, `personalised`).
- Product-page similar items endpoint `GET /v1/items/{item_id}/similar`.
- EU DSA Art. 27 ranking transparency endpoint `GET /v1/transparency` in JSON and HTML.

### Changed
- RAG cache strictly enforces matching user ID, candidate set, and language; wildcard and legacy-table fallbacks removed from lookups. Legacy table writes removed.

## [0.1.0] - 2026-10-10

### Added
- Grounded RAG explanation module rejecting candidate hallucinations.
- `LLMBackend` pluggable interface supporting local instruction-tuned, hosted API, and fake test backends.
- Structured JSON output format for per-item recommendation reasons with template fallbacks.
- Multi-recommender candidate retrieval (SASRec and embedding index) with content and language-keyed caching.
- Initial MovieLens sequential and content embedding recommendation pipeline.
