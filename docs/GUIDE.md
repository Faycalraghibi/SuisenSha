# Developer Guide

---

## Architecture

```
                     ┌───────────────────────┐
                     │  Phase 1: Data & EDA  │
                     │  MovieLens-100K       │
                     └──────────┬────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │ Phase 2:       │  │ Phase 3:       │  │ Phase 4:       │
   │ Embedding      │  │ SASRec         │  │ RAG + LLM      │
   │ + FAISS        │  │ Transformer    │  │ Generation     │
   └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
           └───────────────────┼───────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Phase 5: Evaluation  │
                    │ & Comparison         │
                    └──────────────────────┘
```

---

## Results

| Pipeline | HitRate@5 | HitRate@10 | NDCG@5 | NDCG@10 |
|----------|-----------|------------|--------|---------|
| Embedding + FAISS | 0.0107 | 0.0160 | 0.0068 | 0.0085 |
| **SASRec (Transformer)** | **0.0267** | **0.0352** | **0.0169** | **0.0190** |
| RAG (Retrieval) | 0.0107 | 0.0160 | 0.0068 | 0.0085 |

> SASRec outperforms the embedding baseline by **~2.2×** on HitRate@10.

---

## Project Structure

```
SuisenSha/
├── src/pipeline/
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Frozen dataclass configs
│   ├── data/loader.py       # Phase 1 — download, EDA, sequences
│   ├── models/
│   │   ├── embedding.py     # Phase 2 — Sentence-Transformer + FAISS
│   │   ├── sequential.py    # Phase 3 — SASRec (PyTorch)
│   │   └── rag.py           # Phase 4 — Retrieval + DistilGPT-2
│   └── evaluation/
│       └── metrics.py       # Phase 5 — HR, NDCG, comparison chart
├── tests/                   # 15 unit tests
├── data/                    # Downloaded dataset (gitignored)
├── artifacts/               # Model checkpoints, FAISS index
├── outputs/                 # EDA plots, comparison chart
└── pyproject.toml           # PEP 621 packaging + tool config
```

---

## Pipelines

| Phase | Module | What It Does |
|-------|--------|--------------|
| **1** | `data.loader` | Downloads MovieLens-100K (943 users, 1 682 items), runs EDA, builds per-user interaction sequences |
| **2** | `models.embedding` | Embeds movie descriptions with `all-MiniLM-L6-v2`, indexes 1 682 vectors in FAISS, recommends by user-profile similarity |
| **3** | `models.sequential` | SASRec — causal self-attention over item-ID sequences, trained for next-item prediction (20 epochs on CPU) |
| **4** | `models.rag` | Retrieves top-20 candidates via FAISS, feeds them with user history to DistilGPT-2 for natural-language explained recommendations |
| **5** | `evaluation.metrics` | HitRate@k, NDCG@k, Precision@k, Recall@k — side-by-side comparison with bar chart |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Sequential Model | [PyTorch](https://pytorch.org/) (SASRec) |
| LLM Generation | [Hugging Face Transformers](https://huggingface.co/docs/transformers/) (`distilgpt2`) |
| Data | [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) |

---

## Installation

```bash
# Production install
pip install -e .

# With dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

---

## Usage

```bash
# Run the full pipeline (all 5 phases)
python -m pipeline.cli

# Run a specific phase
python -m pipeline.cli --phase 1   # Data Loading & EDA
python -m pipeline.cli --phase 2   # Embedding Recommender
python -m pipeline.cli --phase 3   # SASRec Transformer
python -m pipeline.cli --phase 4   # RAG + LLM
python -m pipeline.cli --phase 5   # Evaluation & Comparison

# After `pip install -e .`, the CLI is also available as:
suisensha --phase 3
```

---

## Configuration

All hyperparameters live in [`src/pipeline/config.py`](../src/pipeline/config.py), grouped into frozen dataclasses:

| Config Class | Key Settings |
|-------------|--------------|
| `EmbeddingConfig` | model name, embedding dim, top-K |
| `SASRecConfig` | hidden dim, heads, layers, dropout, epochs, batch size |
| `RAGConfig` | LLM model name, max new tokens, retrieval K |
| `EvalConfig` | K values for metrics |

---

## Dev Commands (Makefile)

| Command | Purpose |
|---------|---------|
| `make install` | Install production deps |
| `make dev` | Install with dev deps |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with ruff |
| `make type-check` | Run mypy |
| `make test` | Run pytest suite |
| `make run` | Run full pipeline |
| `make run-phase PHASE=3` | Run a specific phase |
| `make clean` | Remove caches and data |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/eda_distributions.png` | Rating, user-activity & item-popularity histograms |
| `outputs/comparison.png` | Side-by-side metric bar chart across all pipelines |
| `artifacts/faiss_index.bin` | Persisted FAISS inner-product index |
| `artifacts/item_embeddings.npy` | Item embedding matrix (1 682 × 384) |
| `artifacts/sasrec_model.pt` | SASRec model checkpoint |

---

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

15 tests covering:
- **Metrics** — HitRate, NDCG, Precision, Recall (hit/miss/edge cases)
- **Data loader** — filtering by rating, minimum interactions, timestamp ordering
