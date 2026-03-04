from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Two levels up: src/pipeline/config.py → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR: Path = _PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = _PROJECT_ROOT / "artifacts"
OUTPUTS_DIR: Path = _PROJECT_ROOT / "outputs"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_ZIP: Path = DATA_DIR / "ml-100k.zip"
MOVIELENS_DIR: Path = DATA_DIR / "ml-100k"

INTERACTIONS_CSV: Path = DATA_DIR / "interactions.csv"
MOVIES_CSV: Path = DATA_DIR / "movies.csv"
USER_SEQUENCES_PKL: Path = DATA_DIR / "user_sequences.pkl"

FAISS_INDEX_PATH: Path = ARTIFACTS_DIR / "faiss_index.bin"
EMBEDDINGS_NPY: Path = ARTIFACTS_DIR / "item_embeddings.npy"
ITEM_IDS_NPY: Path = ARTIFACTS_DIR / "item_ids.npy"

SEQ_MODEL_PATH: Path = ARTIFACTS_DIR / "sasrec_model.pt"
RAG_CACHE_DB: Path = ARTIFACTS_DIR / "rag_cache.db"

SEED: int = 42


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    top_k: int = 10


@dataclass(frozen=True)
class SASRecConfig:
    max_seq_len: int = 50
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 128


@dataclass(frozen=True)
class RAGConfig:
    llm_model_name: str = "distilgpt2"
    max_new_tokens: int = 300
    retrieval_k: int = 20


@dataclass(frozen=True)
class EvalConfig:
    k_values: tuple[int, ...] = (5, 10)


embedding_cfg = EmbeddingConfig()
sasrec_cfg = SASRecConfig()
rag_cfg = RAGConfig()
eval_cfg = EvalConfig()


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
