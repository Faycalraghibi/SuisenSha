from __future__ import annotations

import logging
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pipeline.config import (
    ARTIFACTS_DIR,
    EMBEDDINGS_NPY,
    FAISS_INDEX_PATH,
    ITEM_IDS_NPY,
    MOVIES_CSV,
    USER_SEQUENCES_PKL,
    embedding_cfg,
    ensure_dirs,
    eval_cfg,
)
from pipeline.evaluation.metrics import hit_rate_at_k, ndcg_at_k

logger = logging.getLogger(__name__)


def generate_item_embeddings(
    movies: pd.DataFrame,
    model_name: str = embedding_cfg.model_name,
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Generating item embeddings with %s …", model_name)
    model = SentenceTransformer(model_name)
    descriptions = movies["description"].tolist()
    # normalize_embeddings=True so inner-product == cosine similarity in FAISS
    embeddings = model.encode(
        descriptions,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    return movies["item_id"].values, np.asarray(embeddings, dtype="float32")  # type: ignore[return-value]


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """IndexFlatIP because embeddings are L2-normalised — IP ≡ cosine."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_artefacts(
    index: faiss.IndexFlatIP,
    item_ids: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(EMBEDDINGS_NPY, embeddings)
    np.save(ITEM_IDS_NPY, item_ids)
    logger.info("Saved FAISS artefacts to %s", ARTIFACTS_DIR)


def load_artefacts() -> tuple[faiss.IndexFlatIP, np.ndarray, np.ndarray]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    embeddings = np.load(EMBEDDINGS_NPY)
    item_ids = np.load(ITEM_IDS_NPY)
    return index, item_ids, embeddings


def recommend_for_user(
    user_history_ids: list[int],
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    top_k: int = embedding_cfg.top_k,
) -> list[int]:
    id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
    hist_idx = [id_to_idx[i] for i in user_history_ids if i in id_to_idx]
    if not hist_idx:
        return []

    user_vec = embeddings[hist_idx].mean(axis=0, keepdims=True)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm

    # Over-fetch to compensate for filtering out already-seen items
    _, indices = index.search(user_vec.astype("float32"), top_k + len(hist_idx))
    history_set = set(user_history_ids)
    return [int(item_ids[i]) for i in indices[0] if i >= 0 and int(item_ids[i]) not in history_set][
        :top_k
    ]


def evaluate(
    sequences: dict[int, list[int]],
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    k: int = 10,
) -> dict[str, float]:
    hits, ndcgs = [], []
    for seq in tqdm(sequences.values(), desc=f"Eval embedding @{k}"):
        if len(seq) < 2:
            continue
        recs = recommend_for_user(seq[:-1], item_ids, embeddings, index, top_k=k)
        hits.append(hit_rate_at_k(recs, seq[-1], k))
        ndcgs.append(ndcg_at_k(recs, seq[-1], k))

    result = {
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
    logger.info(
        "Embedding Recommender — HR@%d: %.4f  NDCG@%d: %.4f",
        k,
        result[f"HitRate@{k}"],
        k,
        result[f"NDCG@{k}"],
    )
    return result


def run_phase2(
    movies: pd.DataFrame | None = None,
    sequences: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    if movies is None:
        movies = pd.read_csv(MOVIES_CSV)
    if sequences is None:
        with open(USER_SEQUENCES_PKL, "rb") as f:
            sequences = pickle.load(f)

    item_ids, embeddings = generate_item_embeddings(movies)
    index = build_faiss_index(embeddings)
    save_artefacts(index, item_ids, embeddings)

    results: dict[str, float] = {}
    for k in eval_cfg.k_values:
        results.update(evaluate(sequences, item_ids, embeddings, index, k=k))
    return results
