from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    ARTIFACTS_DIR,
    EMBEDDINGS_NPY,
    FAISS_INDEX_PATH,
    ITEM_IDS_NPY,
    embedding_cfg,
    ensure_dirs,
    eval_cfg,
)
from pipeline.data.loader import load_processed_movies, load_sequences
from pipeline.evaluation.metrics import evaluate_leave_one_out
from pipeline.models.base import Recommender

logger = logging.getLogger(__name__)


@dataclass
class ItemIndex(Recommender):
    """FAISS index plus the item-id ↔ row mapping needed to query it."""

    index: faiss.Index
    item_ids: np.ndarray
    embeddings: np.ndarray
    index_type: str = "flat"
    id_to_idx: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.id_to_idx = {int(iid): idx for idx, iid in enumerate(self.item_ids)}

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
        index_type: str = "flat",
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ) -> ItemIndex:
        return build_item_index(
            item_ids=item_ids,
            embeddings=embeddings,
            index_type=index_type,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

    def fit(self, train_data: Any) -> None:
        """ItemIndex is typically pre-computed via generate_embeddings and build_faiss_index."""
        pass

    def user_vector(self, history_ids: list[int]) -> np.ndarray | None:
        hist_idx = [self.id_to_idx[i] for i in history_ids if i in self.id_to_idx]
        if not hist_idx:
            return None
        user_vec = self.embeddings[hist_idx].mean(axis=0, keepdims=True)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        return user_vec.astype("float32")

    def search(
        self,
        history_ids: list[int],
        top_k: int,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return (item_id, cosine score) for the top_k unseen items closest to the user."""
        user_vec = self.user_vector(history_ids)
        if user_vec is None:
            return []

        history_set = set(history_ids) | (set(exclude) if exclude else set())
        # Over-fetch to compensate for filtering out already-seen items
        fetch = min(top_k + len(history_set), self.index.ntotal)
        scores, indices = self.index.search(user_vec, fetch)

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            iid = int(self.item_ids[idx])
            if iid in history_set:
                continue
            results.append((iid, float(score)))
            if len(results) >= top_k:
                break
        return results

    def similar_items(
        self,
        item_id: int,
        top_k: int = 10,
        exclude: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return (item_id, cosine score) for items closest to the given item_id."""
        if item_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[item_id]
        item_vec = self.embeddings[idx : idx + 1].astype("float32")
        norm = np.linalg.norm(item_vec)
        if norm > 0:
            item_vec = (item_vec / norm).astype("float32")

        exclude_set = (set(exclude) if exclude else set()) | {item_id}
        fetch = min(top_k + len(exclude_set), self.index.ntotal)
        scores, indices = self.index.search(item_vec, fetch)

        results: list[tuple[int, float]] = []
        for score, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            iid = int(self.item_ids[i])
            if iid in exclude_set:
                continue
            results.append((iid, float(score)))
            if len(results) >= top_k:
                break
        return results

    def recommend(self, history_ids: list[int], top_k: int = embedding_cfg.top_k) -> list[int]:
        return [iid for iid, _ in self.search(history_ids, top_k)]


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


def build_hnsw_index(
    dim: int,
    m: int = 32,
    ef_construction: int = 64,
    ef_search: int = 64,
) -> faiss.IndexHNSWFlat:
    """Build an approximate nearest neighbor index using HNSW with inner product."""
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index


def evaluate_hnsw_recall(
    flat_index: faiss.Index,
    hnsw_index: faiss.Index,
    query_vectors: np.ndarray,
    k: int = 10,
) -> float:
    """Evaluate top-k recall of HNSW approximate index compared to exact flat index."""
    q_vecs = query_vectors.astype("float32")
    _, flat_indices = flat_index.search(q_vecs, k)
    _, hnsw_indices = hnsw_index.search(q_vecs, k)

    recalls: list[float] = []
    for f_row, h_row in zip(flat_indices, hnsw_indices):
        target_set = {int(idx) for idx in f_row if idx >= 0}
        retrieved_set = {int(idx) for idx in h_row if idx >= 0}
        if not target_set:
            continue
        recall = len(target_set & retrieved_set) / len(target_set)
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def build_item_index(
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    index_type: str = "flat",
    m: int = 32,
    ef_construction: int = 64,
    ef_search: int = 64,
) -> ItemIndex:
    """Build ItemIndex using either exact flat inner-product or HNSW ANN index."""
    dim = embeddings.shape[1]
    if index_type.lower() == "hnsw":
        index = build_hnsw_index(
            dim=dim,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    logger.info(
        "FAISS %s index built — %d vectors, dim=%d",
        index_type.upper(),
        index.ntotal,
        dim,
    )
    return ItemIndex(
        index=index,
        item_ids=item_ids,
        embeddings=embeddings,
        index_type=index_type.lower(),
    )


def save_artefacts(item_index: ItemIndex) -> None:
    ensure_dirs()
    faiss.write_index(item_index.index, str(FAISS_INDEX_PATH))
    np.save(EMBEDDINGS_NPY, item_index.embeddings)
    np.save(ITEM_IDS_NPY, item_index.item_ids)
    logger.info("Saved FAISS artefacts to %s", ARTIFACTS_DIR)


def load_artefacts() -> ItemIndex:
    return ItemIndex(
        index=faiss.read_index(str(FAISS_INDEX_PATH)),
        item_ids=np.load(ITEM_IDS_NPY),
        embeddings=np.load(EMBEDDINGS_NPY),
    )


def evaluate(
    sequences: dict[int, list[int]],
    item_index: ItemIndex,
    k_values: tuple[int, ...] = eval_cfg.k_values,
) -> dict[str, float]:
    result = evaluate_leave_one_out(
        sequences, item_index.recommend, k_values, desc="Eval embedding"
    )
    logger.info("Embedding Recommender — %s", result)
    return result


def run_phase2(
    movies: pd.DataFrame | None = None,
    sequences: dict[int, list[int]] | None = None,
) -> dict[str, float]:
    if movies is None:
        movies = load_processed_movies()
    if sequences is None:
        sequences = load_sequences()

    item_ids, embeddings = generate_item_embeddings(movies)
    item_index = build_item_index(item_ids, embeddings)
    save_artefacts(item_index)

    return evaluate(sequences, item_index)
