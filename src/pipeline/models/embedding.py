from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

logger = logging.getLogger(__name__)


@dataclass
class ItemIndex:
    """FAISS index plus the item-id ↔ row mapping needed to query it."""

    index: faiss.Index
    item_ids: np.ndarray
    embeddings: np.ndarray
    id_to_idx: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.id_to_idx = {int(iid): idx for idx, iid in enumerate(self.item_ids)}

    def user_vector(self, history_ids: list[int]) -> np.ndarray | None:
        hist_idx = [self.id_to_idx[i] for i in history_ids if i in self.id_to_idx]
        if not hist_idx:
            return None
        user_vec = self.embeddings[hist_idx].mean(axis=0, keepdims=True)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        return user_vec.astype("float32")

    def search(self, history_ids: list[int], top_k: int) -> list[tuple[int, float]]:
        """Return (item_id, cosine score) for the top_k unseen items closest to the user."""
        user_vec = self.user_vector(history_ids)
        if user_vec is None:
            return []

        history_set = set(history_ids)
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


def build_item_index(item_ids: np.ndarray, embeddings: np.ndarray) -> ItemIndex:
    """IndexFlatIP because embeddings are L2-normalised — IP ≡ cosine."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)
    return ItemIndex(index=index, item_ids=item_ids, embeddings=embeddings)


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
