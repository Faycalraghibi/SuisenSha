from __future__ import annotations

import logging
import pickle

import faiss
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline

from pipeline.config import (
    MOVIES_CSV,
    USER_SEQUENCES_PKL,
    rag_cfg,
)

logger = logging.getLogger(__name__)

# Lazy singleton — avoid reloading the 82M-param model on every call
_generator = None


def retrieve_candidates(
    user_history_ids: list[int],
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    movies_df: pd.DataFrame,
    top_k: int = rag_cfg.retrieval_k,
) -> list[dict]:
    id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
    hist_idx = [id_to_idx[i] for i in user_history_ids if i in id_to_idx]
    if not hist_idx:
        return []

    user_vec = embeddings[hist_idx].mean(axis=0, keepdims=True)
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec /= norm

    scores, indices = index.search(user_vec.astype("float32"), top_k + len(hist_idx))
    movie_info = movies_df.set_index("item_id").to_dict("index")
    history_set = set(user_history_ids)

    candidates: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        iid = int(item_ids[idx])
        if iid in history_set:
            continue
        info = movie_info.get(iid, {})
        candidates.append(
            {
                "item_id": iid,
                "title": info.get("title", f"Item {iid}"),
                "genres": info.get("genres", ""),
                "score": float(score),
            }
        )
        if len(candidates) >= top_k:
            break
    return candidates


def _get_generator():
    global _generator
    if _generator is not None:
        return _generator

    logger.info("Loading LLM: %s …", rag_cfg.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(rag_cfg.llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(rag_cfg.llm_model_name)
    # GPT-2 has no pad token by default — reuse EOS to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device=-1 forces CPU inference (safe on machines without CUDA)
    _generator = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    logger.info("LLM ready.")
    return _generator


def _build_prompt(history_titles: list[str], candidate_titles: list[str]) -> str:
    hist = ", ".join(history_titles[-10:])
    cands = "\n".join(f"  - {t}" for t in candidate_titles[:15])
    return (
        "You are a movie recommendation assistant.\n\n"
        f"The user recently watched: {hist}\n\n"
        f"Here are some candidate movies:\n{cands}\n\n"
        "Based on the user's taste, recommend the best 5 movies from the "
        "candidates above and explain briefly why each is a good match.\n\n"
        "Recommendations:\n"
    )


def generate_recommendations(
    user_history_ids: list[int],
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    movies_df: pd.DataFrame,
) -> str:
    candidates = retrieve_candidates(user_history_ids, item_ids, embeddings, index, movies_df)
    if not candidates:
        return "No candidates found for this user."

    title_lookup = movies_df.set_index("item_id")["title"].to_dict()
    history_titles = [title_lookup.get(i, f"Item {i}") for i in user_history_ids[-10:]]
    candidate_titles = [c["title"] for c in candidates]

    prompt = _build_prompt(history_titles, candidate_titles)
    gen = _get_generator()
    output = gen(
        prompt,
        max_new_tokens=rag_cfg.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        pad_token_id=gen.tokenizer.eos_token_id,
    )
    # Strip the prompt prefix — only return the generated continuation
    return output[0]["generated_text"][len(prompt) :].strip()


def run_phase4(
    movies: pd.DataFrame | None = None,
    sequences: dict[int, list[int]] | None = None,
) -> None:
    if movies is None:
        movies = pd.read_csv(MOVIES_CSV)
    if sequences is None:
        with open(USER_SEQUENCES_PKL, "rb") as f:
            sequences = pickle.load(f)

    from pipeline.models.embedding import load_artefacts

    index, item_ids, embeddings = load_artefacts()

    title_lookup = movies.set_index("item_id")["title"].to_dict()
    sample_users = list(sequences.keys())[:3]

    for uid in sample_users:
        seq = sequences[uid]
        recent = [title_lookup.get(i, f"Item {i}") for i in seq[-5:]]
        logger.info("═" * 65)
        logger.info("  🎬  User %d — Recent: %s", uid, ", ".join(recent))
        logger.info("═" * 65)

        response = generate_recommendations(seq, item_ids, embeddings, index, movies)
        logger.info("🤖 LLM output:\n%s\n", response)
