from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline as hf_pipeline

from pipeline.config import rag_cfg
from pipeline.data.loader import build_movie_lookup, load_processed_movies, load_sequences

if TYPE_CHECKING:
    from pipeline.models.embedding import ItemIndex

logger = logging.getLogger(__name__)

# Lazy singleton — avoid reloading the 82M-param model on every call
_generator = None


def retrieve_candidates(
    user_history_ids: list[int],
    item_index: ItemIndex,
    movie_lookup: dict[int, dict],
    top_k: int = rag_cfg.retrieval_k,
) -> list[dict]:
    candidates: list[dict] = []
    for iid, score in item_index.search(user_history_ids, top_k):
        info = movie_lookup.get(iid, {})
        candidates.append(
            {
                "item_id": iid,
                "title": info.get("title", f"Item {iid}"),
                "genres": info.get("genres", ""),
                "score": score,
            }
        )
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
    item_index: ItemIndex,
    movie_lookup: dict[int, dict],
) -> str:
    candidates = retrieve_candidates(user_history_ids, item_index, movie_lookup)
    if not candidates:
        return "No candidates found for this user."

    history_titles = [
        movie_lookup.get(i, {}).get("title", f"Item {i}") for i in user_history_ids[-10:]
    ]
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
    return output[0]["generated_text"][len(prompt) :].strip()


def run_phase4(
    movies: pd.DataFrame | None = None,
    sequences: dict[int, list[int]] | None = None,
) -> None:
    if movies is None:
        movies = load_processed_movies()
    if sequences is None:
        sequences = load_sequences()

    from pipeline.models.embedding import load_artefacts

    item_index = load_artefacts()
    movie_lookup = build_movie_lookup(movies)

    for uid in list(sequences)[:3]:
        seq = sequences[uid]
        recent = [movie_lookup.get(i, {}).get("title", f"Item {i}") for i in seq[-5:]]
        logger.info("═" * 65)
        logger.info("  🎬  User %d — Recent: %s", uid, ", ".join(recent))
        logger.info("═" * 65)

        response = generate_recommendations(seq, item_index, movie_lookup)
        logger.info("🤖 LLM output:\n%s\n", response)
