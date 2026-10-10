from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel

from pipeline.api.models import ExplainedItem
from pipeline.config import rag_cfg
from pipeline.data.loader import build_movie_lookup, load_processed_movies, load_sequences
from pipeline.models.explanations import generate_template_explanation, get_safe_explanation
from pipeline.models.llm import LLMBackend, get_llm_backend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LLMItemOutput(BaseModel):
    item_id: int
    reason: str


def retrieve_candidates(
    user_history_ids: list[int],
    recommender: Any = "sasrec",
    movie_lookup: dict[int, dict] | None = None,
    top_k: int = rag_cfg.retrieval_k,
) -> list[dict]:
    """Retrieve top-k candidate items from SASRec, embedding index, or a custom recommender."""
    if movie_lookup is None:
        movie_lookup = {}

    candidate_ids: list[int] = []

    if recommender == "sasrec" or (
        isinstance(recommender, str) and recommender.lower() == "sasrec"
    ):
        try:
            from pipeline.models.sequential import load_sasrec_model, predict_next_items

            model, _ = load_sasrec_model()
            candidate_ids = predict_next_items(
                model, user_history_ids, top_k=top_k, exclude=set(user_history_ids)
            )
        except Exception as exc:
            logger.warning(
                "Could not load SASRec checkpoint (%s), falling back to embedding index.", exc
            )
            recommender = "embedding"

    if recommender == "embedding" or (
        isinstance(recommender, str) and recommender.lower() == "embedding"
    ):
        from pipeline.models.embedding import load_artefacts

        item_index = load_artefacts()
        return [
            {
                "item_id": iid,
                "title": movie_lookup.get(iid, {}).get("title", f"Item {iid}"),
                "genres": movie_lookup.get(iid, {}).get("genres", ""),
                "score": score,
            }
            for iid, score in item_index.search(user_history_ids, top_k)
        ]

    if hasattr(recommender, "eval"):
        from pipeline.models.sequential import predict_next_items

        candidate_ids = predict_next_items(
            recommender, user_history_ids, top_k=top_k, exclude=set(user_history_ids)
        )
    elif hasattr(recommender, "search"):
        return [
            {
                "item_id": iid,
                "title": movie_lookup.get(iid, {}).get("title", f"Item {iid}"),
                "genres": movie_lookup.get(iid, {}).get("genres", ""),
                "score": score,
            }
            for iid, score in recommender.search(user_history_ids, top_k)
        ]
    elif callable(recommender):
        res = recommender(user_history_ids, top_k)
        candidate_ids = [r[0] if isinstance(r, tuple | list) else r for r in res]

    candidates: list[dict] = []
    for iid in candidate_ids:
        info = movie_lookup.get(iid, {})
        candidates.append(
            {
                "item_id": iid,
                "title": info.get("title", f"Item {iid}"),
                "genres": info.get("genres", ""),
            }
        )
    return candidates


def _build_structured_prompt(
    history_movies: list[dict],
    candidates: list[dict],
    lang: str = "en",
) -> str:
    recent = ", ".join(f"{m.get('title')} ({m.get('genres')})" for m in history_movies[-10:])
    cand_lines = "\n".join(
        f"  - ID {c['item_id']}: {c.get('title')} ({c.get('genres', '')})" for c in candidates
    )

    if lang == "fr":
        return (
            "Vous êtes un assistant de recommandation de films.\n\n"
            f"L'utilisateur a récemment apprécié: {recent}\n\n"
            f"Voici les films candidats:\n{cand_lines}\n\n"
            "Pour chaque film candidat, fournissez une brève raison (une phrase) "
            "expliquant pourquoi l'utilisateur va l'apprécier.\n"
            "N'inventez aucun film qui ne figure pas dans les candidats ou l'historique.\n"
            "Répondez UNIQUEMENT avec un tableau JSON valide au format suivant:\n"
            '[{"item_id": <int>, "reason": "<explication>"}]\n'
        )

    return (
        "You are a movie recommendation assistant.\n\n"
        f"The user recently watched and enjoyed: {recent}\n\n"
        f"Here are candidate movies:\n{cand_lines}\n\n"
        "For each candidate movie above, provide a brief 1-sentence grounded reason "
        "explaining why the user will enjoy it based on their taste.\n"
        "Do not mention any movie outside the candidates and recent history.\n"
        "Respond ONLY with a valid JSON array of objects:\n"
        '[{"item_id": <int>, "reason": "<explanation>"}]\n'
    )


def parse_and_validate_explanations(
    raw_output: str,
    candidates: list[dict],
    history_movies: list[dict],
    catalogue_titles: set[str] | None = None,
    lang: str = "en",
) -> list[ExplainedItem]:
    """Parse JSON array from LLM output with Pydantic and per-item safety fallback."""
    parsed_reasons: dict[int, str] = {}
    if raw_output:
        clean = raw_output.strip()
        if "```" in clean:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", clean, re.DOTALL)
            if match:
                clean = match.group(1)
        array_match = re.search(r"\[.*\]", clean, re.DOTALL)
        if array_match:
            clean = array_match.group(0)

        try:
            data = json.loads(clean)
            if isinstance(data, list):
                for item in data:
                    try:
                        validated = LLMItemOutput.model_validate(item)
                        parsed_reasons[validated.item_id] = validated.reason
                    except Exception:
                        continue
        except Exception:
            pass

    allowed_titles = {c.get("title", "") for c in candidates if c.get("title")} | {
        m.get("title", "") for m in history_movies if m.get("title")
    }

    explained: list[ExplainedItem] = []
    for c in candidates:
        iid = c["item_id"]
        raw_reason = parsed_reasons.get(iid)
        safe_reason = get_safe_explanation(
            generated_reason=raw_reason,
            candidate=c,
            history_movies=history_movies,
            allowed_titles=allowed_titles,
            catalogue_titles=catalogue_titles,
            lang=lang,
        )
        explained.append(
            ExplainedItem(
                item_id=iid,
                title=c.get("title", f"Item {iid}"),
                genres=c.get("genres", ""),
                reason=safe_reason,
            )
        )
    return explained


def format_explanations(items: list[ExplainedItem]) -> str:
    """Format structured explained items into a readable summary string."""
    lines = [f"• {it.title} ({it.genres}): {it.reason}" for it in items]
    return "\n".join(lines)


def generate_recommendations(
    user_history_ids: list[int],
    recommender: Any = "sasrec",
    movie_lookup: dict[int, dict] | None = None,
    candidates: list[dict] | None = None,
    backend: LLMBackend | None = None,
    top_k: int = 5,
    lang: str = "en",
    catalogue_titles: set[str] | None = None,
) -> list[ExplainedItem]:
    """Generate grounded, structured recommendations with explanations."""
    if movie_lookup is None:
        movie_lookup = {}

    if candidates is None:
        candidates = retrieve_candidates(user_history_ids, recommender, movie_lookup, top_k=top_k)

    if not candidates:
        return []

    history_movies = [
        {
            "item_id": i,
            "title": movie_lookup.get(i, {}).get("title", f"Item {i}"),
            "genres": movie_lookup.get(i, {}).get("genres", ""),
        }
        for i in user_history_ids[-10:]
    ]

    llm = backend if backend is not None else get_llm_backend()
    if llm is None:
        # LLM is disabled: pure template fallback
        return [
            ExplainedItem(
                item_id=c["item_id"],
                title=c.get("title", f"Item {c['item_id']}"),
                genres=c.get("genres", ""),
                reason=generate_template_explanation(c, history_movies, lang=lang),
            )
            for c in candidates
        ]

    prompt = _build_structured_prompt(history_movies, candidates, lang=lang)
    raw_output = llm.generate(prompt)

    if catalogue_titles is None:
        catalogue_titles = {
            info.get("title", "") for info in movie_lookup.values() if info.get("title")
        }

    return parse_and_validate_explanations(
        raw_output=raw_output,
        candidates=candidates,
        history_movies=history_movies,
        catalogue_titles=catalogue_titles,
        lang=lang,
    )


def run_phase4(
    movies: pd.DataFrame | None = None,
    sequences: dict[int, list[int]] | None = None,
) -> None:
    if movies is None:
        movies = load_processed_movies()
    if sequences is None:
        sequences = load_sequences()

    movie_lookup = build_movie_lookup(movies)

    for uid in list(sequences)[:3]:
        seq = sequences[uid]
        recent = [movie_lookup.get(i, {}).get("title", f"Item {i}") for i in seq[-5:]]
        logger.info("═" * 65)
        logger.info("  🎬  User %d — Recent: %s", uid, ", ".join(recent))
        logger.info("═" * 65)

        items = generate_recommendations(
            seq,
            recommender="sasrec",
            movie_lookup=movie_lookup,
            top_k=5,
        )
        logger.info(
            "🤖 Grounded Explanations (SASRec Candidates):\n%s\n",
            format_explanations(items),
        )
