"""Explanation quality evaluation suite.

Evaluates:
- Grounding rate (absence of ungrounded catalog items)
- Hallucination rate (claimed genres and release years verified against catalogue)
- LLM-as-judge relevance (scored on 1-5 scale)
- Length (word and character counts)
- Latency (generation time in ms)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from pipeline.models.explanations import is_repetitive, validate_grounding
from pipeline.models.llm import LLMBackend

logger = logging.getLogger(__name__)

CANONICAL_GENRES = {
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film-noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
}

GENRE_ALIASES = {
    "romantic": "romance",
    "comedic": "comedy",
    "dramatic": "drama",
    "science fiction": "sci-fi",
    "sci fi": "sci-fi",
    "scifi": "sci-fi",
    "animated": "animation",
    "thrilling": "thriller",
    "horrific": "horror",
    "mysterious": "mystery",
    "action-packed": "action",
    "musical": "musical",
    "western": "western",
    "war": "war",
}


def extract_genres_from_text(text: str) -> set[str]:
    """Find mentioned genre keywords in explanation text."""
    lower = text.lower()
    found = set()

    for alias, canonical in GENRE_ALIASES.items():
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, lower):
            found.add(canonical)

    for genre in CANONICAL_GENRES:
        pattern = rf"\b{re.escape(genre)}\b"
        if re.search(pattern, lower):
            found.add(genre)

    return found


def extract_years_from_text(text: str) -> set[int]:
    """Find 4-digit release years (1880-2040) mentioned in text."""
    years = set()
    for match in re.finditer(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text):
        years.add(int(match.group(1)))
    return years


def extract_year_from_movie(movie: dict[str, Any]) -> int | None:
    """Extract release year from title or release_date field."""
    title = str(movie.get("title", ""))
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return int(match.group(1))

    release_date = str(movie.get("release_date", ""))
    match_date = re.search(r"\b(\d{4})\b", release_date)
    if match_date:
        return int(match_date.group(1))

    return None


def detect_explanation_hallucinations(
    explanation: str,
    candidate: dict[str, Any],
    history_movies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify claimed genres and release years against catalogue metadata.

    Returns dict with ``has_hallucination``, ``hallucinated_genres``, and ``hallucinated_years``.
    """
    history = history_movies or []

    # Collect allowed genres from candidate and referenced history
    allowed_genres: set[str] = set()
    raw_cand_genres = str(candidate.get("genres", ""))
    for g in re.split(r"[,|]", raw_cand_genres):
        g_clean = g.strip().lower()
        if g_clean in CANONICAL_GENRES:
            allowed_genres.add(g_clean)
        elif g_clean in GENRE_ALIASES:
            allowed_genres.add(GENRE_ALIASES[g_clean])

    for hm in history:
        raw_hm_genres = str(hm.get("genres", ""))
        for g in re.split(r"[,|]", raw_hm_genres):
            g_clean = g.strip().lower()
            if g_clean in CANONICAL_GENRES:
                allowed_genres.add(g_clean)
            elif g_clean in GENRE_ALIASES:
                allowed_genres.add(GENRE_ALIASES[g_clean])

    # Check extracted genres
    extracted_genres = extract_genres_from_text(explanation)
    hallucinated_genres = extracted_genres - allowed_genres

    # Collect allowed years
    allowed_years: set[int] = set()
    cand_year = extract_year_from_movie(candidate)
    if cand_year is not None:
        allowed_years.add(cand_year)
    for hm in history:
        hm_year = extract_year_from_movie(hm)
        if hm_year is not None:
            allowed_years.add(hm_year)

    # Check extracted years
    extracted_years = extract_years_from_text(explanation)
    hallucinated_years = {y for y in extracted_years if y not in allowed_years}

    has_hallucination = bool(hallucinated_genres or hallucinated_years)
    return {
        "has_hallucination": has_hallucination,
        "hallucinated_genres": sorted(hallucinated_genres),
        "hallucinated_years": sorted(hallucinated_years),
    }


def score_explanation_relevance(
    backend: LLMBackend,
    explanation: str,
    candidate: dict[str, Any],
    history_movies: list[dict[str, Any]] | None = None,
) -> float:
    """Prompt an LLM to evaluate the relevance of an explanation on a 1-5 scale."""
    history_str = ""
    if history_movies:
        titles = [m.get("title", "") for m in history_movies[-5:] if m.get("title")]
        if titles:
            history_str = f"User Recent History: {', '.join(titles)}\n"

    prompt = (
        "You are an expert evaluator for recommender systems.\n"
        f"{history_str}"
        f"Recommended Movie: {candidate.get('title', '')} "
        f"(Genres: {candidate.get('genres', '')})\n"
        f"Generated Explanation: {explanation}\n"
        "Evaluate whether this explanation is relevant and compelling.\n"
        "Rate the explanation on a scale from 1 (irrelevant) to 5 (highly relevant).\n"
        "Respond with only the single integer score (1, 2, 3, 4, or 5):"
    )

    resp = backend.generate(prompt).strip()
    match = re.search(r"\b([1-5])\b", resp)
    if match:
        return float(match.group(1))
    return 3.0


@dataclass(frozen=True)
class ExplanationEvalReport:
    total: int
    grounding_rate: float
    hallucination_rate: float
    llm_judge_relevance: float
    avg_length_words: float
    avg_length_chars: float
    avg_latency_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total_explanations": float(self.total),
            "grounding_rate": self.grounding_rate,
            "hallucination_rate": self.hallucination_rate,
            "llm_judge_relevance": self.llm_judge_relevance,
            "avg_length_words": self.avg_length_words,
            "avg_length_chars": self.avg_length_chars,
            "avg_latency_ms": self.avg_latency_ms,
        }


def evaluate_explanation_suite(
    explanations: list[dict[str, Any]],
    catalogue_titles: set[str] | None = None,
    judge_backend: LLMBackend | None = None,
) -> ExplanationEvalReport:
    """Evaluate a list of explanations.

    Each item in ``explanations`` is a dict with keys:
    - ``explanation``: str
    - ``candidate``: dict (with title, genres, release_date)
    - ``history_movies``: list[dict] (optional)
    - ``allowed_titles``: set[str] (optional)
    - ``latency_ms``: float (optional)
    """
    if not explanations:
        return ExplanationEvalReport(
            total=0,
            grounding_rate=0.0,
            hallucination_rate=0.0,
            llm_judge_relevance=0.0,
            avg_length_words=0.0,
            avg_length_chars=0.0,
            avg_latency_ms=0.0,
        )

    grounded_flags = []
    hallucinated_flags = []
    judge_scores = []
    word_lengths = []
    char_lengths = []
    latencies = []

    for item in explanations:
        exp = str(item.get("explanation", ""))
        cand = item.get("candidate", {})
        hist = item.get("history_movies", [])
        allowed_titles = item.get("allowed_titles")
        if allowed_titles is None:
            cand_title = cand.get("title", "")
            allowed_titles = {cand_title} | {m.get("title", "") for m in hist}

        # 1. Grounding
        is_grounded = not is_repetitive(exp) and validate_grounding(
            exp,
            allowed_titles=allowed_titles,
            catalogue_titles=catalogue_titles,
        )
        grounded_flags.append(1.0 if is_grounded else 0.0)

        # 2. Hallucination
        h_info = detect_explanation_hallucinations(exp, cand, hist)
        hallucinated_flags.append(1.0 if h_info["has_hallucination"] else 0.0)

        # 3. LLM judge
        if judge_backend is not None:
            score = score_explanation_relevance(judge_backend, exp, cand, hist)
            judge_scores.append(score)

        # 4. Length
        words = len(exp.split())
        word_lengths.append(words)
        char_lengths.append(len(exp))

        # 5. Latency
        lat = item.get("latency_ms")
        if lat is not None:
            latencies.append(float(lat))

    total = len(explanations)
    return ExplanationEvalReport(
        total=total,
        grounding_rate=float(sum(grounded_flags) / total),
        hallucination_rate=float(sum(hallucinated_flags) / total),
        llm_judge_relevance=float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0,
        avg_length_words=float(sum(word_lengths) / total),
        avg_length_chars=float(sum(char_lengths) / total),
        avg_latency_ms=float(sum(latencies) / len(latencies)) if latencies else 0.0,
    )
