from __future__ import annotations

import re


def _extract_genres(genres_str: str) -> list[str]:
    if not genres_str:
        return []
    # MovieLens genres are often pipe-separated, comma-separated, or single words
    return [
        g.strip()
        for g in re.split(r"[|,]", genres_str)
        if g.strip() and g.strip().lower() != "unknown"
    ]


def generate_template_explanation(
    candidate: dict,
    history_movies: list[dict],
    lang: str = "en",
) -> str:
    """Generate a deterministic, grounded explanation for a candidate movie
    based on user history.
    """
    candidate_title = candidate.get("title", f"Item {candidate.get('item_id')}")
    candidate_genres = set(_extract_genres(candidate.get("genres", "")))

    # Find history movies sharing at least one genre, ordered recent first
    matching_by_genre: dict[str, list[str]] = {}
    for movie in reversed(history_movies):
        title = movie.get("title")
        if not title:
            continue
        m_genres = set(_extract_genres(movie.get("genres", "")))
        overlap = candidate_genres & m_genres
        for g in overlap:
            matching_by_genre.setdefault(g, [])
            if title not in matching_by_genre[g]:
                matching_by_genre[g].append(title)

    # Pick the genre with the most matching recent movies
    best_genre = None
    best_matches: list[str] = []
    for g, matches in matching_by_genre.items():
        if len(matches) > len(best_matches):
            best_genre = g
            best_matches = matches

    if lang == "fr":
        if len(best_matches) >= 2 and best_genre:
            return (
                f"Parce que vous avez aimé '{best_matches[0]}' et '{best_matches[1]}', "
                f"tous deux du genre {best_genre}."
            )
        if len(best_matches) == 1 and best_genre:
            return (
                f"Parce que vous avez aimé '{best_matches[0]}', qui partage le genre {best_genre}."
            )
        if candidate_genres:
            top_g = sorted(candidate_genres)[0]
            return f"Recommandé en fonction de votre intérêt pour les films du genre {top_g}."
        return "Recommandé selon vos préférences de visionnage."

    # Default to English
    if len(best_matches) >= 2 and best_genre:
        return f"Because you liked '{best_matches[0]}' and '{best_matches[1]}', both {best_genre}."
    if len(best_matches) == 1 and best_genre:
        return f"Because you liked '{best_matches[0]}', which also features {best_genre}."
    if candidate_genres:
        top_g = sorted(candidate_genres)[0]
        return f"Recommended based on your interest in {top_g} movies."
    return f"Recommended based on your viewing history and interest in '{candidate_title}'."


def is_repetitive(text: str, max_ngram_rep: int = 3, min_unique_ratio: float = 0.5) -> bool:
    """Check if generated text contains loops, phrase repetitions, or degenerate patterns."""
    text_clean = text.strip()
    if not text_clean:
        return True

    words = re.findall(r"\b\w+\b", text_clean.lower())
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < min_unique_ratio:
            return True

    # Check 3-grams
    if len(words) >= 6:
        trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
        counts: dict[tuple[str, ...], int] = {}
        for tg in trigrams:
            counts[tg] = counts.get(tg, 0) + 1
            if counts[tg] >= max_ngram_rep:
                return True

    # Check identical sentences
    sentences = [
        s.strip().lower() for s in re.split(r"[.!?]+", text_clean) if len(s.strip().split()) >= 3
    ]
    if len(sentences) >= 2:
        for s in sentences:
            if sentences.count(s) >= 2:
                return True

    return False


def validate_grounding(
    text: str,
    allowed_titles: set[str],
    catalogue_titles: set[str] | None = None,
) -> bool:
    """Validate that text does not hallucinate catalogue titles outside allowed_titles."""
    if not text or len(text.strip()) < 5:
        return False

    if catalogue_titles:
        # Check if any catalogue title that is NOT allowed appears in the text
        text_lower = text.lower()
        for title in catalogue_titles:
            if title in allowed_titles:
                continue
            # Case-insensitive title match with boundary check
            pattern = rf"\b{re.escape(title.lower())}\b"
            if re.search(pattern, text_lower):
                return False

    return True


def get_safe_explanation(
    generated_reason: str | None,
    candidate: dict,
    history_movies: list[dict],
    allowed_titles: set[str],
    catalogue_titles: set[str] | None = None,
    lang: str = "en",
) -> str:
    """Return the generated reason if grounded and non-repetitive,
    otherwise fallback to template.
    """
    if generated_reason and not is_repetitive(generated_reason):
        if validate_grounding(
            generated_reason,
            allowed_titles=allowed_titles,
            catalogue_titles=catalogue_titles,
        ):
            return generated_reason.strip()

    return generate_template_explanation(candidate, history_movies, lang=lang)
