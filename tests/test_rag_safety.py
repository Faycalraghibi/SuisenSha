from __future__ import annotations

from pipeline.models.explanations import (
    generate_template_explanation,
    get_safe_explanation,
    is_repetitive,
    validate_grounding,
)


def test_template_explanation_two_matching_movies():
    candidate = {"item_id": 100, "title": "Star Wars", "genres": "Action|Sci-Fi"}
    history = [
        {"item_id": 1, "title": "Toy Story", "genres": "Animation|Children"},
        {"item_id": 2, "title": "The Matrix", "genres": "Action|Sci-Fi"},
        {"item_id": 3, "title": "Terminator 2", "genres": "Action|Sci-Fi"},
    ]
    explanation = generate_template_explanation(candidate, history, lang="en")
    assert "Because you liked" in explanation
    assert "Terminator 2" in explanation
    assert "The Matrix" in explanation
    assert "Sci-Fi" in explanation or "Action" in explanation


def test_template_explanation_french():
    candidate = {"item_id": 100, "title": "Alien", "genres": "Sci-Fi|Horror"}
    history = [
        {"item_id": 1, "title": "The Thing", "genres": "Horror"},
        {"item_id": 2, "title": "Blade Runner", "genres": "Sci-Fi"},
    ]
    explanation = generate_template_explanation(candidate, history, lang="fr")
    assert "Parce que vous avez aimé" in explanation


def test_is_repetitive_detects_loops():
    clean_text = "A thrilling science-fiction adventure with groundbreaking visual effects."
    assert not is_repetitive(clean_text)

    loop_text = "This movie is good. This movie is good. This movie is good. This movie is good."
    assert is_repetitive(loop_text)

    ngram_loop = "the best of the best of the best of the best of the best"
    assert is_repetitive(ngram_loop)


def test_validate_grounding_rejects_hallucinated_titles():
    allowed_titles = {"The Matrix", "Terminator 2"}
    catalogue = {"The Matrix", "Terminator 2", "Titanic", "Jurassic Park"}

    grounded_text = "Because you enjoyed The Matrix, this cyberpunk film offers similar action."
    assert validate_grounding(
        grounded_text, allowed_titles=allowed_titles, catalogue_titles=catalogue
    )

    hallucinated_text = "Fans of Titanic will love this dramatic journey into space."
    assert not validate_grounding(
        hallucinated_text, allowed_titles=allowed_titles, catalogue_titles=catalogue
    )


def test_get_safe_explanation_falls_back_on_hallucination():
    candidate = {"item_id": 50, "title": "Star Wars", "genres": "Sci-Fi"}
    history = [{"item_id": 10, "title": "The Matrix", "genres": "Sci-Fi"}]
    catalogue = {"Star Wars", "The Matrix", "Titanic"}

    hallucinated_generated = "Watch this because Titanic was an amazing romantic movie."
    safe = get_safe_explanation(
        generated_reason=hallucinated_generated,
        candidate=candidate,
        history_movies=history,
        allowed_titles={"Star Wars", "The Matrix"},
        catalogue_titles=catalogue,
    )
    # Must reject Titanic and return the template
    assert "Titanic" not in safe
    assert "The Matrix" in safe
    assert "Because you liked" in safe


def test_get_safe_explanation_falls_back_on_repetition():
    candidate = {"item_id": 50, "title": "Star Wars", "genres": "Sci-Fi"}
    history = [{"item_id": 10, "title": "The Matrix", "genres": "Sci-Fi"}]

    repetitive_generated = "Great movie great movie great movie great movie great movie."
    safe = get_safe_explanation(
        generated_reason=repetitive_generated,
        candidate=candidate,
        history_movies=history,
        allowed_titles={"Star Wars", "The Matrix"},
    )
    assert "Because you liked" in safe
