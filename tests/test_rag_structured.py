from __future__ import annotations

import json

import pytest

from pipeline.models.explanations import generate_template_explanation
from pipeline.models.rag import (
    ExplainedItem,
    _build_structured_prompt,
    parse_and_validate_explanations,
)


@pytest.fixture
def sample_candidates():
    return [
        {"item_id": 101, "title": "The Matrix", "genres": "Action|Sci-Fi"},
        {"item_id": 102, "title": "Toy Story", "genres": "Animation|Children"},
    ]


@pytest.fixture
def sample_history():
    return [
        {"item_id": 1, "title": "Terminator 2", "genres": "Action|Sci-Fi"},
        {"item_id": 2, "title": "Aliens", "genres": "Action|Sci-Fi"},
    ]


def test_build_structured_prompt(sample_candidates, sample_history):
    prompt = _build_structured_prompt(sample_history, sample_candidates)
    assert "The Matrix" in prompt
    assert "Toy Story" in prompt
    assert "Terminator 2" in prompt
    assert "JSON" in prompt


def test_parse_and_validate_explanations_valid_json(sample_candidates, sample_history):
    raw_llm_json = json.dumps(
        [
            {"item_id": 101, "reason": "Because you love cyberpunk and sci-fi action."},
            {"item_id": 102, "reason": "A heartwarming animation for all ages."},
        ]
    )
    catalogue = {"The Matrix", "Toy Story", "Terminator 2", "Aliens"}
    items = parse_and_validate_explanations(
        raw_output=raw_llm_json,
        candidates=sample_candidates,
        history_movies=sample_history,
        catalogue_titles=catalogue,
    )

    assert len(items) == 2
    assert isinstance(items[0], ExplainedItem)
    assert items[0].item_id == 101
    assert items[0].reason == "Because you love cyberpunk and sci-fi action."
    assert items[1].item_id == 102
    assert items[1].reason == "A heartwarming animation for all ages."


def test_parse_and_validate_per_item_fallback_on_partial_or_malformed(
    sample_candidates, sample_history
):
    # Only item 101 is provided, and text has extra markdown wrapping
    raw_output = '```json\n[{"item_id": 101, "reason": "Great sci-fi choice."}]\n```'
    items = parse_and_validate_explanations(
        raw_output=raw_output,
        candidates=sample_candidates,
        history_movies=sample_history,
    )

    assert len(items) == 2
    # Item 101 gets the LLM reason
    assert items[0].item_id == 101
    assert items[0].reason == "Great sci-fi choice."
    # Item 102 was missing from LLM output, falls back to template
    assert items[1].item_id == 102
    expected_template = generate_template_explanation(sample_candidates[1], sample_history)
    assert items[1].reason == expected_template


def test_parse_and_validate_per_item_fallback_on_hallucination(sample_candidates, sample_history):
    catalogue = {"The Matrix", "Toy Story", "Terminator 2", "Aliens", "Avatar"}
    # Item 101 mentions Avatar (in catalogue, but not in candidates or history)
    raw_output = json.dumps(
        [
            {"item_id": 101, "reason": "Because Avatar was a great visual film."},
            {"item_id": 102, "reason": "Fun family animation."},
        ]
    )
    items = parse_and_validate_explanations(
        raw_output=raw_output,
        candidates=sample_candidates,
        history_movies=sample_history,
        catalogue_titles=catalogue,
    )

    # Item 101 hallucinated Avatar, so it must fall back to template
    assert "Avatar" not in items[0].reason
    assert "Because you liked" in items[0].reason
    # Item 102 is valid
    assert items[1].reason == "Fun family animation."


def test_template_fallback_works_with_llm_disabled(sample_candidates):
    movie_lookup = {
        101: {"title": "The Matrix", "genres": "Action|Sci-Fi"},
        102: {"title": "Toy Story", "genres": "Animation|Children"},
        1: {"title": "Terminator 2", "genres": "Action|Sci-Fi"},
        2: {"title": "Aliens", "genres": "Action|Sci-Fi"},
    }
    from unittest.mock import patch

    with patch("pipeline.models.rag.get_llm_backend", return_value=None):
        from pipeline.models.rag import generate_recommendations

        items = generate_recommendations(
            user_history_ids=[1, 2],
            recommender="embedding",
            movie_lookup=movie_lookup,
            candidates=sample_candidates,
            backend=None,
        )
    assert len(items) == 2
    assert "Because you liked" in items[0].reason
    assert items[0].title == "The Matrix"
