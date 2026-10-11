"""Unit tests for explanation quality evaluation."""

from pipeline.evaluation.explanations import (
    detect_explanation_hallucinations,
    evaluate_explanation_suite,
    extract_genres_from_text,
    extract_years_from_text,
    score_explanation_relevance,
)
from pipeline.models.llm import FakeLLMBackend


def test_extract_genres_and_years():
    text = "A thrilling romantic comedy released in 1994, with great Sci-Fi action."
    genres = extract_genres_from_text(text)
    years = extract_years_from_text(text)

    assert "thriller" in genres
    assert "romance" in genres
    assert "comedy" in genres
    assert "sci-fi" in genres
    assert "action" in genres
    assert 1994 in years


def test_detect_hallucination_clean():
    candidate = {
        "item_id": 1,
        "title": "Toy Story (1995)",
        "genres": "Animation, Children, Comedy",
        "release_date": "01-Jan-1995",
    }
    history = [
        {
            "item_id": 2,
            "title": "Aladdin (1992)",
            "genres": "Animation, Children, Musical",
            "release_date": "01-Jan-1992",
        }
    ]
    explanation = "Because you loved Aladdin (1992), this 1995 animated comedy is a great match."
    result = detect_explanation_hallucinations(explanation, candidate, history)

    assert not result["has_hallucination"]
    assert result["hallucinated_genres"] == []
    assert result["hallucinated_years"] == []


def test_detect_hallucination_genre():
    candidate = {
        "item_id": 10,
        "title": "The Terminator (1984)",
        "genres": "Action, Sci-Fi",
        "release_date": "01-Jan-1984",
    }
    explanation = "A romantic comedy you will surely enjoy."
    result = detect_explanation_hallucinations(explanation, candidate)

    assert result["has_hallucination"]
    assert "comedy" in result["hallucinated_genres"]
    assert "romance" in result["hallucinated_genres"]


def test_detect_hallucination_year():
    candidate = {
        "item_id": 1,
        "title": "Toy Story (1995)",
        "genres": "Animation, Children, Comedy",
        "release_date": "01-Jan-1995",
    }
    explanation = "Enjoy this classic animation released in 2005."
    result = detect_explanation_hallucinations(explanation, candidate)

    assert result["has_hallucination"]
    assert 2005 in result["hallucinated_years"]


def test_score_explanation_relevance():
    backend_good = FakeLLMBackend(canned_response="Score: 4")
    candidate = {"title": "The Matrix", "genres": "Action, Sci-Fi"}
    explanation = "A high-octane sci-fi thriller."

    score = score_explanation_relevance(backend_good, explanation, candidate)
    assert score == 4.0

    backend_bad = FakeLLMBackend(canned_response="I cannot evaluate this.")
    fallback_score = score_explanation_relevance(backend_bad, explanation, candidate)
    assert fallback_score == 3.0


def test_evaluate_explanation_suite():
    candidate = {
        "item_id": 1,
        "title": "Toy Story (1995)",
        "genres": "Animation, Children, Comedy",
    }
    items = [
        {
            "explanation": "Because you love animated comedy films like Toy Story.",
            "candidate": candidate,
            "latency_ms": 12.5,
        },
        {
            "explanation": "A romantic documentary horror from 2020.",
            "candidate": candidate,
            "latency_ms": 15.0,
        },
    ]

    report = evaluate_explanation_suite(
        items,
        catalogue_titles={"Toy Story (1995)", "GoldenEye (1995)"},
        judge_backend=FakeLLMBackend(canned_response="5"),
    )

    assert report.total == 2
    assert report.hallucination_rate == 0.5  # second item hallucinates genres and year
    assert report.grounding_rate == 1.0
    assert report.llm_judge_relevance == 5.0
    assert report.avg_latency_ms == 13.75
    assert report.avg_length_words > 0
    assert report.avg_length_chars > 0

    report_dict = report.to_dict()
    assert "hallucination_rate" in report_dict
    assert "avg_latency_ms" in report_dict
