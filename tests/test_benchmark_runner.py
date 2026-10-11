"""Unit tests for benchmark runner and CLI integration."""

from pathlib import Path

import pandas as pd

from pipeline.benchmark import compute_ci95, run_benchmark
from pipeline.cli import _parse_args, main
from pipeline.models.llm import FakeLLMBackend


def test_compute_ci95():
    assert compute_ci95([]) == (0.0, 0.0)
    assert compute_ci95([0.5]) == (0.5, 0.0)
    assert compute_ci95([0.5, 0.5, 0.5]) == (0.5, 0.0)

    mean, ci = compute_ci95([0.1, 0.2, 0.3])
    assert abs(mean - 0.2) < 1e-6
    assert ci > 0.0


def test_benchmark_synthetic(tmp_path: Path):
    # Construct small synthetic dataset
    sequences = {
        1: [1, 2, 3, 4],
        2: [2, 3, 4, 5],
        3: [1, 3, 5, 6],
        4: [2, 4, 6, 7],
        5: [1, 2, 5, 8],
    }
    movies_df = pd.DataFrame(
        [
            {"item_id": 1, "title": "Movie 1", "genres": "Action", "release_date": "1995"},
            {"item_id": 2, "title": "Movie 2", "genres": "Comedy", "release_date": "1996"},
            {"item_id": 3, "title": "Movie 3", "genres": "Action, Comedy", "release_date": "1997"},
            {"item_id": 4, "title": "Movie 4", "genres": "Drama", "release_date": "1998"},
            {"item_id": 5, "title": "Movie 5", "genres": "Sci-Fi", "release_date": "1999"},
            {"item_id": 6, "title": "Movie 6", "genres": "Action", "release_date": "2000"},
            {"item_id": 7, "title": "Movie 7", "genres": "Drama", "release_date": "2001"},
            {"item_id": 8, "title": "Movie 8", "genres": "Comedy", "release_date": "2002"},
        ]
    )

    results = run_benchmark(
        seeds=[42, 43],
        k_values=(2,),
        output_dir=tmp_path,
        include_models=["Random", "Popularity"],
        judge_backend=FakeLLMBackend(canned_response="4"),
        sequences=sequences,
        movies_df=movies_df,
    )

    csv_file = tmp_path / "benchmark.csv"
    md_file = tmp_path / "benchmark.md"

    assert csv_file.is_file()
    assert md_file.is_file()

    md_content = md_file.read_text(encoding="utf-8")
    assert "SuisenSha Recommender Benchmark Report" in md_content
    assert "Ranking Accuracy" in md_content
    assert "Beyond-Accuracy & Exposure Metrics" in md_content
    assert "Performance by User Activity Quartile" in md_content
    assert "Explanation Quality Evaluation" in md_content

    assert "Random" in results["aggregated_results"]
    assert "Popularity" in results["aggregated_results"]


def test_cli_parse_benchmark_args():
    args = _parse_args(["benchmark", "--seeds", "3", "--output-dir", "custom_reports"])
    assert args.command == "benchmark"
    assert args.seeds == 3
    assert args.output_dir == "custom_reports"


def test_cli_parse_phase_args():
    args = _parse_args(["--phase", "2"])
    assert getattr(args, "command", None) is None
    assert args.phase == 2


def test_cli_main_benchmark_execution(tmp_path: Path, monkeypatch):
    called = {}

    def fake_run_benchmark(seeds=5, output_dir=None):
        called["seeds"] = seeds
        called["output_dir"] = output_dir
        return {
            "md_path": str(tmp_path / "benchmark.md"),
            "csv_path": str(tmp_path / "benchmark.csv"),
        }

    monkeypatch.setattr("pipeline.benchmark.run_benchmark", fake_run_benchmark)
    main(["benchmark", "--seeds", "2", "--output-dir", str(tmp_path)])

    assert called["seeds"] == 2
    assert called["output_dir"] == str(tmp_path)
