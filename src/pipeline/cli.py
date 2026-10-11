from __future__ import annotations

import argparse
import io
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(message)s",
    datefmt="%H:%M:%S",
)

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          SuisenSha — Recommend + LLM + RAG Pipeline         ║
╚══════════════════════════════════════════════════════════════╝"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the recommendation pipeline (MovieLens-100K).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-seed model benchmark.")
    bench_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Number of random seeds (default: 5).",
    )
    bench_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write benchmark reports (default: reports).",
    )
    bench_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom TOML configuration file.",
    )
    bench_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config setting (e.g. --set sasrec.epochs=40). Can be repeated.",
    )

    # Legacy/direct phase flags
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        help="Phase to run (1-8). 0 = run all phases sequentially.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(Phase 8 only) Max number of users to batch-generate RAG for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom TOML configuration file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a config setting (e.g. --set sasrec.epochs=40). Can be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from pipeline.config import load_config, set_active_config

    cfg = load_config(config_path=args.config, overrides=args.overrides)
    set_active_config(cfg)

    # Windows consoles/pipes default to cp1252, which can't encode the banner and log glyphs
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print(_BANNER)

    if getattr(args, "command", None) == "benchmark":
        print("\n▓▓▓  BENCHMARK SUITE — Multi-Seed Model Evaluation  ▓▓▓")
        from pipeline.benchmark import run_benchmark

        seeds = getattr(args, "seeds", None) or cfg.benchmark.seeds
        out_dir = getattr(args, "output_dir", None) or cfg.benchmark.output_dir
        results = run_benchmark(seeds=seeds, output_dir=out_dir)
        print(f"\n✅ Benchmark complete across {seeds} seeds.")
        print(f"  Markdown report → {results['md_path']}")
        print(f"  CSV metrics     → {results['csv_path']}")
        return

    movies = sequences = None

    if args.phase in (0, 1):
        print("\n▓▓▓  PHASE 1 — Data Loading & EDA  ▓▓▓")
        from pipeline.data.loader import run_phase1

        _, movies, sequences = run_phase1()

    if args.phase in (0, 2):
        print("\n▓▓▓  PHASE 2 — Embedding Recommender (FAISS)  ▓▓▓")
        from pipeline.models.embedding import run_phase2

        emb_results = run_phase2(movies, sequences)
        print(f"  → {emb_results}")

    if args.phase in (0, 3):
        print("\n▓▓▓  PHASE 3 — SASRec Sequential Recommender  ▓▓▓")
        from pipeline.models.sequential import run_phase3

        seq_results = run_phase3(sequences)
        print(f"  → {seq_results}")

    if args.phase in (0, 4):
        print("\n▓▓▓  PHASE 4 — RAG + LLM Recommendations  ▓▓▓")
        from pipeline.models.rag import run_phase4

        run_phase4(movies, sequences)

    if args.phase in (0, 5):
        print("\n▓▓▓  PHASE 5 — Evaluation & Comparison  ▓▓▓")
        from pipeline.evaluation.metrics import run_phase5

        run_phase5()

    if args.phase == 6:
        print("\n▓▓▓  PHASE 6 — Starting FastAPI Backend  ▓▓▓")
        import uvicorn

        uvicorn.run("pipeline.api.main:app", host="127.0.0.1", port=8000)

    if args.phase == 7:
        print("\n▓▓▓  PHASE 7 — Starting Streamlit UI  ▓▓▓")
        import subprocess
        from pathlib import Path

        ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])

    if args.phase == 8:
        print("\n▓▓▓  PHASE 8 — Batch RAG Pre-computation  ▓▓▓")
        from pipeline.batch import run_batch

        run_batch(limit=args.limit)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
