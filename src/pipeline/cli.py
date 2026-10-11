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
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default="ml-100k",
        choices=["ml-100k", "ml-1m", "movielens-100k", "movielens-1m"],
        help="Dataset to benchmark on (default: ml-100k).",
    )

    # mlops retrain subcommand
    retrain_parser = subparsers.add_parser(
        "retrain", help="Run automated retraining and gated promotion."
    )
    retrain_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train candidate model.",
    )
    retrain_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force promotion even if candidate metrics degrade.",
    )

    # mlops rollback subcommand
    rollback_parser = subparsers.add_parser(
        "rollback", help="Roll back production model to previous registered version."
    )
    rollback_parser.add_argument(
        "--model",
        type=str,
        default="sasrec",
        help="Model name to rollback (default: sasrec).",
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
        dataset = getattr(args, "dataset", "ml-100k")
        results = run_benchmark(seeds=seeds, output_dir=out_dir, dataset=dataset)
        print(f"\n✅ Benchmark complete across {seeds} seeds on {dataset}.")
        print(f"  Markdown report → {results['md_path']}")
        return

    if getattr(args, "command", None) == "retrain":
        print("\n▓▓▓  MLOPS — Automated Retraining & Gated Promotion  ▓▓▓")
        from pipeline.mlops.pipeline import RetrainingPipeline

        retraining_pipeline = RetrainingPipeline(config=cfg)
        result = retraining_pipeline.run(
            epochs=getattr(args, "epochs", None),
            force_promote=getattr(args, "force", False),
        )
        print(f"\nCandidate Version: v{result.candidate_version}")
        print(f"Candidate Metrics: {result.candidate_metrics}")
        if result.promoted:
            print("Status: ✅ Promoted to Production!")
        else:
            print(f"Status: ❌ Rejected. Reason: {result.rejection_reason}")
        return

    if getattr(args, "command", None) == "rollback":
        print("\n▓▓▓  MLOPS — Model Rollback  ▓▓▓")
        from pipeline.mlops.registry import ModelRegistry

        registry = ModelRegistry()
        model_name = getattr(args, "model", "sasrec")
        rolled_back = registry.rollback(model_name=model_name)
        print(f"\n✅ Successfully rolled back {model_name} to v{rolled_back.version} (Production).")
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
