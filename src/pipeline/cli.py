from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--phase",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Phase to run (1-7). 0 = run all phases sequentially.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(_BANNER)

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



    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
