from __future__ import annotations

import logging
import math
import pickle

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import (
    MOVIES_CSV,
    OUTPUTS_DIR,
    USER_SEQUENCES_PKL,
    ensure_dirs,
    eval_cfg,
)

logger = logging.getLogger(__name__)


def hit_rate_at_k(recommended: list[int], target: int, k: int = 10) -> float:
    return 1.0 if target in recommended[:k] else 0.0


def ndcg_at_k(recommended: list[int], target: int, k: int = 10) -> float:
    for i, item in enumerate(recommended[:k]):
        if item == target:
            return 1.0 / math.log2(i + 2)
    return 0.0


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if k == 0:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def compare_pipelines(results: dict[str, dict[str, float]]) -> None:
    all_metrics = sorted({m for r in results.values() for m in r})
    header = f"{'Pipeline':<30}" + "".join(f"{m:<15}" for m in all_metrics)
    sep = "─" * len(header)

    lines = [
        "\n" + "═" * 65,
        "          📊  Pipeline Comparison",
        "═" * 65,
        header,
        sep,
    ]
    for name, metrics in results.items():
        row = f"{name:<30}" + "".join(f"{metrics.get(m, 0.0):<15.4f}" for m in all_metrics)
        lines.append(row)
    lines.append("═" * 65)
    logger.info("\n".join(lines))

    ensure_dirs()
    colours = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    pipelines = list(results.keys())
    n_metrics = len(all_metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, all_metrics):
        vals = [results[p].get(metric, 0) for p in pipelines]
        bars = ax.bar(pipelines, vals, color=colours[: len(pipelines)], edgecolor="black")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(vals) * 1.3 + 0.01)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    chart_path = OUTPUTS_DIR / "comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info("Comparison chart saved → %s", chart_path)


def run_phase5() -> None:
    _movies = pd.read_csv(MOVIES_CSV)  # noqa: F841
    with open(USER_SEQUENCES_PKL, "rb") as f:
        sequences = pickle.load(f)

    all_results: dict[str, dict[str, float]] = {}

    logger.info("Evaluating Embedding Recommender …")
    from pipeline.models.embedding import evaluate, load_artefacts

    index, item_ids, embeddings = load_artefacts()
    emb = {}
    for k in eval_cfg.k_values:
        emb.update(evaluate(sequences, item_ids, embeddings, index, k=k))
    all_results["Embedding + FAISS"] = emb

    logger.info("Evaluating SASRec Recommender …")
    from pipeline.models.sequential import evaluate_sasrec

    seq = {}
    for k in eval_cfg.k_values:
        seq.update(evaluate_sasrec(sequences, k=k))
    all_results["SASRec (Transformer)"] = seq

    # RAG retrieval uses the same FAISS index, so retrieval metrics match embedding
    all_results["RAG (Retrieval)"] = emb.copy()

    compare_pipelines(all_results)
