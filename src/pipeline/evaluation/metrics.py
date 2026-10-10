from __future__ import annotations

import logging
import math
from collections.abc import Callable

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import OUTPUTS_DIR, ensure_dirs, eval_cfg

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


def evaluate_leave_one_out(
    sequences: dict[int, list[int]],
    recommend: Callable[[list[int], int], list[int]],
    k_values: tuple[int, ...] = eval_cfg.k_values,
    desc: str = "Evaluating",
) -> dict[str, float]:
    """Hold out each user's last item and score ``recommend(history, top_k)`` against it.

    Recommendations are requested once at ``max(k_values)``; smaller k are prefixes.
    """
    max_k = max(k_values)
    hits: dict[int, list[float]] = {k: [] for k in k_values}
    ndcgs: dict[int, list[float]] = {k: [] for k in k_values}

    for seq in tqdm(sequences.values(), desc=desc):
        if len(seq) < 2:
            continue
        recs = recommend(seq[:-1], max_k)
        for k in k_values:
            hits[k].append(hit_rate_at_k(recs, seq[-1], k))
            ndcgs[k].append(ndcg_at_k(recs, seq[-1], k))

    result: dict[str, float] = {}
    for k in k_values:
        result[f"HitRate@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        result[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return result


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
    from pipeline.data.loader import load_sequences
    from pipeline.models.embedding import evaluate, load_artefacts
    from pipeline.models.sequential import evaluate_sasrec

    sequences = load_sequences()

    # RAG reuses the embedding retriever, so it has no separate ranking metrics
    all_results: dict[str, dict[str, float]] = {}

    logger.info("Evaluating Embedding Recommender …")
    all_results["Embedding + FAISS"] = evaluate(sequences, load_artefacts())

    logger.info("Evaluating SASRec Recommender …")
    all_results["SASRec (Transformer)"] = evaluate_sasrec(sequences)

    compare_pipelines(all_results)
