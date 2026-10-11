from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

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


def precision_at_k(
    recommended: list[int],
    relevant: set[int] | list[int] | int,
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0
    rel_set = {relevant} if isinstance(relevant, int) else set(relevant)
    return sum(1 for item in recommended[:k] if item in rel_set) / k


def recall_at_k(
    recommended: list[int],
    relevant: set[int] | list[int] | int,
    k: int = 10,
) -> float:
    rel_set = {relevant} if isinstance(relevant, int) else set(relevant)
    if not rel_set:
        return 0.0
    return sum(1 for item in recommended[:k] if item in rel_set) / len(rel_set)


def mrr_at_k(
    recommended: list[int],
    target: int | set[int] | list[int],
    k: int = 10,
) -> float:
    rel_set = {target} if isinstance(target, int) else set(target)
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def catalog_coverage_at_k(
    all_recommendations: list[list[int]] | dict[Any, list[int]],
    num_items: int,
    k: int = 10,
) -> float:
    if num_items <= 0:
        return 0.0
    unique_items: set[int] = set()
    recs_iter = (
        all_recommendations.values()
        if isinstance(all_recommendations, dict)
        else all_recommendations
    )
    for recs in recs_iter:
        unique_items.update(recs[:k])
    return len(unique_items) / num_items


def intra_list_diversity_at_k(
    recommended: list[int],
    item_features: Mapping[int, Any],
    k: int = 10,
) -> float:
    recs = recommended[:k]
    m = len(recs)
    if m < 2:
        return 0.0

    total_dist = 0.0
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            f_i = item_features.get(recs[i])
            f_j = item_features.get(recs[j])
            pairs += 1

            if f_i is None or f_j is None:
                total_dist += 1.0
                continue

            if isinstance(f_i, np.ndarray) and isinstance(f_j, np.ndarray):
                denom = (np.linalg.norm(f_i) * np.linalg.norm(f_j)) + 1e-9
                cosine = float(np.dot(f_i, f_j) / denom)
                dist = max(0.0, min(2.0, 1.0 - cosine))
            else:
                s_i = set(f_i)
                s_j = set(f_j)
                union = s_i | s_j
                if not union:
                    dist = 1.0
                else:
                    dist = 1.0 - (len(s_i & s_j) / len(union))
            total_dist += dist

    return total_dist / pairs if pairs > 0 else 0.0


def mean_intra_list_diversity_at_k(
    all_recommendations: list[list[int]] | dict[Any, list[int]],
    item_features: Mapping[int, Any],
    k: int = 10,
) -> float:
    recs_iter = (
        all_recommendations.values()
        if isinstance(all_recommendations, dict)
        else all_recommendations
    )
    scores = [intra_list_diversity_at_k(recs, item_features, k=k) for recs in recs_iter]
    return float(np.mean(scores)) if scores else 0.0


def novelty_at_k(
    recommended: list[int],
    item_popularity: dict[int, int],
    total_users: int,
    k: int = 10,
) -> float:
    recs = recommended[:k]
    if not recs or total_users <= 0:
        return 0.0

    self_info = []
    for item in recs:
        pop = max(item_popularity.get(item, 0), 1)
        p = min(pop / total_users, 1.0)
        self_info.append(-math.log2(p))
    return float(np.mean(self_info))


def mean_novelty_at_k(
    all_recommendations: list[list[int]] | dict[Any, list[int]],
    item_popularity: dict[int, int],
    total_users: int,
    k: int = 10,
) -> float:
    recs_iter = (
        all_recommendations.values()
        if isinstance(all_recommendations, dict)
        else all_recommendations
    )
    scores = [novelty_at_k(recs, item_popularity, total_users, k=k) for recs in recs_iter]
    return float(np.mean(scores)) if scores else 0.0


def average_popularity_at_k(
    recommended: list[int],
    item_popularity: dict[int, int],
    k: int = 10,
) -> float:
    recs = recommended[:k]
    if not recs:
        return 0.0
    return float(np.mean([item_popularity.get(item, 0) for item in recs]))


def mean_average_popularity_at_k(
    all_recommendations: list[list[int]] | dict[Any, list[int]],
    item_popularity: dict[int, int],
    k: int = 10,
) -> float:
    recs_iter = (
        all_recommendations.values()
        if isinstance(all_recommendations, dict)
        else all_recommendations
    )
    scores = [average_popularity_at_k(recs, item_popularity, k=k) for recs in recs_iter]
    return float(np.mean(scores)) if scores else 0.0


def gini_exposure(
    all_recommendations: list[list[int]] | dict[Any, list[int]],
    num_items: int,
    k: int = 10,
) -> float:
    if num_items <= 1:
        return 0.0

    counts: dict[int, int] = Counter()
    recs_iter = (
        all_recommendations.values()
        if isinstance(all_recommendations, dict)
        else all_recommendations
    )
    for recs in recs_iter:
        for item in recs[:k]:
            if 1 <= item <= num_items:
                counts[item] += 1

    total_exposure = sum(counts.values())
    if total_exposure == 0:
        return 0.0

    # Include all items in catalog (1..num_items)
    frequencies = np.array([counts.get(i, 0) for i in range(1, num_items + 1)], dtype=float)
    frequencies.sort()
    n = len(frequencies)
    indices = np.arange(1, n + 1)
    gini = float(np.sum((2 * indices - n - 1) * frequencies) / (n * total_exposure))
    return float(np.clip(gini, 0.0, 1.0))


def compute_user_activity_quartiles(
    interactions: dict[int, list[int]],
) -> dict[str, set[int]]:
    if not interactions:
        return {"Q1": set(), "Q2": set(), "Q3": set(), "Q4": set()}

    sorted_users = sorted(interactions.keys(), key=lambda u: len(interactions[u]))
    n = len(sorted_users)
    q1_cut = n // 4
    q2_cut = n // 2
    q3_cut = (3 * n) // 4

    return {
        "Q1": set(sorted_users[:q1_cut]),
        "Q2": set(sorted_users[q1_cut:q2_cut]),
        "Q3": set(sorted_users[q2_cut:q3_cut]),
        "Q4": set(sorted_users[q3_cut:]),
    }


def compute_item_popularity_tiers(
    item_popularity: dict[int, int],
    num_items: int,
) -> dict[str, set[int]]:
    all_items = sorted(
        range(1, num_items + 1),
        key=lambda i: item_popularity.get(i, 0),
        reverse=True,
    )
    n = len(all_items)
    head_cut = max(1, int(n * 0.20))
    torso_cut = max(head_cut + 1, int(n * 0.50))

    return {
        "Head": set(all_items[:head_cut]),
        "Torso": set(all_items[head_cut:torso_cut]),
        "Tail": set(all_items[torso_cut:]),
    }


def evaluate_by_user_activity(
    all_recommendations: dict[int, list[int]],
    targets: dict[int, int],
    user_quartiles: dict[str, set[int]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for q_name, uids in user_quartiles.items():
        q_results: dict[str, float] = {}
        sub_users = [u for u in uids if u in targets and u in all_recommendations]
        for k in k_values:
            if not sub_users:
                q_results[f"HitRate@{k}"] = 0.0
                q_results[f"NDCG@{k}"] = 0.0
                q_results[f"Recall@{k}"] = 0.0
                q_results[f"MRR@{k}"] = 0.0
            else:
                hr = [hit_rate_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                ndcg = [ndcg_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                recall = [recall_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                mrr = [mrr_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                q_results[f"HitRate@{k}"] = float(np.mean(hr))
                q_results[f"NDCG@{k}"] = float(np.mean(ndcg))
                q_results[f"Recall@{k}"] = float(np.mean(recall))
                q_results[f"MRR@{k}"] = float(np.mean(mrr))
        results[q_name] = q_results
    return results


def evaluate_by_item_popularity_tiers(
    all_recommendations: dict[int, list[int]],
    targets: dict[int, int],
    item_tiers: dict[str, set[int]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    total_recs_at_10 = sum(len(recs[:10]) for recs in all_recommendations.values()) or 1

    for tier_name, item_set in item_tiers.items():
        tier_results: dict[str, float] = {}
        sub_users = [
            u for u, tgt in targets.items() if tgt in item_set and u in all_recommendations
        ]

        for k in k_values:
            if not sub_users:
                tier_results[f"HitRate@{k}"] = 0.0
                tier_results[f"NDCG@{k}"] = 0.0
            else:
                hr = [hit_rate_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                ndcg = [ndcg_at_k(all_recommendations[u], targets[u], k) for u in sub_users]
                tier_results[f"HitRate@{k}"] = float(np.mean(hr))
                tier_results[f"NDCG@{k}"] = float(np.mean(ndcg))

        # Calculate exposure share in top 10 recommendations
        rec_in_tier = sum(
            sum(1 for it in recs[:10] if it in item_set) for recs in all_recommendations.values()
        )
        tier_results["exposure_share@10"] = float(rec_in_tier / total_recs_at_10)
        results[tier_name] = tier_results
    return results


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
