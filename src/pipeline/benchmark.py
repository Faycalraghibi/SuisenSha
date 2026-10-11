"""SuisenSha benchmark suite.

Runs multi-seed benchmarks across all recommendation models (Random, Popularity,
ItemKNN, BPR-MF, SASRec), tracking:
- Ranking accuracy: HitRate@k, NDCG@k, Recall@k, MRR@k
- Beyond-accuracy & exposure: Catalog coverage, intra-list diversity, novelty,
  average popularity, Gini index of exposure
- Sliced metrics by user activity quartiles and item popularity tiers
- Training time and peak memory consumption via tracemalloc
- 95% confidence intervals across seeds
- Explanation quality evaluation (grounding rate, hallucination rate, LLM-as-judge,
  length, latency)
- Writes reports/benchmark.md and reports/benchmark.csv
"""

from __future__ import annotations

import csv
import logging
import math
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import (
    REPORTS_DIR,
    SEED,
    benchmark_cfg,
    ensure_dirs,
    eval_cfg,
)
from pipeline.data.loader import load_movies, load_sequences
from pipeline.data.splits import split_leave_two_out
from pipeline.evaluation.explanations import evaluate_explanation_suite
from pipeline.evaluation.metrics import (
    average_popularity_at_k,
    catalog_coverage_at_k,
    compute_item_popularity_tiers,
    compute_user_activity_quartiles,
    evaluate_by_item_popularity_tiers,
    evaluate_by_user_activity,
    gini_exposure,
    hit_rate_at_k,
    intra_list_diversity_at_k,
    mrr_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)
from pipeline.models.base import Recommender
from pipeline.models.baselines import (
    BPRMFRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from pipeline.models.explanations import generate_template_explanation
from pipeline.models.llm import FakeLLMBackend, LLMBackend
from pipeline.models.sequential import train_sasrec
from pipeline.ranking.diversity import maximal_marginal_relevance

logger = logging.getLogger(__name__)


def compute_ci95(values: list[float]) -> tuple[float, float]:
    """Calculate mean and 95% confidence interval half-width.

    Returns (mean, ci95).
    """
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return mean, 0.0
    try:
        from scipy.stats import t

        crit = float(t.ppf(0.975, df=n - 1))
    except Exception:
        crit = 1.96
    ci = crit * (std / math.sqrt(n))
    return mean, float(ci)


def _format_metric(mean: float, ci: float) -> str:
    if ci > 0.0:
        return f"{mean:.4f} ± {ci:.4f}"
    return f"{mean:.4f}"


def run_benchmark(
    seeds: int | list[int] = benchmark_cfg.seeds,
    k_values: tuple[int, ...] = eval_cfg.k_values,
    output_dir: Path | str = REPORTS_DIR,
    include_models: list[str] | None = None,
    judge_backend: LLMBackend | None = None,
    max_eval_users: int | None = None,
    sequences: dict[int, list[int]] | None = None,
    movies_df: Any | None = None,
    dataset: str = "ml-100k",
) -> dict[str, Any]:
    """Run multi-seed benchmark across all models and generate Markdown and CSV reports."""
    ensure_dirs()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(seeds, int):
        seed_list = [SEED + i for i in range(seeds)]
    else:
        seed_list = list(seeds)

    dataset_clean = dataset.lower().replace("_", "-")
    if sequences is None:
        if dataset_clean in ("ml-1m", "movielens-1m"):
            dataset_name = "MovieLens-1M"
            from pipeline.config import DATA_DIR
            from pipeline.data.connectors.movielens import MovieLensConnector

            ml_1m_dir = DATA_DIR / "ml-1m"
            if not (ml_1m_dir / "ratings.dat").is_file():
                ml_1m_dir = MovieLensConnector.download_1m()
            conn = MovieLensConnector(ml_1m_dir, variant="1m")
            movies_df, _ = conn.validate()
            sequences = conn.build_sequences(min_rating=3.5, min_interactions=5)
        else:
            dataset_name = "MovieLens-100K"
            sequences = load_sequences()
            if movies_df is None:
                movies_df = load_movies()
    else:
        if "1m" in dataset_clean:
            dataset_name = "MovieLens-1M"
        elif "100k" in dataset_clean:
            dataset_name = "MovieLens-100K"
        else:
            dataset_name = dataset

        if movies_df is None:
            movies_df = load_movies()

    # Pre-compute catalogue metadata and popularity
    catalogue_titles = set(movies_df["title"].dropna().tolist())
    movie_dict: dict[int, dict[str, Any]] = {}
    item_genres: dict[int, set[str]] = {}
    for _, row in movies_df.iterrows():
        mid = int(row["item_id"])
        genres_set = {g.strip().lower() for g in str(row["genres"]).split(",") if g.strip()}
        movie_dict[mid] = {
            "item_id": mid,
            "title": str(row["title"]),
            "genres": str(row.get("genres", "")),
            "release_date": str(row["release_date"])
            if "release_date" in row and pd.notna(row["release_date"])
            else "",
        }
        item_genres[mid] = genres_set

    num_items = len(movies_df)
    item_popularity: dict[int, int] = {}
    for seq in sequences.values():
        for item in seq:
            item_popularity[item] = item_popularity.get(item, 0) + 1
    total_users = len(sequences)

    # Activity quartiles and item popularity tiers
    user_quartiles = compute_user_activity_quartiles(sequences)
    item_tiers = compute_item_popularity_tiers(item_popularity, num_items=num_items)

    max_k = max(k_values)
    models_to_run = include_models or ["Random", "Popularity", "ItemKNN", "BPR-MF", "SASRec"]

    # Storage for per-run raw results: model -> list of run dicts
    raw_results: dict[str, list[dict[str, Any]]] = {m: [] for m in models_to_run}

    logger.info(
        "Starting benchmark across %d seeds (%s) on %d models",
        len(seed_list),
        seed_list,
        len(models_to_run),
    )

    for seed_idx, seed in enumerate(seed_list):
        logger.info("─── Seed %d/%d (seed=%d) ───", seed_idx + 1, len(seed_list), seed)
        splits = split_leave_two_out(sequences)

        test_uids = list(splits.test_targets.keys())
        if max_eval_users is not None:
            test_uids = test_uids[:max_eval_users]

        for model_name in models_to_run:
            logger.info("Fitting and evaluating %s (seed=%d)...", model_name, seed)
            tracemalloc.start()
            t0 = time.perf_counter()

            model: Recommender
            if model_name == "Random":
                model = RandomRecommender(seed=seed)
                model.fit(splits.train_sequences)
            elif model_name == "Popularity":
                model = PopularityRecommender()
                model.fit(splits.train_sequences)
            elif model_name == "ItemKNN":
                model = ItemKNNRecommender(k_neighbors=20)
                model.fit(splits.train_sequences)
            elif model_name == "BPR-MF":
                model = BPRMFRecommender(seed=seed, epochs=5)
                model.fit(splits.train_sequences)
            elif model_name == "SASRec":
                model = train_sasrec(
                    splits.train_sequences,
                    val_history=splits.val_history,
                    val_targets=splits.val_targets,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            train_time_sec = time.perf_counter() - t0
            peak_mem_mb = peak_mem / (1024 * 1024)

            # Generate recommendations on test split
            all_recs: dict[int, list[int]] = {}
            for uid in test_uids:
                hist = splits.test_history.get(uid, [])
                all_recs[uid] = model.recommend(hist, k=max_k)

            # Compute accuracy metrics
            run_metrics: dict[str, Any] = {
                "model": model_name,
                "seed": seed,
                "train_time_sec": train_time_sec,
                "peak_mem_mb": peak_mem_mb,
            }

            for k in k_values:
                hr_list = [hit_rate_at_k(all_recs[u], splits.test_targets[u], k) for u in test_uids]
                ndcg_list = [ndcg_at_k(all_recs[u], splits.test_targets[u], k) for u in test_uids]
                recall_list = [
                    recall_at_k(all_recs[u], splits.test_targets[u], k) for u in test_uids
                ]
                mrr_list = [mrr_at_k(all_recs[u], splits.test_targets[u], k) for u in test_uids]

                run_metrics[f"HitRate@{k}"] = float(np.mean(hr_list)) if hr_list else 0.0
                run_metrics[f"NDCG@{k}"] = float(np.mean(ndcg_list)) if ndcg_list else 0.0
                run_metrics[f"Recall@{k}"] = float(np.mean(recall_list)) if recall_list else 0.0
                run_metrics[f"MRR@{k}"] = float(np.mean(mrr_list)) if mrr_list else 0.0
                run_metrics[f"Coverage@{k}"] = catalog_coverage_at_k(all_recs, num_items, k=k)

                # Diversity, Novelty, Popularity
                div_scores = [
                    intra_list_diversity_at_k(all_recs[u], item_genres, k=k) for u in test_uids
                ]
                nov_scores = [
                    novelty_at_k(all_recs[u], item_popularity, total_users, k=k) for u in test_uids
                ]
                pop_scores = [
                    average_popularity_at_k(all_recs[u], item_popularity, k=k) for u in test_uids
                ]
                run_metrics[f"Diversity@{k}"] = float(np.mean(div_scores)) if div_scores else 0.0
                run_metrics[f"Novelty@{k}"] = float(np.mean(nov_scores)) if nov_scores else 0.0
                run_metrics[f"AvgPopularity@{k}"] = (
                    float(np.mean(pop_scores)) if pop_scores else 0.0
                )

            run_metrics["GiniExposure@10"] = gini_exposure(all_recs, num_items, k=10)

            # Activity and popularity breakdowns
            activity_breakdown = evaluate_by_user_activity(
                all_recs, splits.test_targets, user_quartiles, k_values=(10,)
            )
            for q_name, q_res in activity_breakdown.items():
                run_metrics[f"Activity_{q_name}_HR@10"] = q_res.get("HitRate@10", 0.0)

            tier_breakdown = evaluate_by_item_popularity_tiers(
                all_recs, splits.test_targets, item_tiers, k_values=(10,)
            )
            for t_name, t_res in tier_breakdown.items():
                run_metrics[f"Tier_{t_name}_HR@10"] = t_res.get("HitRate@10", 0.0)
                run_metrics[f"Tier_{t_name}_Exposure"] = t_res.get("exposure_share@10", 0.0)

            raw_results[model_name].append(run_metrics)

    # Aggregate metrics across seeds (mean ± ci95)
    aggregated_results: dict[str, dict[str, tuple[float, float]]] = {}
    for model_name, runs in raw_results.items():
        if not runs:
            continue
        agg: dict[str, tuple[float, float]] = {}
        first = runs[0]
        metric_keys = [mk for mk in first.keys() if mk not in ("model", "seed")]
        for m_name in metric_keys:
            vals = [float(r[m_name]) for r in runs]
            agg[m_name] = compute_ci95(vals)
        aggregated_results[model_name] = agg

    # Explanation quality evaluation
    logger.info("Evaluating explanation quality...")
    judge = judge_backend or FakeLLMBackend(canned_response="4")
    sample_explanations: list[dict[str, Any]] = []
    splits = split_leave_two_out(sequences)
    sample_uids = list(splits.test_targets.keys())[:25]

    for uid in sample_uids:
        hist = splits.test_history.get(uid, [])
        target = splits.test_targets.get(uid)
        if not target or target not in movie_dict:
            continue
        cand = movie_dict[target]
        hist_movies = [movie_dict[m] for m in hist[-5:] if m in movie_dict]

        t0 = time.perf_counter()
        exp = generate_template_explanation(cand, hist_movies, lang="en")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        sample_explanations.append(
            {
                "explanation": exp,
                "candidate": cand,
                "history_movies": hist_movies,
                "allowed_titles": {cand["title"]} | {m["title"] for m in hist_movies},
                "latency_ms": latency_ms,
            }
        )

    exp_report = evaluate_explanation_suite(
        sample_explanations,
        catalogue_titles=catalogue_titles,
        judge_backend=judge,
    )

    # Diversity Trade-Off Evaluation (MMR re-ranking)
    logger.info("Evaluating accuracy vs. diversity trade-off...")
    diversity_tradeoff: list[dict[str, Any]] = []
    base_rec = PopularityRecommender()
    base_rec.fit(splits.train_sequences)
    test_uids_sample = list(splits.test_targets.keys())
    tradeoff_divs = [0.0, 0.3, 0.7, 1.0]

    for div_val in tradeoff_divs:
        div_recs: dict[int, list[int]] = {}
        for u in test_uids_sample:
            u_hist = splits.test_history.get(u, [])
            cands = base_rec.recommend(u_hist, k=max(max(k_values) * 3, 30))
            scores = {cid: 1.0 / (idx + 1.0) for idx, cid in enumerate(cands)}
            reranked = maximal_marginal_relevance(
                cands,
                item_features=item_genres,
                scores=scores,
                diversity=div_val,
                k=10,
            )
            div_recs[u] = reranked

        hr10 = (
            float(
                np.mean(
                    [
                        hit_rate_at_k(div_recs[u], splits.test_targets[u], 10)
                        for u in test_uids_sample
                    ]
                )
            )
            if test_uids_sample
            else 0.0
        )
        ndcg10 = (
            float(
                np.mean(
                    [ndcg_at_k(div_recs[u], splits.test_targets[u], 10) for u in test_uids_sample]
                )
            )
            if test_uids_sample
            else 0.0
        )
        div10 = (
            float(
                np.mean(
                    [
                        intra_list_diversity_at_k(div_recs[u], item_genres, 10)
                        for u in test_uids_sample
                    ]
                )
            )
            if test_uids_sample
            else 0.0
        )
        nov10 = (
            float(
                np.mean(
                    [
                        novelty_at_k(div_recs[u], item_popularity, total_users, 10)
                        for u in test_uids_sample
                    ]
                )
            )
            if test_uids_sample
            else 0.0
        )
        gini10 = gini_exposure(div_recs, num_items, k=10)

        diversity_tradeoff.append(
            {
                "diversity": div_val,
                "HitRate@10": hr10,
                "NDCG@10": ndcg10,
                "Diversity@10": div10,
                "Novelty@10": nov10,
                "GiniExposure@10": gini10,
            }
        )

    # Write CSV
    csv_path = out_dir / "benchmark.csv"
    _write_csv(raw_results, aggregated_results, csv_path)
    logger.info("Benchmark CSV written → %s", csv_path)

    # Write Markdown
    md_path = out_dir / "benchmark.md"
    _write_markdown(
        aggregated_results,
        exp_report,
        len(seed_list),
        seed_list,
        total_users,
        num_items,
        md_path,
        dataset_name=dataset_name,
        diversity_tradeoff=diversity_tradeoff,
    )
    logger.info("Benchmark report written → %s", md_path)

    return {
        "raw_results": raw_results,
        "aggregated_results": aggregated_results,
        "explanation_report": exp_report.to_dict(),
        "diversity_tradeoff": diversity_tradeoff,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
    }


def _write_csv(
    raw_results: dict[str, list[dict[str, Any]]],
    aggregated_results: dict[str, dict[str, tuple[float, float]]],
    csv_path: Path,
) -> None:
    all_runs: list[dict[str, Any]] = []
    for runs in raw_results.values():
        all_runs.extend(runs)

    if not all_runs:
        return

    fieldnames = list(all_runs[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_runs:
            writer.writerow(row)


def _write_markdown(
    aggregated_results: dict[str, dict[str, tuple[float, float]]],
    exp_report: Any,
    num_seeds: int,
    seed_list: list[int],
    n_users: int,
    n_items: int,
    md_path: Path,
    dataset_name: str = "MovieLens-100K",
    diversity_tradeoff: list[dict[str, Any]] | None = None,
) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# SuisenSha Recommender Benchmark Report",
        "",
        f"- **Generated**: {now_str}",
        "- **Evaluation Protocol**: Leave-two-out sequential split "
        "(validation held-out, test isolated)",
        f"- **Seeds Evaluated**: {num_seeds} ({seed_list})",
        f"- **Dataset**: {dataset_name} ({n_users:,} users, {n_items:,} items)",
        "",
        "## 1. Ranking Accuracy (Mean ± 95% CI)",
        "",
        "| Model | HitRate@5 | HitRate@10 | NDCG@5 | NDCG@10 | Recall@10 | MRR@10 | "
        "Train Time (s) | Peak Mem (MB) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for model, metrics in aggregated_results.items():
        hr5 = _format_metric(*metrics.get("HitRate@5", (0.0, 0.0)))
        hr10 = _format_metric(*metrics.get("HitRate@10", (0.0, 0.0)))
        ndcg5 = _format_metric(*metrics.get("NDCG@5", (0.0, 0.0)))
        ndcg10 = _format_metric(*metrics.get("NDCG@10", (0.0, 0.0)))
        rec10 = _format_metric(*metrics.get("Recall@10", (0.0, 0.0)))
        mrr10 = _format_metric(*metrics.get("MRR@10", (0.0, 0.0)))
        ttime = f"{metrics.get('train_time_sec', (0.0, 0.0))[0]:.2f}s"
        pmem = f"{metrics.get('peak_mem_mb', (0.0, 0.0))[0]:.1f} MB"

        bold = "**" if model == "SASRec" else ""
        lines.append(
            f"| {bold}{model}{bold} | {hr5} | {hr10} | {ndcg5} | {ndcg10} | "
            f"{rec10} | {mrr10} | {ttime} | {pmem} |"
        )

    lines.extend(
        [
            "",
            "## 2. Beyond-Accuracy & Exposure Metrics (Mean ± 95% CI)",
            "",
            "| Model | Coverage@10 | Diversity@10 | Novelty@10 | "
            "Avg Popularity@10 | Gini Exposure@10 |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    for model, metrics in aggregated_results.items():
        cov10 = _format_metric(*metrics.get("Coverage@10", (0.0, 0.0)))
        div10 = _format_metric(*metrics.get("Diversity@10", (0.0, 0.0)))
        nov10 = _format_metric(*metrics.get("Novelty@10", (0.0, 0.0)))
        pop10 = _format_metric(*metrics.get("AvgPopularity@10", (0.0, 0.0)))
        gini = _format_metric(*metrics.get("GiniExposure@10", (0.0, 0.0)))

        bold = "**" if model == "SASRec" else ""
        lines.append(f"| {bold}{model}{bold} | {cov10} | {div10} | {nov10} | {pop10} | {gini} |")

    lines.extend(
        [
            "",
            "## 3. Performance by User Activity Quartile (HitRate@10)",
            "",
            "| Model | Q1 (Least Active) | Q2 | Q3 | Q4 (Most Active) |",
            "|:---|:---:|:---:|:---:|:---:|",
        ]
    )

    for model, metrics in aggregated_results.items():
        q1 = _format_metric(*metrics.get("Activity_Q1_HR@10", (0.0, 0.0)))
        q2 = _format_metric(*metrics.get("Activity_Q2_HR@10", (0.0, 0.0)))
        q3 = _format_metric(*metrics.get("Activity_Q3_HR@10", (0.0, 0.0)))
        q4 = _format_metric(*metrics.get("Activity_Q4_HR@10", (0.0, 0.0)))

        bold = "**" if model == "SASRec" else ""
        lines.append(f"| {bold}{model}{bold} | {q1} | {q2} | {q3} | {q4} |")

    lines.extend(
        [
            "",
            "## 4. Performance by Item Popularity Tier",
            "",
            "| Model | Head HR@10 | Torso HR@10 | Tail HR@10 | "
            "Head Exposure Share | Tail Exposure Share |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ]
    )

    for model, metrics in aggregated_results.items():
        h_hr = _format_metric(*metrics.get("Tier_Head_HR@10", (0.0, 0.0)))
        tor_hr = _format_metric(*metrics.get("Tier_Torso_HR@10", (0.0, 0.0)))
        tail_hr = _format_metric(*metrics.get("Tier_Tail_HR@10", (0.0, 0.0)))
        h_exp = _format_metric(*metrics.get("Tier_Head_Exposure", (0.0, 0.0)))
        tail_exp = _format_metric(*metrics.get("Tier_Tail_Exposure", (0.0, 0.0)))

        bold = "**" if model == "SASRec" else ""
        lines.append(
            f"| {bold}{model}{bold} | {h_hr} | {tor_hr} | {tail_hr} | {h_exp} | {tail_exp} |"
        )

    lines.extend(
        [
            "",
            "## 5. Explanation Quality Evaluation",
            "",
            "| Metric | Result |",
            "|:---|:---:|",
            f"| Evaluated Explanations | {exp_report.total} |",
            f"| Grounding Rate | {exp_report.grounding_rate * 100:.1f}% |",
            f"| Hallucination Rate | {exp_report.hallucination_rate * 100:.1f}% |",
            f"| LLM-as-Judge Relevance | {exp_report.llm_judge_relevance:.2f} / 5.0 |",
            f"| Average Length (words) | {exp_report.avg_length_words:.1f} words |",
            f"| Average Length (characters) | {exp_report.avg_length_chars:.1f} chars |",
            f"| Average Latency | {exp_report.avg_latency_ms:.2f} ms |",
            "",
        ]
    )

    if diversity_tradeoff:
        lines.extend(
            [
                "## 6. Accuracy vs. Diversity Trade-off (MMR Re-ranking)",
                "",
                "| Diversity | HitRate@10 | NDCG@10 | Diversity@10 | Novelty@10 | Gini |",
                "|:---|:---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for row in diversity_tradeoff:
            d_val = row["diversity"]
            hr = f"{row['HitRate@10']:.4f}"
            ndcg = f"{row['NDCG@10']:.4f}"
            div = f"{row['Diversity@10']:.4f}"
            nov = f"{row['Novelty@10']:.2f}"
            gini = f"{row['GiniExposure@10']:.4f}"
            lines.append(f"| {d_val:.1f} | {hr} | {ndcg} | {div} | {nov} | {gini} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
