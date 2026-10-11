"""Retraining pipeline orchestrating ingestion, training, evaluation, and promotion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pipeline.config import PipelineConfig, load_config
from pipeline.data.loader import load_processed_movies, load_sequences
from pipeline.evaluation.metrics import evaluate_leave_one_out
from pipeline.mlops.drift import DriftDetector, DriftReport
from pipeline.mlops.registry import ModelRegistry
from pipeline.models.sequential import train_sasrec

logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """Result of an end-to-end retraining flow."""

    candidate_version: int
    promoted: bool
    rejection_reason: str | None
    candidate_metrics: dict[str, float]
    production_metrics: dict[str, float] | None
    drift_report: DriftReport | None


class RetrainingPipeline:
    """Orchestrates scheduled/triggered retraining and gated promotion."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        config: PipelineConfig | None = None,
        tolerance: float = 0.05,  # Allow up to 5% drop vs production before blocking promotion
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.config = config or load_config()
        self.tolerance = tolerance
        self.drift_detector = DriftDetector()

    def run(
        self,
        epochs: int | None = None,
        force_promote: bool = False,
    ) -> RetrainingResult:
        """Run the full ingestion -> train -> benchmark -> gated promotion pipeline."""
        logger.info("Starting MLOps Retraining Flow...")

        # 1. Ingestion / Data loading
        sequences = load_sequences()
        movies_df = load_processed_movies()
        num_items = len(movies_df)

        # 2. Train candidate SASRec model
        cfg = self.config
        training_epochs = epochs if epochs is not None else cfg.sasrec.epochs
        logger.info("Training candidate SASRec model (epochs=%d)...", training_epochs)
        model = train_sasrec(
            sequences=sequences,
            seed=cfg.seed,
        )

        # 3. Evaluate candidate model on leave-one-out
        logger.info("Evaluating candidate model on validation set...")
        candidate_eval = evaluate_leave_one_out(
            sequences=sequences,
            recommend=lambda hist, k: model.recommend(hist, k=k),
            k_values=(10,),
        )
        candidate_metrics = {
            "HitRate@10": float(candidate_eval.get("HitRate@10", 0.0)),
            "NDCG@10": float(candidate_eval.get("NDCG@10", 0.0)),
        }
        logger.info("Candidate metrics: %s", candidate_metrics)

        # 4. Save candidate checkpoint temporarily for registry
        temp_dir = self.registry.registry_dir / "temp_candidate"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_weights_path = temp_dir / "candidate_sasrec.pt"
        import torch

        ckpt = {"model_state": model.state_dict(), "num_items": num_items}
        torch.save(ckpt, str(temp_weights_path))

        # 5. Register candidate in ModelRegistry

        mv = self.registry.register_model(
            model_name="sasrec",
            artifact_path=temp_weights_path,
            metrics=candidate_metrics,
            parameters={
                "epochs": training_epochs,
                "max_seq_len": cfg.sasrec.max_seq_len,
                "hidden_dim": cfg.sasrec.hidden_dim,
            },
            description="Automated retraining candidate",
        )

        # Clean up temp candidate
        if temp_weights_path.exists():
            temp_weights_path.unlink()

        # 6. Gated promotion check vs current production model
        curr_prod = self.registry.get_production_version("sasrec")
        promoted = False
        rejection_reason: str | None = None
        prod_metrics: dict[str, float] | None = None

        if curr_prod is None:
            # First model automatically promoted
            logger.info(
                "No active production model found. Automatically promoting v%d.", mv.version
            )
            self.registry.promote_to_production(version=mv.version, model_name="sasrec")
            promoted = True
        else:
            prod_metrics = curr_prod.metrics
            prod_hr = prod_metrics.get("HitRate@10", 0.0)
            cand_hr = candidate_metrics.get("HitRate@10", 0.0)

            # Check if candidate satisfies tolerance threshold: cand_hr >= prod_hr * (1 - tolerance)
            min_allowed_hr = prod_hr * (1.0 - self.tolerance)
            if force_promote or cand_hr >= min_allowed_hr:
                logger.info(
                    "Candidate v%d passed gate (HR@10: %.4f vs Prod: %.4f). Promoting!",
                    mv.version,
                    cand_hr,
                    prod_hr,
                )
                self.registry.promote_to_production(version=mv.version, model_name="sasrec")
                promoted = True
            else:
                rejection_reason = (
                    f"Candidate HitRate@10 ({cand_hr:.4f}) below tolerance threshold "
                    f"({min_allowed_hr:.4f}) of current production model v{curr_prod.version}."
                )
                logger.warning("Promotion rejected: %s", rejection_reason)

        return RetrainingResult(
            candidate_version=mv.version,
            promoted=promoted,
            rejection_reason=rejection_reason,
            candidate_metrics=candidate_metrics,
            production_metrics=prod_metrics,
            drift_report=None,
        )
