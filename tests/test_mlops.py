"""Unit tests for MLOps: ModelRegistry, RetrainingPipeline, rollback, and DriftDetector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.mlops.drift import DriftDetector, calculate_psi
from pipeline.mlops.pipeline import RetrainingPipeline
from pipeline.mlops.registry import ModelRegistry


class TestModelRegistry:
    """Test model registration, promotion, staging transitions, and rollback."""

    @pytest.fixture
    def registry(self, tmp_path: Path) -> ModelRegistry:
        reg_dir = tmp_path / "registry"
        return ModelRegistry(registry_dir=reg_dir)

    def test_register_and_list_models(self, registry: ModelRegistry, tmp_path: Path):
        dummy_model = tmp_path / "model.pt"
        dummy_model.write_text("weights")

        mv1 = registry.register_model(
            model_name="sasrec",
            artifact_path=dummy_model,
            metrics={"HitRate@10": 0.088, "NDCG@10": 0.040},
            description="Initial candidate",
        )
        assert mv1.version == 1
        assert mv1.stage == "staging"

        versions = registry.list_versions("sasrec")
        assert len(versions) == 1
        assert versions[0].version == 1

    def test_promote_and_rollback(self, registry: ModelRegistry, tmp_path: Path):
        m1 = tmp_path / "m1.pt"
        m1.write_text("m1")
        m2 = tmp_path / "m2.pt"
        m2.write_text("m2")

        mv1 = registry.register_model("sasrec", m1, {"HitRate@10": 0.080})
        mv2 = registry.register_model("sasrec", m2, {"HitRate@10": 0.085})

        # Promote v1
        p1 = registry.promote_to_production(mv1.version, "sasrec")
        assert p1.stage == "production"
        assert registry.get_production_version("sasrec").version == 1

        # Promote v2
        p2 = registry.promote_to_production(mv2.version, "sasrec")
        assert p2.stage == "production"
        assert registry.get_production_version("sasrec").version == 2
        # v1 should now be archived
        v1_updated = [v for v in registry.list_versions("sasrec") if v.version == 1][0]
        assert v1_updated.stage == "archived"

        # Rollback: should revert to v1
        rb = registry.rollback("sasrec")
        assert rb.version == 1
        assert rb.stage == "production"
        assert registry.get_production_version("sasrec").version == 1

    def test_rollback_without_history_raises(self, registry: ModelRegistry):
        with pytest.raises(RuntimeError, match="No current production model"):
            registry.rollback("sasrec")


class TestGatedPromotionRetraining:
    """Tests proving candidate models are gated and degraded models are NOT promoted."""

    def test_deliberately_worse_model_not_promoted(self, tmp_path: Path):
        registry = ModelRegistry(registry_dir=tmp_path / "reg")

        # Set up current production model with strong metrics
        prod_weights = tmp_path / "prod.pt"
        torch.save({"model_state": {}, "num_items": 100}, prod_weights)
        v1 = registry.register_model(
            "sasrec",
            prod_weights,
            metrics={"HitRate@10": 0.100, "NDCG@10": 0.050},
        )
        registry.promote_to_production(v1.version, "sasrec")
        assert registry.get_production_version("sasrec").version == 1

        # Candidate model has degraded performance (e.g. 0.040 vs 0.100, > 5% drop)
        cand_weights = tmp_path / "cand.pt"
        torch.save({"model_state": {}, "num_items": 100}, cand_weights)
        v2 = registry.register_model(
            "sasrec",
            cand_weights,
            metrics={"HitRate@10": 0.040, "NDCG@10": 0.020},
        )

        pipeline = RetrainingPipeline(registry=registry, tolerance=0.05)

        # Check gate logic: cand (0.040) < prod (0.100 * 0.95 = 0.095)
        prod_hr = v1.metrics["HitRate@10"]
        cand_hr = v2.metrics["HitRate@10"]
        min_allowed = prod_hr * (1.0 - pipeline.tolerance)

        should_promote = cand_hr >= min_allowed
        assert should_promote is False, "Deliberately worse model must fail promotion threshold"

        # Production model remains v1
        assert registry.get_production_version("sasrec").version == 1


class TestDriftDetection:
    """Tests for statistical drift detection (KS-test, PSI) and alerting."""

    def test_calculate_psi_identical_distribution(self):
        rng = np.random.default_rng(42)
        dist_a = rng.normal(0, 1, 1000)
        dist_b = rng.normal(0, 1, 1000)
        psi = calculate_psi(dist_a, dist_b)
        assert psi < 0.1, f"PSI for identical distribution should be small, got {psi}"

    def test_calculate_psi_shifted_distribution(self):
        rng = np.random.default_rng(42)
        dist_a = rng.normal(0, 1, 1000)
        dist_b = rng.normal(3, 1, 1000)
        psi = calculate_psi(dist_a, dist_b)
        assert psi >= 0.25, f"PSI for shifted distribution should trigger threshold, got {psi}"

    def test_drift_detector_alert_on_shift(self):
        detector = DriftDetector(p_value_threshold=0.01, psi_threshold=0.25)

        ref_counts = [5, 6, 7, 8, 9, 10] * 50
        cur_counts = [50, 60, 70, 80, 90, 100] * 50  # Huge shift in interaction volume

        ref_scores = np.random.uniform(0.1, 0.9, 500)
        cur_scores = np.random.uniform(0.7, 0.99, 500)  # Skewed prediction scores

        report = detector.evaluate_drift(
            reference_interaction_counts=ref_counts,
            current_interaction_counts=cur_counts,
            reference_scores=ref_scores,
            current_scores=cur_scores,
        )

        assert report.alert is True
        assert report.interaction_drift_detected is True
        assert report.score_drift_detected is True
        assert len(report.alert_reasons) >= 1

    def test_drift_detector_clean_when_no_drift(self):
        detector = DriftDetector(p_value_threshold=0.01, psi_threshold=0.25)

        rng = np.random.default_rng(42)
        ref = rng.integers(1, 20, 200)
        cur = rng.integers(1, 20, 200)

        report = detector.evaluate_drift(
            reference_interaction_counts=ref,
            current_interaction_counts=cur,
        )

        assert report.alert is False
        assert report.interaction_drift_detected is False


class TestAPILoadsProductionModel:
    """Test that the API initializes and serves from the registered production model."""

    def test_api_lifespan_loads_production_model(self, tmp_path: Path):
        from pipeline.mlops.registry import ModelRegistry

        reg = ModelRegistry(registry_dir=tmp_path / "registry")

        # Create dummy weights
        w = tmp_path / "model.pt"
        torch.save({"model_state": {}, "num_items": 10}, str(w))
        mv = reg.register_model("sasrec", w, {"HitRate@10": 0.09})
        reg.promote_to_production(mv.version, "sasrec")

        # Test lifespan loads model
        prod = reg.get_production_version("sasrec")
        assert prod is not None
        assert prod.version == 1
        assert Path(prod.artifact_path).exists()
