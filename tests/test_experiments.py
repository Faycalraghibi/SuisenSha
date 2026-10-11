"""Unit tests for A/B testing: deterministic assignment, uplift, sample size, and A/A test."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.api.main import app
from pipeline.experiments.manager import (
    ExperimentManager,
    Variant,
    calculate_sample_size,
)


class TestExperimentManager:
    """Tests for ExperimentManager variant assignment, events logging, and uplift."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> ExperimentManager:
        return ExperimentManager(data_dir=tmp_path / "exp")

    def test_deterministic_variant_assignment(self, manager: ExperimentManager):
        variants = [
            Variant(name="control", weight=1.0, model_type="sasrec"),
            Variant(name="treatment", weight=1.0, model_type="hybrid"),
        ]
        manager.create_experiment("exp_det", "Deterministic Test", variants)

        # Repeated assignments for the same user must be strictly identical
        user_1_assignments = [manager.assign_variant("exp_det", 42).name for _ in range(50)]
        assert len(set(user_1_assignments)) == 1, (
            "User 42 must always receive the identical variant"
        )

        user_2_assignments = [manager.assign_variant("exp_det", 999).name for _ in range(50)]
        assert len(set(user_2_assignments)) == 1, (
            "User 999 must always receive the identical variant"
        )

    def test_aa_test_shows_no_significant_uplift(self, manager: ExperimentManager):
        """A/A test: Both control and treatment draw from the identical conversion rate (e.g. 10%).

        Verifies that p-value remains > 0.05 and no spurious significant uplift is reported.
        """
        variants = [
            Variant(name="control", weight=1.0, model_type="sasrec"),
            Variant(name="treatment", weight=1.0, model_type="sasrec"),
        ]
        manager.create_experiment("exp_aa", "A/A Experiment", variants)

        # Simulate 1,000 users per variant with identical 10% conversion rate
        for i in range(1000):
            # Control user
            u_c = f"u_c_{i}"
            manager.log_exposure("exp_aa", "control", u_c)
            if i < 100:  # exactly 10%
                manager.log_conversion("exp_aa", "control", u_c)

            # Treatment user
            u_t = f"u_t_{i}"
            manager.log_exposure("exp_aa", "treatment", u_t)
            if i < 102:  # 10.2% (slight noise)
                manager.log_conversion("exp_aa", "treatment", u_t)

        report = manager.get_uplift_report(
            "exp_aa", control_name="control", treatment_name="treatment"
        )
        assert report.is_significant is False, "A/A test should NOT be statistically significant"
        assert report.p_value > 0.05
        assert -0.05 < report.absolute_uplift < 0.05

    def test_significant_uplift_detected(self, manager: ExperimentManager):
        variants = [
            Variant(name="control", weight=1.0, model_type="sasrec"),
            Variant(name="treatment", weight=1.0, model_type="hybrid"),
        ]
        manager.create_experiment("exp_ab", "A/B Experiment", variants)

        # Control: 5% conversion (50/1000)
        for i in range(1000):
            uid = f"ctrl_{i}"
            manager.log_exposure("exp_ab", "control", uid)
            if i < 50:
                manager.log_conversion("exp_ab", "control", uid)

        # Treatment: 10% conversion (100/1000) -> significant uplift
        for i in range(1000):
            uid = f"treat_{i}"
            manager.log_exposure("exp_ab", "treatment", uid)
            if i < 100:
                manager.log_conversion("exp_ab", "treatment", uid)

        report = manager.get_uplift_report(
            "exp_ab", control_name="control", treatment_name="treatment"
        )
        assert report.is_significant is True
        assert report.p_value < 0.001
        assert report.relative_uplift_pct == pytest.approx(100.0, rel=0.01)
        assert report.ci_lower > 0.0


class TestSampleSizeCalculator:
    """Tests for statistical sample size calculator formulas."""

    def test_calculate_sample_size_formula(self):
        # Baseline = 5%, MDE = 20% relative uplift (p1=0.05, p2=0.06)
        # Two-sided alpha=0.05, power=0.80 yields ~8,158 samples per variant
        n = calculate_sample_size(
            baseline_conversion=0.05,
            minimum_detectable_effect=0.20,
            alpha=0.05,
            power=0.80,
        )
        assert 7000 < n < 10000

    def test_calculate_sample_size_invalid_inputs(self):
        assert calculate_sample_size(0.0, 0.1) == 0
        assert calculate_sample_size(1.0, 0.1) == 0
        assert calculate_sample_size(0.5, 0.0) == 0


class TestExperimentAPI:
    """API endpoints integration tests for /v1/experiments."""

    @pytest.fixture
    def client(self, tmp_path: Path):
        from pipeline.api.dependencies import get_experiment_manager
        from pipeline.api.security import get_api_key

        test_mgr = ExperimentManager(data_dir=tmp_path / "exp_api")
        app.dependency_overrides[get_api_key] = lambda: {
            "key_hash": "dummy",
            "client_name": "test_admin",
            "tier": "admin",
            "is_active": True,
        }
        app.dependency_overrides[get_experiment_manager] = lambda: test_mgr

        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    def test_create_and_list_experiments_api(self, client: TestClient):
        payload = {
            "id": "exp_api_test",
            "name": "API Test Experiment",
            "description": "Testing API creation",
            "variants": [
                {"name": "control", "weight": 1.0, "model_type": "sasrec"},
                {"name": "treatment", "weight": 1.0, "model_type": "hybrid"},
            ],
        }
        create_resp = client.post("/v1/experiments", json=payload)
        assert create_resp.status_code == 201
        data = create_resp.json()
        assert data["id"] == "exp_api_test"
        assert len(data["variants"]) == 2

        list_resp = client.get("/v1/experiments")
        assert list_resp.status_code == 200
        experiments = list_resp.json()
        assert any(e["id"] == "exp_api_test" for e in experiments)

    def test_sample_size_api(self, client: TestClient):
        resp = client.get(
            "/v1/experiments/sample-size",
            params={
                "baseline_conversion": 0.05,
                "mde": 0.10,
                "alpha": 0.05,
                "power": 0.80,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "required_sample_size_per_variant" in data
        assert data["required_sample_size_per_variant"] > 0
        assert data["total_required_sample_size"] == data["required_sample_size_per_variant"] * 2
