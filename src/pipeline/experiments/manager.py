"""A/B testing experiment manager, deterministic assignment, and uplift reporting."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy import stats

from pipeline.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """Experiment variant definition."""

    name: str  # e.g. "control", "treatment_sasrec", "treatment_hybrid"
    weight: float = 1.0  # relative traffic weight
    model_type: str = "sasrec"  # "sasrec", "hybrid", "embedding", "popularity"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Experiment metadata and configuration."""

    id: str
    name: str
    variants: list[Variant]
    status: str = "active"  # "active", "paused", "completed"
    tenant_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""


@dataclass
class VariantMetrics:
    """Live traffic and conversion metrics for an experiment variant."""

    variant_name: str
    exposures: int
    conversions: int
    conversion_rate: float


@dataclass
class UpliftReport:
    """Statistical comparison of treatment vs. control variant."""

    experiment_id: str
    control: VariantMetrics
    treatment: VariantMetrics
    absolute_uplift: float
    relative_uplift_pct: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    confidence_level: float = 0.95


def calculate_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size per variant for a two-proportion test."""
    p1 = baseline_conversion
    p2 = baseline_conversion * (1.0 + minimum_detectable_effect)
    if not (0.0 < p1 < 1.0) or not (0.0 < p2 < 1.0) or p1 == p2:
        return 0

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2.0

    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return int(math.ceil(numerator / denominator))


def compute_uplift(
    control: VariantMetrics,
    treatment: VariantMetrics,
    confidence_level: float = 0.95,
) -> UpliftReport:
    """Compute difference in conversion rates, 95% CIs, and p-value via two-proportion z-test."""
    n1 = max(control.exposures, 1)

    n2 = max(treatment.exposures, 1)
    p1 = control.conversion_rate
    p2 = treatment.conversion_rate

    diff = p2 - p1
    rel_uplift = (diff / p1 * 100.0) if p1 > 0 else 0.0

    # Pooled standard error for hypothesis test
    total_conversions = control.conversions + treatment.conversions
    total_samples = control.exposures + treatment.exposures
    p_pooled = total_conversions / max(total_samples, 1)
    se_pooled = (
        math.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)) if 0 < p_pooled < 1 else 0.0
    )

    if se_pooled > 0:
        z_stat = diff / se_pooled
        p_val = float(2 * (1.0 - stats.norm.cdf(abs(z_stat))))
    else:
        z_stat = 0.0
        p_val = 1.0

    # Unpooled standard error for confidence interval
    se_diff = math.sqrt((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2))
    z_crit = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    margin = z_crit * se_diff

    ci_lower = diff - margin
    ci_upper = diff + margin
    is_significant = p_val < (1.0 - confidence_level)

    return UpliftReport(
        experiment_id="",
        control=control,
        treatment=treatment,
        absolute_uplift=diff,
        relative_uplift_pct=rel_uplift,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        is_significant=is_significant,
        confidence_level=confidence_level,
    )


class ExperimentManager:
    """Manages experiment lifecycle, deterministic assignment, and event tracking."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else ARTIFACTS_DIR / "experiments"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exp_file = self.data_dir / "experiments.json"
        self.events_file = self.data_dir / "experiment_events.json"
        self._ensure_init()

    def _ensure_init(self) -> None:
        if not self.exp_file.exists():
            with open(self.exp_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
        if not self.events_file.exists():
            with open(self.events_file, "w", encoding="utf-8") as f:
                json.dump({"exposures": [], "conversions": []}, f, indent=2)

    def _load_experiments(self) -> dict[str, Any]:
        with open(self.exp_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_experiments(self, data: dict[str, Any]) -> None:
        with open(self.exp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_events(self) -> dict[str, list[dict[str, Any]]]:
        with open(self.events_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_events(self, data: dict[str, list[dict[str, Any]]]) -> None:
        with open(self.events_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        exp_id: str,
        name: str,
        variants: list[Variant],
        description: str = "",
        tenant_id: str = "default",
    ) -> Experiment:
        """Create and store an online experiment."""
        exps = self._load_experiments()
        if exp_id in exps:
            raise ValueError(f"Experiment '{exp_id}' already exists.")
        if len(variants) < 2:
            raise ValueError("Experiment requires at least 2 variants (control and treatment).")

        exp = Experiment(
            id=exp_id,
            name=name,
            variants=variants,
            status="active",
            tenant_id=tenant_id,
            description=description,
        )
        exps[exp_id] = {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "tenant_id": exp.tenant_id,
            "created_at": exp.created_at,
            "description": exp.description,
            "variants": [asdict(v) for v in exp.variants],
        }
        self._save_experiments(exps)
        return exp

    def get_experiment(self, exp_id: str, tenant_id: str | None = None) -> Experiment | None:
        """Retrieve an experiment by its unique ID, optionally matching tenant_id."""
        exps = self._load_experiments()
        raw = exps.get(exp_id)
        if not raw:
            return None
        if tenant_id is not None and raw.get("tenant_id", "default") != tenant_id:
            return None
        variants = [Variant(**v) for v in raw.get("variants", [])]
        return Experiment(
            id=raw["id"],
            name=raw["name"],
            variants=variants,
            status=raw["status"],
            tenant_id=raw.get("tenant_id", "default"),
            created_at=raw["created_at"],
            description=raw.get("description", ""),
        )

    def list_experiments(self, tenant_id: str | None = None) -> list[Experiment]:
        """List all experiments, optionally filtered by tenant."""
        exps = self._load_experiments()
        result = []
        for raw in exps.values():
            if tenant_id is not None and raw.get("tenant_id", "default") != tenant_id:
                continue
            variants = [Variant(**v) for v in raw.get("variants", [])]
            result.append(
                Experiment(
                    id=raw["id"],
                    name=raw["name"],
                    variants=variants,
                    status=raw["status"],
                    tenant_id=raw.get("tenant_id", "default"),
                    created_at=raw["created_at"],
                    description=raw.get("description", ""),
                )
            )
        return result

    def assign_variant(
        self,
        exp_id: str,
        user_id: int | str,
        tenant_id: str | None = None,
    ) -> Variant:
        """Deterministically assign user to a variant using SHA-256 salted hash."""
        exp = self.get_experiment(exp_id, tenant_id=tenant_id)
        if not exp:
            raise ValueError(f"Experiment '{exp_id}' not found.")
        if exp.status != "active":
            return exp.variants[0]  # default to control

        # Salt hash with experiment ID to decouple assignments across experiments
        salt_key = f"{exp_id}:{user_id}".encode()
        hash_val = int(hashlib.sha256(salt_key).hexdigest(), 16)
        bucket = (hash_val % 10000) / 10000.0  # [0.0, 1.0)

        total_weight = sum(v.weight for v in exp.variants)
        cum_threshold = 0.0
        for v in exp.variants:
            norm_weight = v.weight / total_weight
            cum_threshold += norm_weight
            if bucket < cum_threshold:
                return v

        return exp.variants[-1]

    def log_exposure(self, exp_id: str, variant_name: str, user_id: int | str) -> None:
        """Record an exposure event when user views recommendations under a variant."""
        events = self._load_events()
        events["exposures"].append(
            {
                "exp_id": exp_id,
                "variant_name": variant_name,
                "user_id": str(user_id),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_events(events)

    def log_conversion(
        self,
        exp_id: str,
        variant_name: str,
        user_id: int | str,
        event_type: str = "click",
    ) -> None:
        """Record a conversion event (click, purchase, etc.) for an assigned variant."""
        events = self._load_events()
        events["conversions"].append(
            {
                "exp_id": exp_id,
                "variant_name": variant_name,
                "user_id": str(user_id),
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_events(events)

    def get_uplift_report(
        self,
        exp_id: str,
        control_name: str = "control",
        treatment_name: str | None = None,
        tenant_id: str | None = None,
    ) -> UpliftReport:
        """Compute live uplift report between control and treatment variants."""
        exp = self.get_experiment(exp_id, tenant_id=tenant_id)
        if not exp:
            raise ValueError(f"Experiment '{exp_id}' not found.")

        events = self._load_events()
        exposures = events.get("exposures", [])
        conversions = events.get("conversions", [])

        # Count exposures per variant
        exp_counts: dict[str, set[str]] = {}
        for ev in exposures:
            if ev.get("exp_id") == exp_id:
                v = ev.get("variant_name", "")
                exp_counts.setdefault(v, set()).add(str(ev.get("user_id")))

        # Count converted users per variant
        conv_counts: dict[str, set[str]] = {}
        for ev in conversions:
            if ev.get("exp_id") == exp_id:
                v = ev.get("variant_name", "")
                conv_counts.setdefault(v, set()).add(str(ev.get("user_id")))

        # Default treatment name if not specified
        if treatment_name is None:
            treatments = [v.name for v in exp.variants if v.name != control_name]
            treatment_name = treatments[0] if treatments else exp.variants[1].name

        n_ctrl = len(exp_counts.get(control_name, set()))
        c_ctrl = len(conv_counts.get(control_name, set()))
        cr_ctrl = (c_ctrl / n_ctrl) if n_ctrl > 0 else 0.0

        n_treat = len(exp_counts.get(treatment_name, set()))
        c_treat = len(conv_counts.get(treatment_name, set()))
        cr_treat = (c_treat / n_treat) if n_treat > 0 else 0.0

        ctrl_metrics = VariantMetrics(
            variant_name=control_name,
            exposures=n_ctrl,
            conversions=c_ctrl,
            conversion_rate=cr_ctrl,
        )
        treat_metrics = VariantMetrics(
            variant_name=treatment_name,
            exposures=n_treat,
            conversions=c_treat,
            conversion_rate=cr_treat,
        )

        report = compute_uplift(control=ctrl_metrics, treatment=treat_metrics)
        report.experiment_id = exp_id
        return report
