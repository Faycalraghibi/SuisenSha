"""Drift detection module for interactions, catalog distributions, and recommendation scores."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Summary of drift detection across interactions and model scores."""

    timestamp: str
    interaction_drift_detected: bool
    score_drift_detected: bool
    ks_statistic_interactions: float
    p_value_interactions: float
    ks_statistic_scores: float
    p_value_scores: float
    psi_score: float
    alert: bool
    alert_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    eps: float = 1e-4,
) -> float:
    """Calculate the Population Stability Index (PSI) between two 1D numerical distributions."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on reference/expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    # Ensure monotonic unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / max(len(expected), 1) + eps
    act_pct = act_counts / max(len(actual), 1) + eps

    # Re-normalize
    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


class DriftDetector:
    """Detects statistical distribution shift in user activity and recommendation scores."""

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.25,
    ) -> None:
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def evaluate_drift(
        self,
        reference_interaction_counts: list[int] | np.ndarray,
        current_interaction_counts: list[int] | np.ndarray,
        reference_scores: list[float] | np.ndarray | None = None,
        current_scores: list[float] | np.ndarray | None = None,
    ) -> DriftReport:
        """Run two-sample Kolmogorov-Smirnov tests and PSI against baseline distributions."""
        ref_int = np.asarray(reference_interaction_counts, dtype=float)
        cur_int = np.asarray(current_interaction_counts, dtype=float)

        alert_reasons: list[str] = []

        # 1. Interaction count drift
        if len(ref_int) > 0 and len(cur_int) > 0:
            ks_int = stats.ks_2samp(ref_int, cur_int)
            ks_stat_int = float(ks_int.statistic)
            p_val_int = float(ks_int.pvalue)
            int_drift = p_val_int < self.p_value_threshold
            if int_drift:
                alert_reasons.append(
                    f"Interaction drift detected (KS: {ks_stat_int:.4f}, p: {p_val_int:.2e})"
                )
        else:
            ks_stat_int = 0.0
            p_val_int = 1.0
            int_drift = False

        # 2. Score distribution drift and PSI
        ref_sc = np.asarray(reference_scores if reference_scores is not None else [], dtype=float)
        cur_sc = np.asarray(current_scores if current_scores is not None else [], dtype=float)

        if len(ref_sc) > 0 and len(cur_sc) > 0:
            ks_sc = stats.ks_2samp(ref_sc, cur_sc)
            ks_stat_sc = float(ks_sc.statistic)
            p_val_sc = float(ks_sc.pvalue)
            psi = calculate_psi(ref_sc, cur_sc)
            score_drift = (p_val_sc < self.p_value_threshold) or (psi >= self.psi_threshold)
            if score_drift:
                alert_reasons.append(
                    f"Score drift (KS: {ks_stat_sc:.4f}, p: {p_val_sc:.2e}, PSI: {psi:.4f})"
                )

        else:
            ks_stat_sc = 0.0
            p_val_sc = 1.0
            psi = 0.0
            score_drift = False

        alert = len(alert_reasons) > 0
        now_iso = datetime.now(timezone.utc).isoformat()

        report = DriftReport(
            timestamp=now_iso,
            interaction_drift_detected=int_drift,
            score_drift_detected=score_drift,
            ks_statistic_interactions=ks_stat_int,
            p_value_interactions=p_val_int,
            ks_statistic_scores=ks_stat_sc,
            p_value_scores=p_val_sc,
            psi_score=psi,
            alert=alert,
            alert_reasons=alert_reasons,
        )

        if alert:
            logger.warning(
                "MLOps Alert: Data or prediction drift detected! Reasons: %s", alert_reasons
            )
        else:
            logger.info("Drift check passed: no significant distribution shift detected.")

        return report
