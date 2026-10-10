from __future__ import annotations

import logging
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.config import PipelineConfig, tracking_cfg

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


@contextmanager
def start_run(
    run_name: str | None = None,
    cfg: PipelineConfig | None = None,
    tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Start an MLflow run if tracking is enabled, or a no-op context manager if disabled."""
    if not tracking_cfg.enabled:
        yield None
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_cfg.tracking_uri)
    mlflow.set_experiment(tracking_cfg.experiment_name)

    run_tags = {"git_commit": get_git_commit()}
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
        if cfg:
            log_config(cfg)
        yield run


def log_config(cfg: PipelineConfig) -> None:
    """Flatten and log pipeline configuration parameters to MLflow."""
    if not tracking_cfg.enabled:
        return

    import mlflow

    flat_params: dict[str, Any] = {"seed": cfg.seed}
    sub_cfgs = [
        ("embedding", asdict(cfg.embedding)),
        ("sasrec", asdict(cfg.sasrec)),
        ("rag", asdict(cfg.rag)),
        ("eval", asdict(cfg.eval)),
        ("split", asdict(cfg.split)),
        ("tracking", asdict(cfg.tracking)),
    ]
    for section_name, sub_dict in sub_cfgs:
        for k, v in sub_dict.items():
            flat_params[f"{section_name}.{k}"] = str(v)

    mlflow.log_params(flat_params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log evaluation metrics to MLflow."""
    if not tracking_cfg.enabled:
        return

    import mlflow

    clean_metrics = {k: float(v) for k, v in metrics.items()}
    mlflow.log_metrics(clean_metrics, step=step)


def log_artifact(local_path: Path | str, artifact_path: str | None = None) -> None:
    """Log an artifact file to MLflow."""
    if not tracking_cfg.enabled:
        return

    import mlflow

    path = Path(local_path)
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
