from __future__ import annotations

from pipeline.config import PipelineConfig, TrackingConfig, set_active_config
from pipeline.tracking import get_git_commit, log_artifact, log_config, log_metrics, start_run


def test_git_commit_lookup():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_tracking_noop_when_disabled():
    cfg = PipelineConfig(tracking=TrackingConfig(enabled=False))
    set_active_config(cfg)

    with start_run(run_name="test-run") as run:
        assert run is None
        log_config(cfg)
        log_metrics({"ndcg@10": 0.05})
        log_artifact("nonexistent.txt")


def test_tracking_enabled(tmp_path):
    tracking_dir = str(tmp_path / "mlruns")
    cfg = PipelineConfig(tracking=TrackingConfig(enabled=True, tracking_uri=tracking_dir))
    set_active_config(cfg)

    artifact_file = tmp_path / "model.txt"
    artifact_file.write_text("dummy model weight", encoding="utf-8")

    with start_run(run_name="active-test-run", cfg=cfg) as run:
        assert run is not None
        log_metrics({"hitrate@10": 0.12, "ndcg@10": 0.08}, step=1)
        log_artifact(artifact_file)

    # Disable again
    set_active_config(PipelineConfig(tracking=TrackingConfig(enabled=False)))
