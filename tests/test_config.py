from __future__ import annotations

from pipeline.config import DEFAULT_CONFIG_PATH, load_config, set_active_config


def test_default_config_loading():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.embedding.dim == 384
    assert cfg.sasrec.epochs == 20
    assert cfg.sasrec.max_seq_len == 50
    assert cfg.sasrec.patience == 5
    assert cfg.split.strategy == "leave_two_out"
    assert cfg.split.min_rating == 3.5
    assert cfg.tracking.enabled is False


def test_cli_overrides():
    overrides = [
        "sasrec.epochs=40",
        "sasrec.lr=0.005",
        "seed=123",
        "split.strategy=temporal",
        "tracking.enabled=true",
    ]
    cfg = load_config(overrides=overrides)
    assert cfg.sasrec.epochs == 40
    assert cfg.sasrec.lr == 0.005
    assert cfg.seed == 123
    assert cfg.split.strategy == "temporal"
    assert cfg.tracking.enabled is True


def test_custom_toml_file(tmp_path):
    custom_toml = tmp_path / "custom.toml"
    custom_toml.write_text(
        """
[general]
seed = 999

[sasrec]
epochs = 15
max_seq_len = 30
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path=custom_toml)
    assert cfg.seed == 999
    assert cfg.sasrec.epochs == 15
    assert cfg.sasrec.max_seq_len == 30
    # Inherits default fallback
    assert cfg.embedding.dim == 384


def test_set_active_config():
    cfg = load_config(overrides=["sasrec.epochs=55"])
    set_active_config(cfg)
    from pipeline import config as cfg_module

    assert cfg_module.sasrec_cfg.epochs == 55
    # Reset back to default
    default_cfg = load_config(DEFAULT_CONFIG_PATH)
    set_active_config(default_cfg)
    assert cfg_module.sasrec_cfg.epochs == 20
