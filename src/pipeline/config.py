from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

# Two levels up: src/pipeline/config.py → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH: Path = CONFIGS_DIR / "default.toml"

DATA_DIR: Path = _PROJECT_ROOT / "data"
ARTIFACTS_DIR: Path = _PROJECT_ROOT / "artifacts"
OUTPUTS_DIR: Path = _PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = _PROJECT_ROOT / "reports"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_ZIP: Path = DATA_DIR / "ml-100k.zip"
MOVIELENS_DIR: Path = DATA_DIR / "ml-100k"

INTERACTIONS_CSV: Path = DATA_DIR / "interactions.csv"
MOVIES_CSV: Path = DATA_DIR / "movies.csv"
USER_SEQUENCES_PKL: Path = DATA_DIR / "user_sequences.pkl"

FAISS_INDEX_PATH: Path = ARTIFACTS_DIR / "faiss_index.bin"
EMBEDDINGS_NPY: Path = ARTIFACTS_DIR / "item_embeddings.npy"
ITEM_IDS_NPY: Path = ARTIFACTS_DIR / "item_ids.npy"

SEQ_MODEL_PATH: Path = ARTIFACTS_DIR / "sasrec_model.pt"
RAG_CACHE_DB: Path = ARTIFACTS_DIR / "rag_cache.db"

SEED: int = 42


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    top_k: int = 10


@dataclass(frozen=True)
class SASRecConfig:
    max_seq_len: int = 50
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    patience: int = 5
    window_stride: int = 10


@dataclass(frozen=True)
class RAGConfig:
    llm_model_name: str = "distilgpt2"
    llm_backend: str = "local"
    max_new_tokens: int = 300
    retrieval_k: int = 20
    api_base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    k_values: tuple[int, ...] = (5, 10)


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "leave_two_out"
    min_rating: float = 3.5
    min_interactions: int = 5
    temporal_val_ratio: float = 0.1
    temporal_test_ratio: float = 0.1


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    experiment_name: str = "suisensha-benchmark"
    tracking_uri: str = "artifacts/mlruns"


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: int = 5
    k_values: tuple[int, ...] = (5, 10)
    output_dir: str = "reports"


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sasrec: SASRecConfig = field(default_factory=SASRecConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def _cast_val(raw_val: str, target_type: type) -> Any:
    if target_type is bool:
        return raw_val.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(raw_val)
    if target_type is float:
        return float(raw_val)
    if target_type is tuple or target_type is list:
        items = [x.strip() for x in raw_val.strip("[]()").split(",") if x.strip()]
        return tuple(int(x) for x in items)
    return raw_val


def load_config(
    config_path: Path | str | None = None,
    overrides: list[str] | None = None,
) -> PipelineConfig:
    """Load config from a TOML file with optional dotted CLI overrides (e.g. sasrec.epochs=40)."""
    raw_dict: dict[str, Any] = {}
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.is_file():
        with open(path, "rb") as f:
            raw_dict = tomllib.load(f)

    # Apply dotted key overrides: --set section.param=value
    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key_path, val_str = override.split("=", 1)
            parts = key_path.strip().split(".")
            curr = raw_dict
            for p in parts[:-1]:
                curr = curr.setdefault(p, {})
            curr[parts[-1]] = val_str

    gen = raw_dict.get("general", {})
    seed = int(raw_dict.get("seed", gen.get("seed", SEED)))

    emb_raw = raw_dict.get("embedding", {})
    emb = EmbeddingConfig(
        model_name=str(emb_raw.get("model_name", EmbeddingConfig.model_name)),
        dim=int(emb_raw.get("dim", EmbeddingConfig.dim)),
        top_k=int(emb_raw.get("top_k", EmbeddingConfig.top_k)),
    )

    sas_raw = raw_dict.get("sasrec", {})
    sas = SASRecConfig(
        max_seq_len=int(sas_raw.get("max_seq_len", SASRecConfig.max_seq_len)),
        hidden_dim=int(sas_raw.get("hidden_dim", SASRecConfig.hidden_dim)),
        num_heads=int(sas_raw.get("num_heads", SASRecConfig.num_heads)),
        num_layers=int(sas_raw.get("num_layers", SASRecConfig.num_layers)),
        dropout=float(sas_raw.get("dropout", SASRecConfig.dropout)),
        lr=float(sas_raw.get("lr", SASRecConfig.lr)),
        epochs=int(sas_raw.get("epochs", SASRecConfig.epochs)),
        batch_size=int(sas_raw.get("batch_size", SASRecConfig.batch_size)),
        patience=int(sas_raw.get("patience", SASRecConfig.patience)),
        window_stride=int(sas_raw.get("window_stride", SASRecConfig.window_stride)),
    )

    rag_raw = raw_dict.get("rag", {})
    rag = RAGConfig(
        llm_model_name=str(rag_raw.get("llm_model_name", RAGConfig.llm_model_name)),
        llm_backend=str(rag_raw.get("llm_backend", RAGConfig.llm_backend)),
        max_new_tokens=int(rag_raw.get("max_new_tokens", RAGConfig.max_new_tokens)),
        retrieval_k=int(rag_raw.get("retrieval_k", RAGConfig.retrieval_k)),
        api_base_url=rag_raw.get("api_base_url") or None,
        api_key=rag_raw.get("api_key") or None,
    )

    eval_raw = raw_dict.get("eval", {})
    k_vals = eval_raw.get("k_values", EvalConfig.k_values)
    if isinstance(k_vals, list):
        k_vals = tuple(int(x) for x in k_vals)
    elif isinstance(k_vals, str):
        k_vals = tuple(int(x.strip()) for x in k_vals.strip("[]()").split(",") if x.strip())
    ev = EvalConfig(k_values=k_vals)

    split_raw = raw_dict.get("split", {})
    split = SplitConfig(
        strategy=str(split_raw.get("strategy", SplitConfig.strategy)),
        min_rating=float(split_raw.get("min_rating", SplitConfig.min_rating)),
        min_interactions=int(split_raw.get("min_interactions", SplitConfig.min_interactions)),
        temporal_val_ratio=float(
            split_raw.get("temporal_val_ratio", SplitConfig.temporal_val_ratio)
        ),
        temporal_test_ratio=float(
            split_raw.get("temporal_test_ratio", SplitConfig.temporal_test_ratio)
        ),
    )

    track_raw = raw_dict.get("tracking", {})
    track_enabled = track_raw.get("enabled", TrackingConfig.enabled)
    if isinstance(track_enabled, str):
        track_enabled = track_enabled.lower() in ("true", "1", "yes")
    tracking = TrackingConfig(
        enabled=bool(track_enabled),
        experiment_name=str(track_raw.get("experiment_name", TrackingConfig.experiment_name)),
        tracking_uri=str(track_raw.get("tracking_uri", TrackingConfig.tracking_uri)),
    )

    bench_raw = raw_dict.get("benchmark", {})
    bench_k = bench_raw.get("k_values", BenchmarkConfig.k_values)
    if isinstance(bench_k, list):
        bench_k = tuple(int(x) for x in bench_k)
    elif isinstance(bench_k, str):
        bench_k = tuple(int(x.strip()) for x in bench_k.strip("[]()").split(",") if x.strip())
    benchmark = BenchmarkConfig(
        seeds=int(bench_raw.get("seeds", BenchmarkConfig.seeds)),
        k_values=bench_k,
        output_dir=str(bench_raw.get("output_dir", BenchmarkConfig.output_dir)),
    )

    return PipelineConfig(
        seed=seed,
        embedding=emb,
        sasrec=sas,
        rag=rag,
        eval=ev,
        split=split,
        tracking=tracking,
        benchmark=benchmark,
    )


# Active default configuration singletons
_default_cfg = load_config()
pipeline_cfg: PipelineConfig = _default_cfg
embedding_cfg: EmbeddingConfig = _default_cfg.embedding
sasrec_cfg: SASRecConfig = _default_cfg.sasrec
rag_cfg: RAGConfig = _default_cfg.rag
eval_cfg: EvalConfig = _default_cfg.eval
split_cfg: SplitConfig = _default_cfg.split
tracking_cfg: TrackingConfig = _default_cfg.tracking
benchmark_cfg: BenchmarkConfig = _default_cfg.benchmark


def set_active_config(cfg: PipelineConfig) -> None:
    global \
        pipeline_cfg, \
        embedding_cfg, \
        sasrec_cfg, \
        rag_cfg, \
        eval_cfg, \
        split_cfg, \
        tracking_cfg, \
        benchmark_cfg, \
        SEED
    pipeline_cfg = cfg
    embedding_cfg = cfg.embedding
    sasrec_cfg = cfg.sasrec
    rag_cfg = cfg.rag
    eval_cfg = cfg.eval
    split_cfg = cfg.split
    tracking_cfg = cfg.tracking
    benchmark_cfg = cfg.benchmark
    SEED = cfg.seed


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
