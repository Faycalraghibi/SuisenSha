"""Model registry managing versioned recommendation models, stage transitions, and rollback."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import ARTIFACTS_DIR, SEQ_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata describing a registered model version."""

    version: int
    model_name: str
    artifact_path: str
    metrics: dict[str, float]
    stage: str = "staging"  # "production", "staging", "archived"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """File-backed and MLflow-compatible Model Registry for SuisenSha recommendation models."""

    def __init__(self, registry_dir: Path | str | None = None) -> None:
        self.registry_dir = Path(registry_dir) if registry_dir else ARTIFACTS_DIR / "registry"
        self.models_dir = self.registry_dir / "models"
        self.metadata_file = self.registry_dir / "registry.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_init()

    def _ensure_init(self) -> None:
        if not self.metadata_file.exists():
            initial_state: dict[str, Any] = {
                "versions": [],
                "production_version": None,
                "history": [],
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(initial_state, f, indent=2)

    def _load_metadata(self) -> dict[str, Any]:
        with open(self.metadata_file, encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata(self, data: dict[str, Any]) -> None:
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        artifact_path: Path | str,
        metrics: dict[str, float],
        parameters: dict[str, Any] | None = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a new candidate model version."""
        data = self._load_metadata()
        versions = data.get("versions", [])
        new_version_num = max([v["version"] for v in versions], default=0) + 1

        dest_dir = self.models_dir / f"{model_name}_v{new_version_num}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(artifact_path)
        dest_artifact = dest_dir / src_path.name
        if src_path.is_file():
            shutil.copy2(src_path, dest_artifact)
        elif src_path.is_dir():
            shutil.copytree(src_path, dest_artifact, dirs_exist_ok=True)

        mv = ModelVersion(
            version=new_version_num,
            model_name=model_name,
            artifact_path=str(dest_artifact),
            metrics=metrics,
            stage="staging",
            description=description,
            parameters=parameters or {},
        )
        versions.append(asdict(mv))
        data["versions"] = versions
        self._save_metadata(data)
        logger.info(
            "Registered model %s version %d with metrics %s",
            model_name,
            new_version_num,
            metrics,
        )
        return mv

    def get_latest_version(self, model_name: str) -> ModelVersion | None:
        """Return the latest registered version of a model."""
        data = self._load_metadata()
        matching = [v for v in data.get("versions", []) if v["model_name"] == model_name]
        if not matching:
            return None
        latest = max(matching, key=lambda v: v["version"])
        return ModelVersion(**latest)

    def get_production_version(self, model_name: str) -> ModelVersion | None:
        """Return the current production version of a model."""
        data = self._load_metadata()
        prod_ver = data.get("production_version")
        if prod_ver is None:
            # Fallback to searching stage == 'production'
            for v in data.get("versions", []):
                if v["model_name"] == model_name and v["stage"] == "production":
                    return ModelVersion(**v)
            return None
        for v in data.get("versions", []):
            if v["version"] == prod_ver and v["model_name"] == model_name:
                return ModelVersion(**v)
        return None

    def list_versions(self, model_name: str | None = None) -> list[ModelVersion]:
        """List registered model versions."""
        data = self._load_metadata()
        versions = [
            ModelVersion(**v)
            for v in data.get("versions", [])
            if model_name is None or v["model_name"] == model_name
        ]
        return sorted(versions, key=lambda v: v.version)

    def promote_to_production(self, version: int, model_name: str = "sasrec") -> ModelVersion:
        """Promote a model version to production and demote earlier production versions."""
        data = self._load_metadata()

        found = False
        target_mv: ModelVersion | None = None

        prev_prod = data.get("production_version")
        for v in data.get("versions", []):
            if v["model_name"] == model_name:
                if v["version"] == version:
                    v["stage"] = "production"
                    target_mv = ModelVersion(**v)
                    found = True
                elif v["stage"] == "production":
                    v["stage"] = "archived"

        if not found or target_mv is None:
            raise ValueError(f"Model version {version} for '{model_name}' not found.")

        data["production_version"] = version
        history = data.setdefault("history", [])
        history.append(
            {
                "action": "promote",
                "from_version": prev_prod,
                "to_version": version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_metadata(data)

        # Mirror production artifact to active default SEQ_MODEL_PATH if applicable
        if (
            model_name == "sasrec"
            and self.registry_dir == (ARTIFACTS_DIR / "registry")
            and Path(target_mv.artifact_path).exists()
        ):
            shutil.copy2(target_mv.artifact_path, SEQ_MODEL_PATH)

        logger.info("Promoted %s v%d to production (previous: %s)", model_name, version, prev_prod)
        return target_mv

    def rollback(self, model_name: str = "sasrec") -> ModelVersion:
        """Roll back to the previous production model version."""
        data = self._load_metadata()
        curr_prod = data.get("production_version")
        if curr_prod is None:
            raise RuntimeError(f"No current production model to roll back for '{model_name}'.")

        history = data.get("history", [])
        # Find previous production versions from history or versions list
        prev_versions = [
            h["from_version"]
            for h in reversed(history)
            if h.get("action") == "promote"
            and h.get("from_version") is not None
            and h["from_version"] != curr_prod
        ]

        target_version: int | None = None
        if prev_versions:
            target_version = prev_versions[0]
        else:
            # Fallback to highest version < curr_prod
            candidates = [
                v["version"]
                for v in data.get("versions", [])
                if v["model_name"] == model_name and v["version"] < curr_prod
            ]
            if candidates:
                target_version = max(candidates)

        if target_version is None:
            raise RuntimeError(
                f"No prior model version available to roll back to for '{model_name}'."
            )

        rolled_back = self.promote_to_production(version=target_version, model_name=model_name)

        # Reload data after promote_to_production updated metadata
        updated_data = self._load_metadata()
        history = updated_data.setdefault("history", [])
        history.append(
            {
                "action": "rollback",
                "from_version": curr_prod,
                "to_version": target_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_metadata(updated_data)
        logger.warning(
            "Rolled back %s from v%d to v%d",
            model_name,
            curr_prod,
            target_version,
        )
        return rolled_back
