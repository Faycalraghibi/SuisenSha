"""Export the SuisenSha FastAPI OpenAPI JSON specification to docs/openapi.json."""

from __future__ import annotations

from pathlib import Path

from pipeline.api.main import app


def export_openapi(output_path: Path | str = "docs/openapi.json") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    openapi_schema = app.openapi()
    num_paths = len(openapi_schema.get("paths", {}))
    print(f"Exported OpenAPI spec to {out.resolve()} ({num_paths} paths)")
    return out


if __name__ == "__main__":
    export_openapi()
