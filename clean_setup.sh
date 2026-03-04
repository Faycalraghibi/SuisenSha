#!/usr/bin/env bash
set -euo pipefail

VENV_NAME="suisensha-venv"

echo "[1/4] Cleaning previous environment..."
rm -rf "$VENV_NAME" __pycache__ .pytest_cache .mypy_cache .ruff_cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "[2/4] Creating virtual environment: $VENV_NAME"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

echo "[3/4] Installing project with dev dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

echo "[4/4] Installing pre-commit hooks..."
pre-commit install

echo ""
echo "Setup complete! Activate with:"
echo "   source $VENV_NAME/bin/activate"
