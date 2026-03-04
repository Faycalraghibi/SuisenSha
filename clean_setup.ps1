$ErrorActionPreference = "Stop"

$VenvName = "suisensha-venv"

Write-Host "[1/4] Cleaning previous environment..." -ForegroundColor Cyan
if (Test-Path $VenvName) { Remove-Item -Recurse -Force $VenvName }
if (Test-Path __pycache__) { Remove-Item -Recurse -Force __pycache__ }
if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache }
if (Test-Path .mypy_cache) { Remove-Item -Recurse -Force .mypy_cache }
if (Test-Path .ruff_cache) { Remove-Item -Recurse -Force .ruff_cache }
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Creating virtual environment: $VenvName" -ForegroundColor Cyan
python -m venv $VenvName
& ".\$VenvName\Scripts\Activate.ps1"

Write-Host "[3/4] Installing project with dev dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -e ".[dev]"

Write-Host "[4/4] Installing pre-commit hooks..." -ForegroundColor Cyan
pre-commit install

Write-Host ""
Write-Host "Setup complete! Activate with:" -ForegroundColor Green
Write-Host ('   .\' + $VenvName + '\Scripts\Activate.ps1') -ForegroundColor Yellow
