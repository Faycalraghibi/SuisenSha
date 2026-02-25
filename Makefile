.PHONY: install dev lint type-check test run run-phase clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev]"

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Auto-format with ruff
	ruff format src/ tests/

type-check: ## Run mypy
	mypy src/

test: ## Run test suite
	pytest

run: ## Run the full pipeline (all phases)
	python -m pipeline.cli

run-phase: ## Run a specific phase (usage: make run-phase PHASE=2)
	python -m pipeline.cli --phase $(PHASE)

clean: ## Remove artefacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/ artifacts/ outputs/*.png
