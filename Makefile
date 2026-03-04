.PHONY: install dev lint type-check test run run-phase clean help pre-commit batch-rag

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

test:
	pytest

run:
	python -m pipeline.cli

run-phase:
	python -m pipeline.cli --phase $(PHASE)

run-api:
	python -m pipeline.cli --phase 6

run-ui:
	python -m pipeline.cli --phase 7

pre-commit:
	pre-commit install

batch-rag:
	python -m pipeline.cli --phase 8

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf data/ artifacts/ outputs/*.png
