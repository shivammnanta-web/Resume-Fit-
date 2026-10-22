.PHONY: setup test lint format typecheck run eval up down clean fetch-esco

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff
MYPY := uv run mypy

setup:
	uv sync --extra dev

test:
	$(PYTEST) tests/

test-cov:
	$(PYTEST) --cov=core --cov=evaluation --cov-report=term-missing --cov-report=html tests/

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:
	$(MYPY) core resumefit

run:
	$(PYTHON) -m resumefit.app

eval:
	$(PYTHON) -m evaluation.harness

fetch-esco:
	$(PYTHON) scripts/fetch_esco.py

up:
	docker compose up -d

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ build/
