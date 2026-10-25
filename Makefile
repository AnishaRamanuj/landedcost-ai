# Task shortcuts. Every target runs through uv so the locked environment is used.
UV ?= uv
# Defaults to $LANDEDCOST_TEST_DATABASE_URL if exported, else empty: integration tests then
# use Testcontainers (Docker) or skip cleanly. Point it at a disposable database to run them
# locally: make test TEST_DB=postgresql://dev:dev@localhost:5432/landedcost_test
TEST_DB ?= $(LANDEDCOST_TEST_DATABASE_URL)
EVAL_OUT ?= docs/eval/hash-embedder.md
EVAL_DETAILS ?= docs/eval/hash-embedder-details.jsonl

.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test test-unit test-integration check migrate ingest \
        eval rrf-sweep stats serve hnsw-recall docker-up docker-down clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install the locked environment (dev tools included)
	$(UV) sync --locked

fmt: ## Format code
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

lint: ## Ruff lint + format check
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

typecheck: ## mypy --strict
	$(UV) run mypy

test-unit: ## Unit tests (no services needed)
	$(UV) run pytest tests/unit

test-integration: ## PostgreSQL integration tests (TEST_DB=... or Docker for Testcontainers)
	LANDEDCOST_TEST_DATABASE_URL=$(TEST_DB) $(UV) run pytest tests/integration

test: ## All tests with coverage
	LANDEDCOST_TEST_DATABASE_URL=$(TEST_DB) $(UV) run pytest --cov=landedcost --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

migrate: ## Create/upgrade the schema in DATABASE_URL
	$(UV) run landedcost migrate

ingest: ## Load the shipped HTS snapshot into DATABASE_URL
	$(UV) run landedcost ingest-hts

eval: ## Run the retrieval + answer eval and write the report and per-item details
	$(UV) run landedcost eval --answers --output $(EVAL_OUT) --details $(EVAL_DETAILS)

rrf-sweep: ## RRF vector-weight sweep with a two-fold check
	$(UV) run python scripts/rrf_weight_sweep.py --output docs/eval/rrf-weights.md

stats: ## Offline statistics for the HTS snapshot
	$(UV) run landedcost stats

hnsw-recall: ## HNSW recall vs exact scan for the eval queries
	$(UV) run python scripts/hnsw_recall.py --ef-search 40 100 200 400 --rebuild 16/64 16/200

serve: ## Run the API on :8201
	$(UV) run landedcost serve

docker-up: ## Build and start Postgres + API with docker compose
	docker compose up --build

docker-down: ## Stop docker compose and remove volumes
	docker compose down -v

clean: ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
