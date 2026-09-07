COMPOSE ?= docker compose
PY := $(shell command -v .venv/bin/python >/dev/null 2>&1 && echo .venv/bin/python || echo python3)

POSTGRES_USER ?= memo
POSTGRES_PASSWORD ?= memo
POSTGRES_DB ?= memo
POSTGRES_PORT ?= 5432

TEST_DB_URL := postgresql+psycopg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/memo_test

.PHONY: help db-up db-down db-logs db-shell db-reset migrate test test-postgres

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

db-up: ## Start the local PostgreSQL + pgvector container
	$(COMPOSE) up -d

db-down: ## Stop the container; the data volume is kept
	$(COMPOSE) down

db-logs: ## Tail the postgres container logs
	$(COMPOSE) logs -f postgres

db-shell: ## Open a psql shell against the local dev database (memo)
	$(COMPOSE) exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

db-reset: ## DESTRUCTIVE: drop the data volume (memo + memo_test) and start fresh
	@echo "This deletes ALL data in the local compose volume (memo and memo_test)."
	@read -p "Type 'reset' to confirm: " confirmation; \
	test "$$confirmation" = "reset" || { echo "Aborted."; exit 1; }
	$(COMPOSE) down -v
	$(COMPOSE) up -d

migrate: ## Apply Alembic migrations to the local dev database (memo); requires db-up
	$(PY) -m alembic upgrade head

test: ## Run the default suite on in-memory SQLite (no database required)
	$(PY) -m pytest

test-postgres: ## Run the whole suite against memo_test (postgres profile); requires db-up
	MEMO_TEST_DATABASE_URL=$(TEST_DB_URL) $(PY) -m pytest
