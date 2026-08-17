.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help
help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements/dev.txt

.PHONY: install
install: $(BIN)/python  ## Create the virtualenv and install dependencies

.PHONY: up
up:  ## Start Postgres and Adminer
	@test -f .env || cp .env.example .env
	docker compose up -d
	@echo "Adminer: http://localhost:$${ADMINER_PORT:-8080} (server 'postgres')"

.PHONY: down
down:  ## Stop the stack, keeping the volume
	docker compose down

.PHONY: nuke
nuke:  ## Stop the stack and delete the database volume
	docker compose down -v

.PHONY: psql
psql:  ## Open a shell on the database
	docker compose exec postgres psql -U $${POSTGRES_USER:-nemfc} -d $${POSTGRES_DB:-nemfc}

.PHONY: verify
verify: install  ## Day 1 exit condition: confirm the MMS layout against live files
	$(BIN)/python scripts/verify_formats.py

.PHONY: test
test: install  ## Run the offline test suite
	$(BIN)/pytest

.PHONY: lint
lint: install  ## Lint and format-check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: fmt
fmt: install  ## Reformat
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

.PHONY: backfill
backfill:  ## Download the NEMWeb archive (Day 2)
	@echo "Not built yet — Day 2."
	@exit 1

.PHONY: clean
clean:  ## Remove caches and the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
