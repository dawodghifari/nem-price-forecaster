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

.PHONY: poll
poll: install  ## Live 5-minute poller. ARGS="--once" for a single pass
	$(BIN)/python scripts/poll.py $(ARGS)

.PHONY: backfill
backfill: install  ## Backfill the NEMWeb archive. ~2.1 GB DISPATCHIS, ~20.6 GB P5MIN
	$(BIN)/python scripts/backfill.py --what mms $(ARGS)

.PHONY: backfill-history
backfill-history: install  ## Aggregated price and demand, 2015 onward
	$(BIN)/python scripts/backfill.py --what history $(ARGS)

.PHONY: backfill-weather
backfill-weather: install  ## Open-Meteo archived forecasts
	$(BIN)/python scripts/backfill.py --what weather --days 400 $(ARGS)

.PHONY: probe-weather
probe-weather: install  ## Check the weather archive is forecasts, not reanalysis
	$(BIN)/python scripts/probe_weather.py

.PHONY: gaps
gaps:  ## Days AEMO published fewer than 288 intervals, excluding archive edges
	@docker compose exec -T postgres psql -U $${POSTGRES_USER:-nemfc} -d $${POSTGRES_DB:-nemfc} \
	  -v region=$${NEMFC_REGION:-NSW1} -f - < db/queries/gaps.sql

.PHONY: sizes
sizes:  ## Row counts and on-disk size per raw table
	@docker compose exec -T postgres psql -U $${POSTGRES_USER:-nemfc} -d $${POSTGRES_DB:-nemfc} -c "\
	  SELECT relname AS table, n_live_tup AS rows, \
	         pg_size_pretty(pg_total_relation_size(relid)) AS size \
	  FROM pg_stat_user_tables WHERE schemaname = 'raw' \
	  ORDER BY pg_total_relation_size(relid) DESC;"
	@docker compose exec -T postgres psql -U $${POSTGRES_USER:-nemfc} -d $${POSTGRES_DB:-nemfc} -tAc "\
	  SELECT 'database total: ' || pg_size_pretty(pg_database_size(current_database()));"

.PHONY: counts
counts: install  ## Row counts per raw table, and ingest coverage
	@docker compose exec -T postgres psql -U $${POSTGRES_USER:-nemfc} -d $${POSTGRES_DB:-nemfc} -c "\
	  SELECT report, vintage_source, count(*) AS files, \
	         min(known_at) AS earliest, max(known_at) AS latest \
	  FROM raw.ingest_log GROUP BY 1, 2 ORDER BY 1, 2;"

.PHONY: clean
clean:  ## Remove caches and the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
