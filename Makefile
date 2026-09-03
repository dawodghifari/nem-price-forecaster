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

DBT_VENV := .venv-dbt
DBT_BIN := $(DBT_VENV)/bin
DBT := cd dbt && DBT_PROFILES_DIR=. ../$(DBT_BIN)/dbt

$(DBT_BIN)/dbt:
	$(PY) -m venv $(DBT_VENV)
	$(DBT_BIN)/pip install --upgrade pip
	$(DBT_BIN)/pip install -r requirements/dbt.txt

.PHONY: dbt-install
dbt-install: $(DBT_BIN)/dbt  ## Create the dbt virtualenv and fetch packages
	$(DBT) deps

.PHONY: dbt-parse
dbt-parse: dbt-install  ## Validate the project without touching the database
	$(DBT) parse

.PHONY: dbt-build
dbt-build: dbt-install  ## Run every model and test
	$(DBT) build

.PHONY: dbt-test
dbt-test: dbt-install  ## Run tests only
	$(DBT) test

.PHONY: dbt-fresh
dbt-fresh: dbt-install  ## Source freshness. Warns when the poller is stopped
	$(DBT) source freshness

.PHONY: leakage
leakage: install  ## Verify the feature table against raw, independently of dbt
	$(BIN)/pytest tests/test_leakage_against_raw.py -v

.PHONY: baselines
baselines: install  ## Day 5: measure the three baselines, write docs/day5_baselines.md
	$(BIN)/python scripts/baselines.py $(ARGS)

.PHONY: model-install
model-install: install  ## Add LightGBM to the virtualenv
	$(BIN)/pip install -q -r requirements/model.txt

.PHONY: train
train: model-install  ## Day 6: train, backtest, write docs/day6_model.md
	$(BIN)/python scripts/train.py $(ARGS)

.PHONY: fit
fit: model-install  ## Fit the served model on all data, write models/
	$(BIN)/python scripts/fit_production.py $(ARGS)

.PHONY: serve-install
serve-install: install  ## Add the serving dependencies
	$(BIN)/pip install -q -r requirements/serve.txt

.PHONY: serve
serve: serve-install  ## Run the API locally on :8000
	$(BIN)/uvicorn nemfc.serve:app --app-dir src --reload --port 8000

.PHONY: refresh-serving
refresh-serving: dbt-install  ## Rebuild the serving models. Seconds; run on the poller's cadence
	$(DBT) run --select tag:serving

.PHONY: api
api:  ## Build and run the API container, waiting until it answers
	docker compose up -d --build api
	@printf 'waiting for the API'
	@for i in $$(seq 1 40); do \
	  if curl -sf -o /dev/null http://localhost:$${API_PORT:-8000}/health; then \
	    echo " ready"; exit 0; fi; \
	  printf '.'; sleep 1; done; \
	echo " gave up after 40s"; docker compose logs --tail=40 api; exit 1
	@echo "API: http://localhost:$${API_PORT:-8000}/docs"

.PHONY: smoke
smoke:  ## Phase 1 exit condition: a live prediction from real data
	@curl -sf http://localhost:$${API_PORT:-8000}/health | python3 -m json.tool
	@echo
	@curl -sf http://localhost:$${API_PORT:-8000}/forecast/latest | python3 -m json.tool

.PHONY: dbt-docs
dbt-docs: dbt-install  ## Generate and serve the lineage graph
	$(DBT) docs generate && $(DBT) docs serve

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
