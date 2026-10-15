SHELL := /bin/bash
PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
-include .env
export
DEFAULT_DSN := postgresql://rag:rag@localhost:5432/rag

.PHONY: up down setup setup-model corpus demo ingest eval search test test-integration lint clean

up: ## Postgres + pgvector via docker compose
	docker compose up -d --wait

down:
	docker compose down -v

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt -e .

setup: $(BIN)/python

setup-model: setup ## optional: sentence-transformers + CPU torch (USE_ST_MODEL=1)
	$(BIN)/pip install -r requirements-model.txt

corpus: setup ## regenerate the synthetic corpus (md, html, pdf) and eval questions
	$(BIN)/python scripts/generate_corpus.py

demo: setup ## ingest, re-ingest (no-op), edit one doc, re-ingest (only it), run eval
	PG_DSN=$${PG_DSN:-$(DEFAULT_DSN)} $(BIN)/python -m rag_ingest demo

ingest: setup
	PG_DSN=$${PG_DSN:-$(DEFAULT_DSN)} $(BIN)/python -m rag_ingest ingest

eval: setup
	PG_DSN=$${PG_DSN:-$(DEFAULT_DSN)} $(BIN)/python -m rag_ingest eval

search: setup ## make search Q="prior authorization turnaround"
	PG_DSN=$${PG_DSN:-$(DEFAULT_DSN)} $(BIN)/python -m rag_ingest search "$(Q)"

test: setup ## unit tests; the Postgres test is skipped unless PG_DSN is set
	$(BIN)/pytest

test-integration: setup
	$(BIN)/pytest -m integration -v

lint: setup
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

clean:
	rm -rf output .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
