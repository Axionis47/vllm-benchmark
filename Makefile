# Makefile for LLM Inference Benchmark Suite

.PHONY: setup lint typecheck test docker-build smoke clean help

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: $(VENV)/bin/activate ## Create virtual environment and install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

lint: ## Run linting with ruff
	$(RUFF) check bench/ server/ tests/
	$(RUFF) format --check bench/ server/ tests/

lint-fix: ## Run linting and fix issues
	$(RUFF) check --fix bench/ server/ tests/
	$(RUFF) format bench/ server/ tests/

typecheck: ## Run type checking with mypy
	$(MYPY) bench/ server/ tests/

test: ## Run unit tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=bench --cov=server --cov-report=term-missing

docker-build: ## Build Docker image
	docker build -t vllm-bench:latest -f server/docker/Dockerfile server/docker/

smoke: ## Run smoke test with mock server
	@echo "Running smoke test..."
	@mkdir -p results/smoke_test
	$(PYTHON) -m bench.smoke_runner
	@echo "Smoke test complete. Check results/smoke_test/ and docs/REPORT_SMOKE.md"

clean: ## Clean up generated files
	rm -rf $(VENV)
	rm -rf results/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf **/__pycache__/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Dataset targets (now using HuggingFace datasets)
BENCH_DATASETS := bench/datasets
RAW_DIR := $(BENCH_DATASETS)/raw
PROCESSED_DIR := $(BENCH_DATASETS)/processed

hf-setup: ## Verify HuggingFace datasets library installed
	@echo "Checking HuggingFace datasets..."
	@pip install --quiet datasets transformers
	@python3 -c "from datasets import load_dataset; print('HuggingFace datasets OK')"

kaggle-setup: ## Legacy: Verify Kaggle setup (optional)
	@echo "Note: Using HuggingFace datasets, Kaggle credentials optional"
	@test -f ~/.kaggle/kaggle.json && echo "Kaggle credentials found" || echo "Kaggle credentials not found (OK)"

datasets: hf-setup ## Download and preprocess HuggingFace datasets into prompts.jsonl
	@echo "Creating directories..."
	@mkdir -p $(RAW_DIR) $(PROCESSED_DIR)
	@echo "Running dataset preprocessing (arXiv, HotpotQA, OASST1)..."
	python3 bench/datasets/preprocess.py
	@echo "Dataset preprocessing complete!"
	@ls -lh $(PROCESSED_DIR)/prompts.jsonl

# CI targets
ci-lint: setup lint ## CI: setup and lint
ci-typecheck: setup typecheck ## CI: setup and typecheck
ci-test: setup test ## CI: setup and test
ci-all: setup lint typecheck test ## CI: run all checks

# =============================================================================
# BENCHMARK MATRIX TARGETS (VM only)
# =============================================================================

RESULTS_DIR := results
PROMPTS_FILE := bench/datasets/processed/prompts.jsonl

smoke-matrix: ## Run smoke gate for all configs C1-C8
	@echo "============================================================"
	@echo "RUNNING SMOKE MATRIX (C1-C8)"
	@echo "============================================================"
	python3 bench/runner/run_smoke_matrix.py \
		--results-dir $(RESULTS_DIR) \
		--prompts $(PROMPTS_FILE)

full-matrix: ## Run full benchmark matrix (runs smoke gate first)
	@echo "============================================================"
	@echo "RUNNING FULL BENCHMARK MATRIX (C1-C8)"
	@echo "============================================================"
	python3 bench/runner/run_full_matrix.py \
		--results-dir $(RESULTS_DIR) \
		--prompts $(PROMPTS_FILE)

full-matrix-skip-smoke: ## Run full matrix without smoke gate (dangerous!)
	@echo "WARNING: Skipping smoke gate - use at your own risk!"
	python3 bench/runner/run_full_matrix.py \
		--results-dir $(RESULTS_DIR) \
		--prompts $(PROMPTS_FILE) \
		--skip-smoke

report-matrix: ## Generate final report from latest full matrix results
	@echo "Generating matrix report..."
	@LATEST=$$(ls -td $(RESULTS_DIR)/full_matrix_* 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "ERROR: No full_matrix results found in $(RESULTS_DIR)"; \
		exit 1; \
	fi; \
	echo "Using results from: $$LATEST"; \
	python3 bench/analysis/aggregate_matrix.py \
		--results-dir "$$LATEST" \
		--output docs/REPORT_FINAL.md
	@echo "Report generated: docs/REPORT_FINAL.md"

# Quick targets for individual configs
serve-c1: ## Start vLLM with C1 config
	python3 server/vllm/launcher.py --config server/vllm/configs/C1.yaml --action start

serve-c2: ## Start vLLM with C2 config
	python3 server/vllm/launcher.py --config server/vllm/configs/C2.yaml --action start

stop-vllm: ## Stop vLLM container
	docker stop vllm-server 2>/dev/null || true
	docker rm -f vllm-server 2>/dev/null || true

vllm-logs: ## Show vLLM container logs
	docker logs -f vllm-server

