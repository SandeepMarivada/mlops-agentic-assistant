# ── MLOps Agentic AI — Makefile ───────────────────────────────────────────────
# Common commands for local development and deployment.
# Usage: make <target>

.PHONY: install run-api run-ui run test \
        docker-build docker-up docker-down docker-logs \
        clean help

# ── Local development ─────────────────────────────────────────────────────────

install:                          ## Install all Python dependencies
	pip install -r requirements.txt

run-api:                          ## Start FastAPI server (with hot reload)
	python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

run-ui:                           ## Start Streamlit UI
	python -m streamlit run ui.py --server.port 8501 \
		--server.headless true --browser.gatherUsageStats false

run: run-api                      ## Alias for run-api

# ── Tests ─────────────────────────────────────────────────────────────────────

test:                             ## Run all tests with pytest
	pytest tests/ -v --tb=short

test-ci:                          ## Run tests in CI mode (no API calls)
	pytest tests/ -v --tb=short -k "not chat"

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build:                     ## Build both Docker images
	docker compose build

docker-up:                        ## Start all containers in background
	docker compose up -d --build

docker-down:                      ## Stop and remove all containers
	docker compose down

docker-logs:                      ## Tail logs from all containers
	docker compose logs -f

docker-restart:                   ## Rebuild and restart containers
	docker compose down && docker compose up -d --build

docker-status:                    ## Show running containers
	docker compose ps

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:                            ## Remove ChromaDB, logs, and pycache
	rm -rf chroma_db/ __pycache__/ .pytest_cache/ *.log
	find . -name "*.pyc" -delete

# ── Help ──────────────────────────────────────────────────────────────────────

help:                             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
