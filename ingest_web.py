"""
Web Ingestion Script — Fetch real content from official MLOps documentation.
Saves fetched content to data/ folder, then re-indexes into ChromaDB.

Usage:
    python ingest_web.py              # fetch all sources
    python ingest_web.py --list       # show all sources without fetching
    python ingest_web.py --source mlflow  # fetch only mlflow sources

After running, restart the server OR call POST /index-docs to re-index.
"""

import argparse
import os
import time
import requests
from bs4 import BeautifulSoup

DATA_DIR = "./data"

# ── Official documentation URLs ───────────────────────────────────────────────
# Each entry: (filename, url, description)

SOURCES = {

    # GitHub raw markdown — always plain text, no JS rendering needed
    "mlflow": [
        ("mlflow_readme.txt",        "https://raw.githubusercontent.com/mlflow/mlflow/master/README.md",                            "MLflow Overview"),
        ("mlflow_tracking.txt",      "https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/tracking/__init__.py",      "MLflow Tracking Module"),
        ("mlflow_concepts.txt",      "https://raw.githubusercontent.com/mlflow/mlflow/master/CHANGELOG.md",                         "MLflow Changelog"),
    ],

    "dvc": [
        ("dvc_start.txt",            "https://dvc.org/doc/start",                                  "DVC Getting Started"),
        ("dvc_pipeline.txt",         "https://dvc.org/doc/user-guide/pipelines",                        "DVC Pipelines User Guide"),
    ],

    "kubeflow": [
        ("kubeflow_pipelines.txt",   "https://www.kubeflow.org/docs/components/pipelines/overview/", "Kubeflow Pipelines Overview"),
        ("kubeflow_intro.txt",       "https://raw.githubusercontent.com/kubeflow/kubeflow/master/README.md", "Kubeflow Introduction"),
    ],

    "kserve": [
        ("kserve_overview.txt",      "https://raw.githubusercontent.com/kserve/kserve/master/README.md",                           "KServe Overview"),
        ("kserve_concepts.txt",      "https://raw.githubusercontent.com/kserve/kserve/master/python/kserve/README.md",            "KServe Python SDK"),
    ],

    "langgraph": [
        ("langgraph_quickstart.txt", "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",                     "LangGraph Overview"),
        ("langgraph_concepts.txt",   "https://raw.githubusercontent.com/langchain-ai/langgraph/main/libs/langgraph/README.md",      "LangGraph Library README"),
    ],

    "crewai": [
        ("crewai_concepts.txt",      "https://docs.crewai.com/concepts/agents",                    "CrewAI Agents"),
        ("crewai_tasks.txt",         "https://docs.crewai.com/concepts/tasks",                     "CrewAI Tasks"),
        ("crewai_crews.txt",         "https://docs.crewai.com/concepts/crews",                     "CrewAI Crews"),
    ],

    "fastapi": [
        ("fastapi_intro.txt",        "https://fastapi.tiangolo.com/",                              "FastAPI Introduction"),
        ("fastapi_async.txt",        "https://fastapi.tiangolo.com/async/",                        "FastAPI Async"),
    ],

    "chromadb": [
        ("chromadb_usage.txt",       "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md", "ChromaDB README"),
        ("chromadb_guide.txt",       "https://docs.trychroma.com/docs/overview/introduction",      "ChromaDB Introduction"),
    ],

    "openrouter": [
        ("openrouter_docs.txt",      "https://raw.githubusercontent.com/OpenRouterTeam/openrouter-runner/main/README.md", "OpenRouter README"),
    ],

    "github_actions": [
        ("github_actions_ml.txt",    "https://docs.github.com/en/actions/about-github-actions/understanding-github-actions", "GitHub Actions Overview"),
    ],
}


# ── HTML → clean text ─────────────────────────────────────────────────────────

def fetch_and_clean(url: str) -> str:
    """
    Fetch a URL and extract clean text content.
    Removes navigation, scripts, styles, and other noise.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "button", "noscript", "svg",
                     "iframe", "meta", "link"]):
        tag.decompose()

    # Try to get the main content area first
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(class_=["content", "main-content", "doc-content", "md-content"]) or
        soup.find("body")
    )

    if main:
        # Get text, normalize whitespace
        lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
        # Filter out very short lines (likely nav remnants) and blank lines
        lines = [l for l in lines if len(l) > 30 or (l and not l.startswith(("©", "•", "|")))]
        text = "\n".join(lines)
    else:
        text = soup.get_text(separator="\n")

    # Trim excessive whitespace
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_sources(filter_key: str = None):
    os.makedirs(DATA_DIR, exist_ok=True)

    sources_to_run = {}
    if filter_key:
        if filter_key not in SOURCES:
            print(f"Unknown source '{filter_key}'. Available: {list(SOURCES.keys())}")
            return
        sources_to_run[filter_key] = SOURCES[filter_key]
    else:
        sources_to_run = SOURCES

    total = sum(len(v) for v in sources_to_run.values())
    done = 0
    failed = []

    print(f"\nFetching {total} pages from {len(sources_to_run)} source(s)...\n")

    for category, pages in sources_to_run.items():
        print(f"-- {category.upper()} --")

        for filename, url, description in pages:
            filepath = os.path.join(DATA_DIR, filename)
            done += 1
            print(f"  [{done}/{total}] {description}")
            print(f"           {url}")

            try:
                text = fetch_and_clean(url)

                if len(text) < 200:
                    print(f"           WARNING: Very little content ({len(text)} chars) - skipping")
                    failed.append((filename, url, "too little content"))
                    continue

                # Write to file with a header
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {description}\n")
                    f.write(f"# Source: {url}\n\n")
                    f.write(text)

                print(f"           OK Saved {len(text):,} chars -> data/{filename}")
                time.sleep(1)  # polite delay between requests

            except requests.exceptions.HTTPError as e:
                print(f"           FAILED HTTP Error: {e}")
                failed.append((filename, url, str(e)))
            except requests.exceptions.Timeout:
                print(f"           FAILED Timeout")
                failed.append((filename, url, "timeout"))
            except Exception as e:
                print(f"           FAILED Error: {e}")
                failed.append((filename, url, str(e)))

        print()

    # Summary
    print("=" * 50)
    print(f"Done! {total - len(failed)}/{total} pages fetched successfully.")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for fname, url, reason in failed:
            print(f"  - {fname}: {reason}")

    print("\nNext step: Re-index the knowledge base")
    print("  Option 1: Restart the server (auto-indexes on startup)")
    print("  Option 2: Call the API: curl -X POST http://localhost:8000/index-docs")
    print("  Option 3: Click 'Re-index Documents' in the Streamlit sidebar")


def list_sources():
    print("\nAvailable sources:\n")
    for category, pages in SOURCES.items():
        print(f"  {category}:")
        for filename, url, description in pages:
            print(f"    - {description}")
            print(f"      {url}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest web documentation into RAG knowledge base")
    parser.add_argument("--list",   action="store_true", help="List all sources without fetching")
    parser.add_argument("--source", type=str, help="Fetch only one source (e.g. mlflow, langgraph)")
    args = parser.parse_args()

    if args.list:
        list_sources()
    else:
        ingest_sources(filter_key=args.source)
