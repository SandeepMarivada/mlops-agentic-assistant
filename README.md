# MLOps Agentic AI Assistant
**By Sandeep Marivada**

An AI-powered assistant that answers MLOps questions using a RAG knowledge base enriched with real internet documentation, a LangGraph ReAct agent, CrewAI multi-agent collaboration, and a FastAPI + Streamlit interface. Built as part of the MLOps Platform Engineering Course (Assignment 3 — Modules 10–13).

---

## What It Does

Ask it anything about MLOps:
- *"How does MLflow track experiments?"*
- *"What is the difference between LangGraph and CrewAI?"*
- *"How do I deploy a model with KServe?"*
- *"What is RAG and how does it prevent hallucination?"*
- *"Explain DVC pipelines with an example"*

The assistant searches its MLOps knowledge base (RAG), reasons step-by-step using the ReAct pattern (LangGraph), and returns a grounded, accurate answer — backed by both curated docs and real official documentation fetched from the internet.

---

## Architecture

```
Internet Documentation (20 sources)
  MLflow · DVC · Kubeflow · KServe · LangGraph
  CrewAI · FastAPI · ChromaDB · OpenRouter · GitHub Actions
           ↓ (ingest_web.py)
        data/*.txt files
           ↓ (rag.py — indexing)
   ChromaDB Vector Store
   HuggingFace all-MiniLM-L6-v2 embeddings (local)
           ↑
      RAG Retrieval
           ↑
User → Streamlit UI (port 8501)
           ↓ HTTP
    FastAPI Backend (port 8000)
           ↓
   ┌───────────────────┐
   │  LangGraph Agent  │  ←─ POST /chat
   │  or CrewAI Crew   │  ←─ POST /crew-chat
   └───────────────────┘
           ↓
      Tools Layer
   ┌─────────────────────────┐
   │ search_mlops_docs()     │  ← queries ChromaDB
   │ calculate()             │  ← math expressions
   └─────────────────────────┘
           ↓
   OpenRouter LLM (openai/gpt-4o-mini)
```

---

## Modules Covered

| Module | Topic | Problem Solved | Implementation |
|--------|-------|---------------|----------------|
| 10 | LLM & Agentic AI | Chatbots fail to reason autonomously | `agent.py` — LangGraph ReAct agent |
| 11 | Python for Agents | Agents need reliable backend APIs | `main.py` — FastAPI async backend |
| 12 | Agent Frameworks | Manual orchestration is error-prone | `agent.py` (LangGraph) + `crew.py` (CrewAI) |
| 13 | RAG Systems | LLMs hallucinate without grounding | `rag.py` — ChromaDB + HuggingFace embeddings |

---

## Project Structure

```
mlops-agentic-assistant/
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline → EC2 deploy
│
├── tests/
│   ├── __init__.py
│   └── test_api.py             # 10 pytest tests
│
├── data/
│   ├── mlops_docs.txt          # Core knowledge base (all 13 modules)
│   ├── mlflow_readme.txt       # MLflow README (GitHub)
│   ├── mlflow_tracking.txt     # MLflow tracking module
│   ├── mlflow_concepts.txt     # MLflow changelog
│   ├── dvc_start.txt           # DVC Getting Started (dvc.org)
│   ├── dvc_pipeline.txt        # DVC Pipelines User Guide
│   ├── kubeflow_pipelines.txt  # Kubeflow Pipelines Overview
│   ├── kubeflow_intro.txt      # Kubeflow README (GitHub)
│   ├── kserve_overview.txt     # KServe README (GitHub)
│   ├── kserve_concepts.txt     # KServe Python SDK docs
│   ├── langgraph_quickstart.txt # LangGraph README (GitHub)
│   ├── langgraph_concepts.txt  # LangGraph library README
│   ├── crewai_concepts.txt     # CrewAI Agents (docs.crewai.com)
│   ├── crewai_tasks.txt        # CrewAI Tasks (docs.crewai.com)
│   ├── crewai_crews.txt        # CrewAI Crews (docs.crewai.com)
│   ├── fastapi_intro.txt       # FastAPI Introduction (fastapi.tiangolo.com)
│   ├── fastapi_async.txt       # FastAPI Async docs
│   ├── chromadb_usage.txt      # ChromaDB README (GitHub)
│   ├── chromadb_guide.txt      # ChromaDB Introduction (docs.trychroma.com)
│   ├── openrouter_docs.txt     # OpenRouter README (GitHub)
│   └── github_actions_ml.txt  # GitHub Actions Overview (docs.github.com)
│
├── config.py                   # API keys, model name, paths
├── rag.py                      # Module 13: RAG system (ChromaDB + embeddings)
├── tools.py                    # Module 12: Agent tools
├── agent.py                    # Module 10+12: LangGraph ReAct agent
├── crew.py                     # Module 12: CrewAI multi-agent
├── main.py                     # Module 11: FastAPI async backend
├── ui.py                       # Streamlit chat UI
├── ingest_web.py               # Web scraper — fetches real internet docs into data/
│
├── Dockerfile.api              # FastAPI container (multi-stage build)
├── Dockerfile.ui               # Streamlit container
├── docker-compose.yml          # Multi-container setup
├── .dockerignore               # Excludes .env, chroma_db etc
│
├── requirements.txt            # Python dependencies
├── Makefile                    # Common dev commands
├── LEARNING_GUIDE.md           # Concept explanations mapped to code
└── README.md                   # This file
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM Provider | [OpenRouter](https://openrouter.ai) | Access to 100+ LLMs via one API |
| LLM Model | `openai/gpt-4o-mini` | Fast, cheap, supports tool calling |
| Agent Framework | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful ReAct agent with tool use |
| Multi-Agent | [CrewAI](https://crewai.com) | Researcher + Writer collaboration |
| RAG | [ChromaDB](https://chromadb.com) | Local vector database |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Free, local, no API key needed |
| Backend | [FastAPI](https://fastapi.tiangolo.com) | Async Python web API |
| Frontend | [Streamlit](https://streamlit.io) | Chat UI |
| Web Scraping | requests + BeautifulSoup | Fetch official docs into RAG |
| Containerization | Docker + Docker Compose | Reproducible deployment |
| CI/CD | GitHub Actions | Automated test + deploy pipeline |
| Cloud | AWS EC2 | Production hosting |

---

## Prerequisites

- Python 3.12+
- [OpenRouter API key](https://openrouter.ai) (free tier available)
- Docker Desktop (for container deployment)
- Git

---

## Local Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd mlops-agentic-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```bash
cp .env.example .env
```
Edit `.env` and add your OpenRouter API key:
```
OPENROUTER_API_KEY=your_key_here
MODEL_NAME=openai/gpt-4o-mini
```

### 4. Start the API server
```bash
# Option A: using make
make run-api

# Option B: directly
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start the Streamlit UI (new terminal)
```bash
# Option A: using make
make run-ui

# Option B: directly
python -m streamlit run ui.py --server.port 8501
```

### 6. Open in browser
| Service | URL |
|---------|-----|
| Chat UI | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| API health | http://localhost:8000/health |

> **Note:** On first run, the server downloads the HuggingFace embedding model (~90MB) and indexes the knowledge base. This takes ~30-60 seconds.

---

## Enriching the Knowledge Base with Internet Docs

The `ingest_web.py` script fetches real documentation from 10 official sources (20 pages total) and saves them to `data/`.

```bash
# Fetch all sources
python ingest_web.py

# Fetch a specific source only
python ingest_web.py --source mlflow
python ingest_web.py --source langgraph
python ingest_web.py --source crewai

# List all configured sources without fetching
python ingest_web.py --list
```

Available sources: `mlflow`, `dvc`, `kubeflow`, `kserve`, `langgraph`, `crewai`, `fastapi`, `chromadb`, `openrouter`, `github_actions`

After fetching, re-index ChromaDB:
```bash
curl -X POST http://localhost:8000/index-docs
```
Or click **Re-index Documents** in the Streamlit sidebar.

---

## Docker Setup (Recommended)

Run everything with a single command:

```bash
# Build and start both containers
docker compose up --build -d

# View logs
docker compose logs -f

# Stop everything
docker compose down
```

Both services will be available at the same URLs as local setup.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Chat with LangGraph agent |
| `POST` | `/crew-chat` | Chat with CrewAI multi-agent |
| `GET` | `/history/{session_id}` | Get conversation history |
| `DELETE` | `/history/{session_id}` | Clear conversation history |
| `POST` | `/index-docs` | Re-index documents into ChromaDB |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Chat Request
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is MLflow?", "session_id": "my-session"}'
```

### Example: Chat Response
```json
{
  "response": "MLflow is an open-source platform for managing the ML lifecycle...",
  "session_id": "my-session"
}
```

---

## Using the UI

**LangGraph Agent mode:**
- Single agent with RAG search + calculator tools
- Maintains conversation memory per session
- Best for: conversational Q&A, follow-up questions

**CrewAI Multi-Agent mode:**
- Researcher agent searches the knowledge base
- Writer agent synthesizes findings into a clear answer
- Best for: detailed, well-structured explanations

**Session ID:**
- Use different session IDs for separate conversations
- Same session ID = agent remembers previous messages

---

## Running Tests

```bash
# Run all tests
make test

# Or directly
pytest tests/ -v
```

### Test Results
```
tests/test_api.py::test_root_returns_200                    PASSED
tests/test_api.py::test_health_check                        PASSED
tests/test_api.py::test_health_returns_version              PASSED
tests/test_api.py::test_history_empty_for_new_session       PASSED
tests/test_api.py::test_clear_nonexistent_session           PASSED
tests/test_api.py::test_history_response_schema             PASSED
tests/test_api.py::test_chat_endpoint_exists                PASSED
tests/test_api.py::test_chat_response_schema                PASSED
tests/test_api.py::test_chat_default_session_id             PASSED
tests/test_api.py::test_crew_chat_endpoint_exists           PASSED
========================= 10 passed =========================
```

---

## CI/CD Deployment to AWS EC2

### Pipeline Flow
```
Push to main branch
       ↓
[GitHub Actions]
       ↓
 Run 10 tests ──── FAIL → Pipeline stops, no deploy
       │
      PASS
       ↓
 SSH into EC2
       ↓
 git pull latest code
       ↓
 Write .env from GitHub Secrets
       ↓
 docker compose up --build -d
       ↓
 Verify containers running
       ↓
App live at EC2_IP:8000 and EC2_IP:8501
```

### Step 1: Launch EC2 Instance
- **AMI:** Ubuntu 22.04 LTS
- **Instance type:** t3.medium (2 vCPU, 4GB RAM minimum)
- **Security group inbound rules:**

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP | SSH (GitHub Actions deploy) |
| 8000 | TCP | 0.0.0.0/0 | FastAPI backend |
| 8501 | TCP | 0.0.0.0/0 | Streamlit UI |

### Step 2: Install Prerequisites on EC2

SSH into your instance and run the setup script (does everything in one go):

```bash
# Download and run the setup script
curl -o setup-ec2.sh https://raw.githubusercontent.com/<your-username>/<your-repo>/main/setup-ec2.sh
chmod +x setup-ec2.sh
./setup-ec2.sh
```

Or copy the contents of `setup-ec2.sh` from this repo and run it manually. It installs:
- Git
- Docker Engine (from official Docker repo, not Ubuntu's default)
- Docker Compose v2 plugin (`docker compose`, NOT `docker-compose`)
- Creates `/app` directory owned by `ubuntu`

**What it installs and why:**

| Package | Why needed |
|---------|-----------|
| `git` | CI/CD clones the repo on first deploy, pulls on subsequent deploys |
| `docker-ce` | Builds and runs the API + UI containers |
| `docker-compose-plugin` | `docker compose up --build -d` in the deploy script |
| `/app` directory | Where the repo is cloned on EC2 |

> **Critical:** After the script finishes, **log out and log back in** before pushing to deploy.
> This is required for Docker group permissions to take effect in new SSH sessions (including GitHub Actions).
> Without this step, deploy will fail with `permission denied while trying to connect to Docker daemon`.

```bash
# After re-login, verify:
docker run hello-world       # should print "Hello from Docker!"
docker compose version       # should print Docker Compose version 2.x.x
```

### Step 4: Add GitHub Secrets
Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value |
|-------------|-------|
| `EC2_HOST` | Your EC2 public IP (e.g., `54.123.45.67`) |
| `EC2_USERNAME` | `ubuntu` |
| `EC2_SSH_KEY` | Full contents of your `.pem` key file |
| `OPENROUTER_API_KEY` | Your OpenRouter API key |

### Step 5: Push to Deploy
```bash
git add .
git commit -m "deploy: initial production deployment"
git push origin main
```

The pipeline triggers automatically. View progress at:
`https://github.com/<your-username>/<repo>/actions`

### After Deployment
| Service | URL |
|---------|-----|
| Chat UI | `http://<EC2_IP>:8501` |
| API docs | `http://<EC2_IP>:8000/docs` |

---

## Makefile Commands

```bash
make install        # Install all dependencies
make run-api        # Start FastAPI server with hot reload
make run-ui         # Start Streamlit UI
make test           # Run all pytest tests
make docker-build   # Build Docker images
make docker-up      # Start all containers
make docker-down    # Stop all containers
make docker-logs    # Tail container logs
make docker-restart # Rebuild and restart containers
make docker-status  # Show running containers
make clean          # Remove ChromaDB, logs, pycache
make help           # Show all available commands
```

---

## Learning Resources

| Video | Topic | Link |
|-------|-------|------|
| Video 1 | MLOps Fundamentals | https://www.youtube.com/watch?v=NgWujOrCZFo |
| Video 2 | MLflow (Assignment 1) | https://www.youtube.com/watch?v=86BKEv0X2xU |
| Video 3 | Kubeflow (Assignment 2) | https://www.youtube.com/watch?v=5iOQcGfcZe4 |
| Video 4 | Agentic AI (Assignment 3) | https://www.youtube.com/watch?v=jGg_1h0qzaM |
| Video 5 | MCP Servers | https://www.youtube.com/watch?v=MDBG2MOp4Go |
| Video 6 | FastAPI | https://www.youtube.com/watch?v=rvFsGRvj9jo |

For detailed concept explanations mapped to this codebase, see [LEARNING_GUIDE.md](./LEARNING_GUIDE.md).

---

*MLOps Assistant — Sandeep Marivada*
