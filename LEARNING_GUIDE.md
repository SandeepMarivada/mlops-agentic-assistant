# MLOps Learning Guide — Sandeep Marivada
> Read this alongside the YouTube videos. Every concept is mapped to actual code in this project.

---

## How to Use This Guide
1. Watch the video for the module
2. Come back here and read the concept explanation
3. Open the referenced file and find the code
4. Run it and see it working

---

## Module 1 — MLOps Fundamentals
**Video:** https://www.youtube.com/watch?v=NgWujOrCZFo
**Problem:** ML systems fail without structured lifecycle management.

### What is MLOps?
MLOps = DevOps principles applied to Machine Learning.
Without MLOps, teams face:
- "It worked on my machine" — no reproducibility
- Models deployed manually — error-prone and slow
- No monitoring — model quality degrades silently
- No versioning — can't rollback a bad model

### The ML Lifecycle (what we automated)
```
Data → Train → Evaluate → Package → Deploy → Monitor → Retrain
```

### Maturity Levels
| Level | Description | This Project |
|-------|-------------|-------------|
| 0 | Manual everything | — |
| 1 | Automated pipeline | docker-compose.yml |
| 2 | Full CI/CD | .github/workflows/deploy.yml |

### In Our Code
- `docker-compose.yml` — defines the full system as code (infrastructure as code)
- `.github/workflows/deploy.yml` — automates the entire deploy lifecycle
- `Makefile` — standardizes all commands so any team member can run them

---

## Module 2 — Data Versioning (DVC)
**Problem:** Training data changes break reproducibility.

### Why Data Versioning Matters
Code is versioned with Git. But what about data?
- You train model v1 on dataset A → accuracy 90%
- Dataset A gets updated → you retrain → accuracy drops to 80%
- You can't go back because you didn't version the data

### DVC (Data Version Control)
DVC tracks large files (datasets, models) in Git without storing them in Git.
It stores a small `.dvc` pointer file in Git and the actual data in S3/GCS.

```bash
# How DVC works
dvc init                          # initialize DVC in project
dvc add data/train.csv            # track dataset
git add data/train.csv.dvc        # commit the pointer
dvc push                          # upload actual data to S3
dvc pull                          # download data by version
dvc repro                         # reproduce pipeline from scratch
```

### In Our Code
Our `data/` folder holds the RAG knowledge base.
In a full MLOps setup, all data files would be versioned with DVC:
```
data/mlops_docs.txt      →  tracked by DVC
data/mlops_docs.txt.dvc  →  committed to Git (small pointer file)
```
The assistant also has real DVC documentation fetched from dvc.org (`data/dvc_start.txt`, `data/dvc_pipeline.txt`). Ask it: *"How do I create a DVC pipeline?"*

---

## Module 3 — Experiment Tracking (MLflow)
**Video:** https://www.youtube.com/watch?v=86BKEv0X2xU
**Problem:** Experiments are lost without tracking.

### Why Experiment Tracking Matters
When training ML models, you run many experiments:
- Experiment 1: learning_rate=0.01, accuracy=85%
- Experiment 2: learning_rate=0.001, accuracy=91%
- Experiment 3: n_estimators=200, accuracy=93%

Without tracking, you forget what worked and why.

### MLflow Core API
```python
import mlflow

# Start tracking an experiment run
with mlflow.start_run(run_name="random_forest_v1"):

    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.01)

    # Train your model here...
    # model = train(...)

    # Log results/metrics
    mlflow.log_metric("accuracy", 0.93)
    mlflow.log_metric("f1_score", 0.91)
    mlflow.log_metric("loss", 0.07)

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("confusion_matrix.png")

# View all runs: mlflow ui → http://localhost:5000
```

### Key MLflow Concepts
| Concept | What it is |
|---------|-----------|
| Run | One training execution with its params + metrics |
| Experiment | Group of related runs |
| Artifact | Files saved with a run (model, plots, data) |
| Metric | Numeric value tracked over time (loss, accuracy) |
| Parameter | Input config to your training (learning rate, epochs) |

### In Our Code
Real MLflow docs are indexed in the knowledge base (`data/mlflow_readme.txt`, `data/mlflow_concepts.txt`).
Ask the assistant: *"Show me how to log a PyTorch model with MLflow"*

---

## Module 4 — Model Registry (MLflow)
**Problem:** No governance over deployed models.

### The Problem Without a Registry
- Developer A trains a model and deploys it directly to production
- No one knows which version is running
- Can't rollback when the model performs badly
- No approval process before deployment

### MLflow Model Registry — Stages
```
None → Staging → Production → Archived
```

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from a run
mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="customer_churn_model"
)

# Promote to Staging for testing
client.transition_model_version_stage(
    name="customer_churn_model",
    version=1,
    stage="Staging"
)

# After testing passes, promote to Production
client.transition_model_version_stage(
    name="customer_churn_model",
    version=1,
    stage="Production"
)

# Get the current production model
model = mlflow.sklearn.load_model(
    model_uri="models:/customer_churn_model/Production"
)
```

---

## Module 5 — Training Pipelines (Kubeflow)
**Video:** https://www.youtube.com/watch?v=5iOQcGfcZe4
**Problem:** Manual training is not scalable.

### Why Pipelines Matter
Manual training process:
1. Engineer SSHes into a server
2. Runs a Python script
3. Waits hours
4. Manually copies the model file
5. Deploys by hand

This doesn't scale. Pipelines automate every step.

### Kubeflow Pipelines
Kubeflow runs on Kubernetes. Each step is a Docker container.

```python
from kfp import dsl

# Each function becomes a container in the pipeline
@dsl.component
def preprocess_data(input_path: str, output_path: dsl.OutputPath()):
    import pandas as pd
    df = pd.read_csv(input_path)
    # ... preprocessing ...
    df.to_csv(output_path, index=False)

@dsl.component
def train_model(data_path: dsl.InputPath(), model_path: dsl.OutputPath()):
    # ... training ...
    pass

@dsl.component
def evaluate_model(model_path: dsl.InputPath()) -> float:
    # ... evaluation ...
    return accuracy

# Define the pipeline — steps run in order
@dsl.pipeline(name="ML Training Pipeline")
def ml_pipeline(input_data: str = "gs://bucket/data.csv"):
    preprocess = preprocess_data(input_path=input_data)
    train = train_model(data_path=preprocess.output)
    evaluate = evaluate_model(model_path=train.output)
```

### In Our Code
Real Kubeflow docs are in the knowledge base (`data/kubeflow_pipelines.txt`, `data/kubeflow_intro.txt`).
Ask: *"What is the difference between Kubeflow and Argo Workflows?"*

---

## Module 6 — Model Serving (KServe)
**Problem:** Ad-hoc serving lacks scalability.

### KServe on Kubernetes
KServe provides production-grade model serving with:
- Auto-scaling (scale up under load, scale to zero when idle)
- Multiple framework support (sklearn, PyTorch, TensorFlow, ONNX)
- Canary deployments (route 10% traffic to new model, 90% to old)
- A/B testing

```yaml
# Deploy a model with KServe (Kubernetes YAML)
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: customer-churn-model
spec:
  predictor:
    sklearn:
      storageUri: "s3://my-bucket/models/churn/v2"
    # Auto-scaling config
    minReplicas: 1
    maxReplicas: 10
```

```bash
# Make a prediction
curl -X POST http://model-endpoint/v1/models/customer-churn-model:predict \
  -d '{"instances": [[0.5, 1.2, 3.4, 0.1]]}'
```

### In Our Code
Real KServe docs are indexed (`data/kserve_overview.txt`, `data/kserve_concepts.txt`).
Ask: *"How do I do canary deployment with KServe?"*

---

## Module 7 — CI/CD for ML (GitHub Actions)
**Problem:** ML changes need controlled releases.
**In our code:** `.github/workflows/deploy.yml`

### What CI/CD Does
**CI (Continuous Integration):** Every code push runs tests automatically.
**CD (Continuous Deployment):** If tests pass, deploy automatically.

### Our Pipeline — `deploy.yml`
```
Developer pushes code to GitHub
    ↓
[Job 1 - CI] GitHub spins up Ubuntu VM
    → installs Python + dependencies
    → runs: pytest tests/ -v
    → if ANY test fails → pipeline STOPS here
    ↓ (all tests pass)
[Job 2 - CD] SSH into EC2
    → git pull latest code
    → write .env from GitHub Secrets (API keys never in git)
    → docker compose up --build -d
    → verify containers running
    ↓
App is live with new code
```

### Key Concepts in `deploy.yml`

**Triggers:**
```yaml
on:
  push:
    branches: [main]      # runs when you push to main
  pull_request:
    branches: [main]      # runs on PRs (CI only, not CD)
```

**Secrets** (Module 9 — Security):
```yaml
# Never hardcode secrets in code
# Store in GitHub Settings → Secrets → Actions
env:
  OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

**Job dependencies:**
```yaml
deploy:
  needs: test    # deploy only runs AFTER test passes
```

### Our Tests — `tests/test_api.py`
```python
# 10 tests that run in CI
def test_health_check():                    # is the server up?
def test_history_empty_for_new_session():   # does memory work?
def test_chat_endpoint_exists():            # does /chat respond?
# ... 7 more tests
```

---

## Module 8 — Monitoring & Drift Detection
**Problem:** Model quality degrades silently.

### Types of Drift
| Type | What happens | Example |
|------|-------------|---------|
| Data Drift | Input distribution changes | Users start using new slang the model never saw |
| Concept Drift | Input→Output relationship changes | Economic conditions change, old patterns don't hold |
| Prediction Drift | Model output distribution changes | Model starts predicting one class more than others |

### How to Detect Drift
```python
# Using Evidently AI (popular open-source tool)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=train_df,    # what model trained on
    current_data=production_df  # what model sees in production
)
report.save_html("drift_report.html")
```

### Key Metrics to Monitor
- **Model accuracy** — compare live accuracy to baseline
- **Feature statistics** — mean, std dev, min, max of input features
- **Prediction confidence** — are predictions becoming less confident?
- **Data quality** — missing values, outliers, schema violations

---

## Module 9 — Security & Governance
**Problem:** Unsecured ML assets risk compliance.

### In Our Code — Security Practices

**1. Secrets never in code:**
```bash
# BAD — never do this
OPENROUTER_API_KEY = "sk-or-v1-abc123..."

# GOOD — read from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
```

**2. `.env` excluded from git:**
```
# .dockerignore and .gitignore
.env
.env.*
!.env.example   # only the template is committed
```

**3. GitHub Secrets for CI/CD:**
- `EC2_SSH_KEY` — private key never in code
- `OPENROUTER_API_KEY` — injected at deploy time

**4. IAM Least Privilege (AWS):**
- EC2 instance should only have permissions it needs
- No admin access — only specific S3 buckets, specific ECR repos

---

## Module 10 — LLM & Agentic AI Basics
**Video:** https://www.youtube.com/watch?v=jGg_1h0qzaM
**Problem:** Chatbots fail to reason or take actions autonomously.
**In our code:** `agent.py`

### What is an LLM Agent?
A regular chatbot: User → LLM → Answer (one shot, no tools)

An Agent: User → LLM → decides to use a tool → tool runs → result → LLM → better answer

### ReAct Pattern (Reasoning + Acting)
```
User: "What MLflow commands do I use for experiment tracking?"

Agent Thought: "I should search the knowledge base for MLflow tracking info"
Agent Action:  search_mlops_docs("MLflow tracking commands")
Observation:   "mlflow.log_param(), mlflow.log_metric(), mlflow.start_run()..."
Agent Thought: "I have the info, I can now answer"
Agent Answer:  "Here are the MLflow tracking commands: ..."
```

This loop (Thought → Action → Observation) repeats until the agent has enough to answer.

### OpenRouter
OpenRouter is a single API that gives you access to 100+ LLMs:
- GPT-4o, GPT-4o-mini (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Llama 3.1 (Meta)
- Gemini (Google)
- Mistral, Mixtral

```python
# config.py — OpenRouter setup
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-4o-mini"

# agent.py — using OpenRouter via LangChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)
```

---

## Module 11 — Python for Agents (FastAPI)
**Video:** https://www.youtube.com/watch?v=rvFsGRvj9jo
**Problem:** Agents need reliable backend APIs.
**In our code:** `main.py`

### Why FastAPI?
| Feature | Benefit |
|---------|---------|
| Auto docs at `/docs` | No manual documentation |
| Pydantic validation | Bad requests rejected automatically |
| Async/await | Handle many requests concurrently |
| Type hints | Fewer bugs, better IDE support |

### Key FastAPI Concepts in `main.py`

**Pydantic Models (input validation):**
```python
class ChatRequest(BaseModel):
    message: str                    # required
    session_id: str = "default"     # optional, defaults to "default"

# FastAPI automatically validates:
# - message must be a string
# - Returns 422 Unprocessable Entity if message is missing
```

**Async endpoint (non-blocking):**
```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # await = non-blocking, FastAPI handles other requests while waiting
    result = await agent.ainvoke({"messages": messages_in})
    return ChatResponse(response=result["messages"][-1].content)
```

**Session Memory (per-user conversation history):**
```python
from collections import defaultdict
session_store: dict[str, list[BaseMessage]] = defaultdict(list)

# Each session_id gets its own conversation history
# "user-A" and "user-B" have completely separate conversations
# Same session_id = agent remembers your previous messages
```

### All Endpoints
| Method | Endpoint | What it does |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check for load balancer |
| POST | `/chat` | LangGraph agent chat |
| POST | `/crew-chat` | CrewAI multi-agent chat |
| GET | `/history/{id}` | Get conversation history |
| DELETE | `/history/{id}` | Clear session |
| POST | `/index-docs` | Re-index RAG documents |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## Module 12 — Agent Frameworks (LangGraph + CrewAI)
**Problem:** Manual orchestration of agent logic is error-prone.
**In our code:** `agent.py` (LangGraph), `crew.py` (CrewAI)

### LangGraph — `agent.py`
LangGraph models the agent as a **directed graph** with nodes and edges.

```
State: { messages: [...] }
         ↓
    [llm node]          ← calls OpenRouter LLM
         ↓
  tool call? ──Yes──→ [tools node]   ← runs search_mlops_docs or calculate
     ↓ No                   ↓
    END              back to [llm node]
```

```python
# agent.py — the complete LangGraph setup

# 1. Define state (what the graph tracks)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # operator.add means new messages are APPENDED (not replaced)

# 2. Define the LLM node
def call_llm(state: AgentState) -> dict:
    response = llm.invoke([SystemMessage(...)] + state["messages"])
    return {"messages": [response]}

# 3. Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)           # node 1: LLM
graph.add_node("tools", ToolNode(tools))  # node 2: tool executor

graph.set_entry_point("llm")
graph.add_conditional_edges("llm", tools_condition)
# tools_condition: if LLM called a tool → go to tools, else → END
graph.add_edge("tools", "llm")            # after tools, always back to LLM

agent = graph.compile()
```

### CrewAI — `crew.py`
CrewAI is for **multiple specialized agents** collaborating.

```python
# crew.py — two agents work as a team

# Agent 1: Researcher — searches knowledge base
researcher = Agent(
    role="MLOps Research Specialist",
    goal="Search the knowledge base and gather accurate information",
    tools=[search_tool],   # has access to RAG search
    llm=llm,
)

# Agent 2: Writer — explains the findings
writer = Agent(
    role="MLOps Technical Writer",
    goal="Transform research into clear explanations",
    llm=llm,
    # no tools — just writes based on researcher's findings
)

# Task 1: Research
research_task = Task(
    description="Search and gather info about: {question}",
    agent=researcher,
)

# Task 2: Write (receives research as context)
writing_task = Task(
    description="Write a clear answer based on research",
    agent=writer,
    context=[research_task],   # ← gets researcher's output
)

# Crew runs them sequentially
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
)
result = crew.kickoff()
```

### When to Use Which?
| Scenario | Use |
|----------|-----|
| Single agent, complex reasoning with tools | LangGraph |
| Multiple specialized agents collaborating | CrewAI |
| Fine-grained control over flow | LangGraph |
| Role-based agent teams | CrewAI |

Real LangGraph and CrewAI docs are indexed in the knowledge base. Ask:
- *"What are LangGraph edges and how do conditional edges work?"*
- *"How do I add memory to a CrewAI agent?"*

---

## Module 13 — RAG Systems
**Problem:** LLMs hallucinate without grounding in enterprise data.
**In our code:** `rag.py`, `ingest_web.py`

### What is RAG?
RAG = Retrieval-Augmented Generation

Without RAG:
```
User: "What is the MLflow command for logging a model?"
LLM:  "You can use mlflow.model.save() ..."   ← WRONG (hallucinated)
```

With RAG:
```
User: "What is the MLflow command for logging a model?"
RAG:  searches docs → finds "mlflow.sklearn.log_model(model, 'name')"
LLM:  "Use mlflow.sklearn.log_model(model, 'name') ..."   ← CORRECT
```

### RAG Pipeline — Two Phases

**Phase 1: Indexing (done once at startup)**
```python
# rag.py — load_and_index_documents()

# Step 1: Load documents from data/ folder
loader = DirectoryLoader("./data", glob="*.txt",
                         loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"})
documents = loader.load()   # loads all 21 .txt files

# Step 2: Split into chunks (500 chars each, 50 overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Embed each chunk (convert text → number vectors)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 4: Store in ChromaDB (vector database on disk)
vectorstore = Chroma.from_documents(chunks, embeddings,
                                    persist_directory="./chroma_db")
```

**Phase 2: Retrieval (done on every query)**
```python
# rag.py — get_retriever()

# User asks a question → embed it → find similar chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Under the hood:
# 1. Embed the query: "how does MLflow work?" → [0.2, 0.8, 0.1, ...]
# 2. Find top-3 most similar chunks in ChromaDB
# 3. Return those chunks as context for the LLM
```

### Enriching with Internet Data — `ingest_web.py`
The knowledge base is not just static text. `ingest_web.py` fetches real documentation from official sources:

```python
# ingest_web.py — sources configured
SOURCES = {
    "mlflow":   [("mlflow_readme.txt", "https://raw.githubusercontent.com/mlflow/mlflow/master/README.md", ...)],
    "dvc":      [("dvc_start.txt", "https://dvc.org/doc/start", ...)],
    "kubeflow": [...],
    "kserve":   [...],
    "langgraph":[...],
    "crewai":   [...],   # 27k+ chars of real CrewAI agent docs
    "fastapi":  [...],   # 14k+ chars of real FastAPI docs
    "chromadb": [...],
    # ...10 sources total, 20 pages
}
```

```bash
# Fetch all internet docs and save to data/
python ingest_web.py

# Re-index ChromaDB with new data
curl -X POST http://localhost:8000/index-docs
```

### Key RAG Components
| Component | What it does | In our code |
|-----------|-------------|-------------|
| Document Loader | Reads files from disk | `DirectoryLoader` with utf-8 encoding |
| Text Splitter | Breaks docs into chunks | `RecursiveCharacterTextSplitter(500, 50)` |
| Embeddings | Converts text to vectors | `HuggingFaceEmbeddings` (local, free) |
| Vector Store | Stores + searches vectors | `ChromaDB` (persisted to `./chroma_db`) |
| Retriever | Finds relevant chunks | `vectorstore.as_retriever(k=3)` |
| Web Ingestion | Fetches real docs | `ingest_web.py` (requests + BeautifulSoup) |

### Why Chunk Size Matters
- Too small (100 chars) → chunks lose context
- Too large (2000 chars) → retrieval is imprecise, too much noise
- 500 chars with 50 overlap = good balance for technical documentation

---

## Docker — Containerization
**In our code:** `Dockerfile.api`, `Dockerfile.ui`, `docker-compose.yml`

### Why Docker?
"It works on my machine" → Docker makes it work on EVERY machine.
A container packages the app + its dependencies + the runtime together.

### `Dockerfile.api` — Multi-stage Build
```dockerfile
# Stage 1: Builder — install dependencies
FROM python:3.12-slim AS builder
RUN pip install -r requirements.txt
# Pre-download HuggingFace model (so container starts fast)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Runtime — only copy what's needed (smaller image)
FROM python:3.12-slim
COPY --from=builder /root/.local /root/.local
COPY *.py ./
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml` — Multi-container
```yaml
services:
  api:                     # FastAPI container
    build: Dockerfile.api
    ports: ["8000:8000"]
    volumes:
      - chroma_data:/app/chroma_db   # persist data across restarts

  ui:                      # Streamlit container
    build: Dockerfile.ui
    environment:
      - API_URL=http://api:8000      # ui talks to api by container name
    depends_on:
      api:
        condition: service_healthy   # wait for API health check to pass
```

---

## Complete Architecture Diagram

```
     Internet (Official Documentation)
  MLflow  DVC  Kubeflow  KServe  LangGraph
  CrewAI  FastAPI  ChromaDB  GitHub Actions
              |
              | ingest_web.py (requests + BeautifulSoup)
              v
    ┌─────────────────────────────────┐
    │  data/ folder (21 .txt files)   │
    │  mlops_docs.txt                 │
    │  mlflow_readme.txt              │
    │  crewai_concepts.txt            │
    │  fastapi_intro.txt  ... etc     │
    └────────────┬────────────────────┘
                 |
                 | rag.py → load, split, embed
                 v
    ┌─────────────────────────────────┐
    │  ChromaDB (./chroma_db)         │
    │  HuggingFace all-MiniLM-L6-v2  │
    │  (vector embeddings, local)     │
    └────────────┬────────────────────┘
                 |
                 | similarity search (k=3)
                 v
┌────────────────────────────────────────────┐
│           User / Browser                    │
└─────────────────┬──────────────────────────┘
                  | http://localhost:8501
                  v
┌────────────────────────────────────────────┐
│         Streamlit UI  (ui.py)              │
│  Chat interface · LangGraph ↔ CrewAI mode  │
│  Session management · Sidebar controls     │
└─────────────────┬──────────────────────────┘
                  | http://api:8000
                  v
┌────────────────────────────────────────────┐
│       FastAPI Backend  (main.py)           │
│  POST /chat        POST /crew-chat         │
│  GET  /history     DELETE /history         │
│  POST /index-docs  GET /health             │
│  Session memory (defaultdict per session)  │
└──────────┬────────────────────┬────────────┘
           |                    |
           v                    v
┌──────────────────┐  ┌─────────────────────┐
│  LangGraph Agent │  │    CrewAI Crew       │
│  (agent.py)      │  │    (crew.py)         │
│                  │  │  Researcher Agent    │
│  llm → tools     │  │       ↓              │
│  ↑         ↓     │  │  Writer Agent        │
│  ←─────────┘     │  └─────────────────────┘
└──────────┬───────┘
           |
           v
┌────────────────────────────────────────────┐
│           Tools  (tools.py)                │
│  search_mlops_docs()  → queries ChromaDB   │
│  calculate()          → math evaluation    │
└────────────────────────────────────────────┘
           |
           v
┌────────────────────────────────────────────┐
│     OpenRouter LLM  (config.py)            │
│     openai/gpt-4o-mini                     │
│     https://openrouter.ai/api/v1           │
└────────────────────────────────────────────┘
```

---

## Quick Reference — Files & What They Teach

| File | Module | What it teaches |
|------|--------|----------------|
| `config.py` | 10 | Environment config, OpenRouter setup |
| `rag.py` | 13 | Document loading (utf-8), chunking, embeddings, ChromaDB |
| `tools.py` | 12 | LangChain tool definition with @tool decorator |
| `agent.py` | 10, 12 | LangGraph StateGraph, nodes, edges, ReAct tool calling |
| `crew.py` | 12 | CrewAI agents, tasks, crew, sequential process |
| `main.py` | 11 | FastAPI, async/await, Pydantic, session memory |
| `ui.py` | 11 | Streamlit, chat interface, state management |
| `ingest_web.py` | 13 | Web scraping, requests, BeautifulSoup, RAG data enrichment |
| `Dockerfile.api` | 7 | Multi-stage Docker build, healthcheck |
| `Dockerfile.ui` | 7 | Container for Python web app |
| `docker-compose.yml` | 7 | Multi-container orchestration, volumes, depends_on |
| `.github/workflows/deploy.yml` | 7, 9 | CI/CD pipeline, secrets management, SSH deploy to EC2 |
| `tests/test_api.py` | 7 | Pytest, FastAPI TestClient, CI testing (10 tests) |
| `Makefile` | — | Developer workflow automation |
| `data/mlops_docs.txt` | 13 | Core RAG knowledge base (all 13 modules) |
| `data/*.txt (20 files)` | 13 | Real internet docs fetched by ingest_web.py |

---

## Recommended Watch Order

| Step | Watch | Then look at |
|------|-------|-------------|
| 1 | Video 1 — MLOps Fundamentals | `docker-compose.yml`, `Makefile` |
| 2 | Video 2 — MLflow | `data/mlflow_readme.txt`, ask assistant about MLflow |
| 3 | Video 3 — Kubeflow | `data/kubeflow_pipelines.txt`, ask assistant about Kubeflow |
| 4 | Video 6 — FastAPI | `main.py` — every endpoint, async, Pydantic |
| 5 | Video 4 — Agentic AI | `agent.py`, `crew.py`, `tools.py`, `rag.py` |
| 6 | Video 5 — MCP Servers | (extends agent.py tools concept) |
| 7 | Review all | Run `ingest_web.py` → chat with the assistant |

---

*MLOps Assistant — Sandeep Marivada*
