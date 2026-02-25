"""
Module 11: FastAPI Backend for the Agent
Problem solved: Agents need reliable backend APIs to interact with systems.
Solution: FastAPI provides true async, session memory, and auto-documented APIs.

Endpoints:
  GET    /                        - Welcome message
  GET    /health                  - Health check
  POST   /chat                    - LangGraph agent chat (with memory)
  POST   /crew-chat               - CrewAI multi-agent chat
  GET    /history/{session_id}    - Get conversation history
  DELETE /history/{session_id}    - Clear conversation history
  POST   /index-docs              - Re-index documents into ChromaDB
  GET    /docs                    - Auto-generated Swagger UI
"""

import asyncio
import os
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from agent import agent
from rag import load_and_index_documents

# ── Lifespan (startup logic) ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Index documents into ChromaDB on startup if not already indexed."""
    if not os.path.exists("./chroma_db"):
        print("ChromaDB not found. Indexing documents on startup...")
        load_and_index_documents()
        print("Documents indexed successfully!")
    else:
        print("ChromaDB found. Skipping indexing.")
    yield  # app runs here

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MLOps Agentic AI Assistant",
    description=(
        "An AI agent for MLOps questions. "
        "Built with LangGraph + CrewAI + FastAPI + ChromaDB RAG + OpenRouter. "
        "Features: session memory, multi-agent collaboration, grounded answers."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Session Memory (Module 10 + 11) ───────────────────────────────────────────
# Stores full message history per session_id.
# Key insight: passing full history to agent ensures it remembers past turns.
# In production, replace this dict with Redis or a database.

session_store: dict[str, list[BaseMessage]] = defaultdict(list)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

class CrewChatRequest(BaseModel):
    question: str
    session_id: str = "crew-default"

class CrewChatResponse(BaseModel):
    response: str
    session_id: str

class IndexResponse(BaseModel):
    status: str
    message: str

class HistoryMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "MLOps Agentic AI Assistant v2.0 is running!",
        "docs": "Visit /docs for interactive API documentation",
        "features": [
            "LangGraph ReAct Agent",
            "RAG with ChromaDB",
            "Session Conversation Memory",
            "CrewAI Multi-Agent (Researcher + Writer)",
            "True Async API"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MLOps Agentic AI", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the LangGraph MLOps agent.

    Features:
    - Session memory: remembers previous messages per session_id
    - RAG: searches MLOps knowledge base before answering
    - True async: uses ainvoke() so FastAPI event loop is never blocked
    """
    try:
        # Retrieve existing history for this session
        history = session_store[request.session_id]

        # Build input: full history + new user message
        messages_in = history + [HumanMessage(content=request.message)]

        # TRUE ASYNC: ainvoke() doesn't block the FastAPI event loop
        # This allows FastAPI to handle other requests while waiting for LLM
        result = await agent.ainvoke({"messages": messages_in})

        # Store the complete updated conversation back into session
        session_store[request.session_id] = result["messages"]

        # Last message is always the final AI response
        response_text = result["messages"][-1].content
        return ChatResponse(response=response_text, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crew-chat", response_model=CrewChatResponse)
async def crew_chat(request: CrewChatRequest):
    """
    Chat using CrewAI multi-agent collaboration (Module 12).

    Two agents work together:
    1. Researcher: searches the MLOps knowledge base
    2. Writer: synthesizes research into a clear answer

    Note: CrewAI is synchronous, so it runs in a thread pool
    via asyncio.to_thread() to keep FastAPI non-blocking.
    """
    try:
        from crew import run_crew

        # Run synchronous CrewAI in a thread pool — keeps FastAPI async
        response = await asyncio.to_thread(run_crew, request.question)
        return CrewChatResponse(response=response, session_id=request.session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    Get conversation history for a session.
    Only returns Human and AI messages (filters out internal tool call messages).
    """
    messages = session_store.get(session_id, [])
    visible = [
        HistoryMessage(
            role="user" if isinstance(m, HumanMessage) else "assistant",
            content=m.content
        )
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and isinstance(m.content, str)
        and m.content.strip()
    ]
    return HistoryResponse(session_id=session_id, messages=visible)


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in session_store:
        del session_store[session_id]
    return {"status": "cleared", "session_id": session_id}


@app.post("/index-docs", response_model=IndexResponse)
async def index_documents():
    """Re-index all documents in data/ directory into ChromaDB."""
    try:
        load_and_index_documents()
        return IndexResponse(status="success", message="Documents indexed into ChromaDB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
