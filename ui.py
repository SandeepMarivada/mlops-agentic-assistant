"""
Streamlit Chat UI for the MLOps Agentic AI Assistant.
Connects to the FastAPI backend at http://localhost:8000.

Features:
- Chat interface with message history
- Switch between LangGraph Agent and CrewAI Multi-Agent
- Session management (named sessions + clear history)
- Re-index documents button
- API health status indicator
"""

import os

import requests
import streamlit as st

# Reads API_URL from environment — defaults to localhost for local dev.
# In Docker: overridden to http://api:8000 via docker-compose.
# In AWS ECS: overridden to the ALB DNS name via task definition env vars.
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLOps AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    # Agent mode selection
    st.subheader("Agent Mode")
    agent_mode = st.radio(
        "Choose agent",
        ["🔗 LangGraph Agent", "👥 CrewAI Multi-Agent"],
        help=(
            "LangGraph: Single agent with RAG + tools + session memory.\n\n"
            "CrewAI: Two agents collaborate — Researcher finds info, Writer explains it."
        ),
    )

    st.divider()

    # Session management
    st.subheader("Session")
    session_id = st.text_input(
        "Session ID",
        value="default",
        help="Use different session IDs for separate conversations.",
    )

    if st.button("🗑️ Clear History", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/history/{session_id}")
            st.session_state.messages = []
            st.success("History cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    # Document indexing
    st.subheader("Knowledge Base")
    if st.button("🔄 Re-index Documents", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                resp = requests.post(f"{API_URL}/index-docs")
                st.success(resp.json().get("message", "Done!"))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Health check
    st.subheader("API Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.success("Backend is running")
        else:
            st.error("Backend error")
    except Exception:
        st.error("Backend offline — start the FastAPI server first:\n`uvicorn main:app --port 8000`")

    st.divider()
    st.caption("MLOps Assistant By Sandeep Marivada")


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("🤖 MLOps Agentic AI Assistant")

# Show which mode is active
if "LangGraph" in agent_mode:
    st.caption("Mode: **LangGraph Agent** — RAG + tools + session memory")
else:
    st.caption("Mode: **CrewAI Multi-Agent** — Researcher + Writer collaboration")

st.divider()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about MLOps..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the appropriate backend endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if "LangGraph" in agent_mode:
                    # LangGraph agent — maintains memory on the server side
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"message": prompt, "session_id": session_id},
                        timeout=120,
                    )
                    answer = resp.json().get("response", "No response received.")
                else:
                    # CrewAI — Researcher + Writer agents collaborate
                    resp = requests.post(
                        f"{API_URL}/crew-chat",
                        json={"question": prompt, "session_id": session_id},
                        timeout=180,
                    )
                    answer = resp.json().get("response", "No response received.")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.Timeout:
                st.error("Request timed out. The agent is taking too long.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
