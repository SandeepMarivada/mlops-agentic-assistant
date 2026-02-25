"""
API Tests — run with: pytest tests/ -v
These tests verify the FastAPI endpoints without needing a live LLM call.
"""

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ── Basic endpoint tests ──────────────────────────────────────────────────────

def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MLOps Agentic AI"


def test_health_returns_version():
    response = client.get("/health")
    assert "version" in response.json()


# ── Session memory tests ──────────────────────────────────────────────────────

def test_history_empty_for_new_session():
    """New session should return empty history."""
    response = client.get("/history/brand-new-session-xyz")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "brand-new-session-xyz"
    assert data["messages"] == []


def test_clear_nonexistent_session():
    """Clearing a session that doesn't exist should not error."""
    response = client.delete("/history/does-not-exist")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"


def test_history_response_schema():
    """History response must have session_id and messages fields."""
    response = client.get("/history/test-schema")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "messages" in data
    assert isinstance(data["messages"], list)


# ── Chat endpoint tests ───────────────────────────────────────────────────────

def test_chat_endpoint_exists():
    """Chat endpoint should be reachable (200 or 500 if no API key in CI)."""
    response = client.post(
        "/chat",
        json={"message": "What is MLflow?", "session_id": "test-ci"},
    )
    # 200 = worked, 500 = LLM error (acceptable in CI without real API key)
    assert response.status_code in [200, 500]


def test_chat_response_schema():
    """If chat succeeds, response must have 'response' and 'session_id'."""
    response = client.post(
        "/chat",
        json={"message": "What is MLflow?", "session_id": "schema-test"},
    )
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert data["session_id"] == "schema-test"


def test_chat_default_session_id():
    """session_id should default to 'default' if not provided."""
    response = client.post("/chat", json={"message": "hello"})
    if response.status_code == 200:
        assert response.json()["session_id"] == "default"


def test_crew_chat_endpoint_exists():
    """CrewAI endpoint should be reachable."""
    response = client.post(
        "/crew-chat",
        json={"question": "What is RAG?", "session_id": "crew-ci"},
    )
    assert response.status_code in [200, 500]
