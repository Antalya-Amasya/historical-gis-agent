import pytest
from fastapi.testclient import TestClient
import backend.app.main as main
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import RuleBasedFakeLLMProvider
from backend.app.rag.retriever import EmptyHistoricalRetriever

@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    monkeypatch.setattr(main, "agent", HistoricalGisAgent(RuleBasedFakeLLMProvider(), EmptyHistoricalRetriever(), max_steps=4, max_tool_executions=10))

def test_bounded_agent_api_returns_auditable_tool_state_without_key() -> None:
    response = TestClient(main.app).post("/api/v1/agent/chat", json={"session_id": "demo", "message": "How did Polybius describe the Alpine crossing?"})
    assert response.status_code == 200
    state = response.json()["state"]
    assert state["status"] == "completed"
    assert state["tool_history"][0]["tool_name"] == "search_historical_evidence"
    assert state["tool_history"][0]["success"]
    assert state["historical_route"] is None

def test_session_state_persists() -> None:
    client = TestClient(main.app)
    client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "Hannibal Alps"})
    response = client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "continue"})
    assert len(response.json()["state"]["messages"]) == 4
