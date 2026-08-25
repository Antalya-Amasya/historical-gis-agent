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


def test_historical_route_presentation_endpoint_returns_geojson_and_panels() -> None:
    response = TestClient(main.app).get("/api/v1/historical-routes/phase10-demo-route/presentation")
    body = response.json()
    assert response.status_code == 200
    assert body["geojson"]["type"] == "FeatureCollection"
    assert body["geojson"]["features"][0]["geometry"]["type"] == "LineString"
    assert body["knowledge_panels"][0]["evidence_refs"] == ["polybius_iii"]
    unresolved = next(feature for feature in body["geojson"]["features"] if feature["properties"].get("waypoint_id") == "unresolved")
    assert unresolved["geometry"] is None


def test_historical_route_presentation_missing_route_is_404() -> None:
    response = TestClient(main.app).get("/api/v1/historical-routes/missing/presentation")
    assert response.status_code == 404


def test_historical_route_presentation_contract_error_is_422(monkeypatch) -> None:
    def invalid(_route_id):
        raise main.PresentationContractError("invalid")

    monkeypatch.setattr(main.historical_route_presentation_service, "get_presentation", invalid)
    response = TestClient(main.app).get("/api/v1/historical-routes/phase10-demo-route/presentation")
    assert response.status_code == 422
