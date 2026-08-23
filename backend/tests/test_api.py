from fastapi.testclient import TestClient

from backend.app.main import app


def test_mock_agent_returns_auditable_historical_places_without_key() -> None:
    response = TestClient(app).post(
        "/api/v1/agent/chat",
        json={"session_id": "demo", "message": "分析汉尼拔翻越阿尔卑斯"},
    )
    assert response.status_code == 200
    body = response.json()
    event = body["state"]["current_event"]
    assert event["id"] == "hannibal-alps-218-bc"
    assert len(event["places"]) == 2
    for place in event["places"]:
        assert place["source"] == "Pleiades: A Gazetteer of Past Places"
        assert place["source_id"]
        assert place["source_url"].startswith("https://pleiades.stoa.org/places/")
        assert isinstance(place["latitude"], float)
        assert isinstance(place["longitude"], float)
    assert "不主张具体路线" in body["reply"]


def test_session_state_persists() -> None:
    client = TestClient(app)
    client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "Hannibal Alps"})
    response = client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "继续"})
    assert len(response.json()["state"]["messages"]) == 4
