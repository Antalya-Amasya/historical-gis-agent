from fastapi.testclient import TestClient

from backend.app.main import app


def test_mock_agent_runs_without_key() -> None:
    response = TestClient(app).post(
        "/api/v1/agent/chat",
        json={"session_id": "demo", "message": "分析汉尼拔翻越阿尔卑斯"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["state"]["current_event"]["id"] == "hannibal-alps-218-bc"
    assert "不对实际路线作判断" in body["reply"]


def test_session_state_persists() -> None:
    client = TestClient(app)
    client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "Hannibal Alps"})
    response = client.post("/api/v1/agent/chat", json={"session_id": "memory", "message": "继续"})
    assert len(response.json()["state"]["messages"]) == 4

