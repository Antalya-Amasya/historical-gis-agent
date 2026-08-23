from fastapi.testclient import TestClient

from geography_mcp.server import app


def test_resolve_demo_place() -> None:
    response = TestClient(app).post("/tools/resolve_ancient_place", json={"name": "Carthago", "period": "-200"})
    assert response.status_code == 200
    assert response.json()["canonical_name"] == "Carthago"

