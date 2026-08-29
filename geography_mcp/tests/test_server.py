from fastapi.testclient import TestClient

from geography_mcp.server import app


def test_resolve_demo_place_uses_shared_auditable_repository() -> None:
    response = TestClient(app).post("/tools/resolve_ancient_place", json={"name": "Carthago", "period": "-200"})
    assert response.status_code == 200
    place = response.json()
    assert place["canonical_name"] == "Carthago"
    assert place["source_id"] == "314921"
    assert place["source_url"] == "https://pleiades.stoa.org/places/314921"


def test_alesia_resolves_to_auditable_pleiades_representative_point() -> None:
    response = TestClient(app).post("/tools/resolve_ancient_place", json={"name": "Alesia", "period": "52 BCE"})
    assert response.status_code == 200
    place = response.json()
    assert place["found"] is True and place["canonical_name"] == "Alesia"
    assert place["source_id"] == "177434"
    assert place["source_url"] == "https://pleiades.stoa.org/places/177434"
    assert place["coordinate_role"] == "representative_point" and place["uncertain"] is True
