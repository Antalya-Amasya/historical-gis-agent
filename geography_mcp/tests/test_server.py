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


def test_gergovia_resolves_by_authoritative_name_and_alias() -> None:
    client = TestClient(app)
    for name in ("Gergovia", "Gergovie"):
        response = client.post("/tools/resolve_ancient_place", json={"name": name, "period": "52 BCE"})
        assert response.status_code == 200
        place = response.json()
        assert place["found"] is True and place["canonical_name"] == "Gergovia"
        assert place["source_id"] == "138373"
        assert place["coordinate_role"] == "representative_point"
        assert place["spatial_semantics"] == "settlement"


def test_bituriges_is_a_broad_region_not_a_fake_settlement_point() -> None:
    response = TestClient(app).post(
        "/tools/resolve_ancient_place", json={"name": "Bituriges", "period": "52 BCE"}
    )
    assert response.status_code == 200
    place = response.json()
    assert place["found"] is True and place["canonical_name"] == "Bituriges Cubi"
    assert place["source_id"] == "138225"
    assert place["coordinate_role"] == "regional_centroid"
    assert place["spatial_semantics"] == "region"
    assert place["uncertain"] is True


def test_unknown_place_remains_unresolved() -> None:
    response = TestClient(app).post(
        "/tools/resolve_ancient_place", json={"name": "Definitely Not A Registered Ancient Place"}
    )
    assert response.status_code == 200
    assert response.json() == {"found": False}
