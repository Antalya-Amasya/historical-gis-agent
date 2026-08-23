from fastapi.testclient import TestClient
from geography_mcp.service import GeographyService, GeoPoint, MockElevationProvider
from geography_mcp.server import app


def test_service_mock_and_profile():
    service = GeographyService()
    assert isinstance(service.elevation_provider, MockElevationProvider)
    profile = service.get_elevation_profile([GeoPoint(latitude=0, longitude=0), GeoPoint(latitude=1, longitude=1)])
    assert profile.provider == "mock"
    assert len(profile.points) == 2


def test_http_four_tools():
    client = TestClient(app)
    assert client.post("/tools/resolve_ancient_place", json={"name":"Carthago"}).status_code == 200
    assert client.post("/tools/calculate_distance", json={"point_a":{"latitude":0,"longitude":0},"point_b":{"latitude":0,"longitude":1}}).status_code == 200
    assert client.post("/tools/get_elevation", json={"latitude":0,"longitude":0}).status_code == 200
    assert client.post("/tools/get_elevation_profile", json={"points":[{"latitude":0,"longitude":0}]}).status_code == 200
