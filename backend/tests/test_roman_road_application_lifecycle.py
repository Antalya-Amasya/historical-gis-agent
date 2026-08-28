from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def tiny_graph():
    segment = RomanRoadSegment("road", 0, ((0, 0), (1, 0)), "Main Road", 1, "Certain", "road", None, None, RoadChronology(None, None, None, None, {}), None, None, None)
    return RomanRoadGraph.from_segments([segment], snap_tolerance_m=5)


def test_disabled_capability_does_not_load_dataset(monkeypatch):
    monkeypatch.setattr(main.settings, "roman_road_enabled", False)
    monkeypatch.setattr(main.RomanRoadGraph, "load", lambda *_args: (_ for _ in ()).throw(AssertionError("must not load")))
    with TestClient(main.app):
        pass


def test_enabled_capability_loads_once_per_application_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(main.settings, "roman_road_enabled", True)
    monkeypatch.setattr(main.settings, "roman_road_geojson_path", "fixture.geojson")
    monkeypatch.setattr(main.Path, "is_file", lambda _self: True)
    monkeypatch.setattr(main.RomanRoadGraph, "load", lambda path: calls.append(path) or tiny_graph())
    with TestClient(main.app) as client:
        client.get("/health")
        client.get("/health")
    assert calls == [main.Path("fixture.geojson")]
