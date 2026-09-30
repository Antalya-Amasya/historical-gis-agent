from pathlib import Path
from shutil import copytree

from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.gis.natural_earth_surface import NaturalEarthAvailability, NaturalEarthSurfaceClassifier
from backend.app.models import HistoricalTravelMode
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology
from backend.app.routes.canonical_route_adapter import CanonicalRouteCompleteness
from backend.tests.test_r3b2_direct_water_edge import FIXTURE_DIR, LAYER_FILES, _manifest, _point, _route


def _tiny_graph():
    segment = RomanRoadSegment("road", 0, ((0, 0), (1, 0)), "Main Road", 1, "Certain", "road", None, None, RoadChronology(None, None, None, None, {}), None, None, None)
    return RomanRoadGraph.from_segments([segment], snap_tolerance_m=5)


def _enable_roman_roads(monkeypatch):
    monkeypatch.setattr(main.settings, "roman_road_enabled", True)
    monkeypatch.setattr(main.settings, "roman_road_geojson_path", "fixture.geojson")
    monkeypatch.setattr(main.Path, "is_file", lambda self: True)
    monkeypatch.setattr(main.RomanRoadGraph, "load", lambda path: _tiny_graph())


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "surface"
    copytree(FIXTURE_DIR, root)
    _manifest(root)
    return root


def _orchestrator(monkeypatch, data_root=None):
    _enable_roman_roads(monkeypatch)
    monkeypatch.setattr(main.settings, "maritime_surface_data_root", None if data_root is None else str(data_root))
    with TestClient(main.app):
        return main.agent.tools.roman_road_orchestrator


def test_no_data_root_builds_and_sea_gaps_without_affecting_land(monkeypatch):
    orchestrator = _orchestrator(monkeypatch, None)
    assert orchestrator is not None
    assert orchestrator.maritime_surface is None
    sea = _route([_point("P", "leg", -2.0, -2.0), _point("Q", "leg", -5.0, -5.0)], HistoricalTravelMode.SEA)
    land = _route([_point("A", "leg", 0.0, 0.0), _point("B", "leg", 1.0, 0.0)], HistoricalTravelMode.LAND)
    unknown = _route([_point("C", "leg", 0.0, 0.0), _point("D", "leg", 1.0, 0.0)], HistoricalTravelMode.UNKNOWN)
    sea_result = orchestrator.build_roman_road_candidates(sea)
    land_result = orchestrator.build_roman_road_candidates(land)
    unknown_result = orchestrator.build_roman_road_candidates(unknown)
    assert sea_result.legs[0].failure_status == "MARITIME_PLANNER_UNAVAILABLE"
    assert sea_result.geometry_segments[0].coordinates == []
    assert land_result.legs[0].travel_mode is HistoricalTravelMode.LAND
    assert land_result.legs[0].reconstruction_method != "DIRECT_WATER_EDGE"
    assert unknown_result.legs[0].travel_mode is HistoricalTravelMode.UNKNOWN
    assert CanonicalRouteCompleteness.COMPLETE is CanonicalRouteCompleteness.COMPLETE


def test_valid_manifest_injects_available_provider_once(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path)
    loads = []
    original = NaturalEarthSurfaceClassifier.from_manifest

    def counted(dataset_dir, manifest_path):
        loads.append((dataset_dir, manifest_path))
        return original(dataset_dir, manifest_path)

    monkeypatch.setattr(main.NaturalEarthSurfaceClassifier, "from_manifest", staticmethod(counted))
    _enable_roman_roads(monkeypatch)
    monkeypatch.setattr(main.settings, "maritime_surface_data_root", str(root))
    with TestClient(main.app) as client:
        client.get("/health")
        client.get("/health")
        provider = main.agent.tools.roman_road_orchestrator.maritime_surface
    assert len(loads) == 1
    assert provider is not None
    assert provider.audit.availability is NaturalEarthAvailability.AVAILABLE


def test_wired_open_water_sea_leg_is_direct_water_edge(monkeypatch, tmp_path):
    orchestrator = _orchestrator(monkeypatch, _fixture_root(tmp_path))
    sea = _route([_point("P", "leg", -2.0, -2.0), _point("Q", "leg", -5.0, -5.0)], HistoricalTravelMode.SEA)
    result = orchestrator.build_roman_road_candidates(sea)
    assert result.legs[0].reconstruction_method == "DIRECT_WATER_EDGE"
    assert result.legs[0].travel_mode is HistoricalTravelMode.SEA
    presentation = RomanRoadPresentationService().present(sea, result)
    feature = next(item for item in presentation.geojson["features"] if item["properties"].get("segment_role") == "direct_water_edge")
    assert feature["properties"]["layer_type"] == "direct_water_edge"
    assert presentation.road_network["terrain_fallback_used"] is False


def test_missing_hash_mismatch_and_corrupt_do_not_inject_or_crash(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    assert _orchestrator(monkeypatch, missing).maritime_surface is None

    mismatched = _fixture_root(tmp_path / "hash")
    (mismatched / LAYER_FILES["ocean"]).write_bytes(b"changed")
    assert _orchestrator(monkeypatch, mismatched).maritime_surface is None

    corrupt = _fixture_root(tmp_path / "corrupt")
    (corrupt / "surface-manifest.json").write_text("{not-json", encoding="utf-8")
    assert _orchestrator(monkeypatch, corrupt).maritime_surface is None


def test_raw_unhashed_constructor_is_not_used_for_production_surface(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path)
    hashes = []
    original = NaturalEarthSurfaceClassifier.__init__

    def tracked(self, dataset_dir, *, expected_hashes=None, availability=None):
        hashes.append(expected_hashes)
        return original(self, dataset_dir, expected_hashes=expected_hashes, availability=availability)

    monkeypatch.setattr(NaturalEarthSurfaceClassifier, "__init__", tracked)
    orchestrator = _orchestrator(monkeypatch, root)
    assert orchestrator.maritime_surface is not None
    assert hashes and hashes[0]
    assert all(item is not None for item in hashes)


def test_maritime_loads_without_roman_roads_and_land_remains_a_gap(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "roman_road_enabled", False)
    monkeypatch.setattr(main.settings, "dem_hgt_dir", None)
    monkeypatch.setattr(main.settings, "maritime_surface_data_root", str(_fixture_root(tmp_path)))
    monkeypatch.setattr(main.RomanRoadGraph, "load", lambda *_args: (_ for _ in ()).throw(AssertionError("roads disabled")))
    with TestClient(main.app) as client:
        assert client.get("/health").json()["natural_earth"] == "ACTIVE"
        assert client.get("/health").json()["itiner_e"] == "UNAVAILABLE"
        orchestrator = main.agent.tools.roman_road_orchestrator
        assert orchestrator.terrain_route_service is None
        sea = _route([_point("P", "leg", -2.0, -2.0), _point("Q", "leg", -5.0, -5.0)], HistoricalTravelMode.SEA)
        land = _route([_point("A", "leg", 0.0, 0.0), _point("B", "leg", 1.0, 0.0)], HistoricalTravelMode.LAND)
        assert orchestrator.build_roman_road_candidates(sea).geometry_segments[0].segment_type == "direct_water_edge"
        land_result = orchestrator.build_roman_road_candidates(land)
        assert land_result.legs[0].failure_status == "LAND_NETWORK_UNAVAILABLE"
        assert land_result.geometry_segments[0].coordinates == []
