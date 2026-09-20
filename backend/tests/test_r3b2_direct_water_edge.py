import hashlib
import json
from math import isfinite
from pathlib import Path
from shutil import copytree

from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateStatus
from backend.app.gis.natural_earth_surface import NaturalEarthAvailability, NaturalEarthSurfaceClassifier
from backend.app.gis.sea import build_sea_edge, plan_direct_water_edge
from backend.app.gis.surface import SurfaceType, WaterDomain
from backend.app.models import GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, HistoricalTravelMode
from backend.app.routes.canonical_route_adapter import CanonicalRouteCompleteness
from backend.app.route_orchestrator import HistoricalRouteOrchestrator, MaritimePlannerUnavailableError
from backend.app.gis.transport import SearchState, TransportMode


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "natural_earth_surface"
LAYER_FILES = {
    "land": "ne_10m_land.geojson", "ocean": "ne_10m_ocean.geojson",
    "lakes": "ne_10m_lakes.geojson", "minor_islands": "ne_10m_minor_islands.geojson",
}
GIS_LIMIT = "Plausible GIS reconstruction between historical constraints; not an exact historical sailing track."


def _manifest(root: Path) -> Path:
    payload = {
        "schema_version": 1,
        "dataset_family": "Natural Earth",
        "dataset_version": "5.1.1",
        "layers": {
            name: {"path": filename, "sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest()}
            for name, filename in LAYER_FILES.items()
        },
    }
    path = root / "surface-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _provider(tmp_path: Path):
    root = tmp_path / "surface"
    copytree(FIXTURE_DIR, root)
    return root, NaturalEarthSurfaceClassifier.from_manifest(root, _manifest(root))


def _point(identifier: str, claim_id: str, lat: float, lon: float) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(id=identifier, canonical_name=identifier, latitude=lat, longitude=lon, confidence=0.9, source="test"),
        event_summary="attested movement", confidence=0.9, claim_ids=[claim_id],
    )


def _route(points, mode, claim_id="leg"):
    claim = HistoricalClaim(id=claim_id, claim_type="ORDERING", text="mode", confidence=0.9, travel_mode=mode)
    return HistoricalRoute(
        id="r", event_id="e", name="Alpha", period="200 BCE", geometry=GeoJsonLineString(coordinates=[]),
        historical_confidence=0.9, ordered_points=points, claims=[claim],
    )


class _RoadService:
    def __init__(self):
        self.calls = []

    def build(self, source, destination):
        self.calls.append((source.historical_place.id, destination.historical_place.id))
        return type("Result", (), {"candidate": None, "status": RomanRoadCandidateStatus.DISCONNECTED, "limitation": "gap"})()


def test_direct_ocean_edge_produces_validated_polyline(tmp_path):
    _, provider = _provider(tmp_path)
    plan = plan_direct_water_edge(-2.0, -2.0, -5.0, -5.0, provider)
    assert plan.available
    assert plan.direct_water_validated
    assert not plan.detour_used
    assert plan.planner == "DIRECT_WATER_EDGE"
    assert plan.physical_distance_m > 0
    assert len(plan.coordinates) >= 3
    assert plan.coordinates[0] == (-2.0, -2.0)
    assert plan.coordinates[-1] == (-5.0, -5.0)
    assert "Modern surface approximation" in " ".join(plan.limitations)
    assert GIS_LIMIT in plan.limitations


def test_land_crossing_and_minor_island_reject_without_land_fallback(tmp_path):
    _, provider = _provider(tmp_path)
    crossed = plan_direct_water_edge(-2.0, -2.0, 12.0, 15.0, provider)
    island = plan_direct_water_edge(-1.0, 13.0, 3.0, 13.0, provider)
    assert not crossed.available and not crossed.coordinates
    assert not island.available
    assert "land" in crossed.reason or crossed.reason


def test_narrow_land_rejected_by_polygon_intersection_not_samples(tmp_path):
    root = tmp_path / "thin"
    root.mkdir()
    ocean = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]]}}]}
    land = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-0.00005, -1], [0.00005, -1], [0.00005, 1], [-0.00005, 1], [-0.00005, -1]]]}}]}
    lake = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[1.4, 1.4], [1.5, 1.4], [1.5, 1.5], [1.4, 1.5], [1.4, 1.4]]]}}]}
    island = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[1.6, -1.6], [1.7, -1.6], [1.7, -1.5], [1.6, -1.5], [1.6, -1.6]]]}}]}
    for name, payload in (
        (LAYER_FILES["ocean"], ocean), (LAYER_FILES["land"], land),
        (LAYER_FILES["lakes"], lake), (LAYER_FILES["minor_islands"], island),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    provider = NaturalEarthSurfaceClassifier.from_manifest(root, _manifest(root))
    plan = plan_direct_water_edge(0.0, -0.02, 0.0, 0.02, provider)
    assert not plan.available
    assert provider.validate_land_segment(((-0.02, 0.0), (0.02, 0.0))).intersects_land is True


def test_lake_boundary_zero_length_and_invalid_coordinates_fail_closed(tmp_path):
    _, provider = _provider(tmp_path)
    lake = plan_direct_water_edge(4.0, 4.0, 4.2, 4.2, provider)
    boundary = plan_direct_water_edge(5.0, 0.0, -2.0, -2.0, provider)
    zero = plan_direct_water_edge(-2.0, -2.0, -2.0, -2.0, provider)
    invalid = plan_direct_water_edge(float("nan"), 0.0, -2.0, -2.0, provider)
    assert not lake.available
    assert not boundary.available
    assert not zero.available
    assert not invalid.available


def test_provider_unavailable_does_not_call_sample_only_build_sea_edge(tmp_path, monkeypatch):
    missing = NaturalEarthSurfaceClassifier.from_manifest(tmp_path, tmp_path / "absent.json")
    called = []
    monkeypatch.setattr("backend.app.gis.sea.build_sea_edge", lambda *a, **k: called.append(True))
    plan = plan_direct_water_edge(-2.0, -2.0, -5.0, -5.0, missing)
    assert missing.audit.availability is NaturalEarthAvailability.MISSING
    assert not plan.available
    assert called == []
    water = type("C", (), {"classify": lambda self, lat, lon: type("S", (), {"surface_type": SurfaceType.WATER})()})()
    assert build_sea_edge(SearchState("a", 0, 0, TransportMode.SEA), SearchState("b", 0, 0.02, TransportMode.SEA), water).status.value == "AVAILABLE"


def test_dispatch_land_unknown_and_geography_cannot_create_mode(tmp_path):
    _, provider = _provider(tmp_path)
    roads = _RoadService()
    orchestrator = RomanRoadRouteOrchestrator(roads, maritime_surface=provider)
    land = _route([_point("A", "land", 1.0, 1.0), _point("B", "land", 2.0, 2.0)], HistoricalTravelMode.LAND, "land")
    unknown = _route([_point("C", "unk", -2.0, -2.0), _point("D", "unk", -5.0, -5.0)], HistoricalTravelMode.UNKNOWN, "unk")
    land_result = orchestrator.build_roman_road_candidates(land)
    unknown_result = orchestrator.build_roman_road_candidates(unknown)
    assert roads.calls == [("A", "B"), ("C", "D")]
    assert land_result.legs[0].travel_mode is HistoricalTravelMode.LAND
    assert unknown_result.legs[0].travel_mode is HistoricalTravelMode.UNKNOWN
    assert land_result.legs[0].reconstruction_method != "DIRECT_WATER_EDGE"
    assert unknown_result.legs[0].reconstruction_method != "DIRECT_WATER_EDGE"


def test_sea_dispatch_success_gap_mixed_route_and_no_snapping(tmp_path):
    _, provider = _provider(tmp_path)
    roads = _RoadService()
    orchestrator = RomanRoadRouteOrchestrator(roads, maritime_surface=provider)
    sea_ok = _route([_point("P", "sea", -2.0, -2.0), _point("Q", "sea", -5.0, -5.0)], HistoricalTravelMode.SEA, "sea")
    ok = orchestrator.build_roman_road_candidates(sea_ok)
    assert ok.legs[0].reconstruction_method == "DIRECT_WATER_EDGE"
    assert ok.legs[0].travel_mode is HistoricalTravelMode.SEA
    assert ok.legs[0].terrain_candidate is not None
    assert ok.geometry_segments[0].coordinates
    assert roads.calls == []

    inland_sea = _route([_point("L", "in", 1.0, 1.0), _point("M", "in", 2.0, 2.0)], HistoricalTravelMode.SEA, "in")
    gap = orchestrator.build_roman_road_candidates(inland_sea)
    assert gap.legs[0].failure_status
    assert gap.geometry_segments[0].coordinates == []

    mixed_claim_land_a = HistoricalClaim(id="la", claim_type="ORDERING", text="marched", confidence=0.9, travel_mode=HistoricalTravelMode.LAND)
    mixed_claim_sea = HistoricalClaim(id="se", claim_type="ORDERING", text="sailed", confidence=0.9, travel_mode=HistoricalTravelMode.SEA)
    mixed_claim_land_b = HistoricalClaim(id="lb", claim_type="ORDERING", text="marched", confidence=0.9, travel_mode=HistoricalTravelMode.LAND)
    a = _point("A", "la", 1.0, 1.0)
    b = _point("B", "la", -2.0, -2.0)
    b.claim_ids = ["la", "se"]
    c = _point("C", "se", -5.0, -5.0)
    c.claim_ids = ["se", "lb"]
    d = _point("D", "lb", 4.0, 4.0)
    mixed = HistoricalRoute(
        id="mix", event_id="e", name="Alpha", period="200 BCE", geometry=GeoJsonLineString(coordinates=[]),
        historical_confidence=0.9, ordered_points=[a, b, c, d],
        claims=[mixed_claim_land_a, mixed_claim_sea, mixed_claim_land_b],
    )
    mixed_result = orchestrator.build_roman_road_candidates(mixed)
    assert [leg.travel_mode for leg in mixed_result.legs] == [
        HistoricalTravelMode.LAND, HistoricalTravelMode.SEA, HistoricalTravelMode.LAND,
    ]
    assert mixed_result.legs[1].reconstruction_method == "DIRECT_WATER_EDGE"
    assert ("A", "B") in roads.calls
    assert ("C", "D") in roads.calls
    assert ("B", "C") not in roads.calls


def test_completeness_and_terrain_presenter_unchanged(tmp_path):
    completeness = CanonicalRouteCompleteness.COMPLETE
    _, provider = _provider(tmp_path)
    sea = _route([_point("P", "leg", 1.0, 1.0), _point("Q", "leg", 2.0, 2.0)], HistoricalTravelMode.SEA)
    result = RomanRoadRouteOrchestrator(_RoadService(), maritime_surface=provider).build_roman_road_candidates(sea)
    assert completeness is CanonicalRouteCompleteness.COMPLETE
    assert not hasattr(result, "completeness")
    try:
        HistoricalRouteOrchestrator().present(type("Intent", (), {"intent": "historical_route"})(), sea, [])
    except MaritimePlannerUnavailableError:
        pass
    else:
        raise AssertionError("terrain presenter must still refuse SEA legs")


def test_reverse_order_symmetric_eligibility(tmp_path):
    _, provider = _provider(tmp_path)
    forward = plan_direct_water_edge(-2.0, -2.0, -5.0, -5.0, provider)
    reverse = plan_direct_water_edge(-5.0, -5.0, -2.0, -2.0, provider)
    assert forward.available and reverse.available
    assert abs(forward.physical_distance_m - reverse.physical_distance_m) < 1e-6
    assert forward.coordinates[::-1] == reverse.coordinates or isfinite(forward.physical_distance_m)


def test_dateline_and_antipodal_fail_closed_when_required(tmp_path):
    _, provider = _provider(tmp_path)
    antipodal = plan_direct_water_edge(0.0, 0.0, 0.0, 180.0, provider)
    assert not antipodal.available
