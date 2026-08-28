from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def point(name, lon, *, semantic=PlaceSpatialSemantics.SETTLEMENT):
    return HistoricalRoutePoint(sequence=1, historical_place=HistoricalPlace(id=name, canonical_name=name, longitude=lon, latitude=0, source="fixture", confidence=.9, spatial_semantics=semantic), event_summary="evidence-backed input", evidence_refs=[name], confidence=.9)


def graph():
    segment = RomanRoadSegment(record_id="road", part_index=0, geometry=((0, 0), (1, 0)), road_type="Main Road", route_type=1, segment_status="Conjectured", name="road", citation="citation", bibliography="bibliography", chronology=RoadChronology(None, None, None, None, {}), average_slope=None, passability=None, shape_length=None)
    return RomanRoadGraph.from_segments([segment], snap_tolerance_m=5)


def render(points):
    route = HistoricalRoute(id="route", event_id="event", name="contract probe", period="test", ordered_points=points, geometry=GeoJsonLineString(coordinates=[(p.historical_place.longitude, p.historical_place.latitude) for p in points]), historical_confidence=.8)
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph(), max_access_distance_m=500)).build_roman_road_candidates(route)
    return RomanRoadPresentationService().present(route, result)


def test_complete_presentation_separates_anchor_access_and_real_road_geometry():
    payload = render([point("a", .001), point("b", .999)])
    features = payload.geojson["features"]
    assert payload.road_network["route_status"] == "COMPLETE"
    assert [item["geometry"]["coordinates"] for item in features[:2]] == [[.001, 0], [.999, 0]]
    roles = [item["properties"].get("segment_role") for item in features[2:]]
    assert roles == ["access_connector", "roman_road", "access_connector"]
    assert features[3]["geometry"]["coordinates"] == [[0.0, 0.0], [1.0, 0.0]]
    assert payload.road_network["aggregate"]["chronology_counts"] == {"unknown": 1}


def test_partial_presentation_keeps_failed_gap_empty_and_reason_visible():
    payload = render([point("a", 0), point("river", 1, semantic=PlaceSpatialSemantics.RIVER)])
    failed = [item for item in payload.geojson["features"] if item["properties"].get("segment_role") == "failed_gap"]
    assert payload.road_network["route_status"] == "UNAVAILABLE"
    assert failed == [{"type": "Feature", "geometry": None, "properties": {"layer_type": "roman_road_segment", "segment_role": "failed_gap", "leg_index": 1, "source_anchor_id": "a", "destination_anchor_id": "river", "failure_status": "RIVER_GEOMETRY_UNAVAILABLE"}}]
