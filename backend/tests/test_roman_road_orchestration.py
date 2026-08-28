from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator, RomanRoadRouteStatus
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def road(identifier: str, geometry, *, status="Certain", known=False) -> RomanRoadSegment:
    return RomanRoadSegment(
        record_id=identifier, part_index=0, geometry=tuple(geometry), road_type="Main Road", route_type=1,
        segment_status=status, name=identifier, citation=f"citation-{identifier}", bibliography=f"bibliography-{identifier}",
        chronology=RoadChronology(-100 if known else None, None, 50 if known else None, None, {}),
        average_slope=None, passability=None, shape_length=None,
    )


def point(identifier: str, longitude: float, latitude: float, *, semantics=PlaceSpatialSemantics.SETTLEMENT) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=1, historical_place=HistoricalPlace(id=identifier, canonical_name=identifier, longitude=longitude, latitude=latitude, source="fixture", confidence=.9, spatial_semantics=semantics),
        event_summary="caller-provided evidence-backed anchor", evidence_refs=[f"e-{identifier}"], confidence=.9,
    )


def historical_route(points: list[HistoricalRoutePoint]) -> HistoricalRoute:
    return HistoricalRoute(id="fixture-route", event_id="fixture", name="fixture", period="fixture", ordered_points=points, geometry=GeoJsonLineString(coordinates=[(p.historical_place.longitude, p.historical_place.latitude) for p in points]), historical_confidence=.9)


def test_complete_multileg_route_uses_one_injected_graph_and_aggregates_provenance():
    graph = RomanRoadGraph.from_segments([
        road("one", ((0, 0), (1, 0)), status="Certain", known=True),
        road("two", ((1, 0), (2, 0)), status="Conjectured"),
    ], snap_tolerance_m=5)
    route = historical_route([point("a", 0, 0), point("b", 1, 0), point("c", 2, 0)])
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph, max_access_distance_m=500)).build_roman_road_candidates(route)
    assert result.status is RomanRoadRouteStatus.COMPLETE
    assert [leg.candidate.ordered_road_edge_ids for leg in result.legs] == [["itiner-e:one:part:0"], ["itiner-e:two:part:0"]]
    assert result.aggregate.successful_leg_count == 2
    assert result.aggregate.segment_status_counts == {"Certain": 1, "Conjectured": 1}
    assert result.aggregate.chronology_counts == {"known": 1, "unknown": 1}
    assert {item.segment_type for item in result.geometry_segments} == {"roman_road"}


def test_failed_middle_leg_remains_explicit_gap_and_is_not_crossed():
    graph = RomanRoadGraph.from_segments([
        road("ab", ((0, 0), (1, 0))),
        road("cd", ((4, 0), (5, 0))),
    ], snap_tolerance_m=5)
    route = historical_route([point("a", 0, 0), point("b", 1, 0), point("c", 4, 0), point("d", 5, 0)])
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph, max_access_distance_m=500)).build_roman_road_candidates(route)
    assert result.status is RomanRoadRouteStatus.PARTIAL
    assert [leg.candidate is not None for leg in result.legs] == [True, False, True]
    assert result.legs[1].failure_status == "ROMAN_ROAD_DISCONNECTED"
    failed = [item for item in result.geometry_segments if item.segment_type == "failed_gap"]
    assert len(failed) == 1 and failed[0].coordinates == []
    assert result.aggregate.successful_leg_count == 2
    assert result.aggregate.failed_leg_count == 1


def test_unavailable_semantic_leg_propagates_reason_and_preserves_input_route():
    graph = RomanRoadGraph.from_segments([road("ab", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    source, river = point("a", 0, 0), point("river", 1, 0, semantics=PlaceSpatialSemantics.RIVER)
    route = historical_route([source, river])
    before = route.model_dump(mode="json")
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph, max_access_distance_m=500)).build_roman_road_candidates(route)
    assert result.status is RomanRoadRouteStatus.UNAVAILABLE
    assert result.legs[0].failure_status == "RIVER_GEOMETRY_UNAVAILABLE"
    assert route.model_dump(mode="json") == before
    assert source.claim_ids == [] and river.claim_ids == []


def test_access_connectors_remain_separate_from_roman_road_geometry():
    graph = RomanRoadGraph.from_segments([road("ab", ((0, 0), (1, 0)), status="Hypothetical")], snap_tolerance_m=5)
    route = historical_route([point("a", .001, 0), point("b", .999, 0)])
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph, max_access_distance_m=500)).build_roman_road_candidates(route)
    assert [item.segment_type for item in result.geometry_segments] == ["access_connector", "roman_road", "access_connector"]
    assert result.legs[0].candidate.segment_status_counts == {"Hypothetical": 1}
