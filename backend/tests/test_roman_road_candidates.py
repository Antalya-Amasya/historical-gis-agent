from backend.app.candidate_routes.roman_roads import (
    RoadAccessStatus,
    RomanRoadCandidateService,
    RomanRoadCandidateStatus,
)
from backend.app.models import HistoricalPlace, HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def point(identifier: str, longitude: float, latitude: float, *, semantics=PlaceSpatialSemantics.SETTLEMENT, coordinate_role="exact_site") -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(id=identifier, canonical_name=identifier, longitude=longitude, latitude=latitude, source="Pleiades", confidence=0.9, spatial_semantics=semantics, coordinate_role=coordinate_role),
        event_summary="Evidence-backed anchor supplied by caller.", evidence_refs=[f"evidence-{identifier}"], confidence=0.9,
    )


def segment(identifier: str, geometry, *, status="Certain", dates=(9999, 9999)) -> RomanRoadSegment:
    return RomanRoadSegment(
        record_id=identifier, part_index=0, geometry=tuple(geometry), road_type="Main Road", route_type=2,
        segment_status=status, name=identifier, citation=f"citation-{identifier}", bibliography=f"bibliography-{identifier}",
        chronology=RoadChronology(None if dates[0] == 9999 else dates[0], None, None if dates[1] == 9999 else dates[1], None, {}),
        average_slope=None, passability=None, shape_length=None,
    )


def test_access_keeps_historical_coordinate_and_enforces_threshold():
    graph = RomanRoadGraph.from_segments([segment("one", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    anchor = point("anchor", 0.001, 0)
    access = RomanRoadCandidateService(graph, max_access_distance_m=500).resolve_access(anchor)
    assert access.status is RoadAccessStatus.AVAILABLE
    assert anchor.historical_place.longitude == 0.001
    assert access.historical_anchor_coordinate == (0.001, 0)
    assert access.roman_road_access_coordinate == (0, 0)
    assert RomanRoadCandidateService(graph, max_access_distance_m=50).resolve_access(anchor).status is RoadAccessStatus.TOO_FAR


def test_representative_settlement_coordinate_is_not_treated_as_an_exact_point():
    graph = RomanRoadGraph.from_segments([segment("one", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    access = RomanRoadCandidateService(graph).resolve_access(point("representative", 0, 0, coordinate_role="representative_point"))
    assert access.status is RoadAccessStatus.NON_POINT_PLACE
    assert access.road_node_id is None


def test_river_region_and_unknown_semantics_fail_closed_without_point_snapping():
    graph = RomanRoadGraph.from_segments([segment("one", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    service = RomanRoadCandidateService(graph)
    assert service.resolve_access(point("river", 0, 0, semantics=PlaceSpatialSemantics.RIVER)).status is RoadAccessStatus.RIVER_GEOMETRY_UNAVAILABLE
    assert service.resolve_access(point("region", 0, 0, semantics=PlaceSpatialSemantics.MOUNTAIN_REGION, coordinate_role="regional_centroid")).status is RoadAccessStatus.REGION_GEOMETRY_UNAVAILABLE
    assert service.resolve_access(point("unknown", 0, 0, semantics=PlaceSpatialSemantics.UNKNOWN)).status is RoadAccessStatus.UNKNOWN_PLACE_SEMANTICS


def test_disconnected_components_fail_closed_without_candidate():
    graph = RomanRoadGraph.from_segments([segment("one", ((0, 0), (1, 0))), segment("two", ((4, 0), (5, 0)))], snap_tolerance_m=5)
    result = RomanRoadCandidateService(graph, max_access_distance_m=500).build(point("a", 0, 0), point("b", 4, 0))
    assert result.status is RomanRoadCandidateStatus.DISCONNECTED
    assert result.candidate is None


def test_candidate_preserves_road_provenance_and_labels_access_connectors():
    graph = RomanRoadGraph.from_segments([
        segment("certain", ((0, 0), (1, 0)), status="Certain", dates=(-100, 50)),
        segment("conjectured", ((1, 0), (2, 0)), status="Conjectured"),
        segment("hypothetical", ((2, 0), (3, 0)), status="Hypothetical"),
    ], snap_tolerance_m=5)
    result = RomanRoadCandidateService(graph, max_access_distance_m=500).build(point("a", 0.001, 0), point("b", 2.999, 0))
    assert result.status is RomanRoadCandidateStatus.AVAILABLE
    candidate = result.candidate
    assert candidate is not None
    assert candidate.ordered_road_edge_ids == ["itiner-e:certain:part:0", "itiner-e:conjectured:part:0", "itiner-e:hypothetical:part:0"]
    assert candidate.itiner_e_record_ids == ["certain", "conjectured", "hypothetical"]
    assert candidate.segment_status_counts == {"Certain": 1, "Conjectured": 1, "Hypothetical": 1}
    assert candidate.chronology_counts == {"known": 1, "unknown": 2}
    assert [item.segment_type for item in candidate.geometry_segments] == ["access_connector", "roman_road", "access_connector"]
    assert candidate.access_connector_distance_m > 0
    assert "not evidence" in candidate.limitations[0]


def test_self_loop_provenance_cannot_pollute_shortest_path_traversal():
    graph = RomanRoadGraph.from_segments([
        segment("self-loop", ((0, 0), (0.00001, 0)), status="Hypothetical"),
        segment("road", ((0, 0), (1, 0))),
    ], snap_tolerance_m=5)
    result = RomanRoadCandidateService(graph, max_access_distance_m=500).build(point("a", 0, 0), point("b", 1, 0))
    assert result.candidate is not None
    assert result.candidate.ordered_road_edge_ids == ["itiner-e:road:part:0"]
    assert graph.stats.self_loop_segment_count == 1


def test_road_graph_does_not_create_claims_or_historical_points():
    graph = RomanRoadGraph.from_segments([segment("road", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    source = point("a", 0, 0)
    destination = point("b", 1, 0)
    RomanRoadCandidateService(graph, max_access_distance_m=500).build(source, destination)
    assert source.claim_ids == []
    assert destination.claim_ids == []
    assert source.historical_place.id == "a"
    assert destination.historical_place.id == "b"
