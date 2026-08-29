from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator, RomanRoadRouteStatus
from backend.app.candidate_routes.geographic import GeographicCandidateRouteService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.terrain import SyntheticTerrainProvider
from backend.app.models import GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, PlaceSpatialSemantics
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


def test_road_is_preferred_then_terrain_fallback_preserves_waypoint_order_and_ordering_provenance():
    graph = RomanRoadGraph.from_segments([road("ab", ((0, 0), (0.1, 0)))], snap_tolerance_m=5)
    points = [point("a", 0, 0), point("b", .1, 0), point("c", .2, 0)]
    points[0].claim_ids = ["a-b"]
    points[1].claim_ids = ["a-b", "b-c"]
    points[2].claim_ids = ["b-c"]
    route = historical_route(points)
    route.claims = [
        HistoricalClaim(id="a-b", claim_type="ORDERING", text="a precedes b", confidence=.9, movement_relation="SAME_MOVEMENT_EVENT", sequence_status="explicit"),
        HistoricalClaim(id="b-c", claim_type="WAYPOINT_ORDERING", text="b precedes c", confidence=.8, sequence_status="explicit"),
    ]
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(graph, max_access_distance_m=500),
        terrain_route_service=GeographicCandidateRouteService(SyntheticTerrainProvider()),
    ).build_roman_road_candidates(route)

    assert result.status is RomanRoadRouteStatus.COMPLETE
    assert [leg.reconstruction_method for leg in result.legs] == ["ROMAN_ROAD_NETWORK", "TERRAIN_ASTAR_FALLBACK"]
    assert [item.segment_type for item in result.geometry_segments] == ["roman_road", "terrain_candidate"]
    fallback = result.legs[1].terrain_candidate
    assert fallback is not None
    assert fallback.geometry.coordinates[0] == (.1, 0) and fallback.geometry.coordinates[-1] == (.2, 0)
    assert fallback.generation_method == "terrain_astar" and fallback.segment_ledger[0].geometry_role == "algorithmic_candidate"
    assert result.legs[0].ordering_provenance[0]["claim_type"] == "ORDERING"
    assert result.legs[1].ordering_provenance[0]["claim_type"] == "WAYPOINT_ORDERING"
    presentation = RomanRoadPresentationService().present(route, result)
    assert presentation.road_network["terrain_fallback_used"] is True
    assert presentation.presentation_summary["route_method"] == "ROMAN_ROAD_PREFERRED_TERRAIN_FALLBACK"
    assert any(feature["properties"]["layer_type"] == "terrain_reconstruction_segment" for feature in presentation.geojson["features"])


def test_failed_terrain_fallback_remains_an_explicit_gap_without_skipping_waypoints():
    class FailingTerrain:
        def build_between(self, *_args, **_kwargs):
            raise ValueError("blocked terrain")

    graph = RomanRoadGraph.from_segments([road("far", ((5, 0), (6, 0)))], snap_tolerance_m=5)
    route = historical_route([point("a", 0, 0), point("b", .1, 0), point("c", .2, 0)])
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(graph, max_access_distance_m=500), terrain_route_service=FailingTerrain(),
    ).build_roman_road_candidates(route)

    assert result.status is RomanRoadRouteStatus.UNAVAILABLE
    assert [leg.terrain_candidate for leg in result.legs] == [None, None]
    assert [item.segment_type for item in result.geometry_segments] == ["failed_gap", "failed_gap"]
    assert all(leg.failure_status == "TERRAIN_RECONSTRUCTION_FAILED" for leg in result.legs)
