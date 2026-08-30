from pathlib import Path

import pytest

from backend.app.candidate_routes.barrier_crossings import (
    BarrierCrossingService,
    CrossingCandidateSource,
    is_broad_mountain_constraint,
)
from backend.app.candidate_routes.geographic import GeographicCandidateRouteService
from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator, RomanRoadRouteStatus
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.candidate_routes.terrain import SyntheticTerrainProvider
from backend.app.models import Evidence, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRouteIntent, HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.route_orchestrator import BarrierCrossingConstraintError, HistoricalRouteOrchestrator
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def road(identifier: str, geometry) -> RomanRoadSegment:
    return RomanRoadSegment(
        record_id=identifier,
        part_index=0,
        geometry=tuple(geometry),
        road_type="Main Road",
        route_type=1,
        segment_status="Certain",
        name=identifier,
        citation=f"citation-{identifier}",
        bibliography=None,
        chronology=RoadChronology(None, None, None, None, {}),
        average_slope=None,
        passability=None,
        shape_length=None,
    )


def point(identifier: str, longitude: float, latitude: float, *, sequence: int = 1) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=sequence,
        historical_place=HistoricalPlace(
            id=identifier,
            canonical_name=identifier,
            longitude=longitude,
            latitude=latitude,
            source="fixture",
            confidence=0.9,
            spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
            spatial_semantics_provenance="fixture audited settlement",
        ),
        event_summary="evidence-backed fixture anchor",
        evidence_refs=[f"e-{identifier}"],
        confidence=0.9,
    )


def barrier(identifier: str = "generic-range", longitude: float = 1.05, latitude: float = 0.0) -> HistoricalRoutePoint:
    item = point(identifier, longitude, latitude, sequence=2)
    item.historical_place.coordinate_role = "regional_centroid"
    item.historical_place.spatial_semantics = PlaceSpatialSemantics.MOUNTAIN_REGION
    item.historical_place.spatial_semantics_provenance = "audited gazetteer mountain-region identity"
    return item


def route(points: list[HistoricalRoutePoint]) -> HistoricalRoute:
    for index, item in enumerate(points, start=1):
        item.sequence = index
    return HistoricalRoute(
        id="generic-crossing-route",
        event_id="generic-event",
        name="Generic ordered crossing",
        period="fixture",
        ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[
            (item.historical_place.longitude, item.historical_place.latitude) for item in points
        ]),
        evidence_refs=[reference for item in points for reference in item.evidence_refs],
        historical_confidence=0.9,
    )


def connected_service(*, terrain: bool = False) -> BarrierCrossingService:
    graph = RomanRoadGraph.from_segments([
        road("west", ((0.0, 0.0), (1.0, 0.0))),
        road("east", ((1.0, 0.0), (2.0, 0.0))),
    ], snap_tolerance_m=5)
    terrain_service = GeographicCandidateRouteService(SyntheticTerrainProvider()) if terrain else None
    return BarrierCrossingService(
        RomanRoadCandidateService(graph, max_access_distance_m=500),
        terrain_route_service=terrain_service,
    )


def test_classification_is_conservative_and_exact_settlement_remains_exact():
    assert is_broad_mountain_constraint(point("ordinary-settlement", 1.0, 0.0)) is False
    assert is_broad_mountain_constraint(barrier()) is True
    unproven = barrier("unproven")
    unproven.historical_place.spatial_semantics_provenance = None
    assert is_broad_mountain_constraint(unproven) is False


def test_road_supported_crossing_is_algorithmic_and_preserves_evidence_set():
    approach, constraint, exit_point = point("approach", 0.0, 0.0), barrier(), point("exit", 2.0, 0.0)
    result = connected_service(terrain=True).build(approach, constraint, exit_point)
    crossing = result.crossing
    assert crossing is not None
    assert crossing.source is CrossingCandidateSource.ANCIENT_ROAD_BARRIER_CANDIDATE
    assert crossing.coordinate == (1.0, 0.0)
    assert crossing.coordinate != (constraint.historical_place.longitude, constraint.historical_place.latitude)
    assert crossing.road_support is True and crossing.terrain_support is True
    assert crossing.authority == "algorithmic_gis_candidate"
    assert crossing.historical_evidence_refs == ["e-approach", "e-generic-range", "e-exit"]
    assert set(crossing.road_edge_ids) == {"itiner-e:west:part:0", "itiner-e:east:part:0"}
    assert not isinstance(crossing, HistoricalRoutePoint)


def test_terrain_fallback_is_used_only_when_road_candidate_is_unavailable():
    graph = RomanRoadGraph.from_segments([road("remote", ((10.0, 10.0), (11.0, 10.0)))], snap_tolerance_m=5)
    service = BarrierCrossingService(
        RomanRoadCandidateService(graph, max_access_distance_m=500),
        terrain_route_service=GeographicCandidateRouteService(SyntheticTerrainProvider()),
    )
    result = service.build(point("approach", 0.0, 0.0), barrier(), point("exit", 2.0, 0.0))
    assert result.crossing is not None
    assert result.crossing.source is CrossingCandidateSource.TERRAIN_DERIVED_CROSSING
    assert result.crossing.road_support is False and result.crossing.terrain_support is True
    assert result.road_candidate is None and result.terrain_candidate is not None


def test_orchestrator_consumes_ordered_constraint_triple_without_mutating_historical_route():
    points = [point("approach", 0.0, 0.0), barrier(), point("exit", 2.0, 0.0)]
    historical_route = route(points)
    before = historical_route.model_dump(mode="json")
    service = connected_service()
    result = RomanRoadRouteOrchestrator(service.road_service).build_roman_road_candidates(historical_route)
    assert result.status is RomanRoadRouteStatus.COMPLETE
    assert historical_route.model_dump(mode="json") == before
    assert [item.historical_place.id for item in historical_route.ordered_points] == ["approach", "generic-range", "exit"]
    assert len(result.legs) == 1
    assert result.legs[0].source_anchor_id == "approach"
    assert result.legs[0].barrier_anchor_id == "generic-range"
    assert result.legs[0].destination_anchor_id == "exit"
    assert result.legs[0].reconstruction_method == "ANCIENT_ROAD_BARRIER_CROSSING"


def test_missing_exit_constraint_is_explicit_and_centroid_is_not_routed_to():
    historical_route = route([point("approach", 0.0, 0.0), barrier()])
    service = connected_service()
    result = RomanRoadRouteOrchestrator(
        service.road_service,
        terrain_route_service=GeographicCandidateRouteService(SyntheticTerrainProvider()),
    ).build_roman_road_candidates(historical_route)
    assert result.status is RomanRoadRouteStatus.UNAVAILABLE
    assert result.legs[0].failure_status == "BARRIER_EXIT_CONSTRAINT_UNAVAILABLE"
    assert result.legs[0].crossing_candidate is None
    assert result.geometry_segments[0].segment_type == "failed_gap"
    assert result.geometry_segments[0].coordinates == []


def test_crossing_presentation_is_visually_distinct_and_not_a_knowledge_waypoint():
    historical_route = route([point("approach", 0.0, 0.0), barrier(), point("exit", 2.0, 0.0)])
    service = connected_service()
    result = RomanRoadRouteOrchestrator(service.road_service).build_roman_road_candidates(historical_route)
    payload = RomanRoadPresentationService().present(historical_route, result)
    crossing = next(feature for feature in payload.geojson["features"] if feature["properties"].get("layer_type") == "reconstructed_crossing")
    assert crossing["properties"]["authority"] == "algorithmic_gis_candidate"
    assert "waypoint_id" not in crossing["properties"]
    assert [panel["waypoint_id"] for panel in payload.knowledge_panels] == ["approach", "generic-range", "exit"]


def test_no_defensible_crossing_remains_unavailable_without_name_specific_logic():
    graph = RomanRoadGraph.from_segments([road("distant", ((0.0, 0.0), (2.0, 0.0)))], snap_tolerance_m=5)
    service = BarrierCrossingService(RomanRoadCandidateService(graph, max_access_distance_m=500), search_radius_m=10_000)
    result = service.build(point("west", 0.0, 0.0), barrier("arbitrary-range", 20.0, 20.0), point("east", 2.0, 0.0))
    assert result.crossing is None
    assert result.failure_reason == "NO_PATH_SUPPORTED_CROSSING_WITHIN_BARRIER_SEARCH_AREA"
    implementation = Path(__file__).parents[1] / "app" / "candidate_routes" / "barrier_crossings.py"
    assert "hannibal" not in implementation.read_text(encoding="utf-8").casefold()


def test_default_terrain_orchestrator_skips_centroid_and_adds_algorithmic_crossing():
    points = [point("approach", 0.0, 45.0), barrier(latitude=45.0), point("exit", 2.0, 45.0)]
    historical_route = route(points)
    evidence = [
        Evidence(id=f"e-{name}", author="Fixture", work="Fixture", locator=name, excerpt=name)
        for name in ("approach", "generic-range", "exit")
    ]
    response = HistoricalRouteOrchestrator(cell_size_m=25_000).present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence,
    )
    crossing = next(feature for feature in response.geojson["features"] if feature["properties"].get("layer_type") == "reconstructed_crossing")
    assert crossing["properties"]["authority"] == "algorithmic_gis_candidate"
    assert crossing["geometry"]["coordinates"] != [1.05, 45.0]
    assert [1.05, 45.0] not in response.route_geojson["geometry"]["coordinates"]
    assert [item.name for item in response.waypoints] == ["approach", "generic-range", "exit"]
    assert response.route_geojson["properties"]["route_quality"]["waypoint_order_preserved"] is True


def test_default_terrain_orchestrator_fails_when_barrier_is_not_between_trusted_neighbors():
    historical_route = route([
        point("approach", 0.0, 0.0),
        barrier("off-direction-range", 20.0, 20.0),
        point("exit", 2.0, 0.0),
    ])
    evidence = [
        Evidence(id=reference, author="Fixture", work="Fixture", locator=reference, excerpt=reference)
        for reference in historical_route.evidence_refs
    ]
    with pytest.raises(BarrierCrossingConstraintError):
        HistoricalRouteOrchestrator(cell_size_m=25_000).present(
            HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
            historical_route,
            evidence,
        )
