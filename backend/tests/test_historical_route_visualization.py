from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.evaluation import evaluate_route
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.location import LocationConfidence, ResolvedCoordinate
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor
from backend.app.candidate_routes.presentation import LocationAwarePresentationService
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointGraph,
    HistoricalWaypointRole,
    HistoricalWaypointSegment,
)
from backend.app.rag.campaign_route_adapter import MockHistoricalCoordinateResolver


def graph(include_external_reference: bool) -> HistoricalWaypointGraph:
    first = HistoricalWaypoint(id="a", canonical_name="A", order=1, role=HistoricalWaypointRole.START, evidence_refs=["internal-a"], source_book="I", source_chapter="1")
    second = HistoricalWaypoint(id="b", canonical_name="B", order=2, role=HistoricalWaypointRole.END, evidence_refs=["internal-b"], source_book="I", source_chapter="2")
    references = [HistoricalExternalReference(id="reading", reference_type=HistoricalExternalReferenceType.WIKIPEDIA, title="Reading", url="https://example.invalid/reading")] if include_external_reference else []
    annotation = HistoricalAnnotation(id="a", title="Event A", event_type=HistoricalEventType.CAMPAIGN, period="58 BCE", description="Reviewed event.", source_refs=["internal-a"], external_references=references)
    waypoints = HistoricalWaypointBuilder.attach_annotations([first, second], [annotation])
    return HistoricalWaypointGraph(waypoints=waypoints, segments=[HistoricalWaypointSegment(from_waypoint=waypoints[0], to_waypoint=waypoints[1], evidence_refs=["internal-a", "internal-b"])])


def route_and_evaluation():
    route = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["internal-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["internal-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    ).model_copy(update={"confidence": 0.8})
    return route, evaluate_route(route, ArmyProfile())


def resolver():
    return MockHistoricalCoordinateResolver({
        "a": ResolvedCoordinate(GridPoint(0, 0), LocationConfidence.EXACT, display_coordinate=(4.0, 46.0)),
        "b": ResolvedCoordinate(GridPoint(1, 0), LocationConfidence.EXACT, display_coordinate=(5.0, 47.0)),
    })


def test_schematic_route_feature_and_waypoint_points_are_serialized_separately():
    route, evaluation = route_and_evaluation()
    presentation = LocationAwarePresentationService().present(graph(True), route, evaluation, resolver())
    assert presentation.route_geojson["geometry"]["type"] == "LineString"
    assert presentation.route_geojson["properties"]["route_type"] == "schematic_historical_route"
    assert "does not represent an exact marching path" in presentation.route_geojson["properties"]["explanation"]
    assert presentation.geojson["features"][0] == presentation.route_geojson
    assert all(feature["geometry"]["type"] == "Point" for feature in presentation.geojson["features"][1:])


def test_external_references_do_not_change_route_geometry_or_ranking_input():
    route, evaluation = route_and_evaluation()
    service = LocationAwarePresentationService()
    with_links = service.present(graph(True), route, evaluation, resolver())
    without_links = service.present(graph(False), route, evaluation, resolver())
    assert with_links.route_geojson["geometry"] == without_links.route_geojson["geometry"]
    assert with_links.waypoints[0].external_references
    assert not without_links.waypoints[0].external_references


def test_human_source_locator_hides_internal_evidence_id_from_display_references():
    route, evaluation = route_and_evaluation()
    presentation = LocationAwarePresentationService().present(graph(True), route, evaluation, resolver())
    assert presentation.waypoints[0].source_references == ["Book I", "Chapter 1"]
    assert "internal-a" not in presentation.waypoints[0].source_references
