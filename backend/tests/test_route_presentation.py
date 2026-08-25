from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.evaluation import evaluate_route
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.presentation import HistoricalRoutePresentationService
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointGraph,
    HistoricalWaypointRole,
)


def graph():
    start = HistoricalWaypoint(id="a", canonical_name="A", order=1, role=HistoricalWaypointRole.START, evidence_refs=["e-a"])
    end = HistoricalWaypoint(id="b", canonical_name="B", order=2, role=HistoricalWaypointRole.END, evidence_refs=["e-b"])
    annotation = HistoricalAnnotation(
        id="a", title="Start event", event_type=HistoricalEventType.CAMPAIGN,
        period="218 BCE", description="Evidence-backed event", source_refs=["e-a"],
        external_references=[HistoricalExternalReference(
            id="reading", reference_type=HistoricalExternalReferenceType.ENCYCLOPEDIA,
            title="Reading", url="https://example.invalid/reading", language="en", description="Display link",
        )],
    )
    bound = HistoricalWaypointBuilder.attach_annotations([start, end], [annotation])
    from backend.app.candidate_routes.waypoint_graph import HistoricalWaypointSegment
    return HistoricalWaypointGraph(
        waypoints=bound,
        segments=[HistoricalWaypointSegment(from_waypoint=bound[0], to_waypoint=bound[1], evidence_refs=["e-a", "e-b"])],
    )


def candidate_and_evaluation():
    grid = SyntheticGrid.flat(2, 1)
    grid.set_cell(GridPoint(1, 0), elevation_m=10, terrain="hill", terrain_multiplier=3)
    route = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=grid, profile=ArmyProfile(),
    )
    return route, evaluate_route(route, ArmyProfile(name="demo"))


def test_complete_route_presentation_preserves_annotation_external_reference_and_evidence():
    route, evaluation = candidate_and_evaluation()
    presentation = HistoricalRoutePresentationService().present(graph(), route, evaluation, route_name="Demo route", period="218 BCE")
    assert presentation.route_id == route.id and presentation.route_name == "Demo route"
    assert presentation.waypoints[0].event_type is HistoricalEventType.CAMPAIGN
    assert presentation.waypoints[0].external_references[0].id == "reading"
    assert presentation.waypoints[0].evidence_refs == ["e-a"]
    assert presentation.segments[0].evidence_refs == ["e-a", "e-b"]


def test_presentation_geojson_has_route_cost_and_no_fake_waypoint_geometry():
    route, evaluation = candidate_and_evaluation()
    presentation = HistoricalRoutePresentationService().present(graph(), route, evaluation)
    features = presentation.geojson["features"]
    assert features[0]["geometry"]["type"] == "LineString"
    assert features[0]["properties"]["route_id"] == route.id
    assert features[0]["properties"]["total_cost"] == evaluation.score.total_cost
    assert all(feature["geometry"] is None for feature in features[1:])
    assert features[1]["properties"]["external_references"][0]["id"] == "reading"


def test_response_contract_and_deterministic_explanations():
    route, evaluation = candidate_and_evaluation()
    service = HistoricalRoutePresentationService()
    first = service.present(graph(), route, evaluation)
    second = service.present(graph(), route, evaluation)
    response = service.to_response(first)
    assert first.model_dump() == second.model_dump()
    assert response.route.route_id == route.id
    assert "2 supplied historical waypoint" in response.explanations.distance_reason
    assert "evidence reference" in response.explanations.historical_reason


def test_presentation_does_not_change_route_ranking():
    route, evaluation = candidate_and_evaluation()
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    HistoricalRoutePresentationService().present(graph(), route, evaluation)
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    assert before == after
