from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.evaluation import evaluate_route
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.location import (
    LocationConfidence,
    MockCoordinateResolver,
    ResolvedCoordinate,
)
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.presentation import LocationAwarePresentationService
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointGraph,
    HistoricalWaypointRole,
)


def graph():
    waypoints = [
        HistoricalWaypoint(id="exact", canonical_name="Exact", order=1, role=HistoricalWaypointRole.START, evidence_refs=["e-exact"]),
        HistoricalWaypoint(id="approx", canonical_name="Approx", order=2, role=HistoricalWaypointRole.WAYPOINT, evidence_refs=["e-approx"]),
        HistoricalWaypoint(id="disputed", canonical_name="Disputed", order=3, role=HistoricalWaypointRole.WAYPOINT, evidence_refs=["e-disputed"]),
        HistoricalWaypoint(id="unknown", canonical_name="Unknown", order=4, role=HistoricalWaypointRole.END, evidence_refs=["e-unknown"]),
    ]
    return HistoricalWaypointGraph(waypoints=waypoints)


def route_and_evaluation():
    route = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    return route, evaluate_route(route, ArmyProfile(name="demo"))


def resolver():
    return MockCoordinateResolver({
        "exact": ResolvedCoordinate(GridPoint(1, 1), LocationConfidence.EXACT, None, (4.0, 45.0)),
        "approx": ResolvedCoordinate(GridPoint(2, 1), LocationConfidence.APPROXIMATE, "Approximate area", (4.1, 45.1)),
        "disputed": ResolvedCoordinate(GridPoint(3, 1), LocationConfidence.DISPUTED, "Disputed location", (4.2, 45.2)),
        "unknown": ResolvedCoordinate(GridPoint(4, 1), LocationConfidence.UNKNOWN, "Unknown location", (4.3, 45.3)),
    })


def features(*, allow_disputed=False):
    route, evaluation = route_and_evaluation()
    presentation = LocationAwarePresentationService().present(
        graph(), route, evaluation, resolver(), allow_disputed_locations=allow_disputed,
    )
    return presentation, presentation.geojson["features"]


def test_exact_and_approximate_waypoints_emit_points_with_approximate_warning():
    presentation, items = features()
    exact, approximate = items[1], items[2]
    assert exact["geometry"] == {"type": "Point", "coordinates": [4.0, 45.0]}
    assert exact["properties"]["warning"] is None
    assert approximate["geometry"] == {"type": "Point", "coordinates": [4.1, 45.1]}
    assert approximate["properties"]["warning"] == "Approximate area"
    assert "Approximate area" in presentation.location_warnings


def test_disputed_default_withholds_geometry_but_explicit_allowance_emits_point():
    _, default_items = features()
    assert default_items[3]["geometry"] is None
    assert default_items[3]["properties"]["location_confidence"] == "DISPUTED"
    assert default_items[3]["properties"]["warning"] == "Disputed location"
    _, allowed_items = features(allow_disputed=True)
    assert allowed_items[3]["geometry"] == {"type": "Point", "coordinates": [4.2, 45.2]}
    assert allowed_items[3]["properties"]["warning"] == "Disputed location"


def test_unknown_or_missing_resolution_never_generates_geometry_and_line_remains():
    presentation, items = features()
    assert items[4]["geometry"] is None
    assert items[4]["properties"]["location_confidence"] == "UNKNOWN"
    route, evaluation = route_and_evaluation()
    missing = LocationAwarePresentationService().present(graph(), route, evaluation, MockCoordinateResolver({}))
    assert missing.geojson["features"][0]["geometry"]["type"] == "LineString"
    assert all(feature["geometry"] is None for feature in missing.geojson["features"][1:])


def test_coordinate_aware_presentation_does_not_change_ranking():
    route, evaluation = route_and_evaluation()
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    LocationAwarePresentationService().present(graph(), route, evaluation, resolver())
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    assert before == after
