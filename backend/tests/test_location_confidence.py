import pytest

from backend.app.candidate_routes import (
    ArmyProfile,
    CoordinateResolutionError,
    HistoricalRouteAdapter,
    HistoricalRoutePlanningService,
    LocationConfidence,
    MockCoordinateResolver,
    PlanningConstraints,
    ResolvedCoordinate,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint


def route():
    points = [
        HistoricalRoutePoint(
            sequence=index,
            historical_place=HistoricalPlace(id=identifier, canonical_name=name, longitude=float(index), latitude=45, source="fixture", confidence=0.9),
            event_summary="Evidence", evidence_refs=[f"e-{identifier}"], confidence=0.9,
        )
        for index, (identifier, name) in enumerate((("a", "A"), ("b", "B")), start=1)
    ]
    return HistoricalRoute(
        id="route", event_id="event", name="route", period="test", ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[(1, 45), (2, 45)]),
        evidence_refs=["e-a", "e-b"], historical_confidence=0.8,
    )


def adapt(start_confidence, end_confidence, *, constraints=None):
    adapter = HistoricalRouteAdapter(MockCoordinateResolver({
        "a": ResolvedCoordinate(GridPoint(0, 1), start_confidence, "A location note"),
        "b": ResolvedCoordinate(GridPoint(3, 1), end_confidence, "B location note"),
    }))
    return adapter.to_planning_request(
        route(), grid=SyntheticGrid.flat(4, 3), army_profile=ArmyProfile(name="demo"), constraints=constraints,
    )


def test_exact_locations_plan_without_warnings_and_confidence_is_resolved():
    request = adapt(LocationConfidence.EXACT, LocationConfidence.EXACT)
    result = HistoricalRoutePlanningService(HistoricalRouteAdapter(MockCoordinateResolver({
        "a": ResolvedCoordinate(GridPoint(0, 1), LocationConfidence.EXACT),
        "b": ResolvedCoordinate(GridPoint(3, 1), LocationConfidence.EXACT),
    }))).plan(route(), grid=SyntheticGrid.flat(4, 3), army_profile=ArmyProfile(name="demo"))
    assert request.location_warnings == []
    assert result.location_warnings == []


def test_approximate_location_plans_with_deterministic_warning():
    request = adapt(LocationConfidence.APPROXIMATE, LocationConfidence.EXACT)
    assert request.location_warnings == ["A: A location note"]


def test_disputed_requires_explicit_allowance_and_unknown_is_rejected():
    with pytest.raises(CoordinateResolutionError):
        adapt(LocationConfidence.DISPUTED, LocationConfidence.EXACT)
    allowed = adapt(
        LocationConfidence.DISPUTED,
        LocationConfidence.EXACT,
        constraints=PlanningConstraints(allow_disputed_locations=True),
    )
    assert allowed.location_warnings == ["A: A location note"]
    with pytest.raises(CoordinateResolutionError):
        adapt(LocationConfidence.UNKNOWN, LocationConfidence.EXACT)


def test_location_confidence_behavior_is_deterministic():
    first = adapt(LocationConfidence.APPROXIMATE, LocationConfidence.EXACT)
    second = adapt(LocationConfidence.APPROXIMATE, LocationConfidence.EXACT)
    assert first.location_warnings == second.location_warnings
