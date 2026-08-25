import inspect

import pytest

from backend.app.candidate_routes import (
    ArmyProfile,
    CoordinateResolutionError,
    HistoricalRouteAdapter,
    HistoricalRouteEvidenceError,
    HistoricalRoutePlanningService,
    MockCoordinateResolver,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
import backend.app.candidate_routes.historical_adapter as adapter_module
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint


def point(sequence, identifier, name, evidence):
    return HistoricalRoutePoint(
        sequence=sequence,
        historical_place=HistoricalPlace(
            id=identifier, canonical_name=name, longitude=float(sequence), latitude=45.0,
            source="auditable fixture", confidence=0.9,
        ),
        event_summary="Evidence-grounded anchor", evidence_refs=evidence, confidence=0.9,
    )


def historical_route(*, route_evidence=True):
    start = point(1, "pleiades-start", "New Carthage", ["evidence-start"])
    middle = point(2, "pleiades-middle", "Rhodanus", ["evidence-middle"])
    end = point(3, "pleiades-end", "Alps", ["evidence-end"])
    return HistoricalRoute(
        id="historical-demo", event_id="event-1", name="Demo", period="218 BCE",
        ordered_points=[start, middle, end],
        geometry=GeoJsonLineString(coordinates=[(1, 45), (2, 45), (3, 45)]),
        evidence_refs=["evidence-start", "evidence-middle", "evidence-end"] if route_evidence else [],
        historical_confidence=0.8,
    )


def grid():
    result = SyntheticGrid.flat(7, 3)
    for x in (2, 3, 4):
        result.set_cell(GridPoint(x, 1), elevation_m=100, terrain="mountain", terrain_multiplier=20)
    return result


def adapter():
    return HistoricalRouteAdapter(MockCoordinateResolver({
        "pleiades-start": GridPoint(0, 1), "pleiades-end": GridPoint(6, 1),
    }))


def test_historical_route_adapts_to_planning_request_and_preserves_evidence():
    request = adapter().to_planning_request(historical_route(), grid=grid(), army_profile=ArmyProfile(name="demo"))
    assert request.start_anchor.historical_place_id == "pleiades-start"
    assert request.end_anchor.canonical_name == "Alps"
    assert request.start_anchor.evidence_refs == ["evidence-start"]
    assert request.end_anchor.evidence_refs == ["evidence-end"]
    assert request.start_grid_point == GridPoint(0, 1) and request.end_grid_point == GridPoint(6, 1)


def test_adapter_rejects_missing_route_or_anchor_evidence_and_unknown_coordinates():
    with pytest.raises(HistoricalRouteEvidenceError):
        adapter().to_planning_request(historical_route(route_evidence=False), grid=grid(), army_profile=ArmyProfile())
    broken = historical_route()
    broken.ordered_points[-1].evidence_refs = []
    with pytest.raises(HistoricalRouteEvidenceError):
        adapter().to_planning_request(broken, grid=grid(), army_profile=ArmyProfile())
    with pytest.raises(CoordinateResolutionError):
        MockCoordinateResolver({}).resolve(adapter().to_anchor(historical_route().ordered_points[0]))


def test_offline_historical_route_planning_pipeline_is_deterministic():
    service = HistoricalRoutePlanningService(adapter())
    first = service.plan(historical_route(), grid=grid(), army_profile=ArmyProfile(name="demo", mountain_tolerance=0.3))
    second = service.plan(historical_route(), grid=grid(), army_profile=ArmyProfile(name="demo", mountain_tolerance=0.3))
    assert len(first.ranked_routes.routes) == 3
    assert first.selected_route is first.ranked_routes.routes[0]
    assert first.model_dump() == second.model_dump()


def test_adapter_has_no_agent_rag_mcp_or_http_dependency():
    source = inspect.getsource(adapter_module).lower()
    for forbidden in ("agent", "rag", "mcp", "http"):
        assert forbidden not in source
