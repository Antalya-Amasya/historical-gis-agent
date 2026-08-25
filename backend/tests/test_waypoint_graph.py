import inspect

import pytest

from backend.app.candidate_routes import (
    ArmyProfile,
    HistoricalWaypointEvidenceError,
    MockCoordinateResolver,
    MultiSegmentHistoricalRoutePlanner,
    WaypointBuilder,
    WaypointGraphPlanningAdapter,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
import backend.app.candidate_routes.waypoint_graph as graph_module
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint


def route(*, missing_evidence_at=None, route_evidence=True):
    points = []
    for sequence, name in enumerate(("A", "B", "C"), start=1):
        points.append(HistoricalRoutePoint(
            sequence=sequence,
            historical_place=HistoricalPlace(
                id=f"place-{name.lower()}", canonical_name=name, longitude=float(sequence), latitude=45,
                source="fixture", confidence=0.9,
            ),
            event_summary="Evidence anchor",
            evidence_refs=[] if missing_evidence_at == sequence else [f"e-{name.lower()}"],
            confidence=0.9,
        ))
    return HistoricalRoute(
        id="r", event_id="e", name="route", period="test", ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[(1, 45), (2, 45), (3, 45)]),
        evidence_refs=["e-a", "e-b", "e-c"] if route_evidence else [], historical_confidence=0.8,
    )


def test_builder_turns_each_historical_point_into_one_ordered_waypoint_and_segment():
    graph = WaypointBuilder().build(route())
    assert [waypoint.id for waypoint in graph.waypoints] == ["place-a", "place-b", "place-c"]
    assert [waypoint.role.value for waypoint in graph.waypoints] == ["START", "WAYPOINT", "END"]
    assert len(graph.segments) == 2
    assert graph.segments[0].from_waypoint.id == "place-a"
    assert graph.segments[0].to_waypoint.id == "place-b"
    assert graph.segments[0].evidence_refs == ["e-a", "e-b"]


def test_builder_rejects_missing_evidence_and_never_adds_waypoints():
    with pytest.raises(HistoricalWaypointEvidenceError):
        WaypointBuilder().build(route(missing_evidence_at=2))
    with pytest.raises(HistoricalWaypointEvidenceError):
        WaypointBuilder().build(route(route_evidence=False))
    graph = WaypointBuilder().build(route())
    assert len(graph.waypoints) == len(route().ordered_points)
    assert {waypoint.canonical_name for waypoint in graph.waypoints} == {"A", "B", "C"}


def test_graph_adapter_integrates_with_multi_segment_planner_deterministically():
    graph = WaypointBuilder().build(route())
    adapter = WaypointGraphPlanningAdapter(MockCoordinateResolver({
        "place-a": GridPoint(0, 1), "place-b": GridPoint(2, 1), "place-c": GridPoint(4, 1),
    }))
    request = adapter.to_multi_segment_request(graph, grid=SyntheticGrid.flat(5, 3), army_profile=ArmyProfile(name="demo"))
    first = MultiSegmentHistoricalRoutePlanner().plan(request)
    second = MultiSegmentHistoricalRoutePlanner().plan(request)
    assert len(request.segments) == 2
    assert len(first.selected_segments) == 2
    assert first.selected_segments[0].route.evidence_refs == ["e-a", "e-b"]
    assert first.selected_segments[1].route.evidence_refs == ["e-b", "e-c"]
    assert first.model_dump() == second.model_dump()


def test_waypoint_graph_has_no_agent_rag_mcp_or_http_dependency():
    source = inspect.getsource(graph_module).lower()
    for forbidden in ("agent", "rag", "mcp", "http"):
        assert forbidden not in source
