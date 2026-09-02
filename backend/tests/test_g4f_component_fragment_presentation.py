"""G4F component-fragment presentation adapter tests."""
from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.candidate_routes.component_fragment_presentation import ComponentFragmentPresentationAdapter
from backend.app.models import (
    AgentState,
    Evidence,
    GeoJsonLineString,
    HistoricalPlace,
    HistoricalRoute,
    HistoricalRouteBranchRelation,
    HistoricalRouteComponent,
    HistoricalRouteIntent,
    HistoricalRoutePoint,
    PlaceSpatialSemantics,
)
from backend.app.route_orchestrator import HistoricalRouteOrchestrator


def settlement(identifier: str, lon: float, lat: float, *, sequence: int = 1) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=sequence,
        historical_place=HistoricalPlace(
            id=identifier,
            canonical_name=identifier,
            longitude=lon,
            latitude=lat,
            source="fixture",
            confidence=0.9,
            spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
            spatial_semantics_provenance="fixture audited settlement",
        ),
        event_summary="fixture movement",
        evidence_refs=[f"e-{identifier}"],
        confidence=0.9,
    )


def river(identifier: str, lon: float, lat: float, *, sequence: int = 1) -> HistoricalRoutePoint:
    point = settlement(identifier, lon, lat, sequence=sequence)
    point.historical_place.coordinate_role = "representative_point"
    point.historical_place.spatial_semantics = PlaceSpatialSemantics.RIVER
    return point


def mountain(identifier: str, lon: float, lat: float, *, sequence: int = 1) -> HistoricalRoutePoint:
    point = settlement(identifier, lon, lat, sequence=sequence)
    point.historical_place.coordinate_role = "regional_centroid"
    point.historical_place.spatial_semantics = PlaceSpatialSemantics.MOUNTAIN_REGION
    point.historical_place.spatial_semantics_provenance = "audited mountain-region identity"
    return point


def component(component_id: str, points: list[HistoricalRoutePoint]) -> HistoricalRouteComponent:
    for index, point in enumerate(points, start=1):
        point.sequence = index
    refs = sorted({ref for point in points for ref in point.evidence_refs})
    return HistoricalRouteComponent(
        component_id=component_id,
        ordered_points=points,
        evidence_refs=refs,
    )


def route(
    *,
    ordered_points: list[HistoricalRoutePoint] | None = None,
    components: list[HistoricalRouteComponent] | None = None,
    branches: list[HistoricalRouteBranchRelation] | None = None,
) -> HistoricalRoute:
    ordered = ordered_points or []
    for index, point in enumerate(ordered, start=1):
        point.sequence = index
    all_points = ordered or [point for comp in (components or []) for point in comp.ordered_points]
    return HistoricalRoute(
        id="g4f-route",
        event_id="g4f-event",
        name="G4F route",
        period="fixture",
        ordered_points=ordered,
        geometry=GeoJsonLineString(
            coordinates=[(point.historical_place.longitude, point.historical_place.latitude) for point in all_points],
        ),
        evidence_refs=sorted({ref for point in all_points for ref in point.evidence_refs}),
        historical_confidence=0.8,
        route_components=components or [],
        branch_relations=branches or [],
    )


def evidence() -> list[Evidence]:
    return [
        Evidence(id=identifier, author="Fixture", work="Fixture", locator=identifier, excerpt=identifier)
        for identifier in (
            "e-Alpha", "e-Beta", "e-Charlie", "e-Delta", "e-Druentia", "e-Alpes",
            "e-approach", "e-exit",
        )
    ]


def adapter() -> ComponentFragmentPresentationAdapter:
    return ComponentFragmentPresentationAdapter(HistoricalRouteOrchestrator(cell_size_m=25_000))


def linestrings(payload: dict) -> list[list[list[float]]]:
    return [
        feature["geometry"]["coordinates"]
        for feature in payload["geojson"]["features"]
        if feature.get("geometry", {}).get("type") == "LineString"
    ]


def test_two_safe_components_produce_two_fragments():
    historical_route = route(components=[
        component("comp-b", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)]),
        component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
    ])
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    )
    assert result.presentation is not None
    complete = [item for item in result.fragments if item.status == "COMPLETE"]
    assert len(complete) == 2
    assert len(linestrings(result.presentation.model_dump(mode="json"))) == 2
    ends = {tuple(line[-1]) for line in linestrings(result.presentation.model_dump(mode="json"))}
    starts = {tuple(line[0]) for line in linestrings(result.presentation.model_dump(mode="json"))}
    assert ends.isdisjoint(starts)


def test_fragment_set_is_independent_of_component_input_order():
    comp_a = component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])
    comp_b = component("comp-b", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)])
    first = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        route(components=[comp_a, comp_b]),
        evidence(),
    )
    second = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        route(components=[comp_b, comp_a]),
        evidence(),
    )
    assert first.presentation is not None and second.presentation is not None
    first_ids = {item.component_id for item in first.fragments if item.status == "COMPLETE"}
    second_ids = {item.component_id for item in second.fragments if item.status == "COMPLETE"}
    assert first_ids == second_ids == {"comp-a", "comp-b"}
    first_lines = sorted(linestrings(first.presentation.model_dump(mode="json")))
    second_lines = sorted(linestrings(second.presentation.model_dump(mode="json")))
    assert first_lines == second_lines


def test_partial_success_preserves_safe_component_when_barrier_component_fails():
    historical_route = route(components=[
        component("safe", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
        component("barrier", [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]),
    ])
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    )
    assert result.presentation is not None
    assert len([item for item in result.fragments if item.status == "COMPLETE"]) == 1
    failed = [item for item in result.fragments if item.status == "FAILED"]
    assert len(failed) == 1
    assert failed[0].component_id == "barrier"
    assert failed[0].reason_code == "BarrierCrossingConstraintError"
    assert result.diagnostics["components_presented"] == 1
    assert result.diagnostics["components_failed"] == 1


def test_all_components_fail_returns_null_with_diagnostics():
    historical_route = route(components=[
        component("barrier", [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]),
    ])
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    )
    assert result.presentation is None
    assert result.diagnostics["status"] == "FAILED"
    assert result.diagnostics["components_failed"] == 1


def test_single_global_chain_regression():
    historical_route = route(ordered_points=[
        settlement("Alpha", 0.0, 44.0, sequence=1),
        settlement("Beta", 1.0, 44.5, sequence=2),
        settlement("Gamma", 2.0, 45.0, sequence=3),
    ])
    response = adapter().orchestrator.present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    )
    assert response.route_geojson["geometry"]["type"] == "LineString"
    assert len(response.waypoints) == 3


def test_empty_ordered_points_with_components_no_longer_fails_closed_immediately():
    class Geography:
        def call(self, tool, arguments):
            return {"found": True, "id": "x", "canonical_name": arguments["name"], "latitude": 44.0, "longitude": 1.0, "source": "fixture", "confidence": 0.8, "uncertain": True, "coordinate_role": "exact_site"}

    class Retriever:
        def retrieve(self, query, top_k=5, filters=None):
            return []

    registry = AgentToolRegistry(Retriever(), Geography())
    state = AgentState(session_id="g4f-empty-chain", requested_output="historical_route")
    historical_route = route(components=[
        component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
        component("comp-b", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)]),
    ])
    state.historical_route = historical_route
    reconstruction = registry._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["pipeline"] == "terrain_component_fragments"
    assert reconstruction["diagnostics"]["components_presented"] == 2


def test_no_component_with_two_points_returns_null():
    historical_route = route(
        components=[component("short", [settlement("Alpha", 0.0, 44.0)])],
    )
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    )
    assert result.presentation is None
    assert result.diagnostics["components_skipped"] == 1


def test_fragment_preserves_component_evidence_refs():
    comp = component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        route(components=[comp]),
        evidence(),
    )
    assert result.presentation is not None
    fragment = next(item for item in result.fragments if item.component_id == "comp-a")
    assert "e-Alpha" in fragment.evidence_refs
    assert "e-Beta" in fragment.evidence_refs


def test_no_cross_fragment_geometry():
    historical_route = route(components=[
        component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
        component("comp-b", [settlement("Charlie", 10.0, 50.0), settlement("Delta", 11.0, 50.5)]),
    ])
    payload = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        historical_route,
        evidence(),
    ).presentation.model_dump(mode="json")
    lines = linestrings(payload)
    assert len(lines) == 2
    assert lines[0][-1] != lines[1][0]
    assert lines[0][-1] != lines[1][-1]


def test_branch_metadata_preserved_on_route_state():
    branches = [HistoricalRouteBranchRelation(earlier="Alpha", later="Gamma", rule="SOURCE_STRUCTURAL_ORDER", evidence_refs=["e-Alpha"])]
    historical_route = route(
        components=[component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])],
        branches=branches,
    )
    state = AgentState(session_id="g4f-branch", requested_output="historical_route")
    state.historical_route = historical_route
    registry = AgentToolRegistry(type("R", (), {"retrieve": lambda *a, **k: []})(), type("G", (), {"call": lambda *a, **k: {"found": True, "id": "x", "canonical_name": "Alpha", "latitude": 44.0, "longitude": 0.0, "source": "fixture", "confidence": 0.8, "uncertain": True, "coordinate_role": "exact_site"}})())
    registry._reconstruct_candidate_route(historical_route, state)
    assert len(state.historical_route.branch_relations) == 1
    assert state.historical_route.branch_relations[0].earlier == "Alpha"


def test_single_fragment_still_exposes_legacy_fields():
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
        route(components=[component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])]),
        evidence(),
    )
    payload = result.presentation.model_dump(mode="json")
    assert payload["route"]["route_id"]
    assert payload["route_geojson"]["geometry"]["type"] == "LineString"
    assert payload["geojson"]["type"] == "FeatureCollection"
    assert payload["fragments"]
