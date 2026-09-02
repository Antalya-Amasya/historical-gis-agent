"""G4G-2 barrier-failure fallback to safe route component fragments."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.app.agent.tools import AgentToolRegistry
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
from backend.app.route_orchestrator import (
    BarrierCrossingConstraintError,
    HistoricalRouteOrchestrator,
    RouteOrchestrationError,
)


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
        id="g4g2-route",
        event_id="g4g2-event",
        name="G4G2 route",
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
            "e-CarthagoNova", "e-Rhodanus",
        )
    ]


def registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        type("R", (), {"retrieve": lambda *a, **k: []})(),
        type("G", (), {"call": lambda *a, **k: {"found": True, "id": "x", "canonical_name": "Alpha", "latitude": 44.0, "longitude": 0.0, "source": "fixture", "confidence": 0.8, "uncertain": True, "coordinate_role": "exact_site"}})(),
    )


def linestrings(payload: dict) -> list[list[list[float]]]:
    return [
        feature["geometry"]["coordinates"]
        for feature in payload["geojson"]["features"]
        if feature.get("geometry", {}).get("type") == "LineString"
    ]


def test_global_barrier_fail_with_one_safe_component_produces_presentation():
    """TEST 1: global barrier fail + one safe component → presentation exists."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[
            component("global-dup", global_chain),
            component("safe", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)]),
        ],
    )
    state = AgentState(session_id="g4g2-1", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    diag = reconstruction["diagnostics"]
    assert diag["component_fallback"]["attempted"] is True
    assert diag["component_fallback"]["fragments_presented"] == 1
    assert diag["global_presentation"]["status"] == "FAILED"
    assert diag["global_presentation"]["reason"] == "BARRIER_CROSSING_CONSTRAINT"
    assert len(linestrings(reconstruction["presentation"])) == 1


def test_global_barrier_fail_duplicate_only_returns_null_without_retry():
    """TEST 2: global barrier fail + duplicate global component only → presentation null."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[component("global-dup", global_chain)],
    )
    state = AgentState(session_id="g4g2-2", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is None
    assert reconstruction["diagnostics"]["reason_code"] == "BarrierCrossingConstraintError"
    assert "component_fallback" not in reconstruction["diagnostics"]


def test_global_barrier_fail_safe_and_unsafe_components_partial_success():
    """TEST 3: global barrier fail + safe + unsafe components → partial success."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[
            component("safe", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
            component("barrier", global_chain),
        ],
    )
    state = AgentState(session_id="g4g2-3", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    diag = reconstruction["diagnostics"]
    assert diag["component_fallback"]["fragments_presented"] == 1
    assert diag["component_fallback"]["components_failed"] == 0
    assert diag["component_fallback"]["components_skipped_duplicate"] == 1
    assert diag["global_presentation"]["reason"] == "BARRIER_CROSSING_CONSTRAINT"


def test_global_success_with_components_does_not_fallback():
    """TEST 4: global success + components → no fallback, existing behavior unchanged."""
    ordered = [
        settlement("Alpha", 0.0, 44.0, sequence=1),
        settlement("Beta", 1.0, 44.5, sequence=2),
        settlement("Gamma", 2.0, 45.0, sequence=3),
    ]
    historical_route = route(
        ordered_points=ordered,
        components=[component("extra", [settlement("Charlie", 10.0, 50.0), settlement("Delta", 11.0, 50.5)])],
    )
    state = AgentState(session_id="g4g2-4", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["pipeline"] == "terrain_candidate_orchestrator"
    assert reconstruction["diagnostics"]["status"] == "COMPLETE"
    assert "component_fallback" not in reconstruction["diagnostics"]
    assert len(reconstruction["presentation"]["waypoints"]) == 3


def test_global_non_barrier_failure_not_silently_swallowed():
    """TEST 5: global non-barrier RouteOrchestrationError → no fallback."""
    ordered = [
        settlement("Alpha", 0.0, 44.0, sequence=1),
        settlement("Beta", 1.0, 44.5, sequence=2),
    ]
    historical_route = route(
        ordered_points=ordered,
        components=[component("safe", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)])],
    )
    state = AgentState(session_id="g4g2-5", requested_output="historical_route")
    state.historical_evidence = evidence()
    reg = registry()
    with patch.object(
        reg.route_orchestrator,
        "present",
        side_effect=RouteOrchestrationError("terrain reconstruction returned no candidate paths"),
    ):
        reconstruction = reg._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is None
    assert reconstruction["diagnostics"]["reason_code"] == "RouteOrchestrationError"
    assert "component_fallback" not in reconstruction["diagnostics"]


def test_all_fallback_components_barrier_fail_returns_null():
    """TEST 6: all fallback components barrier fail → presentation null."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    barrier_b = [river("Pyrenaei", 1.0, 43.0), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[component("barrier-b", barrier_b)],
    )
    state = AgentState(session_id="g4g2-6", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is None
    diag = reconstruction["diagnostics"]
    assert diag["reason_code"] == "NO_SAFE_COMPONENT_FALLBACK"
    assert diag["global_presentation"]["status"] == "FAILED"
    assert diag["component_fallback"]["components_failed"] == 1


def test_no_cross_component_geometry_in_fallback():
    """TEST 7: no cross-component geometry."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[
            component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
            component("comp-b", [settlement("Charlie", 10.0, 50.0), settlement("Delta", 11.0, 50.5)]),
        ],
    )
    state = AgentState(session_id="g4g2-7", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    lines = linestrings(reconstruction["presentation"])
    assert len(lines) == 2
    assert lines[0][-1] != lines[1][0]


def test_fragment_provenance_preserved_in_fallback():
    """TEST 8: provenance preserved per fallback fragment."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    safe = component("safe", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])
    historical_route = route(ordered_points=global_chain, components=[safe])
    state = AgentState(session_id="g4g2-8", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    fragments = reconstruction["presentation"]["fragments"]
    complete = [item for item in fragments if item["status"] == "COMPLETE"]
    assert len(complete) == 1
    assert "e-Alpha" in complete[0]["evidence_refs"]
    assert "e-Beta" in complete[0]["evidence_refs"]


def test_branch_relations_preserved_in_fallback():
    """TEST 9: branch_relations preserved."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    branches = [HistoricalRouteBranchRelation(earlier="Alpha", later="Gamma", rule="SOURCE_STRUCTURAL_ORDER", evidence_refs=["e-Alpha"])]
    historical_route = route(
        ordered_points=global_chain,
        components=[component("safe", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])],
        branches=branches,
    )
    state = AgentState(session_id="g4g2-9", requested_output="historical_route")
    state.historical_route = historical_route
    state.historical_evidence = evidence()
    registry()._reconstruct_candidate_route(historical_route, state)
    assert len(state.historical_route.branch_relations) == 1
    assert state.historical_route.branch_relations[0].earlier == "Alpha"


def test_diagnostics_include_original_global_failure():
    """TEST 10: diagnostics include original global failure."""
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[component("safe", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)])],
    )
    state = AgentState(session_id="g4g2-10", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    global_diag = reconstruction["diagnostics"]["global_presentation"]
    assert global_diag["status"] == "FAILED"
    assert global_diag["reason_code"] == "BarrierCrossingConstraintError"
    assert global_diag["reason"] == "BARRIER_CROSSING_CONSTRAINT"


def test_g4f_empty_ordered_points_behavior_unchanged():
    """TEST 11: G4F ordered_points=[] + components unchanged."""
    historical_route = route(components=[
        component("comp-a", [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)]),
        component("comp-b", [settlement("Charlie", 2.0, 45.0), settlement("Delta", 3.0, 45.5)]),
    ])
    state = AgentState(session_id="g4g2-11", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["pipeline"] == "terrain_component_fragments"
    assert reconstruction["diagnostics"]["components_presented"] == 2
    assert "global_presentation" not in reconstruction["diagnostics"]


def test_single_global_chain_regression_unchanged():
    """TEST 12: single global chain regression unchanged."""
    historical_route = route(ordered_points=[
        settlement("Alpha", 0.0, 44.0, sequence=1),
        settlement("Beta", 1.0, 44.5, sequence=2),
        settlement("Gamma", 2.0, 45.0, sequence=3),
    ])
    state = AgentState(session_id="g4g2-12", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["status"] == "COMPLETE"
    assert len(reconstruction["presentation"]["waypoints"]) == 3


def test_hannibal_shaped_fixture_safe_fragment_survives():
    """Hannibal-shaped fixture: river→mountain global fail; independent safe fragment survives."""
    # Observed Hannibal shape: Druentia→Alpes global chain fails at terminal mountain barrier.
    global_chain = [river("Druentia", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = route(
        ordered_points=global_chain,
        components=[
            component("global-dup", global_chain),
            component(
                "independent-fragment",
                [settlement("Alpha", 0.0, 44.0), settlement("Beta", 1.0, 44.5)],
            ),
        ],
    )
    state = AgentState(session_id="g4g2-hannibal-shape", requested_output="historical_route")
    state.historical_evidence = evidence()
    reconstruction = registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["component_fallback"]["fragments_presented"] == 1
    with pytest.raises(BarrierCrossingConstraintError):
        HistoricalRouteOrchestrator(cell_size_m=25_000).present(
            HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
            historical_route,
            evidence(),
        )
