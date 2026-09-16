"""V1H: complete admitted routes must not be downgraded by broad diagnostics."""
from __future__ import annotations

from backend.app.models import AgentState, HistoricalRouteComponent
from backend.app.route_result_status import RouteResultStatus, derive_route_result_status
from backend.tests.test_g4d_route_preservation import route_point, structured_route


def _route_state(**overrides) -> AgentState:
    state = AgentState(
        session_id="v1h-status",
        requested_output="historical_route",
        status="completed",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_complete_route_with_partial_route_diagnostic_is_full_route():
    route = structured_route()
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={
                "reason_codes": ["PARTIAL_ROUTE"],
                "component_count": 1,
                "branch_relation_count": 0,
                "contradictory_relation_count": 0,
                "ordered_place_count": 2,
            },
        ),
    )
    assert status is RouteResultStatus.FULL_ROUTE


def test_disconnected_components_remain_partial():
    route = structured_route()
    route.route_components = [
        HistoricalRouteComponent(
            component_id="ab",
            ordered_points=[route_point("Alpha"), route_point("Beta", 2)],
            evidence_refs=["one"],
        ),
        HistoricalRouteComponent(
            component_id="cd",
            ordered_points=[route_point("Charlie"), route_point("Delta", 2)],
            evidence_refs=["two"],
        ),
    ]
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={
                "reason_codes": ["PARTIAL_ROUTE"],
                "component_count": 2,
                "contradictory_relation_count": 0,
            },
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_branched_route_remains_partial():
    route = structured_route(branches=True)
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={
                "reason_codes": ["PARTIAL_ROUTE"],
                "branch_relation_count": 1,
                "contradictory_relation_count": 0,
            },
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_contradictory_relation_remains_partial():
    route = structured_route()
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={
                "reason_codes": ["PARTIAL_ROUTE"],
                "contradictory_relation_count": 1,
            },
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_no_route_without_route_or_presentation():
    status = derive_route_result_status(
        _route_state(
            historical_route=None,
            historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_ORDERING"]},
        ),
    )
    assert status is RouteResultStatus.NO_ROUTE


def test_error_from_provider_failure():
    status = derive_route_result_status(
        _route_state(status="provider_error", historical_route=None),
    )
    assert status is RouteResultStatus.ERROR


def test_non_route_request_returns_none():
    status = derive_route_result_status(
        AgentState(session_id="answer", requested_output="answer", status="completed"),
    )
    assert status is None


def test_presentation_only_partial_artifact_without_route():
    status = derive_route_result_status(
        _route_state(
            historical_route=None,
            historical_route_presentation={"route": {"route_id": "fragment"}},
            historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_ORDERING"]},
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_component_only_route_without_global_ordered_points_remains_partial():
    route = structured_route(components=True)
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={"reason_codes": ["PARTIAL_ROUTE"]},
        ),
    )
    assert status is RouteResultStatus.PARTIAL
