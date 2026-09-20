"""R2-FINAL-C2: canonical completeness propagates to route_result_status."""

from __future__ import annotations

from backend.app.models import AgentState
from backend.app.route_result_status import RouteResultStatus, derive_route_result_status
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_g6dr_connector_relation_admission import QUERY, _pair as g6ds_pair
from backend.tests.test_g6ds_route_assembly_acceptance import POSITIVE_FIRST
from backend.tests.test_r2_final_c1_coordinate_authority import (
    _build,
    _evidence,
    _movement,
)

SOFT_QUERY = "Trace Commander Delta from Port A toward City D."


def _chain_events(segments: list[tuple[str, str]], *, actor: str = "Commander Delta"):
    events = []
    evidence_by_id = {}
    passage_parts = []
    for index, (origin, destination) in enumerate(segments):
        prefix = "" if index == 0 else "Then "
        text = f"{prefix}{actor} marched from {origin} to {destination}."
        passage_parts.append(text)
        evidence_id = f"ev{index + 1}"
        events.append(
            _movement(
                f"e{index + 1}",
                origin,
                destination,
                [evidence_id],
                actor_text=actor,
            ),
        )
    passage = " ".join(passage_parts)
    for index in range(len(segments)):
        evidence_by_id[f"ev{index + 1}"] = _evidence(f"ev{index + 1}", passage)
    return events, evidence_by_id


def _build_chain(segments: list[tuple[str, str]], *, query: str = SOFT_QUERY, event_id: str = "c2"):
    events, evidence_by_id = _chain_events(segments)
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        list(evidence_by_id.values()),
        event_id=event_id,
        name="Commander Delta",
        period="49 BCE",
        query_contexts=(query,),
    )


def _route_state(**overrides) -> AgentState:
    state = AgentState(
        session_id="c2-status",
        requested_output="historical_route",
        status="completed",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _status_from_outcome(outcome) -> RouteResultStatus | None:
    return derive_route_result_status(
        _route_state(
            historical_route=outcome.route,
            historical_route_diagnostics=outcome.diagnostics,
        ),
    )


def _status_from_diagnostics(route, diagnostics) -> RouteResultStatus | None:
    return derive_route_result_status(
        _route_state(historical_route=route, historical_route_diagnostics=diagnostics),
    )


def test_shape_only_single_component_must_not_upgrade_partial_to_full():
    route = object()
    status = _status_from_diagnostics(
        route,
        {
            "canonical_completeness": "PARTIAL",
            "reason_codes": [],
        },
    )
    assert status is RouteResultStatus.PARTIAL


def test_origin_prefix_partial_never_publishes_full_route():
    outcome = _build_chain([("Port A", "River B"), ("River B", "Island C")], event_id="c2-origin")
    assert outcome.diagnostics["canonical_completeness"] == "PARTIAL"
    assert outcome.route is not None
    assert len(outcome.route.ordered_points) >= 2
    assert _status_from_outcome(outcome) is RouteResultStatus.PARTIAL


def test_destination_suffix_partial_never_publishes_full_route():
    outcome = _build_chain([("River B", "Island C"), ("Island C", "City D")], event_id="c2-dest")
    assert outcome.diagnostics["canonical_completeness"] == "PARTIAL"
    assert outcome.route is not None
    assert _status_from_outcome(outcome) is RouteResultStatus.PARTIAL


def test_complete_control_still_publishes_full_route():
    query = ("Trace Commander Delta from Port A toward City D.",)
    text = "Commander Delta marched from Port A to City D."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City D", ["ev1"], actor_text="Commander Delta")
    outcome = _build([event], {"ev1": evidence}, query_contexts=query)
    assert outcome.diagnostics["canonical_completeness"] == "COMPLETE"
    assert outcome.route is not None
    assert _status_from_outcome(outcome) is RouteResultStatus.FULL_ROUTE


def test_absent_control_publishes_no_route():
    status = _status_from_diagnostics(
        None,
        {"canonical_completeness": "ABSENT", "reason_codes": ["CANONICAL_ROUTE_ABSENT"]},
    )
    assert status is RouteResultStatus.NO_ROUTE


def test_disconnected_partial_publishes_partial_route():
    outcome = _build_chain(
        [
            ("Port A", "River B"),
            ("River B", "Island C"),
            ("Fort X", "Gate Q"),
            ("Gate Q", "City D"),
        ],
        event_id="c2-disconnected",
    )
    assert outcome.diagnostics["canonical_completeness"] == "PARTIAL"
    assert outcome.route is not None
    assert _status_from_outcome(outcome) is RouteResultStatus.PARTIAL


def test_approximate_complete_still_publishes_full_route():
    query = ("Trace Commander Alpha from Port A to City D.",)
    text = "Commander Alpha marched from Port A to City D."
    evidence = _evidence("ev1", text)
    event = _movement(
        "e1",
        "Port A",
        "City D",
        ["ev1"],
        destination_role="representative_point",
        actor_text="Commander Alpha",
    )
    outcome = _build([event], {"ev1": evidence}, query_contexts=query)
    assert outcome.diagnostics["canonical_completeness"] == "COMPLETE"
    assert outcome.route is not None
    assert _status_from_outcome(outcome) is RouteResultStatus.FULL_ROUTE


def test_mandatory_unrelated_negative_remains_no_route():
    query = ("Trace Commander Delta from Port A toward City D.",)
    text = "Commander Delta marched from Station B to Gate C."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Station B", "Gate C", ["ev1"], actor_text="Commander Delta")
    outcome = _build([event], {"ev1": evidence}, query_contexts=query)
    assert outcome.diagnostics["canonical_completeness"] == "ABSENT"
    assert outcome.route is None
    assert _status_from_outcome(outcome) is RouteResultStatus.NO_ROUTE


def test_direct_positive_strict_query_remains_full_route():
    query = ("Trace Commander Alpha from Port A to City D.",)
    text = "Commander Alpha marched from Port A to City D."
    evidence = _evidence("ev1", text)
    event = _movement("e1", "Port A", "City D", ["ev1"], actor_text="Commander Alpha")
    outcome = _build([event], {"ev1": evidence}, query_contexts=query)
    assert outcome.diagnostics["canonical_completeness"] == "COMPLETE"
    assert _status_from_outcome(outcome) is RouteResultStatus.FULL_ROUTE


def test_g6ds_standard_control_remains_full_route():
    first = POSITIVE_FIRST
    second = "Then in 200 BCE Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = g6ds_pair(first, second)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2],
        list(evidence_by_id.values()),
        event_id="c2-g6ds",
        name="Ariston",
        period="200 BCE",
        query_contexts=QUERY,
    )
    assert outcome.route is not None
    assert outcome.diagnostics["canonical_completeness"] == "COMPLETE"
    assert _status_from_outcome(outcome) is RouteResultStatus.FULL_ROUTE


def test_legacy_fallback_without_canonical_completeness_unchanged():
    from backend.tests.test_g4d_route_preservation import structured_route

    route = structured_route()
    status = derive_route_result_status(
        _route_state(historical_route=route, historical_route_diagnostics={"reason_codes": []}),
    )
    assert status is RouteResultStatus.FULL_ROUTE
