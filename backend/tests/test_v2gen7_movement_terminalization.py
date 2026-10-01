"""Explicit movement-display intent survives auxiliary analytical clauses."""
import pytest

from backend.app.agent.loop import infer_requested_output
from backend.app.models import AgentState, HistoricalTravelMode, HistoricalClaim
from backend.app.route_result_status import derive_route_result_status, RouteResultStatus
from backend.tests.test_agent_loop import agent, call, ev, terminal
from backend.tests.test_v2gen4_answer_support import _loop, _route_state, structured_route
from time import perf_counter

LIBO_QUERY = "Trace Libo's movement from Oricum to Brundisium and explain separately the historically supported movement, historical travel mode, representative endpoint coordinates, simulated coastal access, and reconstructed water geometry. Do not treat the simulated coastal-access points as historical embarkation or landing sites, and do not claim that the displayed maritime line is the exact historical sailing track."


@pytest.mark.parametrize("query", [LIBO_QUERY,
    "Trace Ariston's movement from New Carthage to the Rhone and explain the coordinates.",
    "Show Bion's journey and describe the evidence limitations."])
def test_explicit_display_intent_is_not_vetoed_by_explanation(query):
    assert infer_requested_output(query) == "historical_route"


@pytest.mark.parametrize("query,expected", [
    ("Explain the political consequences of Ariston's movement.", "answer"),
    ("How important were Bion's movements during the war?", "answer"),
    ("What are the coordinates of New Carthage?", "geography_fact")])
def test_analytical_only_and_geography_requests_keep_their_contract(query, expected):
    assert infer_requested_output(query) == expected


def test_repeated_nonprogress_tools_end_in_route_closure_without_budget_increase():
    query = "Trace Ariston's movement from New Carthage to the Rhone and explain the coordinates."
    evidence = ev("movement", "Ariston sailed from New Carthage to the Rhone.")
    distance = {"point_a": {"latitude": 40, "longitude": 19}, "point_b": {"latitude": 40, "longitude": 18}}
    script = [call("search_historical_evidence", {"query": query, "top_k": 8}, "search")]
    script += [call("calculate_distance", distance, f"distance-{i}") for i in range(7)]
    reply, state = agent(script, [evidence], max_steps=8).respond(query, AgentState(session_id="nonprogress"))
    assert state.requested_output == "historical_route"
    assert state.status == "completed"
    assert "safe step limit" not in reply
    assert state.tool_execution_stats["route_builder_attempted"] == 1
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2
    assert state.tool_execution_stats["suppressed_tool_calls"] >= 1
    assert [e.id for e in state.supporting_evidence] == [evidence.id]


@pytest.mark.parametrize("completeness,expected", [("PARTIAL", RouteResultStatus.PARTIAL), ("COMPLETE", RouteResultStatus.FULL_ROUTE)])
def test_partial_and_full_routes_finish_with_cited_support(completeness, expected):
    source = ev("movement", "Ariston sailed from New Carthage to the Rhone.")
    route = structured_route()
    route.evidence_refs = [source.id]
    state = _route_state(historical_evidence=[source], historical_route=route,
        historical_route_diagnostics={"canonical_completeness": completeness})
    from backend.app.agent.evidence_support import render_evidence_citations
    answer = "Evidence inspected. " + render_evidence_citations((source.id,), [source])
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply == answer
    assert derive_route_result_status(state) is expected
    assert state.supporting_evidence == [source]


def test_no_route_stays_safe_without_simulation_creating_history():
    state = _route_state()
    reply, state = _loop()._finish("Evidence is insufficient.", state, perf_counter())
    assert derive_route_result_status(state) is RouteResultStatus.NO_ROUTE
    assert state.historical_route is None and state.supporting_evidence == []
    assert reply == "Evidence is insufficient."


def test_partial_sea_summary_keeps_simulated_access_out_of_historical_waypoints():
    from backend.app.agent.route_summary import admitted_route_summary
    route = structured_route()
    names = [point.historical_place.canonical_name for point in route.ordered_points]
    source = ev("movement", f"Ariston sailed from {names[0]} to {names[-1]}.")
    route.evidence_refs = [source.id]
    route.claims = [HistoricalClaim(id="sea", claim_type="ORDERING", text="sailed", confidence=.8, travel_mode=HistoricalTravelMode.SEA)]
    state = _route_state(historical_evidence=[source], historical_route=route,
        historical_route_diagnostics={"canonical_completeness": "PARTIAL"},
        historical_route_presentation={"geojson": {"features": [
            {"properties": {"segment_role": "simulated_coastal_access"}},
            {"properties": {"segment_role": "direct_water_edge"}}
        ]}})
    reply, state = _loop()._finish(admitted_route_summary(state), state, perf_counter())
    assert state.status == "completed"
    assert derive_route_result_status(state) is RouteResultStatus.PARTIAL
    assert [point.historical_place.canonical_name for point in state.historical_route.ordered_points] == names
    assert "not a historical embarkation point" in reply
    assert "not a documented exact sailing track" in reply
    assert state.supporting_evidence == [source]
