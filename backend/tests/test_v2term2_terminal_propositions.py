"""Terminal propositions borrow only admitted route support, never mention overlap."""
from copy import deepcopy
from time import perf_counter
import socket

import pytest

from backend.app.agent.loop import BoundedAgentLoop, NO_ROUTE_TERMINAL_GUARDRAIL
from backend.app.models import AgentState
from backend.tests.test_agent_loop import agent, call, ev, terminal
from backend.tests.test_g4d_route_preservation import structured_route, route_point
from backend.tests.test_v1_1h2_no_route_terminal_arbitration import _loop, _route_state
from backend.tests.test_v2term1_terminal_route_safety import negative_state, SOURCES, geometry

LEAKS = [
    "The route cannot be reconstructed. One may still infer Darsena as an intermediate stop.",
    "The route cannot be established. One can nevertheless infer Darsena as the likely waypoint.",
    "We cannot know the road, yet Darsena was probably used.",
    "The route is unknown. Darsena was probably used.",
    "There is no reliable itinerary. The army likely passed through Darsena.",
    "路线无法确定。尽管如此，可以推测达尔塞纳是途中地点。",
    "史料无法确定路线。可以推测经过达尔塞纳。",
    "没有证据说明具体道路。达尔塞纳很可能是中途节点。",
    "路线不明。推测部队经由达尔塞纳前进。",
    "No route is established. Darsena was probably used. The army then reached Neralon.",
    "There is no evidence for the itinerary. Darsena appears to have been a waypoint.",
    "The exact road cannot be identified. The army likely passed through Darsena.",
    "Darsena was a waypoint.",
    "Aelius Cassianus embarked at Darsena.",
    "The journey continued to Neralon.",
    "Disembarkation occurred at Darsena.",
    "The most plausible historical route used Darsena.",
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External network forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def admitted_state(names=("Veloria", "Darsena", "Neralon"), partial=False):
    route = structured_route()
    route.ordered_points = [route_point(name, i + 1) for i, name in enumerate(names)]
    return _route_state(historical_route=route, historical_route_diagnostics={
        "canonical_completeness": "PARTIAL" if partial else "COMPLETE",
    })


@pytest.mark.parametrize("answer", LEAKS)
def test_no_route_cannot_borrow_place_mention(answer):
    source = ev("mention", "Darsena was a prosperous town. 达尔塞纳是一座繁荣城镇。")
    state = _route_state(historical_evidence=[source])
    before = deepcopy(state.historical_event_diagnostics)
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route(answer)
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL
    assert state.historical_route is None and state.historical_events == []
    assert state.historical_event_diagnostics == before


@pytest.mark.parametrize("answer", LEAKS)
def test_public_grounded_submission_does_not_license_unsupported_proposition(answer):
    source = ev("mention", "Aelius Cassianus remained in camp at Veloria. Darsena was a prosperous town. Neralon was nearby. 达尔塞纳是一座繁荣城镇。")
    reply, state = agent([
        call("search_historical_evidence", {"query": "Aelius Cassianus"}),
        terminal(answer, [source.id]),
    ], [source], max_grounding_corrections=0).respond(
        "Trace Aelius Cassianus from Veloria to Neralon.", AgentState(session_id="term2-public"))
    assert answer not in reply
    assert state.historical_route is None
    assert not any(e.actor.actor_text == "Nikanor Philotas" for e in state.historical_events)


@pytest.mark.parametrize("answer", [
    "The evidence supports movement through Darsena.",
    "Darsena was likely a waypoint.",
    "One may infer Darsena as an intermediate stop.",
    "The evidence supports movement from Veloria through Darsena to Neralon.",
])
def test_supported_positive_is_preserved(answer):
    state = admitted_state()
    before = deepcopy(state.historical_route.model_dump())
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply == answer
    assert state.historical_route.model_dump() == before


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("answer", [
    "The route then continued from Darsena to Neralon.",
    "The evidence supports movement from Veloria through Galveth to Darsena.",
    "Darsena was an intermediate stop.",
    "The route went from Darsena to Veloria.",
])
def test_one_supported_fragment_does_not_authorize_an_extra_edge(partial, answer):
    state = admitted_state(("Veloria", "Darsena"), partial)
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply != answer and "Neralon" not in reply and "Galveth" not in reply
    assert state.historical_route is not None


def test_supported_partial_fragment_keeps_its_wording():
    answer = "The evidence supports movement from Veloria to Darsena."
    reply, _ = _loop()._finish(answer, admitted_state(("Veloria", "Darsena"), True), perf_counter())
    assert reply == answer


@pytest.mark.parametrize("answer", [
    "The journey was abandoned.", "The force was prevented from entering the city.",
    "The force failed to reach the destination.", "The march was planned.",
])
def test_negative_requires_existing_documentary_state(answer):
    reply, _ = _loop()._finish(answer, _route_state(), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


def test_supported_prevented_fact_is_not_erased_by_positive_rewrite():
    answer = "The force was prevented from entering the city. Darsena was likely a waypoint."
    state = negative_state("PREVENTED")
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert SOURCES["PREVENTED"] in reply and "waypoint" not in reply
    assert state.historical_route is None


@pytest.mark.parametrize("answer", [
    "The historical route is not established. The GIS simulation passes through Darsena.",
    "A plausible simulated path uses Darsena.",
])
def test_simulation_only_assertions_require_existing_simulation(answer):
    reply, _ = _loop()._finish(answer, _route_state(historical_route_presentation=geometry()), perf_counter())
    assert reply == answer
    reply, _ = _loop()._finish(answer, _route_state(), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


@pytest.mark.parametrize("answer", [
    "The GIS simulation confirms Darsena as a historical waypoint.",
    "Darsena was probably used according to GIS simulation.",
    "The GIS simulation passes through Darsena. The army then reached Neralon.",
])
def test_simulation_cannot_authorize_history(answer):
    reply, _ = _loop()._finish(answer, _route_state(historical_route_presentation=geometry()), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


@pytest.mark.parametrize("text", [
    "Marcus marched from Rome to Capua.", "The senate elected a consul.",
])
def test_ordinary_source_grounded_qa_is_preserved(text):
    reply, state = agent([
        call("search_historical_evidence", {"query": "What happened?"}),
        terminal(text, ["qa"]),
    ], [ev("qa", text)], max_grounding_corrections=0).respond(
        "What happened?", AgentState(session_id="term2-qa"))
    assert reply.startswith(text) and "[Evidence: qa" in reply
    assert state.requested_output == "answer"


def test_terminal_support_does_not_reparse_upstream_scope(monkeypatch):
    import backend.app.routes.query_scope_parser as parser
    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal must not reparse endpoint or subject authority")
    monkeypatch.setattr(parser, "parse_query_scope", forbidden)
    answer = "The evidence supports movement from Veloria through Darsena to Neralon."
    reply, _ = _loop()._finish(answer, admitted_state(), perf_counter())
    assert reply == answer


@pytest.mark.parametrize("answer", [
    "Darsena was not a waypoint.",
    "One may not infer Darsena as an intermediate stop.",
])
def test_admitted_positive_state_does_not_authorize_opposite_polarity(answer):
    reply, _ = _loop()._finish(answer, admitted_state(), perf_counter())
    assert reply != answer


def test_ordinary_qa_exception_requires_prior_grounding_and_exact_source_text():
    source = ev("qa", "Marcus marched from Rome to Capua. Darsena was nearby.")
    answer = "Marcus marched from Rome to Capua."
    state = _route_state(requested_output="answer", historical_evidence=[source])
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert not reply.startswith(answer)
    state.final_grounding_status = "grounded"
    reply, _ = _loop()._finish("Marcus marched from Rome through Darsena to Capua.", state, perf_counter())
    assert "through Darsena" not in reply


def test_grounded_qa_exception_cannot_bypass_route_admission():
    answer = "Marcus marched from Rome to Capua."
    state = _route_state(historical_evidence=[ev("qa", answer)], final_grounding_status="grounded")
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


@pytest.mark.parametrize("status", ["accepted", "rejected"])
def test_specific_detail_consumes_existing_claim_status(status):
    from backend.app.models import HistoricalClaim
    answer = "Darsena was an embarkation point."
    state = admitted_state()
    state.historical_route.claims = [HistoricalClaim(
        id="detail", claim_type="WAYPOINT", text=answer,
        supporting_evidence_ids=["one"], confidence=0.9, status=status,
    )]
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert (reply == answer) is (status == "accepted")
