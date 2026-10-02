"""NO_ROUTE terminal polarity is local; grounded negative prose keeps citations."""
from time import perf_counter

import pytest

from backend.app.agent.loop import BoundedAgentLoop, NO_ROUTE_TERMINAL_GUARDRAIL
from backend.app.route_result_status import derive_route_result_status, RouteResultStatus
from backend.app.models import AgentState
from backend.tests.test_agent_loop import agent, ev, terminal
from backend.tests.test_v1_1h2_no_route_terminal_arbitration import _loop, _route_state


@pytest.mark.parametrize("text,reject", [
    ("The route went from A to B.", True),
    ("The evidence indicates the attempted movement was prevented, so no completed route should be reconstructed.", False),
    ("The evidence does not establish a completed movement, so a route cannot be reconstructed.", False),
    ("The actor likely followed a route from A to B.", True),
    ("The source does not establish movement from Setovia to Rome and no route between those places should be drawn.", False),
    ("It is not impossible that the route went from A to B.", True),
    ("No route was established. The actor likely followed a route from A to B.", True),
    ("Evidence is insufficient, but the route went from A to B.", True),
    ("No route was established; the actor sailed from A to B.", True),
    ("The route includes an invented waypoint at C.", True),
    ("The geometry follows A → B.", True),
    ("The actor travelled by sea from A to B.", True),
    ("The movement was aborted, so the route should not be reconstructed.", False),
    ("The movement was abandoned.", False),
    ("The movement was proposed but not executed, so no route should be reconstructed.", False),
    ("当前证据不足以生成可靠路线。", False),
    ("The current retrieved historical evidence is insufficient. The route went from A to B.", True),
    ("The route was not executed.", False),
    ("This is not supported by evidence and is not a route node.", False),
    ("We cannot rule out a route from A to B.", True),
    ("We cannot deny that the route went from A to B.", True),
    ("The route from A to B is not exact.", True),
    ("The route went from A to B although entry was prevented.", True),
])
def test_terminal_polarity_controls(text, reject):
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route(text) is reject
    state = _route_state(historical_evidence=[ev("negative-control", text)] if not reject else [])
    reply, finished = _loop()._finish(text, state, perf_counter())
    assert reply == (NO_ROUTE_TERMINAL_GUARDRAIL if reject else text)
    assert derive_route_result_status(finished) is RouteResultStatus.NO_ROUTE
    assert finished.historical_route is None
    assert finished.historical_route_presentation is None


def test_prevented_entry_survives_grounded_submission_with_valid_citation():
    text = "The attempted entry into Setovia was prevented."
    source = ev("prevented-entry", text)
    answer = text + " A completed route cannot be reconstructed."
    reply, state = agent([terminal(answer, [source.id])], [source]).respond(
        "Did evidence support entry into Setovia? Reconstruct a route only if completed.", AgentState(session_id="prevented"))
    assert text in reply
    assert source.id in reply
    assert "Final answer mentioned a route without route state" not in state.warnings
    assert state.historical_route is None
    assert state.historical_route_presentation is None
    assert derive_route_result_status(state) is RouteResultStatus.NO_ROUTE


def test_negative_prose_does_not_bypass_citation_id_validation():
    source = ev("valid", "The attempted entry into Setovia was prevented.")
    subject = agent([terminal("A completed route cannot be reconstructed.", ["fabricated-id"])], [source])
    subject.max_grounding_corrections = 0
    reply, state = subject.respond("Reconstruct the route into Setovia.", AgentState(session_id="bad-citation"))
    assert "fabricated-id" not in reply
    assert state.historical_route is None
    assert state.historical_route_presentation is None
