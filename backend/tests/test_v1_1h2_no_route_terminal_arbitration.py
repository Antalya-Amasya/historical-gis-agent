"""V1.1H2: deterministic NO_ROUTE terminal answer arbitration."""
from __future__ import annotations

from time import perf_counter

from backend.app.agent.loop import NO_ROUTE_TERMINAL_GUARDRAIL, BoundedAgentLoop
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState
from backend.app.route_result_status import RouteResultStatus, derive_route_result_status
from backend.tests.test_agent_loop import Geo, Retriever
from backend.tests.test_g4d_route_preservation import structured_route


def _loop() -> BoundedAgentLoop:
    return BoundedAgentLoop(
        ScriptedLLMProvider([]),
        AgentToolRegistry(Retriever([]), Geo()),
        max_steps=1,
    )


def _route_state(**overrides) -> AgentState:
    state = AgentState(
        session_id="h2",
        requested_output="historical_route",
        status="completed",
        historical_route=None,
        historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_ORDERING"]},
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_no_route_unsafe_route_prose_is_replaced_before_serialization():
    loop = _loop()
    unsafe = (
        "Caesar's route from Brundisium → Epirus → Apollonia → Lissus → Oricum → Dyrrachium "
        "shows the campaign movement."
    )
    state = _route_state()
    reply, finished = loop._finish(unsafe, state, perf_counter())
    assert derive_route_result_status(finished) is RouteResultStatus.NO_ROUTE
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL
    assert finished.final_answer == NO_ROUTE_TERMINAL_GUARDRAIL
    assert "Brundisium" not in reply and "→" not in reply
    assert "Final answer mentioned a route without route state" in finished.warnings


def test_no_route_safe_insufficient_prose_is_not_replaced():
    loop = _loop()
    safe = "Current evidence is insufficient to reconstruct a reliable route."
    state = _route_state()
    reply, finished = loop._finish(safe, state, perf_counter())
    assert reply == safe
    assert finished.final_answer == safe
    assert "Final answer mentioned a route without route state" not in finished.warnings


def test_full_route_affirmative_prose_is_preserved():
    loop = _loop()
    route = structured_route()
    prose = "Libo's route proceeds from Oricum to Brundisium along the attested movement."
    state = _route_state(
        historical_route=route,
        historical_route_diagnostics={"reason_codes": []},
    )
    reply, finished = loop._finish(prose, state, perf_counter())
    assert derive_route_result_status(finished) is RouteResultStatus.FULL_ROUTE
    assert reply == prose
    assert finished.final_answer == prose
