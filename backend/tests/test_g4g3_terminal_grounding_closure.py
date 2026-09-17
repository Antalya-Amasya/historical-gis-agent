"""G4G-3: deterministic route-preserving closure when submit_grounded_answer is missing."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.loop import (
    ROUTE_PROSE_GROUNDING_FALLBACK,
    ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION,
)
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentModelResponse, AgentState

from test_agent_loop import Geo, Retriever, agent, call, ev, route_ev, terminal
from test_g4d_route_preservation import (
    GENERIC_GUARDRAIL,
    inject_structured_route_execute,
    presentation_payload,
    structured_route,
)

PROSE = "Here is the route with unsupported historical narrative."
EMPTY = AgentModelResponse(content="")


def route_script(*extra, with_presentation: bool = False, components: bool = False, branches: bool = False):
    registry = AgentToolRegistry(Retriever(route_ev()), Geo())
    execute = inject_structured_route_execute(
        registry,
        with_presentation=with_presentation,
        components=components,
        branches=branches,
    )

    def build_agent(script):
        subject = HistoricalGisAgent(ScriptedLLMProvider(script), Retriever(route_ev()), Geo())
        subject.tools = registry
        registry.execute = execute
        return subject

    return build_agent, registry


def test_route_presentation_free_form_prose_triggers_guardrail_closure():
    """TEST 1: route + presentation + free-form prose → guardrail, state preserved."""
    build_agent, _ = route_script(with_presentation=True)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        AgentModelResponse(content=PROSE),
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-1"))
    assert state.status == "completed_with_guardrail"
    assert state.final_grounding_status == "guardrail_fallback"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert PROSE not in reply
    assert state.historical_route is not None
    assert state.historical_route_presentation is not None
    assert "route_terminal_submission_missing" in state.warnings


def test_route_empty_terminal_response_uses_route_fallback():
    """TEST 2: route + empty response → deterministic fallback, not generic empty message."""
    build_agent, _ = route_script()
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-2"))
    assert state.status == "completed_with_guardrail"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION
    assert reply != "The agent completed without a final answer."
    assert state.historical_route is not None


def test_route_components_only_preserved_on_missing_submit():
    """TEST 3: route_components only → preserved."""
    build_agent, _ = route_script(components=True)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-3"))
    assert state.historical_route is not None
    assert state.historical_route.ordered_points == []
    assert len(state.historical_route.route_components) == 1


def test_route_without_presentation_does_not_claim_map_exists():
    """TEST 4: route exists + presentation null → no presentation claim."""
    build_agent, _ = route_script(with_presentation=False)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-4"))
    assert state.historical_route_presentation is None
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION
    assert "fragments" not in reply


def test_malformed_submit_path_unchanged():
    """TEST 5: malformed submit → existing G4D behavior unchanged."""
    build_agent, _ = route_script(with_presentation=True)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Unsupported prose.", []),
    ])
    subject.max_grounding_corrections = 0
    reply, state = subject.respond("show a historical route", AgentState(session_id="g4g3-5"))
    assert state.status == "completed_with_guardrail"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert "unsupported_historical_answer_discarded" in state.warnings
    assert state.historical_route is not None


def test_successful_submit_grounded_answer_unchanged():
    """TEST 6: successful submit_grounded_answer → normal grounded completion."""
    script = [
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ]
    reply, state = agent(script, route_ev()).respond("show a route", AgentState(session_id="g4g3-6"))
    assert state.status == "completed"
    assert state.final_grounding_status == "grounded"
    assert reply.startswith("Grounded route summary.")


def test_ordinary_answer_without_route_unchanged():
    """TEST 7: ordinary answer request without route → unchanged."""
    reply, state = agent(
        [call("search_historical_evidence", {"query": "Cannae"}), terminal("Bad answer.", [])],
        [ev("one", "The battle occurred.")],
        max_grounding_corrections=0,
    ).respond("What happened at Cannae?", AgentState(session_id="g4g3-7"))
    assert state.historical_route is None
    assert reply == GENERIC_GUARDRAIL


def test_provider_error_not_converted_to_route_success():
    """TEST 8: provider_error → not fake route success."""
    subject = agent([call("search_historical_evidence", {"query": "route"})], route_ev())
    with patch.object(subject.provider, "complete", side_effect=TimeoutError("timed out")):
        reply, state = subject.respond("show a route", AgentState(session_id="g4g3-8"))
    assert state.status == "provider_error"
    assert "unavailable" in reply.lower()


def test_successful_submit_after_route_build_does_not_trigger_closure():
    """TEST 9: successful submit after route build → fallback NOT triggered."""
    script = [
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ]
    reply, state = agent(script, route_ev()).respond("show a route", AgentState(session_id="g4g3-9"))
    assert state.status == "completed"
    assert state.final_grounding_status == "grounded"
    assert reply.startswith("Grounded route summary.")
    assert "route_terminal_submission_missing" not in state.warnings


def test_deterministic_route_build_then_missing_submit_triggers_fallback():
    """TEST 10: deterministic route build + missing submit → fallback."""
    build_agent, _ = route_script(with_presentation=True)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        AgentModelResponse(content="I will answer without a route."),
    ])
    subject.max_steps = 4
    reply, state = subject.respond("show a historical route", AgentState(session_id="g4g3-10"))
    assert state.status == "completed_with_guardrail"
    assert state.tool_execution_stats["completion_corrections"] == 0
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert state.historical_route is not None


def test_branch_relations_preserved_on_terminal_closure():
    """TEST 11: branch_relations preserved."""
    build_agent, _ = route_script(branches=True)
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-11"))
    assert len(state.historical_route.branch_relations) == 1


def test_route_diagnostics_preserved_on_terminal_closure():
    """TEST 12: route diagnostics preserved."""
    build_agent, _ = route_script()
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-12"))
    assert state.historical_route_diagnostics is not None
    assert state.historical_route_diagnostics.get("route_source") == "event_anchor"


def test_fallback_adds_zero_new_evidence_claims():
    """TEST 13: fallback adds zero new evidence claims."""
    build_agent, _ = route_script()
    subject = build_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        AgentModelResponse(content=PROSE),
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4g3-13"))
    assert "Polybius" not in reply
    assert "Hannibal" not in reply
    assert "Genava" not in reply
    assert state.evidence_grounded_claim_count == 0


def test_sulla_shaped_component_fragment_fixture():
    """Sulla-shaped: components + presentation, free-form terminal, no submit."""
    build_agent, _ = route_script(with_presentation=True, components=True)
    presentation = presentation_payload()
    presentation["fragments"] = [{"component_id": "main", "status": "COMPLETE"}, {"component_id": "alt", "status": "COMPLETE"}]

    registry = AgentToolRegistry(Retriever(route_ev()), Geo())
    original = inject_structured_route_execute(registry, components=True, with_presentation=True)

    def execute(name, arguments, state):
        result = original(name, arguments, state)
        if name == "build_historical_route":
            state.historical_route_presentation = presentation
        return result

    registry.execute = execute
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            AgentModelResponse(content="Sulla marched through Italy with his legions."),
        ]),
        Retriever(route_ev()),
        Geo(),
    )
    subject.tools = registry
    reply, state = subject.respond("show Sulla route", AgentState(session_id="g4g3-sulla"))
    assert state.historical_route is not None
    assert state.historical_route_presentation is not None
    assert len(state.historical_route.route_components) == 1
    assert state.status == "completed_with_guardrail"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert "Sulla" not in reply


def test_hannibal_shaped_empty_terminal_closure():
    """Hannibal-shaped: route exists, empty terminal → route-preserving fallback."""
    build_agent, _ = route_script()
    subject = build_agent([
        call("search_historical_evidence", {"query": "Hannibal Alps"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        EMPTY,
    ])
    reply, state = subject.respond("show Hannibal route through the Alps", AgentState(session_id="g4g3-hannibal"))
    assert state.historical_route is not None
    assert reply != "The agent completed without a final answer."
    assert state.status == "completed_with_guardrail"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION
