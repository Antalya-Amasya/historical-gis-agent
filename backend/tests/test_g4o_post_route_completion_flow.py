"""G4O: suppress upstream tool churn after a historical route is built."""
from __future__ import annotations

import json

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.loop import (
    POST_ROUTE_SUPPRESSION_MESSAGE,
    ROUTE_PROSE_GROUNDING_FALLBACK,
    ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION,
)
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentModelResponse, AgentState

from test_agent_loop import Geo, Retriever, agent, call, ev, route_ev, terminal
from test_g4d_route_preservation import inject_structured_route_execute, structured_route


def route_agent(script, *, max_steps=8, components=False, branches=False, with_presentation=False):
    geo = Geo()
    registry = AgentToolRegistry(Retriever(route_ev()), geo)
    execute = inject_structured_route_execute(
        registry,
        components=components,
        branches=branches,
        with_presentation=with_presentation,
    )
    subject = HistoricalGisAgent(ScriptedLLMProvider(script), Retriever(route_ev()), geo, max_steps=max_steps)
    subject.tools = registry
    registry.execute = execute
    return subject, registry, geo


def test_route_built_resolve_suppressed():
    """TEST 1: route built → resolve_ancient_place suppressed, route preserved."""
    subject, registry, geo = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}),
        terminal("Grounded route summary.", ["one"]),
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-1"))
    assert state.historical_route is not None
    resolve_entry = next(entry for entry in state.tool_history if entry.tool_name == "resolve_ancient_place")
    assert resolve_entry.outcome == "route_already_built"
    assert geo.calls == []
    assert state.status == "completed"
    assert reply.startswith("Grounded route summary.")


def test_route_built_search_suppressed():
    """TEST 2: route built → search_historical_evidence suppressed."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("search_historical_evidence", {"query": "more"}),
        terminal("Grounded route summary.", ["one"]),
    ])
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-2"))
    searches = [entry for entry in state.tool_history if entry.tool_name == "search_historical_evidence"]
    assert len(searches) == 2
    assert searches[0].outcome == "success"
    assert searches[1].outcome == "route_already_built"


def test_route_built_duplicate_build_suppressed():
    """TEST 3: route built → duplicate build_historical_route suppressed."""
    subject, registry, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}, "dup"),
        terminal("Grounded route summary.", ["one"]),
    ])
    builds = []
    original_execute = registry.execute

    def counting_execute(name, arguments, state):
        if name == "build_historical_route":
            builds.append(arguments)
        return original_execute(name, arguments, state)

    registry.execute = counting_execute
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-3"))
    assert len(builds) == 1
    assert [entry.outcome for entry in state.tool_history if entry.tool_name == "build_historical_route"] == [
        "success", "route_already_built",
    ]


def test_resolve_allowed_before_route_built():
    """TEST 4: route not built → resolve_ancient_place still allowed."""
    geo = Geo()
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "Hannibal"}),
            call("resolve_ancient_place", {"name": "Carthago Nova"}),
            AgentModelResponse(content="Evidence is insufficient to generate a route."),
        ]),
        Retriever([ev("n", "New Carthage")]),
        geo,
        max_steps=4,
    )
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-4"))
    assert geo.calls == [("resolve_ancient_place", {"name": "Carthago Nova"})]
    assert state.historical_route is None
    assert all(entry.outcome != "route_already_built" for entry in state.tool_history)


def test_route_components_only_counts_as_built():
    """TEST 5: ordered_points=[] + route_components → POST_ROUTE_PHASE."""
    subject, _, geo = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Rhodanus"}),
        terminal("Component route summary.", ["one"]),
    ], components=True)
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-5"))
    assert state.historical_route.ordered_points == []
    assert len(state.historical_route.route_components) == 1
    resolve_entry = next(entry for entry in state.tool_history if entry.tool_name == "resolve_ancient_place")
    assert resolve_entry.outcome == "route_already_built"
    assert geo.calls == []


def test_empty_route_object_not_treated_as_complete():
    """TEST 6: empty route object → upstream tools still allowed."""
    geo = Geo()
    registry = AgentToolRegistry(Retriever(route_ev()), geo)
    original_execute = registry.execute

    def empty_route_execute(name, arguments, state):
        if name == "build_historical_route":
            state.historical_route = structured_route()
            state.historical_route.ordered_points = []
            state.historical_route.route_components = []
            state.historical_route.branch_relations = []
            return {"success": True, "result": {"route_points": 0}, "duration_ms": 1}, "build empty"
        return original_execute(name, arguments, state)

    registry.execute = empty_route_execute
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            call("resolve_ancient_place", {"name": "Carthago Nova"}),
            AgentModelResponse(content="Evidence is insufficient to generate a route."),
        ]),
        Retriever(route_ev()),
        geo,
        max_steps=6,
    )
    subject.tools = registry
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-6"))
    assert geo.calls == [("resolve_ancient_place", {"name": "Carthago Nova"})]
    resolve_entry = next(entry for entry in state.tool_history if entry.tool_name == "resolve_ancient_place")
    assert resolve_entry.outcome == "success"


def test_presentation_path_preserved_inside_route_build():
    """TEST 7: presentation orchestration inside build_historical_route remains unchanged."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Grounded route summary.", ["one"]),
    ], with_presentation=True)
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-7"))
    assert state.historical_route_presentation is not None
    assert state.historical_route is not None


def test_submit_grounded_answer_allowed_after_route_built():
    """TEST 8: submit_grounded_answer allowed after route built."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Grounded route summary.", ["one"]),
    ])
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-8"))
    assert state.status == "completed"
    assert state.final_grounding_status == "grounded"
    assert reply.startswith("Grounded route summary.")


def test_g4g3_fallback_preserved_after_suppression():
    """TEST 9: route built + suppressed upstream + no submit → G4G-3 closure."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}),
        AgentModelResponse(content=""),
    ], with_presentation=True)
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-9"))
    assert state.status == "completed_with_guardrail"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert state.historical_route is not None
    assert "route_terminal_submission_missing" in state.warnings


def test_suppressed_tool_does_not_erase_route():
    """TEST 10: suppressed upstream call preserves route state."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Druentia"}),
        call("resolve_ancient_place", {"name": "Alpes"}, "two"),
        terminal("Grounded route summary.", ["one"]),
    ])
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-10"))
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2
    assert state.tool_execution_stats["suppressed_tool_calls"] == 2


def test_suppressed_tool_does_not_consume_execution_budget():
    """TEST 11: suppressed calls do not consume real tool execution budget."""
    subject, _, _ = route_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}),
        call("resolve_ancient_place", {"name": "Druentia"}),
        call("resolve_ancient_place", {"name": "Alpes"}),
        terminal("Grounded route summary.", ["one"]),
    ], max_steps=8)
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4o-11"))
    assert state.tool_execution_stats["actual_tool_executions"] == 2
    assert state.tool_execution_stats["suppressed_tool_calls"] == 3
    suppressed = [entry for entry in state.tool_history if entry.outcome == "route_already_built"]
    assert len(suppressed) == 3


def test_g4l_shaped_sequence_no_longer_hits_max_steps_from_redundant_resolves():
    """TEST 12: G4L-shaped churn no longer ends at max_steps."""
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal Iberia"}, "s1"),
        call("search_historical_evidence", {"query": "Rhodanus crossing"}, "s2"),
        call("search_historical_evidence", {"query": "Alpine crossing"}, "s3"),
        call("search_historical_evidence", {"query": "Italian campaign"}, "s4"),
        call("build_historical_route", {"event_id": "hannibal", "name": "Hannibal route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}, "r1"),
        call("resolve_ancient_place", {"name": "Druentia"}, "r2"),
        call("resolve_ancient_place", {"name": "Alpes"}, "r3"),
        terminal("Hannibal marched from Iberia through the Alps.", ["one"]),
    ])
    subject, _, _ = route_agent([], max_steps=8)
    subject.provider = provider
    reply, state = subject.respond("show Hannibal historical route", AgentState(session_id="g4o-12"))
    assert state.status != "max_steps"
    assert state.status in {"completed", "completed_with_guardrail"}
    assert state.tool_execution_stats["suppressed_tool_calls"] == 3
    assert state.historical_route is not None
    if state.status == "completed":
        assert reply.startswith("Hannibal marched")
    else:
        assert reply in {ROUTE_PROSE_GROUNDING_FALLBACK, ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION}


def test_suppression_tool_message_contains_guidance():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}),
        terminal("Done.", ["one"]),
    ])
    subject, _, _ = route_agent([])
    subject.provider = provider
    subject.respond("show route", AgentState(session_id="g4o-msg"))
    tool_messages = [
        json.loads(message["content"])
        for request in provider.requests
        for message in request["messages"]
        if message.get("role") == "tool"
    ]
    suppressed = next(item for item in tool_messages if item.get("outcome") == "route_already_built")
    assert suppressed["result"]["reason"] == "ROUTE_ALREADY_BUILT"
    assert POST_ROUTE_SUPPRESSION_MESSAGE in suppressed["result"]["message"]
