"""G5K: live-vs-deterministic orchestration gates and pre-route churn control."""
from __future__ import annotations

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.loop import POST_ROUTE_SUPPRESSION_MESSAGE, ROUTE_PROSE_GROUNDING_FALLBACK
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.route_orchestration import PRE_ROUTE_SUPPRESSION_MESSAGE, route_is_ready
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import (
    AgentModelResponse,
    AgentState,
    AgentToolHistoryEntry,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)

from test_agent_loop import Geo, Retriever, agent, call, ev, route_ev, terminal
from test_g4d_route_preservation import inject_structured_route_execute, presentation_payload, structured_route


def movement_event(
    identifier: str,
    *,
    origin: str = "Athenae",
    destination: str = "Peloponnesus",
    evidence_refs: list[str] | None = None,
) -> HistoricalEvent:
    refs = evidence_refs or ["move"]
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=f"{origin} to {destination}",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        place_mentions=[
            HistoricalEventPlaceMention(raw_text=origin, role=EventPlaceRole.ORIGIN, evidence_refs=refs),
            HistoricalEventPlaceMention(raw_text=destination, role=EventPlaceRole.DESTINATION, evidence_refs=refs),
        ],
    )


def inject_route_ready_execute(registry, *, components=False, branches=False, with_presentation=False):
    original_execute = registry.execute

    def execute(name, arguments, state):
        if name == "search_historical_evidence":
            state.historical_evidence = [ev("move", "Hannibal marched from Genava to the Alpes.")]
            state.historical_events = [movement_event("m1", evidence_refs=["move"])]
            return {
                "success": True,
                "result": {"result_count": 1},
                "summary": "search_historical_evidence evidence_count=1 accumulated_evidence_count=1",
                "duration_ms": 1,
            }, "search_historical_evidence evidence_count=1 accumulated_evidence_count=1"
        if name == "build_historical_route":
            state.historical_route = structured_route(components=components, branches=branches)
            state.historical_route_diagnostics = {"route_source": "event_anchor", "reason_codes": ["PARTIAL_ROUTE"]}
            if with_presentation:
                state.historical_route_presentation = presentation_payload()
            return {"success": True, "result": {"route_points": 2}, "duration_ms": 1}, "build"
        return original_execute(name, arguments, state)

    registry.execute = execute
    return execute


def route_ready_agent(script, *, evidence=None, max_steps=8):
    geo = Geo()
    registry = AgentToolRegistry(Retriever(evidence or route_ev()), geo)
    inject_route_ready_execute(registry, components=True, branches=False, with_presentation=False)
    subject = HistoricalGisAgent(ScriptedLLMProvider(script), Retriever(evidence or route_ev()), geo, max_steps=max_steps)
    subject.tools = registry
    return subject, registry, geo


def test_route_ready_state_detects_origin_destination_pair():
    state = AgentState(session_id="g5k-ready", requested_output="historical_route", user_query="trace route")
    state.historical_evidence = [ev("move", "The army marched from Athenae into Peloponnesus.")]
    state.historical_events = [movement_event("m1")]
    assert route_is_ready(state) is True


def test_route_ready_search_suppressed_but_build_path_preserved():
    subject, _, _ = route_ready_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("search_historical_evidence", {"query": "more"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "401 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ])
    _, state = subject.respond("show Hannibal historical route", AgentState(session_id="g5k-1"))
    searches = [entry for entry in state.tool_history if entry.tool_name == "search_historical_evidence"]
    assert any(entry.outcome == "route_ready_suppressed" for entry in searches)
    assert state.historical_route is not None
    assert state.status == "completed"


def test_route_ready_duplicate_resolve_suppressed_without_tool_failure():
    geo = Geo()
    subject = agent([
        call("search_historical_evidence", {"query": "route"}),
        call("resolve_ancient_place", {"name": "Adriatic Sea"}),
        call("resolve_ancient_place", {"name": "Adriatic Sea"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "48 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ], route_ev(), max_steps=8)
    _, state = subject.respond("show Hannibal historical route", AgentState(session_id="g5k-2"))
    resolves = [entry for entry in state.tool_history if entry.tool_name == "resolve_ancient_place"]
    assert len(resolves) == 2
    assert resolves[0].outcome == "success"
    assert resolves[1].outcome in {"duplicate", "duplicate_resolve", "route_ready_suppressed"}
    assert state.status != "tool_failure"


def test_not_route_ready_search_still_allowed():
    subject = agent([
        call("search_historical_evidence", {"query": "first"}),
        call("search_historical_evidence", {"query": "second"}),
        terminal("Insufficient evidence.", ["one"], True),
    ], [ev("one", "Generic camp evidence without movement verbs.")])
    _, state = subject.respond("trace obscure route", AgentState(session_id="g5k-3"))
    searches = [entry for entry in state.tool_history if entry.tool_name == "search_historical_evidence"]
  # bootstrap + two scripted searches
    assert len(searches) >= 2
    assert all(entry.outcome == "success" for entry in searches[:2])
    assert route_is_ready(state) is False


def test_route_ready_does_not_erase_evidence_or_events():
    subject, _, _ = route_ready_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("search_historical_evidence", {"query": "more"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "401 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ])
    _, state = subject.respond("show Hannibal historical route", AgentState(session_id="g5k-4"))
    assert state.historical_evidence
    assert any(event.event_type is HistoricalEventType.MOVEMENT for event in state.historical_events)


def test_route_built_post_route_suppression_unchanged():
    subject, _, _ = route_ready_agent([
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "401 BCE"}),
        call("resolve_ancient_place", {"name": "Carthago Nova"}),
        terminal("Grounded route summary.", ["move"]),
    ])
    _, state = subject.respond("trace route", AgentState(session_id="g5k-5"))
    resolve_entry = next(entry for entry in state.tool_history if entry.tool_name == "resolve_ancient_place")
    assert resolve_entry.outcome == "route_already_built"
    assert POST_ROUTE_SUPPRESSION_MESSAGE


def test_pre_route_churn_trace_reaches_build_without_max_steps():
    subject, _, _ = route_ready_agent([
        call("resolve_ancient_place", {"name": "Athenae"}),
        call("resolve_ancient_place", {"name": "Peloponnesus"}),
        call("search_historical_evidence", {"query": "more"}),
        call("resolve_ancient_place", {"name": "Athenae"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "401 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ])
    _, state = subject.respond("show Hannibal historical route", AgentState(session_id="g5k-6"))
    assert state.status != "max_steps"
    assert state.historical_route is not None


def test_deterministic_builder_runs_when_route_ready_before_llm_finish():
    subject = agent([
        call("search_historical_evidence", {"query": "route"}),
        AgentModelResponse(content="Route summary without explicit build call."),
    ], route_ev())
    _, state = subject.respond("show a historical route", AgentState(session_id="g5k-7"))
    assert any(entry.tool_name == "build_historical_route" for entry in state.tool_history)
    assert state.historical_route is not None
    assert state.status == "completed_with_guardrail"
    assert ROUTE_PROSE_GROUNDING_FALLBACK


def test_lookup_prior_resolve_reuses_event_binding():
    from backend.app.agent.route_orchestration import lookup_prior_resolve

    place = HistoricalPlace(
        id="pleiades-1",
        canonical_name="Adriatic Sea",
        latitude=43.0,
        longitude=15.0,
        source="Pleiades",
        confidence=0.9,
        spatial_semantics=PlaceSpatialSemantics.SEA,
    )
    state = AgentState(session_id="g5k-8")
    state.historical_events = [
        HistoricalEvent(
            id="m1",
            name="crossing",
            summary="crossed sea",
            event_type=HistoricalEventType.MOVEMENT,
            place_bindings=[
                HistoricalEventPlaceBinding(
                    mention=HistoricalEventPlaceMention(raw_text="Adriatic Sea", canonical_hint="Adriatic Sea"),
                    place=place,
                    role=EventPlaceRole.RELATED_PLACE,
                    resolution_status=EventPlaceResolutionStatus.RESOLVED,
                )
            ],
        )
    ]
    payload, summary = lookup_prior_resolve({"name": "Adriatic Sea"}, state)
    assert payload is not None
    assert payload["duplicate_resolve_suppressed"] is True
    assert payload["canonical_name"] == "Adriatic Sea"


def test_pre_route_suppression_message_exposed():
    assert "build_historical_route" in PRE_ROUTE_SUPPRESSION_MESSAGE
