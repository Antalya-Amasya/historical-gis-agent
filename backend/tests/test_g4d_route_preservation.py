"""G4D: preserve completed historical routes through prose grounding failure."""
from __future__ import annotations

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.loop import ROUTE_PROSE_GROUNDING_FALLBACK, ROUTE_PROSE_GROUNDING_FALLBACK_NO_PRESENTATION
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import (
    AgentModelResponse,
    AgentState,
    GeoJsonLineString,
    HistoricalPlace,
    HistoricalRoute,
    HistoricalRouteBranchRelation,
    HistoricalRouteComponent,
    HistoricalRoutePoint,
)

from test_agent_loop import Geo, Retriever, agent, call, ev, route_ev, terminal

GENERIC_GUARDRAIL = "The current retrieved historical evidence is insufficient to support a reliable answer."


def place(name: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=name.lower(), canonical_name=name, latitude=46.0, longitude=6.0,
        source="test", confidence=0.8, coordinate_role="exact_site",
    )


def route_point(name: str, sequence: int = 1) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=sequence, historical_place=place(name),
        event_summary=f"Movement at {name}", confidence=0.8,
    )


def structured_route(*, components: bool = False, branches: bool = False) -> HistoricalRoute:
    points = [route_point("Genava"), route_point("Alpes", 2)]
    route = HistoricalRoute(
        id="route-test", event_id="test", name="Test route", period="218 BCE",
        ordered_points=points if not components else [],
        geometry=GeoJsonLineString(coordinates=[(6.0, 46.0), (7.0, 47.0)]),
        evidence_refs=["one"], historical_confidence=0.7,
    )
    if components:
        route.route_components = [
            HistoricalRouteComponent(
                component_id="main", ordered_points=points, evidence_refs=["one"],
            ),
        ]
    if branches:
        route.branch_relations = [
            HistoricalRouteBranchRelation(
                earlier="Genava", later="Alpes", rule="SAME_MOVEMENT_EVENT",
                evidence_refs=["one"],
            ),
        ]
    return route


def presentation_payload() -> dict:
    return {
        "route": {"route_id": "route-test", "confidence": 0.7},
        "geojson": {"type": "FeatureCollection", "features": []},
    }


def inject_structured_route_execute(
    registry: AgentToolRegistry,
    *,
    components: bool = False,
    branches: bool = False,
    with_presentation: bool = False,
):
    original_execute = registry.execute

    def execute(name, arguments, state):
        if name == "search_historical_evidence":
            state.historical_evidence = [ev("one", "Hannibal marched from Genava to the Alpes.")]
            return {"success": True, "result": {"result_count": 1}, "duration_ms": 1}, "search"
        if name == "build_historical_route":
            state.historical_route = structured_route(components=components, branches=branches)
            state.historical_route_diagnostics = {"route_source": "event_anchor", "reason_codes": ["PARTIAL_ROUTE"]}
            if with_presentation:
                state.historical_route_presentation = presentation_payload()
            return {"success": True, "result": {"route_points": 2}, "duration_ms": 1}, "build"
        return original_execute(name, arguments, state)

    return execute


def test_route_built_malformed_submit_preserves_structured_route():
    script = [
        call("search_historical_evidence", {"query": "Hannibal route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Unsupported prose about Hannibal.", []),
    ]
    subject = agent(script, route_ev(), max_grounding_corrections=0)
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4d-route-submit"))
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2
    assert state.status == "completed_with_guardrail"
    assert state.final_grounding_status == "guardrail_fallback"
    assert reply == ROUTE_PROSE_GROUNDING_FALLBACK
    assert reply != GENERIC_GUARDRAIL
    assert "unsupported_historical_answer_discarded" in state.warnings


def test_ordinary_answer_malformed_submit_keeps_generic_guardrail():
    script = [
        call("search_historical_evidence", {"query": "Cannae"}),
        terminal("Unsupported prose.", []),
    ]
    subject = agent(script, [ev("one", "The battle occurred.")], max_grounding_corrections=0)
    reply, state = subject.respond("What happened at Cannae?", AgentState(session_id="g4d-answer-guardrail"))
    assert state.requested_output == "answer"
    assert state.historical_route is None
    assert reply == GENERIC_GUARDRAIL


def test_route_not_built_grounding_failure_still_guardrails():
    subject = agent(
        [call("search_historical_evidence", {"query": "Hannibal"}), terminal("Route prose.", [])],
        [ev("one", "Hannibal crossed the Rhone.")],
        max_grounding_corrections=0,
    )
    reply, state = subject.respond("show Hannibal route", AgentState(session_id="g4d-no-route"))
    assert state.historical_route is None
    assert reply == GENERIC_GUARDRAIL


def test_valid_grounded_route_answer_unchanged():
    script = [
        call("search_historical_evidence", {"query": "route"}),
        call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
        terminal("Grounded route summary.", ["move"]),
    ]
    subject = agent(script, route_ev())
    reply, state = subject.respond("show a route", AgentState(session_id="g4d-valid"))
    assert state.historical_route is not None
    assert state.status == "completed"
    assert reply.startswith("Grounded route summary.")
    assert state.final_grounding_status == "grounded"


def test_partial_route_components_preserved_through_grounding_failure(monkeypatch):
    registry = AgentToolRegistry(Retriever(route_ev()), Geo())
    monkeypatch.setattr(registry, "execute", inject_structured_route_execute(registry, components=True))
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            terminal("Unsupported prose.", []),
        ]),
        Retriever(route_ev()), Geo(), max_grounding_corrections=0,
    )
    subject.tools = registry
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4d-components"))
    assert state.historical_route is not None
    assert len(state.historical_route.route_components) == 1
    assert state.historical_route.ordered_points == []
    assert len(state.historical_route.route_components[0].ordered_points) == 2


def test_branch_relations_preserved_through_grounding_failure(monkeypatch):
    registry = AgentToolRegistry(Retriever(route_ev()), Geo())
    monkeypatch.setattr(registry, "execute", inject_structured_route_execute(registry, branches=True))
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            terminal("Unsupported prose.", []),
        ]),
        Retriever(route_ev()), Geo(), max_grounding_corrections=0,
    )
    subject.tools = registry
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4d-branches"))
    assert state.historical_route is not None
    assert len(state.historical_route.branch_relations) == 1
    assert state.historical_route.branch_relations[0].rule == "SAME_MOVEMENT_EVENT"


def test_presentation_preserved_through_grounding_failure(monkeypatch):
    registry = AgentToolRegistry(Retriever(route_ev()), Geo())
    monkeypatch.setattr(registry, "execute", inject_structured_route_execute(registry, with_presentation=True))
    subject = HistoricalGisAgent(
        ScriptedLLMProvider([
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            terminal("Unsupported prose.", []),
        ]),
        Retriever(route_ev()), Geo(), max_grounding_corrections=0,
    )
    subject.tools = registry
    _, state = subject.respond("show Hannibal route", AgentState(session_id="g4d-presentation"))
    assert state.historical_route_presentation is not None
    assert state.historical_route_presentation["route"]["route_id"] == "route-test"


def test_no_stale_route_leakage_on_new_request():
    route_subject = agent(
        [
            call("search_historical_evidence", {"query": "route"}),
            call("build_historical_route", {"event_id": "test", "name": "Test route", "period": "218 BCE"}),
            AgentModelResponse(content="Route complete."),
        ],
        route_ev(),
    )
    state = AgentState(session_id="g4d-stale")
    _, state = route_subject.respond("show a route", state)
    assert state.historical_route is not None

    answer_subject = agent(
        [call("search_historical_evidence", {"query": "Cannae"}), terminal("Bad answer.", [])],
        [ev("battle", "The battle occurred.")],
        max_grounding_corrections=0,
    )
    reply, state = answer_subject.respond("What happened at Cannae?", state)
    assert state.historical_route is None
    assert reply == GENERIC_GUARDRAIL
