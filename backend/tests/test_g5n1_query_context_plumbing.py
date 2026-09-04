"""G5N-1: live build path passes canonical user query to event-anchor admission."""
from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry, _route_admission_query_contexts
from backend.app.models import (
    AgentState,
    AgentToolHistoryEntry,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
)
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g5l_movement_parity_episode_safety import (
    CAESAR_QUERY,
    Geography,
    POMPEY_QUERY,
    Retriever,
    ev,
)
from backend.tests.test_g5n_event_anchor_episode_gate import (
    XENOPHON_QUERY,
    _xenophon_athens_peloponnesus_event,
)


def _search_entry(query: str) -> AgentToolHistoryEntry:
    return AgentToolHistoryEntry(
        tool_name="search_historical_evidence",
        arguments={"query": query},
        result_summary="search_historical_evidence evidence_count=5 accumulated_evidence_count=5",
        success=True,
        outcome="success",
        duration_ms=1,
    )


def _live_build(state: AgentState, *, event_id: str, name: str, period: str):
    tools = AgentToolRegistry(Retriever(state.historical_evidence), Geography())
    payload, _ = tools.execute(
        "build_historical_route",
        {"event_id": event_id, "name": name, "period": period},
        state,
    )
    return payload["result"]


def test_route_admission_contexts_use_only_user_query():
    state = AgentState(session_id="g5n1", user_query=POMPEY_QUERY, requested_output="historical_route")
    state.tool_history.append(_search_entry("Xenophon march Athens Peloponnesus"))
    assert _route_admission_query_contexts(state) == (POMPEY_QUERY,)


def test_pompey_live_build_rejects_xenophon_athenae_peloponnesus_despite_misleading_subquery():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    state = AgentState(session_id="pompey", user_query=POMPEY_QUERY, requested_output="historical_route")
    state.tool_history.append(_search_entry("Xenophon march Athens Peloponnesus"))
    state.historical_evidence = list(evidence_by_id.values())
    state.historical_events = [event]
    result = _live_build(state, event_id="pompey-48", name="Pompey flight", period="48 BCE")
    ead = (state.historical_route_diagnostics or {}).get("event_anchor_diagnostics", {})
    rejected = ead.get("rejected_relations") or []
    assert any(
        item.get("reason") == "EPISODE_RELEVANCE_REJECTED"
        and item.get("earlier") == "Athenae"
        for item in rejected
    )
    route = result.get("route")
    assert route is None or not any(
        point["historical_place"]["canonical_name"] == "Athenae"
        for point in route.get("ordered_points", [])
    )


def test_xenophon_live_build_admits_athenae_peloponnesus():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    state = AgentState(session_id="xen", user_query=XENOPHON_QUERY, requested_output="historical_route")
    state.tool_history.append(_search_entry(XENOPHON_QUERY))
    state.historical_evidence = list(evidence_by_id.values())
    state.historical_events = [event]
    result = _live_build(state, event_id="xen-401", name="Ten Thousand", period="401 BCE")
    route = result.get("route")
    assert route is not None
    names = [point["historical_place"]["canonical_name"] for point in route["ordered_points"]]
    assert "Athenae" in names
    assert any("Peloponnesus" in name for name in names)


def test_omitted_query_context_preserves_conservative_admission():
    event, evidence_by_id = _xenophon_athens_peloponnesus_event()
    state = AgentState(session_id="open", requested_output="historical_route")
    state.historical_evidence = list(evidence_by_id.values())
    state.historical_events = [event]
    result = _live_build(state, event_id="open", name="Open admission", period="401 BCE")
    route = result.get("route")
    assert route is not None
    names = [point["historical_place"]["canonical_name"] for point in route["ordered_points"]]
    assert names == ["Athenae", "Peloponnesus/Peloponnesos/Peloponnese"]


def test_caesar_live_build_rejects_rhodanus_italia_event_anchor():
    movement = "Hannibal crossed from Rhodanus into Italia."
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Rhodanus", role=EventPlaceRole.ORIGIN, evidence_refs=["ev1"]),
            place=_place("Rhodanus"),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Italia", role=EventPlaceRole.DESTINATION, evidence_refs=["ev1"]),
            place=_place("Italia"),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
    ]
    event = _event("hannibal", movement, source_statements=[movement], refs=["ev1"], bindings=bindings)
    evidence = [_evidence("ev1", movement)]
    state = AgentState(session_id="caesar", user_query=CAESAR_QUERY, requested_output="historical_route")
    state.historical_evidence = evidence
    state.historical_events = [event]
    result = _live_build(state, event_id="caesar-48", name="Caesar 48", period="48 BCE")
    route = result.get("route")
    if route is not None:
        names = [point["historical_place"]["canonical_name"] for point in route["ordered_points"]]
        assert names != ["Rhodanus", "Italia"]


def test_caesar_live_build_rejects_hispania_italia_legacy():
    items = [ev("poly", "He did not think that Caesar had yet arrived in Italy from Spain.")]
    state = AgentState(session_id="caesar-legacy", user_query=CAESAR_QUERY, requested_output="historical_route")
    state.historical_evidence = items
    result = _live_build(state, event_id="caesar-48", name="Caesar 48", period="48 BCE")
    assert result.get("route") is None


def test_caesar_live_build_admits_current_episode_positive_edge():
    movement = "Caesar marched from Macedonia to Apollonia before the campaign in Epirus."
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Macedonia", role=EventPlaceRole.ORIGIN, evidence_refs=["ev1"]),
            place=_place("Macedonia"),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Apollonia", role=EventPlaceRole.DESTINATION, evidence_refs=["ev1"]),
            place=_place("Apollonia"),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
    ]
    event = _event("caesar-e1", movement, source_statements=[movement], refs=["ev1"], bindings=bindings)
    evidence = [_evidence("ev1", movement)]
    state = AgentState(session_id="caesar-pos", user_query=CAESAR_QUERY, requested_output="historical_route")
    state.historical_evidence = evidence
    state.historical_events = [event]
    result = _live_build(state, event_id="caesar-48", name="Caesar 48", period="48 BCE")
    route = result.get("route")
    assert route is not None
    names = [point["historical_place"]["canonical_name"] for point in route["ordered_points"]]
    assert names == ["Macedonia", "Apollonia"]
