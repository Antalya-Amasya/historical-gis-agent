"""Direct command intent reaches ordinary orchestration, without adding authority."""
import socket

import pytest

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.loop import infer_requested_output
from backend.app.models import AgentState
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_agent_loop import Retriever, call, ev, terminal
from backend.tests.test_v2auth1_strict_endpoint_admission import CanonicalPlaces

QUERY = "Trace Aelius Cassianus from Veloria to Neralon"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External network calls forbidden")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("query", [
    QUERY,
    QUERY + ".",
    QUERY + "?",
    QUERY + "!",
    QUERY + ", then summarize the evidence.",
    QUERY + " and provide citations.",
    "Trace Aelius Cassianus's route from Veloria to Neralon.",
    "Reconstruct Aelius Cassianus from Veloria to Neralon.",
    "Reconstruct the route of Aelius Cassianus from Veloria to Neralon.",
    "Follow Aelius Cassianus from Veloria to Neralon.",
    "Show Aelius Cassianus's movement from Veloria to Neralon.",
    "How did Aelius Cassianus move from Veloria to Neralon?",
    "Trace Nikanor Philotas from Darsena to Veloria.",
    "Follow Dorieus Varro from Neralon to Darsena.",
    "Show the route from Veloria to Neralon.",
])
def test_direct_route_and_existing_controls(query):
    assert infer_requested_output(query) == "historical_route"


@pytest.mark.parametrize("query", [
    "Trace the political career of Aelius Cassianus.",
    "Trace the causes of the war.",
    "Trace references to Aelius Cassianus in Livy.",
    "Show evidence about Aelius Cassianus.",
    "Reconstruct the political events in Galveth.",
    "Trace Aelius Cassianus from Veloria.",
    "Trace Aelius Cassianus to Neralon.",
    "Trace Aelius Cassianus from Veloria to.",
    "Trace references from Livy to Polybius.",
])
def test_command_alone_or_incomplete_frame_does_not_request_route(query):
    assert infer_requested_output(query) != "historical_route"


def test_bare_generic_command_keeps_existing_non_route_behavior():
    # Bare Trace-from was not supported before this named-subject fix.
    assert infer_requested_output("Trace from Veloria to Neralon.") == "answer"


@pytest.mark.parametrize("text,expected_actor,expected_points", [
    ("Aelius Cassianus marched from Veloria to Neralon.", "Aelius Cassianus", ["Veloria", "Neralon"]),
    ("Nikanor Philotas marched from Veloria to Neralon.", "Nikanor Philotas", []),
    ("Dorieus Varro marched from Veloria to Neralon.", "Dorieus Varro", []),
    ("Aelius Cassianus marched from Veloria to Neralon or Darsena.", "Aelius Cassianus", []),
])
def test_ordinary_agent_automatically_builds_without_provider_route_call(text, expected_actor, expected_points):
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": QUERY}),
        terminal(text, ["direct"]),
    ])
    agent = HistoricalGisAgent(
        provider, Retriever([ev("direct", text)]), CanonicalPlaces(),
        max_grounding_corrections=0,
    )
    _, state = agent.respond(QUERY, AgentState(session_id="direct-route"))
    assert state.requested_output == "historical_route"
    assert sum(t.tool_name == "build_historical_route" for t in state.tool_history) == 1
    source_events, _ = EvidenceGroundedHistoricalEventExtractor().extract([ev("direct", text)])
    assert source_events and all(e.actor.actor_text == expected_actor for e in source_events)
    if expected_actor == "Aelius Cassianus":
        assert state.historical_events
        assert all(e.actor.actor_text == expected_actor for e in state.historical_events)
    else:
        assert state.historical_events == []
    points = [p.historical_place.canonical_name for p in state.historical_route.ordered_points] if state.historical_route else []
    assert points == expected_points
    if not expected_points:
        assert state.historical_route is None
