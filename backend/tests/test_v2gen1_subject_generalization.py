"""Registry misses permit investigation without supplying actor authority."""
from pathlib import Path

import pytest

from backend.app.agent.evidence_support import (
    _subject_alias_registry, assess_evidence_support, extract_subject_terms,
)
from backend.app.agent.loop import BoundedAgentLoop
from backend.app.models import AgentState, EventActorStatus
from backend.app.rag.query_bridge import V1_ENTRIES
from backend.tests.test_agent_loop import agent, ev, terminal


@pytest.mark.parametrize("name", ["Neralis Vexon", "Terenos Qavik"])
def test_unseen_subject_support_is_unassessed_not_insufficient(name):
    query = f"Trace {name}'s route from New Carthage to the Rhone."
    assert extract_subject_terms(query) == ()
    assert all(name.lower() not in str(forms).lower() for forms in _subject_alias_registry().values())
    assert name.lower() not in str(V1_ENTRIES).lower()
    fixtures = Path(__file__).parent / "fixtures"
    for filename in ("g7_broad_full_chain_evaluation.json", "g5r_trusted_movement_benchmark.json"):
        assert name.lower() not in (fixtures / filename).read_text(encoding="utf-8").lower()
    result = assess_evidence_support(query, "historical_route", [ev("source", "A source awaits investigation.")])
    assert result.status == "unassessed"
    assert result.relevant_evidence_ids == ()
    assert result.matched_subject_terms == ()
    assert "UNKNOWN_SUBJECT" in result.reason


def test_loop_registry_miss_does_not_force_insufficiency_after_bounded_builder_attempt():
    subject = agent([])
    loop = BoundedAgentLoop(subject.provider, subject.tools)
    state = AgentState(session_id="unknown", user_query="Trace Neralis Vexon's route.", requested_output="historical_route",
                       historical_evidence=[ev("source", "A source awaits investigation.")],
                       tool_execution_stats={"route_builder_attempted": 1})
    loop._refresh_evidence_support(state)
    assert loop._route_completion_action("Evidence inspection is pending.", state, 0) == "finish"


def test_unknown_subject_without_evidence_still_fails_closed():
    query = "Trace Neralis Vexon's route from New Carthage to the Rhone."
    assert assess_evidence_support(query, "historical_route", []).status == "insufficient"
    _, state = agent([terminal("Evidence is insufficient.", [], True)]).respond(query, AgentState(session_id="empty"))
    assert state.historical_events == []
    assert state.historical_route is None
    assert state.historical_route_presentation is None


def test_explicit_unseen_actor_reaches_route_admission():
    query = "Trace Neralis Vexon's route from New Carthage to the Rhone."
    source = ev("unseen", "Neralis Vexon marched from New Carthage to the Rhone.")
    _, state = agent([terminal("Evidence inspected.", [source.id])], [source]).respond(query, AgentState(session_id="positive"))
    assert state.evidence_support_status == "unassessed"
    assert any(event.actor.actor_status is EventActorStatus.EXPLICIT and event.actor.actor_text == "Neralis Vexon"
               and event.evidence_refs == [source.id] for event in state.historical_events)
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2


def test_query_name_cannot_replace_unrelated_evidence_actor():
    query = "Trace Neralis Vexon's route from New Carthage to the Rhone."
    source = ev("other", "Terenos Qavik marched from New Carthage to the Rhone.")
    _, state = agent([terminal("Evidence is insufficient.", [], True)], [source]).respond(query, AgentState(session_id="wrong-actor"))
    assert all(event.actor.actor_text != "Neralis Vexon" for event in state.historical_events)
    assert state.historical_route is None
    assert state.historical_route_presentation is None


def test_known_alias_support_remains_unchanged():
    query = "展示汉尼拔翻越阿尔卑斯的路线"
    result = assess_evidence_support(query, "historical_route", [ev("known", "Hannibal crossed the Alps.")])
    assert result.status == "sufficient"
    assert set(result.matched_subject_terms) == {"hannibal", "alps"}
