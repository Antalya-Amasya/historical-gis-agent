"""Identity normalization must not supply evidence-local movement actors."""
import pytest

from backend.app.agent.evidence_support import (
    _subject_alias_registry, assess_evidence_support, extract_subject_terms,
)
from backend.app.models import EventActorStatus
from backend.app.rag import query_bridge
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_agent_loop import agent, ev, terminal
from backend.app.models import AgentState


def source(text):
    return ev("source", text).model_copy(update={"author": "Neutral author", "work": "Neutral source"})


@pytest.mark.parametrize("name,alias", [("Polybius", "波利比乌斯"), ("Livy", "李维")])
def test_out_of_fixture_alias_and_explicit_actor_reach_admission(name, alias):
    assert name.lower() not in str(query_bridge.V1_ENTRIES).lower()
    assert extract_subject_terms(f"展示{alias}相关路线") == (name.lower(),)
    evidence = source(f"{name} marched from New Carthage to the Rhone.")
    _, state = agent([terminal("Evidence inspected.", [evidence.id])], [evidence]).respond(
        f"Trace {name}'s route from New Carthage to the Rhone.", AgentState(session_id=name))
    assert any(event.actor.actor_status is EventActorStatus.EXPLICIT and event.actor.actor_text == name
               and event.evidence_refs == [evidence.id] for event in state.historical_events)
    assert state.historical_route is not None
    assert len(state.historical_route.ordered_points) == 2


@pytest.mark.parametrize("name", ["Polybius", "Livy"])
def test_query_identity_cannot_supply_missing_or_wrong_actor(name):
    query = f"Trace {name}'s route from New Carthage to the Rhone."
    for text in ["The army marched from New Carthage to the Rhone.", "Ariston marched from New Carthage to the Rhone."]:
        evidence = source(text)
        _, state = agent([terminal("Evidence is insufficient.", [], True)], [evidence]).respond(query, AgentState(session_id=name))
        assert all(event.actor.actor_text != name for event in state.historical_events)
        assert state.historical_route is None
        assert state.historical_route_presentation is None


def test_unknown_identity_is_investigable_without_becoming_actor_proof():
    evidence = source("The army marched from New Carthage to the Rhone.")
    assert assess_evidence_support("Trace Neralis Vexon's route.", "historical_route", [evidence]).status == "unassessed"
    actor = EvidenceGroundedHistoricalEventExtractor._ground_movement_actor(evidence.text)
    assert actor.actor_status is not EventActorStatus.EXPLICIT
    assert actor.actor_text != "Neralis Vexon"


def test_legacy_inventory_membership_no_longer_changes_grounding(monkeypatch):
    registry = _subject_alias_registry()
    evidence = source("Ariston marched from New Carthage to the Rhone.")
    query = "Trace Neralis Vexon's route."
    before = assess_evidence_support(query, "historical_route", [evidence])
    monkeypatch.setattr(query_bridge, "V1_ENTRIES", (("测试别名", ("Neralis Vexon",), "person"),))
    assert _subject_alias_registry() == registry
    assert assess_evidence_support(query, "historical_route", [evidence]) == before
    assert before.status == "unassessed"


@pytest.mark.parametrize("alias,name", [("庞培", "pompey"), ("朱古达", "jugurtha"), ("凯撒", "caesar")])
def test_existing_aliases_use_shared_production_resource(alias, name):
    assert extract_subject_terms(alias) == (name,)
    assert name in query_bridge.HistoricalQueryBridge().transform(alias).retrieval_query.lower()
    assert assess_evidence_support(alias, "historical_route", [source(f"{name} marched.")]).status == "sufficient"
