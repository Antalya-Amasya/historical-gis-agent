"""Incomplete facts remain documentary diagnostics, never route constraints."""
import pytest

from backend.app.models import EventActorStatus, EventPlaceRole, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_agent_loop import ev
from backend.app.agent.loop import BoundedAgentLoop


def extract(text):
    source = ev("source", text)
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([source])
    return source, events, diagnostics["incomplete_movement_facts"]


@pytest.mark.parametrize("text,outcome", [
    ("Marcus attempted to enter City Alpha but was prevented.", "PREVENTED"),
    ("Marcus attempted to enter City Alpha.", "ATTEMPTED"),
    ("Marcus marched toward City Alpha but abandoned the advance before reaching it.", "ABORTED"),
    ("Marcus planned to march to City Alpha.", "PLANNED"),
    ("Marcus did not enter City Alpha.", "NEGATED"),
    ("Marcus marched toward City Alpha but stopped before reaching it.", "ABORTED"),
])
def test_non_completion_retains_actor_destination_without_arrival(text, outcome):
    source, events, facts = extract(text)
    assert len(facts) == 1 and facts[0]["outcome"] == outcome
    assert facts[0]["actor"]["actor_text"] == "Marcus"
    assert facts[0]["actor"]["actor_status"] == "EXPLICIT"
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]
    assert all(p["role"] == "UNKNOWN" for p in facts[0]["destination_mentions"])
    assert facts[0]["evidence_refs"] == [source.id]
    assert events == []
    events, _ = HistoricalEventConsolidator().consolidate(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(events, [source], event_id="x", name="test", period="unspecified")
    assert outcome.route is None
    assert outcome.diagnostics["transition_constraint_count"] == 0
    assert outcome.diagnostics["observation_count"] == 0


def test_completed_entry_unchanged():
    _, events, facts = extract("Marcus entered City Alpha.")
    assert facts == []
    assert len(events) == 1 and events[0].event_type is HistoricalEventType.MOVEMENT
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert events[0].place_mentions[0].role is EventPlaceRole.DESTINATION


def test_prevented_then_completed_keeps_clauses_distinct():
    _, events, facts = extract("Marcus was initially prevented, but later entered City Alpha.")
    assert facts[0]["outcome"] == "PREVENTED"
    assert facts[0]["destination_mentions"] == []
    assert "entered" not in facts[0]["source_statement"]
    assert len(events) == 1 and events[0].place_mentions[0].role is EventPlaceRole.DESTINATION
    # No new actor propagation across clauses was introduced.
    assert events[0].actor.actor_status is EventActorStatus.UNKNOWN


def test_appian_shape_preserves_context_without_inventing_destination_or_actor():
    text = "While he was besieging the city Setovia a force came to its assistance, which he met and prevented from entering the place."
    source, events, facts = extract(text)
    assert facts[0]["outcome"] == "PREVENTED"
    assert facts[0]["actor"]["actor_status"] == "UNKNOWN"
    assert facts[0]["destination_mentions"] == []
    assert any(p["raw_text"] == "Setovia" and p["role"] == "UNKNOWN" for p in facts[0]["context_place_mentions"])
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(events, [source], event_id="x", name="test", period="unspecified")
    assert outcome.route is None and outcome.diagnostics["transition_constraint_count"] == 0
    assert not BoundedAgentLoop._final_answer_asserts_unsupported_route("The attempted movement was prevented, so no completed route should be reconstructed.")


def test_neighboring_places_are_not_attempted_destinations():
    _, _, facts = extract("Marcus waited in City Beta. Marcus planned to march to City Alpha.")
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]


def test_completed_neighbor_is_not_a_negative_destination():
    _, _, facts = extract("Marcus entered City Beta and Neralis attempted to enter City Alpha but was prevented.")
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]


def test_other_actor_abandonment_does_not_erase_completed_movement():
    _, events, facts = extract("Marcus marched to City Alpha but Bion abandoned the advance before reaching it.")
    assert facts[0]["actor"]["actor_text"] == "Bion"
    assert facts[0]["destination_mentions"] == []
    assert events


@pytest.mark.parametrize("subject", ["He", "The commander", "The army"])
def test_negative_fact_does_not_promote_pronouns_offices_or_collectives(subject):
    _, _, facts = extract(f"{subject} planned to march to City Alpha.")
    assert facts[0]["actor"]["actor_status"] == "UNKNOWN"


@pytest.mark.parametrize("text", [
    "If Marcus planned to march to City Alpha, the plan would fail.",
    'A witness said that Marcus did not enter City Alpha.',
])
def test_hypothetical_or_reported_negative_does_not_gain_fact_authority(text):
    _, _, facts = extract(text)
    assert facts == []


def test_later_negative_clause_does_not_get_replaced_by_earlier_positive_text():
    _, _, facts = extract("Marcus marched to City Beta but later was prevented from entering City Alpha.")
    assert "prevented" in facts[0]["source_statement"]
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]


def test_diagnostic_fact_survives_agent_state_without_route_or_geometry():
    from backend.tests.test_agent_loop import agent, terminal
    from backend.app.models import AgentState
    text = "Marcus planned to march to City Alpha."
    source = ev("planned", text)
    reply, state = agent([terminal("The movement was planned, so no completed route should be reconstructed.", [source.id])], [source]).respond(
        "Trace Marcus's route to City Alpha.", AgentState(session_id="incomplete-state"))
    assert state.historical_event_diagnostics["extraction"]["incomplete_movement_facts"][0]["outcome"] == "PLANNED"
    assert state.historical_route is None
    assert state.historical_route_presentation is None
    assert source.id in reply


def test_completed_actor_does_not_replace_negated_actor():
    _, _, facts = extract("Marcus entered City Beta and Neralis did not enter City Alpha.")
    assert facts[0]["actor"]["actor_text"] != "Marcus"
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]


def test_active_preventing_actor_is_not_promoted_to_prevented_mover():
    _, _, facts = extract("Marcus prevented Bion from entering City Alpha.")
    assert facts[0]["actor"]["actor_status"] == "UNKNOWN"
    assert [p["raw_text"] for p in facts[0]["destination_mentions"]] == ["City Alpha"]
