"""G6DR: connector-authorized SOURCE_STRUCTURAL_ORDER survives real admission."""

from __future__ import annotations

from backend.app.models import EventActorStatus
from backend.app.routes.event_route_orchestration import (
    EventAnchorRouteBuilder,
    OrderingRule,
    _filter_relations_for_query,
)
from backend.app.routes.evidence_relevance import relation_admission_allowed
from backend.tests.test_g6ae_relation_participant_authority import QUERY_200, classify_movement, movement_event
from backend.tests.test_g6dq_explicit_connector_source_chronology import (
    explicit_actor,
    passage_evidence,
    _inter_order,
    movement_event as dq_movement_event,
    unknown_actor,
)

QUERY = (QUERY_200,)


def _explicit_move(identifier: str, statement: str, origin: str, destination: str, refs: list[str], *, year: str | None = None):
    event = movement_event(identifier, statement, origin, destination, year=year, refs=refs)
    return event.model_copy(update={"actor": explicit_actor("Ariston"), "source_statements": [statement], "summary": statement})


def _pair(first: str, second: str):
    passage = f"{first} {second}"
    e1 = _explicit_move("e1", first, "Roma", "Capua", ["ev1"])
    e2 = _explicit_move("e2", second, "Brundisium", "Corcyra", ["ev2"])
    evidence_by_id = {
        "ev1": passage_evidence("ev1", passage, offset=100),
        "ev2": passage_evidence("ev2", passage, offset=200),
    }
    return e1, e2, evidence_by_id


def _structural(relations):
    return [rel for rel in relations if rel.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER and rel.earlier == "Capua" and rel.later == "Brundisium"]


def test_connector_structural_order_survives_same_episode_admission():
    first = "In 200 BCE Ariston marched from Roma to Capua."
    second = "Then in 200 BCE Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = _pair(first, second)
    events_by_id = {e1.id: e1, e2.id: e2}

    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, QUERY)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, QUERY)
    assert e1_detail["admitted"] is True
    assert e2_detail["admitted"] is True
    assert e1.actor.actor_status is EventActorStatus.EXPLICIT
    assert e2.actor.actor_status is EventActorStatus.EXPLICIT
    assert e1.actor.actor_text == e2.actor.actor_text == "Ariston"

    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6dr", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    structural = _structural(outcome.relations)
    assert len(structural) == 1
    bridge = structural[0]
    assert bridge.event_ids == (e1.id, e2.id)
    assert relation_admission_allowed(bridge, events_by_id, evidence_by_id, QUERY, rule=bridge.rule)

    kept, _ = _filter_relations_for_query(list(outcome.relations), events_by_id, evidence_by_id, QUERY)
    assert bridge in kept
    provenance = next(item for item in outcome.diagnostics["ordering_provenance"] if item["rule"] == OrderingRule.SOURCE_STRUCTURAL_ORDER.value)
    assert provenance["historical_authority"] == "EVIDENCE_GROUNDED_WAYPOINT_ORDERING"
    assert provenance["rule"] == OrderingRule.SOURCE_STRUCTURAL_ORDER.value


def test_connector_structural_order_rejected_when_participant_episode_mismatch():
    first = "In 200 BCE Ariston marched from Roma to Capua."
    second = "Then Ariston sailed from Brundisium to Corcyra."
    e1, e2, evidence_by_id = _pair(first, second)
    events_by_id = {e1.id: e1, e2.id: e2}

    _, e1_detail = classify_movement(e1, events_by_id, evidence_by_id, QUERY)
    _, e2_detail = classify_movement(e2, events_by_id, evidence_by_id, QUERY)
    assert e1_detail["admitted"] is True
    assert e2_detail["admitted"] is False
    assert _inter_order(e1, e2, evidence_by_id)[2] is OrderingRule.SOURCE_STRUCTURAL_ORDER

    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2], list(evidence_by_id.values()), event_id="g6dr", name="Ariston", period="200 BCE", query_contexts=QUERY,
    )
    assert _structural(outcome.relations) == []
    rejected = outcome.diagnostics["rejected_relations"]
    assert any(item["rule"] == OrderingRule.SOURCE_STRUCTURAL_ORDER.value and item["earlier"] == "Capua" for item in rejected)


def test_unknown_actor_connector_does_not_authorize_structural_order():
    first = "In 200 BCE Ariston marched from Roma to Capua."
    second = "Then he sailed from Brundisium to Corcyra in 200 BCE."
    passage = f"{first} {second}"
    e1 = _explicit_move("e1", first, "Roma", "Capua", ["ev1"])
    e2 = movement_event("e2", second, "Brundisium", "Corcyra", year=None, refs=["ev2"])
    e2 = e2.model_copy(update={"source_statements": [second], "summary": second})
    evidence_by_id = {"ev1": passage_evidence("ev1", passage, offset=100), "ev2": passage_evidence("ev2", passage, offset=200)}
    assert _inter_order(e1, e2, evidence_by_id) is None


def test_pompey_cyprus_pelusium_remain_uncomposed_through_admission():
    cyprus = dq_movement_event("event-4b6ee897cebc", "Pompey sailed toward Cyprus.", "Rhodes", "Cyprus", refs=["ev-cyprus"], actor=unknown_actor())
    pelusium = dq_movement_event("event-bcd445242ec8", "He steered his course that way toward Pelusium.", "Alpha", "Pelusium", refs=["ev-pelusium"], actor=unknown_actor())
    evidence_by_id = {
        "ev-cyprus": passage_evidence("ev-cyprus", cyprus.summary, offset=100, document="pompey-doc"),
        "ev-pelusium": passage_evidence("ev-pelusium", pelusium.summary, offset=200, document="pompey-doc"),
    }
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [cyprus, pelusium], list(evidence_by_id.values()), event_id="g6dr", name="Pompey", period="48 BCE",
    )
    assert _structural(outcome.relations) == []
    assert not any(rel.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER for rel in outcome.relations)
