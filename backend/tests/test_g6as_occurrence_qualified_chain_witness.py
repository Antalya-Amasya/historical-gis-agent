"""G6AS: chain witnesses must be qualified per movement occurrence."""

from __future__ import annotations

import backend.app.routes.episode_relevance as er
from backend.app.routes.episode_relevance import classify_event_anchor_episode
from backend.app.routes.evidence_relevance import event_relevance, relation_admission_allowed
from backend.app.routes.event_route_orchestration import OrderingRule
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g6ag_query_subject_scope_normalization import COORDS, evidence, movement_event

COORDS.update({"Brundisium": (40.6, 17.9), "Capua": (41.08, 14.25), "Roma": (41.9, 12.5)})


def classify_legs(query: str, statements: list[tuple[str, str, str]]):
    events = {}
    for idx, (text, origin, destination) in enumerate(statements, 1):
        event = movement_event(f"e{idx}", text, origin, destination)
        events[event.id] = event
    ev = evidence("ev1", " ".join(text for text, _, _ in statements))
    out = []
    for event, (text, origin, destination) in zip(events.values(), statements):
        rel = relation(origin, destination, OrderingRule.SAME_MOVEMENT_EVENT, refs=(ev.id,), event_ids=(event.id,))
        tag = event_relevance(event, events, (query,))
        episode, detail = classify_event_anchor_episode(
            rel, events, {ev.id: ev}, (query,), subject_relevance=tag,
        )
        admitted = relation_admission_allowed(rel, events, {ev.id: ev}, (query,), rule=rel.rule)
        out.append((episode, detail, admitted))
    return out


def test_cross_subject_second_leg_is_not_a_chain_witness():
    query = "Trace Ariston from Rome to Capua."
    result = classify_legs(query, [
        ("Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("Bion marched from Brundisium to Capua.", "Brundisium", "Capua"),
    ])
    assert result[0][1]["admitted"] is False
    assert result[1][1]["admitted"] is False
    assert result[1][2] is False


def test_cross_campaign_second_leg_is_not_a_chain_witness():
    query = "Trace Ariston during Campaign Alpha from Rome to Capua."
    result = classify_legs(query, [
        ("During Campaign Alpha, Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("During Campaign Beta, Ariston marched from Brundisium to Capua.", "Brundisium", "Capua"),
    ])
    assert result[1][1]["admitted"] is False
    assert result[1][2] is False


def test_wrong_period_second_leg_is_not_a_chain_witness():
    query = "Trace Ariston in 200 BCE from Rome to Capua."
    result = classify_legs(query, [
        ("In 200 BCE, Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("In 100 BCE, Ariston marched from Brundisium to Capua.", "Brundisium", "Capua"),
    ])
    assert result[1][1]["admitted"] is False
    assert result[1][2] is False


def test_negated_second_leg_is_not_a_chain_witness():
    query = "Trace Ariston from Rome to Capua."
    result = classify_legs(query, [
        ("Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("Ariston did not march from Brundisium to Capua.", "Brundisium", "Capua"),
    ])
    assert result[1][1]["admitted"] is False
    assert result[1][2] is False


def test_same_subject_campaign_and_period_valid_two_leg_chain_remains():
    query = "Trace Ariston during Campaign Alpha in 200 BCE from Rome to Capua."
    result = classify_legs(query, [
        ("During Campaign Alpha in 200 BCE, Ariston marched from Rome to Brundisium.", "Roma", "Brundisium"),
        ("During Campaign Alpha in 200 BCE, Ariston marched from Brundisium to Capua.", "Brundisium", "Capua"),
    ])
    assert all(detail["admitted"] and admitted for _, detail, admitted in result)


def test_matching_time_wrong_endpoint_cannot_complete_route():
    query = "Trace Ariston in 200 BCE from Rome to Capua."
    assert er._proven_query_chain_member(
        "Brundisium", "Rome", "In 200 BCE, Ariston marched from Brundisium to Rome.",
        "In 200 BCE, Ariston marched from Brundisium to Rome.", (query,),
    ) is False


def test_unrelated_edge_inside_window_is_not_a_fragment_witness():
    query = "Trace Ariston from Rome to Capua."
    text = "Ariston marched from Rome to Brundisium. Athens marched to Rome. Ariston marched from Brundisium to Capua."
    assert er._proven_query_chain_member("Brundisium", "Capua", text, text, (query,)) is True
    assert er._proven_query_chain_member("Athens", "Rome", text, text, (query,)) is False


def test_three_leg_chain_remains_occurrence_qualified():
    query = "Trace Ariston from Rome to Capua."
    text = (
        "Ariston marched from Rome to Brundisium. "
        "Ariston marched from Brundisium to Corcyra. "
        "Ariston marched from Corcyra to Capua."
    )
    assert er._proven_query_chain_member("Brundisium", "Corcyra", text, text, (query,)) is True
