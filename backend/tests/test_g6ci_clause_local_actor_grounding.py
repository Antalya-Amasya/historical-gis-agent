"""G6CI: structured clause-local actor grounding on HistoricalEvent."""
from __future__ import annotations

import pytest

from backend.app.models import EventActorStatus, EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_g6bx_clause_local_modality_eligibility import POMPEY_PASSAGE


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_event(text: str):
    events, _ = extract(text)
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert len(movement) == 1
    return movement[0]


def assert_explicit_actor(text: str, expected_actor: str):
    event = movement_event(text)
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == expected_actor
    statement = event.source_statements[0]
    assert event.actor.actor_span is not None
    start, end = event.actor.actor_span
    assert statement[start:end] == expected_actor
    assert event.actor.actor_tokens == expected_actor.split()
    assert event.actor.actor_clause_span is not None
    clause_start, clause_end = event.actor.actor_clause_span
    assert clause_start <= start < end <= clause_end


def assert_unknown_actor(text: str):
    event = movement_event(text)
    assert event.actor.actor_status is EventActorStatus.UNKNOWN
    assert event.actor.actor_text is None
    assert event.actor.actor_span is None


def test_ariston_sailed_explicit_actor():
    assert_explicit_actor("Ariston sailed from Rhodes to Cyprus.", "Ariston")


@pytest.mark.parametrize(
    ("text", "actor"),
    [
        ("Marcus Tullius Cicero returned to Rome.", "Marcus Tullius Cicero"),
        ("Marcus Tullius Cicero, the envoy, returned to Rome.", "Marcus Tullius Cicero"),
        ("Lucius Cornelius Sulla, commander of the army, marched toward Athens.", "Lucius Cornelius Sulla"),
        ("Aurelian Varus, a veteran guide, sailed from Rhodes to Cyprus.", "Aurelian Varus"),
        ("After Bion waited, Ariston, the envoy, sailed from Rhodes to Cyprus.", "Ariston"),
    ],
)
def test_same_clause_noun_appositive_keeps_explicit_actor(text: str, actor: str):
    assert_explicit_actor(text, actor)


@pytest.mark.parametrize(
    "text",
    [
        "He returned to Rome.",
        "The soldiers, the army, marched toward Athens.",
        "Ariston and Bion, the envoys, returned to Rome.",
        "Bion said, Ariston, the envoy, returned to Rome.",
        "Ariston, the companion of Bion, returned to Rome.",
        "Ariston, who was the envoy, returned to Rome.",
        "Ariston, the envoy; returned to Rome.",
        "Ariston, the envoy—returned to Rome.",
    ],
)
def test_appositive_bridge_rejects_ambiguous_or_cross_clause_actor(text: str):
    assert EvidenceGroundedHistoricalEventExtractor._ground_movement_actor(text).actor_status is EventActorStatus.UNKNOWN


@pytest.mark.parametrize(
    "text",
    [
        "Ariston, the envoy, did not return to Rome.",
        "Ariston, the envoy, was prevented from entering Rome.",
        "Ariston, the envoy, planned to return to Rome.",
        'Bion said, "Ariston, the envoy, returned to Rome."',
    ],
)
def test_appositive_bridge_does_not_turn_non_assertion_into_movement(text: str):
    events, _ = extract(text)
    assert not any(event.event_type is HistoricalEventType.MOVEMENT for event in events)


def test_appositive_actor_change_keeps_directional_roles():
    event = movement_event("Ariston, the envoy, sailed from Rhodes to Cyprus.")
    assert event.actor.actor_text == "Ariston"
    assert {mention.raw_text: mention.role for mention in event.place_mentions} == {
        "Rhodes": EventPlaceRole.ORIGIN,
        "Cyprus": EventPlaceRole.DESTINATION,
    }


def test_appositive_movement_fix_does_not_change_presence_actor_grounding():
    actor = EvidenceGroundedHistoricalEventExtractor._ground_presence_actor(
        "Ariston, the envoy, was at Rome."
    )
    assert actor.actor_status is EventActorStatus.UNKNOWN


def test_bion_crossed_while_ariston_waited():
    assert_explicit_actor("Bion crossed the river while Ariston waited.", "Bion")


def test_bion_crossed_after_ariston_waited():
    assert_explicit_actor("Ariston waited while Bion crossed the river.", "Bion")


def test_after_defeating_bion_ariston_crossed():
    assert_explicit_actor("After defeating Bion, Ariston crossed the river.", "Ariston")


def test_pronoun_he_sailed_unknown():
    assert_unknown_actor("He sailed from Cyprus.")


def test_possessive_rival_unknown():
    assert_unknown_actor("His rival crossed the sea.")


def test_collective_soldiers_unknown():
    assert_unknown_actor("The soldiers marched toward Artaxata.")


def test_coordinated_pair_unknown():
    assert_unknown_actor("Ariston and Bion marched toward Capua.")


def test_coordinated_triple_unknown():
    assert_unknown_actor("Ariston, Bion, and Cato sailed from Rhodes.")


def test_pronoun_after_named_clause_unknown():
    assert_unknown_actor("After Ariston defeated Bion, he crossed the river.")


def test_pompey_passage_actor_unknown():
    event = movement_event(POMPEY_PASSAGE)
    assert event.actor.actor_status is EventActorStatus.UNKNOWN
    assert event.place_mentions
    roles = {mention.raw_text: mention.role for mention in event.place_mentions}
    assert roles.get("Cyprus") is EventPlaceRole.ORIGIN
    assert roles.get("Egypt") is not EventPlaceRole.DESTINATION


def test_mithridates_cos_explicit_actor():
    text = "In the meantime Mithridates crossed over to the island of Cos."
    assert_explicit_actor(text, "Mithridates")


def test_lucullus_march_explicit_actor():
    assert_explicit_actor("Lucullus marched toward Artaxata.", "Lucullus")
