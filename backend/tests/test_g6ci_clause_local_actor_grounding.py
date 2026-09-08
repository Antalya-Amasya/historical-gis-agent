"""G6CI: structured clause-local actor grounding on HistoricalEvent."""
from __future__ import annotations

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
