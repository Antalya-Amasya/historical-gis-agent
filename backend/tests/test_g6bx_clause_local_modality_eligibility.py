"""G6BX: clause-local movement modality eligibility."""
from __future__ import annotations

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor

POMPEY_PASSAGE = (
    "As soon, therefore, as it was resolved upon, that he should fly into Egypt, "
    "setting sail from Cyprus in a galley of Seleucia, together with Cornelia, "
    "while the rest of his company sailed along near him, some in ships of war, "
    "and others in merchant vessels, he passed over sea without danger."
)


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_events(text: str):
    events, _ = extract(text)
    return [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]


def test_modal_only_movement_rejected():
    assert movement_events("He should cross the sea.") == []


def test_modal_clause_does_not_veto_later_completed_crossing():
    text = "He might march tomorrow, but Ariston crossed the River Alpha that evening."
    assert movement_events(text)


def test_modal_thought_clause_does_not_veto_later_sailing():
    text = "He thought Bion should leave, but Bion later sailed from Cyprus."
    assert movement_events(text)


def test_intent_clause_does_not_veto_later_sailing():
    text = "He did not intend to flee; instead he sailed to Egypt."
    assert movement_events(text)


def test_future_modal_sailing_rejected():
    assert movement_events("Bion may sail to Cyprus tomorrow.") == []


def test_ordinary_asserted_movement_unchanged():
    text = "Ariston marched from Rhodes to Cyprus."
    events = movement_events(text)
    assert len(events) == 1
    roles = {mention.raw_text: mention.role for mention in events[0].place_mentions}
    assert roles.get("Rhodes") is EventPlaceRole.ORIGIN
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_mixed_modal_and_asserted_movement_only_asserted_contributes():
    text = "Bion should sail to Rhodes tomorrow, but Bion later crossed to Cyprus."
    events = movement_events(text)
    assert len(events) == 1
    destinations = [
        mention.raw_text
        for mention in events[0].place_mentions
        if mention.role is EventPlaceRole.DESTINATION
    ]
    assert "Cyprus" in destinations
    assert "Rhodes" not in destinations


def test_pompey_trusted_passage_emits_completed_movement():
    events = movement_events(POMPEY_PASSAGE)
    assert events
    summaries = " ".join(event.summary for event in events).lower()
    assert "passed" in summaries or "sail" in summaries
