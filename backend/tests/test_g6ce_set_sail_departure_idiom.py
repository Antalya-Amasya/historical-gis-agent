"""G6CE: set/setting sail from X departure idiom."""
from __future__ import annotations

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_g6bx_clause_local_modality_eligibility import POMPEY_PASSAGE


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_events(text: str):
    events, _ = extract(text)
    return [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events = movement_events(text)
    assert len(events) == 1
    return {mention.raw_text: mention.role for mention in events[0].place_mentions}


def test_set_sail_from_assigns_origin():
    assert movement_roles("Ariston set sail from Rhodes.").get("Rhodes") is EventPlaceRole.ORIGIN


def test_was_setting_sail_from_assigns_origin():
    assert movement_roles("Ariston was setting sail from Rhodes.").get("Rhodes") is EventPlaceRole.ORIGIN


def test_participial_setting_sail_from_assigns_origin():
    text = "Setting sail from Rhodes, Ariston headed east."
    assert movement_roles(text).get("Rhodes") is EventPlaceRole.ORIGIN


def test_had_set_sail_from_assigns_origin():
    assert movement_roles("Ariston had set sail from Rhodes.").get("Rhodes") is EventPlaceRole.ORIGIN


def test_discussed_setting_sail_rejected():
    assert movement_events("Ariston discussed setting sail from Rhodes.") == []


def test_planned_to_set_sail_rejected():
    assert movement_events("Ariston planned to set sail from Rhodes.") == []


def test_prevented_from_setting_sail_rejected():
    assert movement_events("Ariston was prevented from setting sail from Rhodes.") == []


def test_negated_set_sail_rejected():
    assert movement_events("Ariston did not set sail from Rhodes.") == []


def test_setting_the_sails_is_not_movement():
    assert movement_events("Ariston was setting the sails on the ship.") == []


def test_pompey_passage_cyprus_origin_egypt_not_destination():
    events = movement_events(POMPEY_PASSAGE)
    assert events
    roles = {mention.raw_text: mention.role for mention in events[0].place_mentions}
    assert roles.get("Cyprus") is EventPlaceRole.ORIGIN
    assert roles.get("Egypt") is not EventPlaceRole.DESTINATION
