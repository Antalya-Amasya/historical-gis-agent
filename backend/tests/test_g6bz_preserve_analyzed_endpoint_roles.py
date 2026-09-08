"""G6BZ: preserve movement_semantics endpoint roles from duplicate demotion."""
from __future__ import annotations

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = extract(text)
    assert len(events) == 1
    assert events[0].event_type is HistoricalEventType.MOVEMENT
    return {mention.raw_text: mention.role for mention in events[0].place_mentions}


def test_sailed_to_preserves_destination():
    roles = movement_roles("Ariston sailed to Cyprus.")
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_returned_to_preserves_destination():
    roles = movement_roles("Ariston returned to Pontus.")
    assert roles.get("Pontus") is EventPlaceRole.DESTINATION


def test_sailed_from_to_preserves_origin_and_destination():
    roles = movement_roles("Ariston sailed from Rhodes to Cyprus.")
    assert roles.get("Rhodes") is EventPlaceRole.ORIGIN
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_marched_toward_preserves_destination():
    roles = movement_roles("Ariston marched toward Artaxata.")
    assert roles.get("Artaxata") is EventPlaceRole.DESTINATION


def test_passed_through_into_preserves_directional_roles():
    roles = movement_roles("Ariston passed through Cilicia into Cappadocia.")
    assert roles.get("Cilicia") is EventPlaceRole.ORIGIN
    assert roles.get("Cappadocia") is EventPlaceRole.DESTINATION


def test_negated_movement_endpoints_remain_fail_closed():
    text = "Ariston did not march from Rome to Capua, but returned home."
    events, _ = extract(text)
    assert events
    roles = {mention.raw_text: mention.role for mention in events[0].place_mentions}
    assert roles.get("Rome") is not EventPlaceRole.ORIGIN
    assert roles.get("Capua") is not EventPlaceRole.DESTINATION


def test_alternative_negated_destination_remains_fail_closed():
    text = "Ariston marched from Rome to Capua, not to Athens."
    roles = movement_roles(text)
    assert roles.get("Capua") is EventPlaceRole.DESTINATION
    assert roles.get("Athens") is not EventPlaceRole.DESTINATION


def test_modal_only_movement_still_rejected():
    events, _ = extract("Bion may sail to Delos tomorrow.")
    assert events == []
