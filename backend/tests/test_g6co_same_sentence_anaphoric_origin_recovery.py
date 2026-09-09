"""G6CO: recover ORIGIN from bounded same-sentence 'as far as X ... from there to Y'."""
from __future__ import annotations

from backend.app.models import EventPlaceRole, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_historical_events import evidence, extract


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = extract(evidence("g6co", text))
    movement = next(event for event in events if event.event_type is HistoricalEventType.MOVEMENT)
    return {mention.raw_text: mention.role for mention in movement.place_mentions}


def test_as_far_as_and_sailed_from_there_to():
    roles = movement_roles(
        "Ariston travelled as far as Rhodes and sailed from there to Cyprus.",
    )
    assert roles.get("Rhodes") is EventPlaceRole.ORIGIN
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_as_far_as_comma_then_from_there_to():
    roles = movement_roles(
        "Ariston marched as far as Capua, then went from there to Neapolis.",
    )
    assert roles.get("Capua") is EventPlaceRole.ORIGIN
    assert roles.get("Neapolis") is EventPlaceRole.DESTINATION


def test_from_there_without_bounded_antecedent_leaves_origin_unknown():
    roles = movement_roles("Ariston sailed from there to Cyprus.")
    assert all(role is not EventPlaceRole.ORIGIN for role in roles.values())
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_near_rhodes_does_not_become_origin():
    roles = movement_roles(
        "Ariston travelled near Rhodes and sailed from there to Cyprus.",
    )
    assert roles.get("Rhodes") is not EventPlaceRole.ORIGIN
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_multiple_bounded_antecedents_fail_closed():
    roles = movement_roles(
        "Ariston travelled as far as Rhodes and as far as Crete and sailed from there to Cyprus.",
    )
    assert all(role is not EventPlaceRole.ORIGIN for role in roles.values())
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION


def test_pompey_cilicia_pelusium_control():
    roles = movement_roles(
        "after coasting along the shore as far as Cilicia went across from there to Pelusium",
    )
    assert roles.get("Cilicia") is EventPlaceRole.ORIGIN
    assert roles.get("Pelusium") is EventPlaceRole.DESTINATION
