"""V1.1G9A: Hannibal route-observation extraction contract."""
from __future__ import annotations

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventType,
)
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_aliases import HistoricalPlaceAlias

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
    HistoricalPlaceAlias("Padus", ("padus",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(
    HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES),
)


def _evidence(text: str, *, eid: str = "ev") -> Evidence:
    return Evidence(
        id=eid,
        author="Polybius",
        work="Histories",
        locator="III",
        excerpt=text,
        text=text,
    )


def _extract(text: str):
    return _EXTRACTOR.extract([_evidence(text)])


def _movement(text: str):
    events, _ = _extract(text)
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert len(movement) == 1, text
    return movement[0]


def _roles(event) -> dict[str, EventPlaceRole]:
    return {mention.raw_text: mention.role for mention in event.place_mentions}


def test_hannibal_arrived_at_place_called_the_island():
    event = _movement("Hannibal arrived at the place called the Island.")
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert any("island" in name.casefold() for name in roles)
    assert EventPlaceRole.ORIGIN not in roles.values()
    assert EventPlaceRole.DESTINATION in roles.values()


def test_hannibal_crossed_the_rhone():
    event = _movement("Hannibal crossed the Rhone.")
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert "Rhone" in roles
    assert roles["Rhone"] is EventPlaceRole.RELATED_PLACE
    assert EventPlaceRole.ORIGIN not in roles.values()


def test_hannibal_crossed_the_alps():
    event = _movement("Hannibal crossed the Alps.")
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert "Alps" in roles
    assert roles["Alps"] is EventPlaceRole.RELATED_PLACE


def test_after_leaving_new_carthage_compound_movement():
    event = _movement(
        "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    )
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert roles.get("New Carthage") is EventPlaceRole.ORIGIN
    assert roles.get("Alps") is EventPlaceRole.RELATED_PLACE
    assert roles.get("Italy") is EventPlaceRole.DESTINATION


def test_padus_plains_after_crossing_pass():
    event = _movement(
        "After crossing the pass, Hannibal came into the plains of the Padus."
    )
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    roles = _roles(event)
    assert "Padus" in roles
    assert roles["Padus"] is EventPlaceRole.DESTINATION
    assert "Hannibal" not in roles


def test_hannibal_is_never_emitted_as_origin_place():
    event = _movement("Hannibal came into Italy.")
    roles = _roles(event)
    assert "Hannibal" not in roles
    assert EventPlaceRole.ORIGIN not in roles.values()
    assert roles.get("Italy") is EventPlaceRole.DESTINATION


def test_attribution_source_is_not_extracted_as_place():
    event = _movement("According to Polybius, Hannibal crossed the Rhone.")
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert "Polybius" not in roles
    assert roles.get("Rhone") is EventPlaceRole.RELATED_PLACE


@pytest.mark.parametrize(
    "text",
    [
        "He crossed the Rhone.",
        "The commander crossed the Alps.",
    ],
)
def test_hannibal_extraction_actor_safety_negatives(text: str):
    event = _movement(text)
    assert event.actor.actor_status is not EventActorStatus.EXPLICIT


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal did not cross the Rhone.",
        "If Hannibal crossed the Alps, he would enter Italy.",
        "The pass, being crossed, would bring him into the plains of the Padus.",
        "According to Polybius, the Rhone was wide in spring.",
        "Polybius described the Rhone.",
    ],
)
def test_hannibal_extraction_movement_safety_negatives(text: str):
    events, _ = _extract(text)
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert not movement
    for event in events:
        roles = _roles(event) if event.place_mentions else {}
        assert "Hannibal" not in roles
        assert "Polybius" not in roles
