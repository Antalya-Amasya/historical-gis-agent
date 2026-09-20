"""V1.1G9E: actor authority repair — explicit grounding and false-EXPLICIT guard."""
from __future__ import annotations

import pytest

from backend.app.models import EventActorStatus, EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_aliases import HistoricalPlaceAlias

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(
    HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES),
)

ISLAND_PASSAGE = (
    "Meanwhile, after four days' march from the passage of the Rhone, "
    "Hannibal arrived at the place called the Island, "
    "Hannibal's march to the foot of the Alps."
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


def test_island_passage_grounds_hannibal_explicit_actor():
    event = _movement(ISLAND_PASSAGE)
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    statement = event.source_statements[0]
    start, end = event.actor.actor_span
    assert statement[start:end] == "Hannibal"
    roles = _roles(event)
    assert any("island" in name.casefold() for name in roles)
    assert EventPlaceRole.DESTINATION in roles.values()
    assert any("rhone" in name.casefold() for name in roles)
    assert any("alps" in name.casefold() for name in roles)


def test_participial_nested_structure_grounds_hannibal():
    event = _movement(
        "Hannibal, having advanced before the standards, ordered the soldiers to halt."
    )
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"


def test_perfect_arrival_grounds_hannibal():
    event = _movement("Hannibal had arrived in Italy with his army.")
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    roles = _roles(event)
    assert roles.get("Italy") is EventPlaceRole.DESTINATION


@pytest.mark.parametrize(
    "text",
    [
        "than news came that Hannibal had arrived in Italy",
        "Than news came that Hannibal had arrived in Italy.",
    ],
)
def test_clause_fragment_never_becomes_explicit_actor(text: str):
    event = _movement(text)
    assert event.actor.actor_text != "than news"
    if event.actor.actor_status is EventActorStatus.EXPLICIT:
        assert event.actor.actor_text == "Hannibal"
    else:
        assert event.actor.actor_status is EventActorStatus.UNKNOWN


def test_pronoun_continuation_stays_unknown():
    event = _movement("He crossed the Rhone after Hannibal arrived at the Island.")
    assert event.actor.actor_status is not EventActorStatus.EXPLICIT

