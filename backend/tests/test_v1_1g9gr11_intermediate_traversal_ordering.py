"""V1.1G9G-R1.1: preserve intermediate traversal ordering under predicate ownership."""
from __future__ import annotations

import pytest

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias

ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
    HistoricalPlaceAlias("Capua", ("capua",), "audited"),
    HistoricalPlaceAlias("Roma", ("rome", "roma"), "audited"),
)

LIVE_ISLAND_PASSAGE = (
    "Meanwhile, after four days\u2019 march from the passage of the Rhone, "
    "Hannibal arrived at the place called the Island, "
    "Hannibal\u2019s march to the foot of the Alps."
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(HistoricalPlaceMentionExtractor(ALIASES))
_MENTION_EXTRACTOR = HistoricalPlaceMentionExtractor(ALIASES)


def _evidence(text: str, *, eid: str = "ev") -> Evidence:
    return Evidence(
        id=eid,
        author="Polybius",
        work="Histories",
        locator="III",
        excerpt=text,
        text=text,
    )


def _movement(text: str):
    events, _ = _EXTRACTOR.extract([_evidence(text)])
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert len(movement) == 1, text
    return movement[0]


def _order_pairs(text: str) -> list[tuple[str | None, str | None]]:
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    return [
        (item.earlier.canonical or item.earlier.surface, item.later.canonical or item.later.surface)
        for item in semantics.route_orderings
    ]


def _has_ordering(text: str, earlier: str, later: str) -> bool:
    return (earlier, later) in _order_pairs(text)


def test_after_leaving_crossed_and_came_into_emits_intermediate_traversal_chain():
    text = "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    pairs = _order_pairs(text)
    assert ("New Carthage", "Alpes") in pairs
    assert ("Alpes", "Italia") in pairs
    assert not _has_ordering(text, "Rhodanus", "Alpes")


def test_left_crossed_then_reached_emits_intermediate_traversal_chain():
    text = "Left New Carthage, crossed the Alps, then reached Italy."
    pairs = _order_pairs(text)
    assert ("New Carthage", "Alpes") in pairs
    assert ("Alpes", "Italia") in pairs


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal arrived at the place called the Island, his march toward the Alps continuing.",
        "Hannibal reached Capua, the route to Rome being difficult.",
    ],
)
def test_appositive_route_context_does_not_emit_context_destination_ordering(text: str):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert not any(
        item.later.canonical in {"Alpes", "Roma"}
        for item in semantics.route_orderings
    )


def test_sequential_reached_and_then_marched_retains_second_destination_ordering():
    text = "Hannibal reached Capua and then marched to Rome."
    assert _has_ordering(text, "Capua", "Roma")


def test_live_island_passage_preserves_rhone_to_island_without_alps_skip():
    pairs = _order_pairs(LIVE_ISLAND_PASSAGE)
    assert ("Rhodanus", "Island") in pairs
    assert not _has_ordering(LIVE_ISLAND_PASSAGE, "Rhodanus", "Alpes")
    event = _movement(LIVE_ISLAND_PASSAGE)
    destinations = [
        mention.raw_text
        for mention in event.place_mentions
        if mention.role is EventPlaceRole.DESTINATION
    ]
    assert any("island" in name.casefold() for name in destinations)


def test_event_route_orderings_include_alps_intermediate_for_new_carthage_passage():
    text = "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    event = _movement(text)
    pairs = [
        (ordering.earlier.canonical, ordering.later.canonical)
        for ordering in event.route_orderings
    ]
    assert ("New Carthage", "Alpes") in pairs
    assert ("Alpes", "Italia") in pairs
