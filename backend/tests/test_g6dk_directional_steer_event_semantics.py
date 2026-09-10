"""G6DK: directional steering movement event semantics."""

from __future__ import annotations

import pytest

from backend.app.models import Evidence, EventPlaceRole
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias

_ALIASES = (
    HistoricalPlaceAlias("Cyprus", ("Cyprus",)),
    HistoricalPlaceAlias("Rhodes", ("Rhodes",)),
    HistoricalPlaceAlias("Pelusium", ("Pelusium",)),
    HistoricalPlaceAlias("Rome", ("Rome",)),
    HistoricalPlaceAlias("Alexandria", ("Alexandria",)),
)


def _extractor() -> HistoricalPlaceMentionExtractor:
    return HistoricalPlaceMentionExtractor(_ALIASES)


def _movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = EvidenceGroundedHistoricalEventExtractor(_extractor()).extract(
        [Evidence(id="e1", author="Plutarch", work="Lives", locator="1", excerpt=text, text=text)]
    )
    movement = next((event for event in events if event.event_type.value == "MOVEMENT"), None)
    if movement is None:
        return {}
    return {mention.raw_text: mention.role for mention in movement.place_mentions}


@pytest.mark.parametrize(
    ("sentence", "place", "role"),
    [
        ("Ariston steered toward Cyprus.", "Cyprus", EventPlaceRole.DESTINATION),
        ("Ariston steered to Rhodes.", "Rhodes", EventPlaceRole.DESTINATION),
        ("Ariston steered his course toward Pelusium.", "Pelusium", EventPlaceRole.DESTINATION),
    ],
)
def test_explicit_steer_destinations(sentence: str, place: str, role: EventPlaceRole):
    roles = _movement_roles(sentence)
    assert roles.get(place) is role


def test_was_steering_preserves_movement():
    sentence = "Ariston was steering toward Cyprus."
    semantics = analyze_sentence(sentence, _extractor().aliases_in(sentence))
    assert semantics.is_movement
    assert _movement_roles(sentence).get("Cyprus") is EventPlaceRole.DESTINATION


@pytest.mark.parametrize(
    "sentence",
    [
        "Ariston should steer toward Cyprus.",
        "Ariston discussed steering toward Cyprus.",
        "Ariston did not steer toward Cyprus.",
    ],
)
def test_non_assertive_steer_rejected(sentence: str):
    assert _movement_roles(sentence) == {}


@pytest.mark.parametrize(
    "sentence",
    [
        "He steered the conversation toward Rome.",
        "The debate steered toward reform.",
        "The policy steered the state toward collapse.",
        "The general steered policy toward peace.",
        "She steered the discussion to Alexandria.",
    ],
)
def test_figurative_steer_not_physical_movement(sentence: str):
    assert _movement_roles(sentence) == {}


def test_pompey_steering_continuation_binds_pelusium():
    sentence = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He steered his course that way toward Pelusium."
    )
    events, _ = EvidenceGroundedHistoricalEventExtractor(_extractor()).extract(
        [Evidence(id="e1", author="Plutarch", work="Lives", locator="1", excerpt=sentence, text=sentence)]
    )
    steering = next(
        (
            event
            for event in events
            if event.event_type.value == "MOVEMENT" and "steered his course" in event.summary.casefold()
        ),
        None,
    )
    assert steering is not None
    roles = {mention.raw_text: mention.role for mention in steering.place_mentions}
    assert roles.get("Pelusium") is EventPlaceRole.DESTINATION
    assert "Cyprus" not in roles
