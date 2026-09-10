"""G6DM: bounded same-sentence backward steering destination recovery."""
from __future__ import annotations

import hashlib

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
    HistoricalPlaceAlias("Alexandria", ("Alexandria",)),
    HistoricalPlaceAlias("Cilicia", ("Cilicia",)),
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


def _steering_event(text: str):
    events, _ = EvidenceGroundedHistoricalEventExtractor(_extractor()).extract(
        [Evidence(id="e1", author="Plutarch", work="Lives", locator="1", excerpt=text, text=text)]
    )
    return next(
        (
            event
            for event in events
            if event.event_type.value == "MOVEMENT" and "steer" in event.summary.casefold()
        ),
        None,
    )


def test_single_prior_pelusium_binds_destination():
    sentence = "At the city of Pelusium stood the army; Ariston steered his course that way."
    roles = _movement_roles(sentence)
    assert roles.get("Pelusium") is EventPlaceRole.DESTINATION
    assert EventPlaceRole.ORIGIN not in roles.values()


def test_two_prior_places_fail_closed():
    sentence = (
        "At the city of Pelusium stood one army, and at the city of Alexandria stood another; "
        "Ariston steered his course that way."
    )
    roles = _movement_roles(sentence)
    assert EventPlaceRole.DESTINATION not in roles.values()


def test_cross_sentence_antecedent_rejected():
    sentence = "He heard of Pelusium. Later he steered his course that way."
    roles = _movement_roles(sentence)
    assert EventPlaceRole.DESTINATION not in roles.values()


def test_explicit_forward_destination_wins():
    sentence = "At the city of Rhodes stood the fleet; he steered that way to Cyprus."
    roles = _movement_roles(sentence)
    assert roles.get("Cyprus") is EventPlaceRole.DESTINATION
    assert roles.get("Rhodes") is not EventPlaceRole.DESTINATION


@pytest.mark.parametrize(
    "sentence",
    [
        "At the city of Pelusium stood the army; Ariston should steer that way.",
        "At the city of Pelusium stood the army; Ariston discussed steering that way.",
        "At the city of Pelusium stood the army; Ariston planned to steer that way.",
    ],
)
def test_modal_steering_rejected(sentence: str):
    assert _movement_roles(sentence) == {}


def test_negated_steering_rejected():
    sentence = "At the city of Pelusium stood the army; Ariston did not steer that way."
    assert _movement_roles(sentence) == {}


def test_figurative_steering_rejected():
    sentence = "At the city of Pelusium stood the army; he steered the conversation that way."
    assert _movement_roles(sentence) == {}


def test_pompey_production_backward_steering_binds_pelusium():
    sentence = (
        "At the city of Pelusium in Egypt the fleet lay at anchor, and he steered his course that way."
    )
    steering = _steering_event(sentence)
    assert steering is not None
    roles = {mention.raw_text: mention.role for mention in steering.place_mentions}
    assert roles.get("Pelusium") is EventPlaceRole.DESTINATION
    assert "Cyprus" not in roles
    assert EventPlaceRole.ORIGIN not in roles.values()
    digest = hashlib.sha256(f"e1:0:{steering.summary}".encode("utf-8")).hexdigest()[:12]
    assert steering.id == f"event-{digest}"


def test_movement_semantics_single_antecedent_endpoint():
    sentence = "At the city of Pelusium stood the army; Ariston steered his course that way."
    semantics = analyze_sentence(sentence, _extractor().aliases_in(sentence))
    assert semantics.is_movement
    assert any(
        endpoint.surface == "Pelusium" and endpoint.role == "destination"
        for endpoint in semantics.endpoints
    )
