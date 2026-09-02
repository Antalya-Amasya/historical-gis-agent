"""G4H movement endpoint promotion: RELATED → ORIGIN under governed movement-from."""
from __future__ import annotations

import pytest

from backend.app.models import (
    Evidence,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Genava", ("genava", "geneva"), "audited"),
    HistoricalPlaceAlias("Lutetia", ("lutetia", "paris"), "audited"),
)

COORDINATES = {
    "Rhodanus": (45.05, 4.83),
    "Italia": (42.5, 12.5),
    "Alpes": (46.0, 7.0),
    "Genava": (46.2, 6.1),
    "Lutetia": (48.9, 2.35),
}


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier, author="Polybius", work="Histories", locator="III",
        excerpt=text, text=text,
    )


def extractor() -> HistoricalPlaceMentionExtractor:
    return HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES)


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([evidence("e1", text)])
    movement = next((event for event in events if event.event_type.value == "MOVEMENT"), None)
    if movement is None:
        return {}
    return {mention.raw_text: mention.role for mention in movement.place_mentions}


def place(name: str) -> HistoricalPlace:
    lat, lon = COORDINATES[name]
    return HistoricalPlace(
        id=name.lower(), canonical_name=name, latitude=lat, longitude=lon,
        source="test", confidence=0.9, coordinate_role="exact_site",
    )


def binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    mention = HistoricalEventPlaceMention(
        raw_text=name, canonical_hint=name, role=role, evidence_refs=refs,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
    )
    return HistoricalEventPlaceBinding(
        mention=mention, place=place(name), role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs, resolver_provenance="test",
    )


def build_route(text: str):
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([evidence("h1", text)])
    movement_events = [event for event in events if event.event_type.value == "MOVEMENT"]
    resolved = []
    for event in movement_events:
        bindings = []
        for mention in event.place_mentions:
            canonical = mention.canonical_hint or mention.raw_text
            if canonical in COORDINATES:
                bindings.append(binding(canonical, mention.role, list(event.evidence_refs)))
        if bindings:
            resolved.append(HistoricalEvent(
                id=event.id, name=event.name, summary=event.summary,
                event_type=HistoricalEventType.MOVEMENT,
                evidence_refs=list(event.evidence_refs),
                place_bindings=bindings,
                temporal_grounding=HistoricalEventTemporalGrounding(),
                grounding_status=event.grounding_status,
                source_statements=event.source_statements,
            ))
    return EventAnchorRouteBuilder().build_with_diagnostics(
        resolved, [evidence("h1", text)], event_id="hannibal", name="Hannibal", period="218 BCE",
    )


# --- Positive cases ---

def test_case1_crossed_from_rhone_valley_into_italy():
    text = "Hannibal crossed from the Rhone valley into Italy."
    roles = movement_roles(text)
    assert roles.get("Rhone") is EventPlaceRole.ORIGIN or roles.get("Rhodanus") is EventPlaceRole.ORIGIN
    assert roles.get("Italy") is EventPlaceRole.DESTINATION or roles.get("Italia") is EventPlaceRole.DESTINATION
    outcome = build_route(text)
    same = [r for r in outcome.relations if r.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert len(same) >= 1
    assert {same[0].earlier, same[0].later} == {"Rhodanus", "Italia"}


def test_case2_march_from_passage_of_rhone_toward_alps():
    text = "After four days' march from the passage of the Rhone the army advanced toward the Alps."
    roles = movement_roles(text)
    assert roles.get("Rhone") is EventPlaceRole.ORIGIN or roles.get("Rhodanus") is EventPlaceRole.ORIGIN
    assert roles.get("Alps") is EventPlaceRole.DESTINATION or roles.get("Alpes") is EventPlaceRole.DESTINATION
    outcome = build_route(text)
    same = [r for r in outcome.relations if r.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert len(same) >= 1
    assert {same[0].earlier, same[0].later} == {"Rhodanus", "Alpes"}


def test_case3_marched_from_x_to_y_unchanged():
    text = "The army marched from Genava to Lutetia."
    roles = movement_roles(text)
    assert roles.get("Genava") is EventPlaceRole.ORIGIN
    assert roles.get("Lutetia") is EventPlaceRole.DESTINATION


def test_departed_from_x_without_destination_is_origin_only():
    text = "Hannibal departed from Rhodanus."
    roles = movement_roles(text)
    assert roles.get("Rhodanus") is EventPlaceRole.ORIGIN
    outcome = build_route(text)
    same = [r for r in outcome.relations if r.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert same == []


def test_crossed_rhone_traversal_not_origin():
    text = "Hannibal crossed the Rhone."
    roles = movement_roles(text)
    assert EventPlaceRole.ORIGIN not in roles.values()


def test_passed_through_gaul_not_origin():
    text = "The army passed through Gaul."
    roles = movement_roles(text)
    assert EventPlaceRole.ORIGIN not in roles.values()


# --- Negative matrix (non-spatial / reference from) ---

NEGATIVE_FROM_CASES = [
    ("The army suffered from disease.", False),
    ("He learned from Caesar.", False),
    ("They escaped from danger.", False),
    ("Rome benefited from Italy.", False),
    ("This is different from Italy.", False),
    ("The story is known from Livy.", False),
    ("The account derived from Polybius.", False),
    ("From this account we know the route.", False),
    ("The camp was four miles from Rome.", False),
    ("News from Italy reached the senate.", False),
    ("The report from Rome was alarming.", False),
    ("Near the road from Rome they halted.", False),
]


@pytest.mark.parametrize("text,should_promote", NEGATIVE_FROM_CASES)
def test_negative_from_does_not_promote_origin(text: str, should_promote: bool):
    roles = movement_roles(text)
    assert (EventPlaceRole.ORIGIN in roles.values()) is should_promote


def test_cross_clause_no_spurious_origin_destination_pair():
    text = "He had earlier come from Spain. Later the army entered Italy."
    roles = movement_roles(text)
    origins = [name for name, role in roles.items() if role is EventPlaceRole.ORIGIN]
    destinations = [name for name, role in roles.items() if role is EventPlaceRole.DESTINATION]
    assert len(origins) <= 1
    assert len(destinations) <= 1
    if origins and destinations:
        assert not (origins[0] != destinations[0] and len(origins) == 1 and len(destinations) == 1)


# --- Movement semantics unit coverage ---

def test_analyze_crossed_from_valley_into():
    text = "Hannibal crossed from the Rhone valley into Italy."
    aliases = extractor().aliases_in(text)
    semantics = analyze_sentence(text, aliases)
    origins = [e for e in semantics.endpoints if e.role == "origin"]
    destinations = [e for e in semantics.endpoints if e.role == "destination"]
    assert origins and destinations
    assert origins[0].canonical == "Rhodanus"
    assert destinations[0].canonical == "Italia"


def test_analyze_mediated_passage_of_origin():
    text = "After four days' march from the passage of the Rhone the army advanced toward the Alps."
    aliases = extractor().aliases_in(text)
    semantics = analyze_sentence(text, aliases)
    origins = [e for e in semantics.endpoints if e.role == "origin"]
    destinations = [e for e in semantics.endpoints if e.role == "destination"]
    assert origins and destinations
    assert origins[0].canonical == "Rhodanus"
    assert destinations[0].canonical == "Alpes"
