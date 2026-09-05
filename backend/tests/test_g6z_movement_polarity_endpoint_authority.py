"""G6Z: movement predicate polarity must govern endpoint role authority."""

from __future__ import annotations

from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator

COORDINATES = {
    "Roma": (41.9, 12.5),
    "Capua": (41.08, 14.25),
    "Brundisium": (40.6, 17.9),
    "Corcyra": (39.6, 19.9),
    "Athens": (37.98, 23.73),
}


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract_all(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def extract_consolidated(text: str) -> list[HistoricalEvent]:
    events, _ = extract_all(text)
    consolidated, _ = HistoricalEventConsolidator().consolidate(events)
    return consolidated


def movement_endpoint_pairs(text: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for event in extract_consolidated(text):
        origin = next(
            (mention for mention in event.place_mentions if mention.role is EventPlaceRole.ORIGIN),
            None,
        )
        destination = next(
            (mention for mention in event.place_mentions if mention.role is EventPlaceRole.DESTINATION),
            None,
        )
        if origin is None or destination is None:
            continue
        pairs.add(
            (
                (origin.canonical_hint or origin.raw_text).casefold(),
                (destination.canonical_hint or destination.raw_text).casefold(),
            )
        )
    return pairs


def destination_names(text: str) -> set[str]:
    names: set[str] = set()
    for event in extract_consolidated(text):
        for mention in event.place_mentions:
            if mention.role is EventPlaceRole.DESTINATION:
                names.add((mention.canonical_hint or mention.raw_text).casefold())
    return names


def build_route(text: str):
    events = extract_consolidated(text)
    resolved: list[HistoricalEvent] = []
    for event in events:
        bindings: list[HistoricalEventPlaceBinding] = []
        for mention in event.place_mentions:
            name = mention.canonical_hint or mention.raw_text
            if name not in COORDINATES:
                continue
            latitude, longitude = COORDINATES[name]
            bindings.append(
                HistoricalEventPlaceBinding(
                    mention=mention,
                    place=HistoricalPlace(
                        id=name.lower().replace(" ", "-"),
                        canonical_name=name,
                        latitude=latitude,
                        longitude=longitude,
                        source="test",
                        confidence=0.8,
                    ),
                    role=mention.role,
                    resolution_status=EventPlaceResolutionStatus.RESOLVED,
                    evidence_refs=list(event.evidence_refs),
                    resolver_provenance="test",
                )
            )
        if bindings:
            resolved.append(
                HistoricalEvent(
                    id=event.id,
                    name=event.name,
                    summary=event.summary,
                    event_type=event.event_type,
                    evidence_refs=list(event.evidence_refs),
                    place_bindings=bindings,
                    temporal_grounding=HistoricalEventTemporalGrounding(),
                    grounding_status=event.grounding_status,
                    source_statements=event.source_statements,
                    place_mentions=event.place_mentions,
                )
            )
    return EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        [evidence("ev", text)],
        event_id="g6z",
        name="G6Z",
        period="100 BCE",
    )


def same_movement_relations(text: str):
    outcome = build_route(text)
    return [relation for relation in outcome.relations if relation.rule is OrderingRule.SAME_MOVEMENT_EVENT]


def test_a_mixed_clause_negated_movement_blocks_rome_capua_route():
    text = "Ariston did not march from Rome to Capua, but returned home."
    assert ("roma", "capua") not in movement_endpoint_pairs(text)
    assert same_movement_relations(text) == []


def test_b_military_classification_does_not_bypass_movement_polarity():
    text = "The army did not march from Rome to Capua."
    events = extract_consolidated(text)
    assert events
    assert events[0].event_type is HistoricalEventType.MILITARY
    assert ("roma", "capua") not in movement_endpoint_pairs(text)
    assert same_movement_relations(text) == []


def test_c_positive_control_preserves_rome_capua_movement():
    text = "Ariston marched from Rome to Capua."
    events = extract_consolidated(text)
    assert len(events) == 1
    assert events[0].event_type is HistoricalEventType.MOVEMENT
    assert events[0].grounding_status is EventGroundingStatus.EVIDENCE_GROUNDED
    assert ("roma", "capua") in movement_endpoint_pairs(text)
    assert same_movement_relations(text)


def test_d_unrelated_negation_preserves_positive_movement():
    text = "Ariston did not hesitate and marched from Rome to Capua."
    assert ("roma", "capua") in movement_endpoint_pairs(text)


def test_e_alternative_destination_excludes_negated_athens():
    text = "Ariston marched from Rome to Capua, not to Athens."
    for event in extract_consolidated(text):
        for mention in event.place_mentions:
            if mention.raw_text.casefold() == "athens":
                assert mention.role is not EventPlaceRole.DESTINATION
    assert ("roma", "capua") in movement_endpoint_pairs(text)


def test_f_mixed_negative_and_positive_movement_clauses_are_separated():
    text = "Ariston did not march from Rome to Capua, but later sailed from Brundisium to Corcyra."
    pairs = movement_endpoint_pairs(text)
    assert ("roma", "capua") not in pairs
    assert ("brundisium", "corcyra") in pairs


def test_g_collective_positive_control_preserves_army_movement():
    text = "The army marched from Rome to Capua."
    assert ("roma", "capua") in movement_endpoint_pairs(text)
    assert same_movement_relations(text)
