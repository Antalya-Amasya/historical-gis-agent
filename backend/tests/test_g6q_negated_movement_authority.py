"""G6Q: explicitly negated movement must not become positive movement authority."""
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
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor

COORDINATES = {
    "Place Rome": (41.9, 12.5),
    "Place Capua": (41.08, 14.25),
    "Place Sicily": (37.5, 14.0),
    "Place Africa": (36.8, 10.2),
}


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_events(text: str) -> list[HistoricalEvent]:
    events, _ = extract(text)
    return [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]


def endpoint_roles(text: str) -> dict[str, EventPlaceRole]:
    roles: dict[str, EventPlaceRole] = {}
    for event in movement_events(text):
        for mention in event.place_mentions:
            roles[mention.raw_text] = mention.role
    return roles


def build_route(text: str):
    events = movement_events(text)
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
                )
            )
    return EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        [evidence("ev", text)],
        event_id="g6q",
        name="G6Q",
        period="100 BCE",
    )


def test_a_gpt6_counterexample_blocks_positive_movement():
    text = "Ariston did not march from Rome to Capua."
    events, diagnostics = extract(text)

    assert movement_events(text) == []
    assert events == []
    assert diagnostics["reason_codes"] == ["NO_EVENT_EVIDENCE", "INSUFFICIENT_GROUNDING"]


def test_b_positive_control_still_extracts_movement():
    text = "Ariston marched from Place Rome to Place Capua."
    events = movement_events(text)

    assert len(events) == 1
    assert events[0].grounding_status is EventGroundingStatus.EVIDENCE_GROUNDED
    roles = endpoint_roles(text)
    assert roles["Place Rome"] is EventPlaceRole.ORIGIN
    assert roles["Place Capua"] is EventPlaceRole.DESTINATION


def test_c_never_blocks_sailing_movement():
    text = "Ariston never sailed from Place Sicily to Place Africa."

    assert movement_events(text) == []


def test_d_unrelated_negation_does_not_suppress_movement():
    text = "Ariston did not hesitate and marched from Place Rome to Place Capua."
    roles = endpoint_roles(text)

    assert roles["Place Rome"] is EventPlaceRole.ORIGIN
    assert roles["Place Capua"] is EventPlaceRole.DESTINATION


def test_e_alternative_destination_negation_does_not_suppress_primary_movement():
    text = "Ariston marched from Place Rome to Place Capua, not to Naples."
    roles = endpoint_roles(text)

    assert roles["Place Rome"] is EventPlaceRole.ORIGIN
    assert roles["Place Capua"] is EventPlaceRole.DESTINATION


def test_f_mixed_sentence_keeps_positive_movement_after_negated_clause():
    text = "Ariston did not remain in Place Rome, but marched to Place Capua."
    roles = endpoint_roles(text)

    assert roles["Place Capua"] is EventPlaceRole.DESTINATION


def test_route_level_negated_movement_is_not_accepted():
    text = "Ariston did not march from Place Rome to Place Capua."
    outcome = build_route(text)
    same = [relation for relation in outcome.relations if relation.rule is OrderingRule.SAME_MOVEMENT_EVENT]

    assert movement_events(text) == []
    assert same == []
    assert outcome.route is None or len(outcome.route.ordered_points) < 2
