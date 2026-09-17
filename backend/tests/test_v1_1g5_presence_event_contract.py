"""V1.1G5/G6B: presence extractor and route pipeline contracts."""
from __future__ import annotations

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
    TemporalGroundingStatus,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor()
_DOC = "doc-1"


class _Geography:
    def __init__(self) -> None:
        self.entries = {
            "Harbor One": _place("harbor-one", "Harbor One"),
            "Fort Two": _place("fort-two", "Fort Two"),
        }

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        place = self.entries.get(arguments["name"])
        return {"found": False} if place is None else {"found": True, **place.model_dump(mode="json")}


def _place(identifier: str, name: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=identifier, canonical_name=name, latitude=41.0, longitude=12.0, source="Audited registry",
        source_id=identifier, confidence=0.9, coordinate_role="exact_site",
        spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
    )


def _evidence(eid: str, text: str, *, offset: int = 100) -> Evidence:
    return Evidence(
        id=eid, author="Fixture", work="Test", locator="1", excerpt=text, text=text,
        metadata={"document_id": _DOC, "spine_index": 1, "start_offset": offset},
    )


def _extract(text: str):
    return _EXTRACTOR.extract([_evidence("ev", text)])


def _resolved(items: list[Evidence]):
    candidates, _ = _EXTRACTOR.extract(items)
    consolidated, _ = HistoricalEventConsolidator().consolidate(candidates)
    return HistoricalEventPlaceResolver(_Geography()).resolve(consolidated)[0]


def _anchors(events, items):
    anchors, _ = project_event_anchors(events, items)
    return {(anchor.canonical_name, anchor.role) for anchor in anchors}


def _build(events, items):
    return EventAnchorRouteBuilder().build_with_diagnostics(
        events, items, event_id="g5", name="Alpha", period="49 BCE",
    )


def _presence_events(text: str):
    events, _ = _extract(text)
    return [event for event in events if event.event_type is HistoricalEventType.PRESENCE]


def _assert_presence_event(text: str, *, actor: str = "Commander Alpha", place: str = "Harbor One"):
    events = _presence_events(text)
    assert len(events) == 1
    event = events[0]
    assert event.event_type is HistoricalEventType.PRESENCE
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == actor
    assert event.source_statements == [text]
    assert event.evidence_refs == ["ev"]
    roles = {mention.raw_text: mention.role for mention in event.place_mentions}
    assert roles.get(place) is EventPlaceRole.EVENT_SITE
    assert EventPlaceRole.ORIGIN not in roles.values()
    assert EventPlaceRole.DESTINATION not in roles.values()


def _assert_no_presence(text: str):
    events, _ = _extract(text)
    assert not any(event.event_type is HistoricalEventType.PRESENCE for event in events)


def test_affirmative_was_at_establishes_presence_event():
    _assert_presence_event("Commander Alpha was at Harbor One.")


def test_affirmative_remained_at_establishes_presence_event():
    _assert_presence_event("Commander Alpha remained at Harbor One.")


@pytest.mark.parametrize(
    "text",
    [
        "Commander Alpha was not at Harbor One.",
        "If Commander Alpha was at Harbor One, the army could regroup.",
        "Commander Alpha planned to go to Harbor One.",
        "Commander Alpha heard news from Harbor One.",
        "Harbor One was important to Commander Alpha.",
    ],
)
def test_non_assertive_presence_statements_emit_no_presence_event(text: str):
    _assert_no_presence(text)


def test_pronoun_actor_without_authority_emits_no_route_eligible_presence():
    events, _ = _extract("He was at Harbor One.")
    presence = [event for event in events if event.event_type is HistoricalEventType.PRESENCE]
    if presence:
        assert all(event.actor.actor_status is not EventActorStatus.EXPLICIT for event in presence)
    else:
        assert presence == []


def test_temporal_presence_retains_year_grounding():
    text = "In 49 BCE, Commander Alpha was at Harbor One."
    events = _presence_events(text)
    assert len(events) == 1
    grounding = events[0].temporal_grounding
    assert grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert grounding.normalized_start == "-49"


def test_presence_resolves_through_normal_geography_pipeline():
    text = "Commander Alpha was at Harbor One."
    item = _evidence("ev", text)
    events = _resolved([item])
    binding = next(binding for binding in events[0].place_bindings if binding.role is EventPlaceRole.EVENT_SITE)
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert binding.place and binding.place.coordinate_role == "exact_site"
    assert _anchors(events, [item]) == {("Harbor One", EventPlaceRole.EVENT_SITE)}


def test_movement_reached_resolves_through_normal_geography_pipeline():
    text = "Commander Alpha reached Fort Two."
    item = _evidence("ev", text)
    events = _resolved([item])
    binding = next(binding for binding in events[0].place_bindings if binding.role is EventPlaceRole.DESTINATION)
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert binding.place and binding.place.coordinate_role == "exact_site"
    assert _anchors(events, [item]) == {("Fort Two", EventPlaceRole.DESTINATION)}


def test_dated_presence_and_movement_form_temporal_waypoint_route():
    items = [
        _evidence("ev-a", "In 49 BCE, Commander Alpha was at Harbor One.", offset=100),
        _evidence("ev-b", "In 48 BCE, Commander Alpha reached Fort Two.", offset=200),
    ]
    events = _resolved(items)
    assert {event.event_type for event in events} == {HistoricalEventType.PRESENCE, HistoricalEventType.MOVEMENT}
    assert _anchors(events, items) == {
        ("Harbor One", EventPlaceRole.EVENT_SITE),
        ("Fort Two", EventPlaceRole.DESTINATION),
    }
    outcome = _build(events, items)
    assert outcome.route is not None and len(outcome.route.route_components) == 1
    relation = next(rel for rel in outcome.relations if rel.earlier == "Harbor One" and rel.later == "Fort Two")
    assert relation.rule is OrderingRule.TEMPORAL_ORDER
    claim = next(item for item in outcome.route.claims if item.claim_type == "WAYPOINT_ORDERING")
    assert claim.movement_relation is None and "no direct movement is asserted" in claim.text


def test_structural_presence_then_movement_forms_waypoint_ordering_route():
    stmt_a = "Commander Alpha was at Harbor One."
    stmt_b = "Then Commander Alpha reached Fort Two."
    items = [_evidence("ev-a", stmt_a, offset=100), _evidence("ev-b", stmt_b, offset=200)]
    events = _resolved(items)
    presence = [event for event in events if event.event_type is HistoricalEventType.PRESENCE]
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert len(presence) == 1 and len(movement) == 1
    assert presence[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert movement[0].actor.actor_status is EventActorStatus.EXPLICIT
    anchors = _anchors(events, items)
    assert ("Harbor One", EventPlaceRole.EVENT_SITE) in anchors
    assert ("Fort Two", EventPlaceRole.DESTINATION) in anchors
    outcome = _build(events, items)
    assert outcome.route is not None, "movement-only structural-order gate"
    relation = next(rel for rel in outcome.relations if rel.earlier == "Harbor One" and rel.later == "Fort Two")
    assert relation.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER
    claim = next(item for item in outcome.route.claims if item.claim_type == "WAYPOINT_ORDERING")
    assert claim.movement_relation is None
