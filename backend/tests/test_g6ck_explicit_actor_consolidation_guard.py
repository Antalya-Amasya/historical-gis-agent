"""G6CK: block consolidation when explicit structured actors conflict."""
from __future__ import annotations

from backend.app.models import (
    EventActorStatus,
    EventGroundingStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventActorGrounding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    TemporalGroundingStatus,
)
from backend.app.routes.events import HistoricalEventConsolidator


def explicit_actor(name: str) -> HistoricalEventActorGrounding:
    tokens = name.split()
    return HistoricalEventActorGrounding(
        actor_text=name,
        actor_tokens=tokens,
        actor_span=(0, len(name)),
        actor_status=EventActorStatus.EXPLICIT,
        actor_clause_span=(0, len(name)),
    )


def unknown_actor() -> HistoricalEventActorGrounding:
    return HistoricalEventActorGrounding(actor_status=EventActorStatus.UNKNOWN)


def shared_identity_movement(
    event_id: str,
    *,
    actor: HistoricalEventActorGrounding,
    evidence_ref: str,
    summary: str = "The army marched from Rome to Capua.",
) -> HistoricalEvent:
    refs = [evidence_ref]
    return HistoricalEvent(
        id=event_id,
        name=event_id,
        summary=summary,
        event_type=HistoricalEventType.MOVEMENT,
        place_mentions=[
            HistoricalEventPlaceMention(raw_text="Rome", role=EventPlaceRole.ORIGIN, evidence_refs=refs),
            HistoricalEventPlaceMention(raw_text="Capua", role=EventPlaceRole.DESTINATION, evidence_refs=refs),
        ],
        evidence_refs=refs,
        source_statements=[summary],
        temporal_grounding=HistoricalEventTemporalGrounding(status=TemporalGroundingStatus.UNRESOLVED),
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        actor=actor,
    )


def consolidate(*events: HistoricalEvent):
    consolidator = HistoricalEventConsolidator()
    keys = {consolidator._movement_key(event) for event in events}
    assert len(keys) == 1
    return consolidator.consolidate(list(events))


def test_explicit_same_actor_still_consolidates():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Ariston"), evidence_ref="ev-b"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1


def test_conflicting_explicit_actors_do_not_consolidate():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
    )
    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_explicit_and_unknown_behavior_unchanged():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=unknown_actor(), evidence_ref="ev-b"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1


def test_unknown_and_unknown_behavior_unchanged():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=unknown_actor(), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=unknown_actor(), evidence_ref="ev-b"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1


def test_multi_token_same_actor_still_consolidates():
    name = "Marcus Licinius Crassus"
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor(name), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor(name), evidence_ref="ev-b"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1


def test_multi_token_different_actors_blocked():
    events, diagnostics = consolidate(
        shared_identity_movement(
            "a",
            actor=explicit_actor("Marcus Licinius Crassus"),
            evidence_ref="ev-a",
        ),
        shared_identity_movement(
            "b",
            actor=explicit_actor("Gaius Julius Caesar"),
            evidence_ref="ev-b",
        ),
    )
    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_order_invariance_ariston_then_bion():
    first, diagnostics_first = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
    )
    second, diagnostics_second = consolidate(
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
    )
    assert len(first) == len(second) == 2
    assert diagnostics_first["merged_candidate_count"] == diagnostics_second["merged_candidate_count"] == 0
