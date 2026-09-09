"""G6CM: UNKNOWN candidates must not duplicate across conflicting explicit groups."""
from __future__ import annotations

from backend.app.models import EventActorStatus, HistoricalEvent
from backend.app.routes.events import HistoricalEventConsolidator
from backend.tests.test_g6ck_explicit_actor_consolidation_guard import (
    consolidate,
    explicit_actor,
    shared_identity_movement,
    unknown_actor,
)


def output_candidate_ids(events: list[HistoricalEvent]) -> list[str]:
    ids: list[str] = []
    for event in events:
        ids.extend(event.candidate_ids or [event.id])
    return ids


def assert_unique_candidates(events: list[HistoricalEvent]) -> None:
    ids = output_candidate_ids(events)
    assert len(ids) == len(set(ids))


def test_conflicting_explicit_and_unknown_never_duplicates_unknown():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
        shared_identity_movement("u", actor=unknown_actor(), evidence_ref="ev-u"),
    )
    assert len(events) == 3
    assert diagnostics["merged_candidate_count"] == 0
    assert_unique_candidates(events)
    refs = [ref for event in events for ref in event.evidence_refs]
    assert refs.count("ev-u") == 1
    explicit = [event for event in events if event.actor.actor_status is EventActorStatus.EXPLICIT]
    assert len(explicit) == 2
    assert {event.actor.actor_text for event in explicit} == {"Ariston", "Bion"}


def test_conflicting_explicit_unknown_order_invariant():
    forward, forward_diag = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
        shared_identity_movement("u", actor=unknown_actor(), evidence_ref="ev-u"),
    )
    reverse, reverse_diag = consolidate(
        shared_identity_movement("u", actor=unknown_actor(), evidence_ref="ev-u"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
    )
    assert len(forward) == len(reverse) == 3
    assert forward_diag["merged_candidate_count"] == reverse_diag["merged_candidate_count"] == 0
    assert sorted(output_candidate_ids(forward)) == sorted(output_candidate_ids(reverse))


def test_single_explicit_and_unknown_still_merges():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("u", actor=unknown_actor(), evidence_ref="ev-u"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1
    assert set(events[0].evidence_refs) == {"ev-a", "ev-u"}


def test_duplicate_explicit_and_unknown_still_merges():
    events, diagnostics = consolidate(
        shared_identity_movement("a1", actor=explicit_actor("Ariston"), evidence_ref="ev-a1"),
        shared_identity_movement("a2", actor=explicit_actor("Ariston"), evidence_ref="ev-a2"),
        shared_identity_movement("u", actor=unknown_actor(), evidence_ref="ev-u"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 2


def test_unknown_and_unknown_still_merges():
    events, diagnostics = consolidate(
        shared_identity_movement("u1", actor=unknown_actor(), evidence_ref="ev-u1"),
        shared_identity_movement("u2", actor=unknown_actor(), evidence_ref="ev-u2"),
    )
    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1


def test_multiple_unknowns_not_duplicated_across_conflicting_explicit_groups():
    events, diagnostics = consolidate(
        shared_identity_movement("a", actor=explicit_actor("Ariston"), evidence_ref="ev-a"),
        shared_identity_movement("b", actor=explicit_actor("Bion"), evidence_ref="ev-b"),
        shared_identity_movement("u1", actor=unknown_actor(), evidence_ref="ev-u1"),
        shared_identity_movement("u2", actor=unknown_actor(), evidence_ref="ev-u2"),
    )
    assert len(events) == 4
    assert diagnostics["merged_candidate_count"] == 0
    assert_unique_candidates(events)
    assert {ref for event in events for ref in event.evidence_refs} == {"ev-a", "ev-b", "ev-u1", "ev-u2"}
