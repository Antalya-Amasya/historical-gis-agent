"""G6T: movement consolidation must preserve occurrence identity."""
from __future__ import annotations

from backend.app.models import (
    EventGroundingStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    TemporalGroundingStatus,
)
from backend.app.routes.events import HistoricalEventConsolidator


def _temporal(year: str) -> HistoricalEventTemporalGrounding:
    return HistoricalEventTemporalGrounding(
        raw_expression=year,
        normalized_start=year,
        normalized_end=year,
        status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
    )


def movement_event(
    event_id: str,
    *,
    summary: str,
    origin: str,
    destination: str,
    evidence_refs: list[str],
    temporal: HistoricalEventTemporalGrounding | None = None,
) -> HistoricalEvent:
    refs = list(evidence_refs)
    return HistoricalEvent(
        id=event_id,
        name=event_id,
        summary=summary,
        event_type=HistoricalEventType.MOVEMENT,
        place_mentions=[
            HistoricalEventPlaceMention(raw_text=origin, role=EventPlaceRole.ORIGIN, evidence_refs=refs),
            HistoricalEventPlaceMention(raw_text=destination, role=EventPlaceRole.DESTINATION, evidence_refs=refs),
        ],
        evidence_refs=refs,
        source_statements=[summary],
        temporal_grounding=temporal or HistoricalEventTemporalGrounding(status=TemporalGroundingStatus.UNRESOLVED),
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
    )


def consolidate(*events: HistoricalEvent):
    return HistoricalEventConsolidator().consolidate(list(events))


def test_a_gpt6_different_actor_and_reverse_direction_remain_separate():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Bion marched from Capua to Rome.",
        origin="Capua",
        destination="Rome",
        evidence_refs=["ev-b"],
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0
    routes = {
        (
            next(m.raw_text for m in event.place_mentions if m.role is EventPlaceRole.ORIGIN),
            next(m.raw_text for m in event.place_mentions if m.role is EventPlaceRole.DESTINATION),
        )
        for event in events
    }
    assert routes == {("Rome", "Capua"), ("Capua", "Rome")}


def test_b_same_route_different_actor_remain_separate():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Bion marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-b"],
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_c_same_actor_reverse_direction_remain_separate():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Ariston marched from Capua to Rome.",
        origin="Capua",
        destination="Rome",
        evidence_refs=["ev-b"],
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_d_same_actor_same_direction_different_time_remain_separate():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua in 100 BCE.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
        temporal=_temporal("-100"),
    )
    event_b = movement_event(
        "b",
        summary="Ariston marched from Rome to Capua in 90 BCE.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-b"],
        temporal=_temporal("-90"),
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_e_distinct_undated_occurrences_do_not_merge_on_route_alone():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua after the assembly adjourned.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Ariston marched from Rome to Capua before winter set in.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-b"],
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 2
    assert diagnostics["merged_candidate_count"] == 0


def test_f_genuine_duplicate_statement_still_consolidates():
    statement = "Ariston marched from Rome to Capua."
    event_a = movement_event(
        "a",
        summary=statement,
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary=statement,
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-b"],
    )

    events, diagnostics = consolidate(event_a, event_b)

    assert len(events) == 1
    assert diagnostics["merged_candidate_count"] == 1
    assert set(events[0].evidence_refs) == {"ev-a", "ev-b"}
    assert events[0].source_statements == [statement]


def test_route_input_receives_distinct_movement_events():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Bion marched from Capua to Rome.",
        origin="Capua",
        destination="Rome",
        evidence_refs=["ev-b"],
    )

    events, _ = consolidate(event_a, event_b)

    assert len(events) == 2
    origins = [sum(1 for m in event.place_mentions if m.role is EventPlaceRole.ORIGIN) for event in events]
    destinations = [sum(1 for m in event.place_mentions if m.role is EventPlaceRole.DESTINATION) for event in events]
    assert origins == [1, 1]
    assert destinations == [1, 1]


def test_red_reproduction_old_key_is_unordered_place_set():
    event_a = movement_event(
        "a",
        summary="Ariston marched from Rome to Capua.",
        origin="Rome",
        destination="Capua",
        evidence_refs=["ev-a"],
    )
    event_b = movement_event(
        "b",
        summary="Bion marched from Capua to Rome.",
        origin="Capua",
        destination="Rome",
        evidence_refs=["ev-b"],
    )
    consolidator = HistoricalEventConsolidator()

    old_style_key_a = "|".join(
        (
            consolidator._family(event_a.event_type),
            ",".join(consolidator._place_key(event_a)),
            consolidator._temporal_key(event_a),
        )
    )
    old_style_key_b = "|".join(
        (
            consolidator._family(event_b.event_type),
            ",".join(consolidator._place_key(event_b)),
            consolidator._temporal_key(event_b),
        )
    )

    assert old_style_key_a == "MOVEMENT|capua,rome|<unresolved>"
    assert old_style_key_b == old_style_key_a
