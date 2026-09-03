"""G5F relation pipeline integrity — SAME_MOVEMENT emission and duplicate survival."""
from __future__ import annotations

from collections import defaultdict

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
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import (
    EventAnchorRouteBuilder,
    OrderingRule,
    _filter_relations_for_query,
    _merge_same_movement_relations,
)
from backend.app.routes.evidence_relevance import relation_admission_allowed
from backend.tests.test_event_anchor_routes import build, evidence, movement
from backend.tests.test_g4b_route_components import assemble, relation


def _relations_by_event(relations) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for rel in relations:
        for event_id in rel.event_ids:
            grouped[event_id].append(rel)
    return grouped


def _same_relations(outcome):
    return [rel for rel in outcome.relations if rel.rule is OrderingRule.SAME_MOVEMENT_EVENT]


def test_a_one_valid_event_emits_one_same_relation():
    outcome = build([movement("march", "Alpha", "Beta", ["a"])], [evidence("a")])
    same = _same_relations(outcome)
    assert len(same) == 1
    assert same[0].event_ids == ("march",)
    assert names(outcome) == ["Alpha", "Beta"]


def test_b_duplicate_valid_events_emit_one_same_relation_for_all_event_ids():
    events = [
        movement("e1", "Alpha", "Beta", ["a"]),
        movement("e2", "Alpha", "Beta", ["a"]),
    ]
    outcome = build(events, [evidence("a")])
    same = _same_relations(outcome)
    assert len(same) == 1
    assert set(same[0].event_ids) == {"e1", "e2"}
    grouped = _relations_by_event(outcome.relations)
    assert grouped["e1"]
    assert grouped["e2"]


def test_c_duplicate_valid_and_invalid_events_keep_one_valid_relation():
    valid_text = "Subject Alpha marched from Alpha City to Beta Province."
    invalid_text = "Commander Hannibal marched from Alpha City to Beta Province."
    valid = HistoricalEvent(
        id="valid",
        name="valid",
        summary=valid_text,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev-valid"],
        source_statements=[valid_text],
        place_bindings=movement("valid", "Alpha", "Beta", ["ev-valid"]).place_bindings,
        temporal_grounding=HistoricalEventTemporalGrounding(
            raw_expression="100",
            normalized_start="100",
            normalized_end="100",
            precision=TemporalPrecision.YEAR,
            evidence_refs=["ev-valid"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        ),
    )
    invalid = HistoricalEvent(
        id="invalid",
        name="invalid",
        summary=invalid_text,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev-invalid"],
        source_statements=[invalid_text],
        place_bindings=movement("invalid", "Alpha", "Beta", ["ev-invalid"]).place_bindings,
        temporal_grounding=HistoricalEventTemporalGrounding(
            raw_expression="218",
            normalized_start="218",
            normalized_end="218",
            precision=TemporalPrecision.YEAR,
            evidence_refs=["ev-invalid"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        ),
    )
    events = [valid, invalid]
    items = [
        Evidence(
            id="ev-valid",
            author="Author",
            work="Work",
            locator="1",
            excerpt=valid_text,
            text=valid_text,
            metadata={"document_id": "doc-valid", "spine_index": 1, "start_offset": 0},
        ),
        Evidence(
            id="ev-invalid",
            author="Author",
            work="Work",
            locator="2",
            excerpt=invalid_text,
            text=invalid_text,
            metadata={"document_id": "doc-invalid", "spine_index": 1, "start_offset": 0},
        ),
    ]
    from backend.app.routes.event_anchors import project_event_anchors

    generated = EventAnchorRouteBuilder()._relations(
        events,
        project_event_anchors(events, items, allow_contextual_related_places=True)[0],
        {item.id: item for item in items},
    )
    merged = _merge_same_movement_relations(generated)
    same = [rel for rel in merged if rel.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert len(same) == 2
    query = ("Subject Alpha campaign route",)
    events_by_id = {event.id: event for event in events}
    evidence_by_id = {item.id: item for item in items}
    admitted, rejected = _filter_relations_for_query(same, events_by_id, evidence_by_id, query)
    assert len(admitted) == 1
    assert admitted[0].event_ids == ("valid",)
    assert len(rejected) == 1


def test_d_strong_same_movement_survives_structural_reverse():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("e1",)),
        relation("Beta", "Alpha", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert assembly.usable[("Alpha", "Beta")].rule is OrderingRule.SAME_MOVEMENT_EVENT
    assert ("Beta", "Alpha") not in assembly.usable
    assert assembly.suppressed


def test_e_same_endpoints_different_relation_types_preserve_authority():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("e1",)),
        relation("Alpha", "Beta", OrderingRule.TEMPORAL_ORDER, refs=("b",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert assembly.usable[("Alpha", "Beta")].rule is OrderingRule.SAME_MOVEMENT_EVENT


def test_f_other_campaign_same_movement_rejected_by_admission():
    foreign = (
        "Commander Hannibal escaped through the straits of Cadiz toward Spain."
    )
    event = HistoricalEvent(
        id="foreign",
        name="foreign",
        summary=foreign,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=[foreign],
        place_bindings=[
            HistoricalEventPlaceBinding(
                mention=HistoricalEventPlaceMention(raw_text="Cadiz", role=EventPlaceRole.ORIGIN, evidence_refs=["ev1"]),
                place=HistoricalPlace(id="cadiz", canonical_name="Gades", latitude=36.5, longitude=-6.2, source="test", confidence=0.8),
                role=EventPlaceRole.ORIGIN,
                resolution_status=EventPlaceResolutionStatus.RESOLVED,
                evidence_refs=["ev1"],
            ),
            HistoricalEventPlaceBinding(
                mention=HistoricalEventPlaceMention(raw_text="Spain", role=EventPlaceRole.DESTINATION, evidence_refs=["ev1"]),
                place=HistoricalPlace(id="hispania", canonical_name="Hispania", latitude=42.5, longitude=-7.5, source="test", confidence=0.8),
                role=EventPlaceRole.DESTINATION,
                resolution_status=EventPlaceResolutionStatus.RESOLVED,
                evidence_refs=["ev1"],
            ),
        ],
        temporal_grounding=HistoricalEventTemporalGrounding(
            raw_expression="218",
            normalized_start="218",
            normalized_end="218",
            precision=TemporalPrecision.YEAR,
            evidence_refs=["ev1"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        ),
    )
    rel = relation("Gades", "Hispania", OrderingRule.SAME_MOVEMENT_EVENT, refs=("ev1",), event_ids=("foreign",))
    evidence_by_id = {
        "ev1": Evidence(
            id="ev1",
            author="Author",
            work="Work",
            locator="1",
            excerpt=foreign,
            text=foreign,
            metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 0},
        ),
    }
    query = ("Trace Sertorius in Hispania",)
    assert not relation_admission_allowed(
        rel, {"foreign": event}, evidence_by_id, query, rule=rel.rule,
    )


def test_g_ambiguous_endpoint_event_emits_no_same_relation():
    ambiguous = movement(
        "ambiguous",
        "Alpha",
        "Beta",
        ["a"],
    )
    ambiguous.place_bindings[0].resolution_status = EventPlaceResolutionStatus.AMBIGUOUS
    outcome = build([ambiguous], [evidence("a")])
    assert _same_relations(outcome) == []


def test_h_only_origin_or_destination_emits_no_same_relation():
    origin_only = movement("origin-only", "Alpha", "Beta", ["a"])
    origin_only.place_bindings = [origin_only.place_bindings[0]]
    destination_only = movement("destination-only", "Alpha", "Beta", ["b"])
    destination_only.place_bindings = [destination_only.place_bindings[1]]
    outcome = build([origin_only, destination_only], [evidence("a"), evidence("b")])
    assert _same_relations(outcome) == []


def test_merge_same_movement_preserves_provenance_and_event_ids():
    generated = [
        relation("Alpha", "Beta", refs=("a", "b"), event_ids=("e1",)),
        relation("Alpha", "Beta", refs=("a", "b"), event_ids=("e2",)),
        relation("Alpha", "Gamma", refs=("c",), event_ids=("e3",)),
    ]
    merged = _merge_same_movement_relations(generated)
    same_alpha_beta = [rel for rel in merged if rel.earlier == "Alpha" and rel.later == "Beta"]
    assert len(same_alpha_beta) == 1
    assert set(same_alpha_beta[0].event_ids) == {"e1", "e2"}
    assert set(same_alpha_beta[0].evidence_refs) == {"a", "b"}
    assert any(rel.later == "Gamma" for rel in merged)


def names(outcome):
    return [point.historical_place.canonical_name for point in outcome.route.ordered_points]
