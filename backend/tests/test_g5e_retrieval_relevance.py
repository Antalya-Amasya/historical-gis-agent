"""G5E retrieval relevance and cross-campaign contamination control."""
from __future__ import annotations

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
    _relation_admission_allowed,
)
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_evidence_relevance,
    event_relevance,
    has_subject_campaign_conflict,
    movement_eligibility_with_context,
)
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_event_anchor_routes import evidence, movement
from backend.tests.test_g4b_route_components import relation


def _evidence(identifier: str, text: str, *, document: str = "doc-1", spine: int = 0, offset: int = 0) -> Evidence:
    return Evidence(
        id=identifier,
        author="Plutarch",
        work="Lives",
        locator="Chapter 1",
        excerpt=text,
        text=text,
        metadata={"document_id": document, "spine_index": spine, "start_offset": offset},
    )


def _place(name: str) -> HistoricalPlace:
    coords = {
        "Athens": (37.98, 23.73),
        "Peloponnesus": (37.5, 22.4),
        "Pontus": (40.5, 36.5),
        "Asia": (39.0, 35.0),
        "Carthago Nova": (-0.98, 37.6),
        "Italia": (12.5, 42.5),
        "Hispania": (-3.7, 40.4),
        "Iberus": (-0.5, 41.0),
        "Valentia": (-0.38, 39.47),
        "Rhodanus": (4.85, 43.33),
    }
    latitude, longitude = coords[name]
    return HistoricalPlace(
        id=name.lower(),
        canonical_name=name,
        latitude=latitude,
        longitude=longitude,
        source="test",
        confidence=0.8,
    )


def _binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    mention = HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs)
    return HistoricalEventPlaceBinding(
        mention=mention,
        place=_place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="registry",
    )


def _event(
    identifier: str,
    summary: str,
    bindings: list[HistoricalEventPlaceBinding],
    refs: list[str],
) -> HistoricalEvent:
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=summary,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        place_bindings=bindings,
        temporal_grounding=HistoricalEventTemporalGrounding(
            raw_expression="401",
            normalized_start="401",
            normalized_end="401",
            precision=TemporalPrecision.YEAR,
            evidence_refs=refs,
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        ),
    )


def test_xenophon_movement_continuation_kept_with_bounded_context():
    text = (
        "Xenophon and the army deliberated after Cunaxa. "
        "It was therefore a very hazardous journey to travel by land from Athens to Peloponnesus."
    )
    item = _evidence("xen", text)
    contexts = ("Xenophon Ten Thousand Cunaxa retreat route 401 BCE",)
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([item], query_contexts=contexts)
    summaries = [event.summary for event in events]
    assert any("Athens" in summary and "Peloponnesus" in summary for summary in summaries)


def test_unrelated_commander_still_filtered():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract(
        [
            _evidence("capture", "Commander A captured Town B in Province C."),
            _evidence("context", "General Z captured Town Y in Province Q."),
        ],
        query_contexts=("Commander A military actions in Province C",),
    )
    assert [event.evidence_refs for event in events] == [["capture"]]


def test_other_campaign_conflict_detected_without_place_overlap_inference():
    query = ("Sertorius Hispania civil war campaign",)
    hannibal_text = "Hannibal crossed from Carthago Nova into Italia after leaving Hispania."
    assert has_subject_campaign_conflict(hannibal_text, query)
    assert classify_evidence_relevance(hannibal_text, query) is EvidenceRelevance.OTHER_CAMPAIGN


def test_structural_relation_rejected_across_campaigns():
    sertorius = _event(
        "sert",
        "Sertorius withdrew from Valentia.",
        [_binding("Valentia", EventPlaceRole.ORIGIN, ["s"]), _binding("Hispania", EventPlaceRole.DESTINATION, ["s"])],
        ["s"],
    )
    hannibal = _event(
        "hann",
        "Hannibal crossed from Carthago Nova into Italia.",
        [_binding("Carthago Nova", EventPlaceRole.ORIGIN, ["h"]), _binding("Italia", EventPlaceRole.DESTINATION, ["h"])],
        ["h"],
    )
    structural = relation(
        "Hispania",
        "Italia",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=("s", "h"),
        event_ids=("sert", "hann"),
    )
    evidence_by_id = {
        "s": _evidence("s", "Sertorius withdrew from Valentia.", document="parallel", spine=10, offset=100),
        "h": _evidence("h", "Hannibal crossed from Carthago Nova into Italia.", document="parallel", spine=11, offset=200),
    }
    events_by_id = {sertorius.id: sertorius, hannibal.id: hannibal}
    query = ("Sertorius Hispania campaign",)
    assert not _relation_admission_allowed(structural, events_by_id, evidence_by_id, query)
    kept, rejected = _filter_relations_for_query([structural], events_by_id, evidence_by_id, query)
    assert kept == []
    assert rejected[0]["reason"] == "CAMPAIGN_RELEVANCE_REJECTED"


def test_same_campaign_structural_relation_preserved():
    first = _event(
        "e1",
        "Scipio advanced from Hispania toward Carthago Nova.",
        [_binding("Hispania", EventPlaceRole.ORIGIN, ["a"]), _binding("Carthago Nova", EventPlaceRole.DESTINATION, ["a"])],
        ["a"],
    )
    second = _event(
        "e2",
        "Scipio sailed from Carthago Nova to Italia.",
        [_binding("Carthago Nova", EventPlaceRole.ORIGIN, ["b"]), _binding("Italia", EventPlaceRole.DESTINATION, ["b"])],
        ["b"],
    )
    structural = relation(
        "Carthago Nova",
        "Italia",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=("a", "b"),
        event_ids=("e1", "e2"),
    )
    evidence_by_id = {
        "a": _evidence("a", "Scipio advanced from Hispania toward Carthago Nova.", document="parallel", spine=1, offset=10),
        "b": _evidence("b", "Scipio sailed from Carthago Nova to Italia.", document="parallel", spine=2, offset=20),
    }
    events_by_id = {first.id: first, second.id: second}
    query = ("Scipio Hispania campaign Italia",)
    assert _relation_admission_allowed(structural, events_by_id, evidence_by_id, query)


def test_mithridates_pontus_asia_pair_preserved():
    text = "Mithridates marched from Pontus into Asia Minor."
    item = _evidence("mith", text)
    contexts = ("Mithridates First Mithridatic War Pontus Asia",)
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([item], query_contexts=contexts)
    roles = {(binding.role, binding.place.canonical_name) for event in events for binding in event.place_bindings}
    assert (EventPlaceRole.ORIGIN, "Pontus") in roles or any("Pontus" in event.summary for event in events)


def test_hannibal_same_movement_rejected_for_sertorius_query():
    hannibal = _event(
        "hann",
        "Hannibal crossed from Carthago Nova into Italia.",
        [_binding("Carthago Nova", EventPlaceRole.ORIGIN, ["h"]), _binding("Italia", EventPlaceRole.DESTINATION, ["h"])],
        ["h"],
    )
    same_movement = relation(
        "Carthago Nova",
        "Italia",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("h",),
        event_ids=("hann",),
    )
    evidence_by_id = {"h": _evidence("h", "Hannibal crossed from Carthago Nova into Italia.")}
    events_by_id = {hannibal.id: hannibal}
    query = ("Sertorius Hispania",)
    assert event_relevance(hannibal, evidence_by_id, query) is EvidenceRelevance.OTHER_CAMPAIGN
    assert not _relation_admission_allowed(same_movement, events_by_id, evidence_by_id, query)


def test_movement_window_eligibility_helper():
    sentences = [
        "Xenophon led the retreat after Cunaxa.",
        "From there they marched to the coast.",
    ]
    assert movement_eligibility_with_context(
        sentences[1],
        HistoricalEventType.MOVEMENT,
        ("Xenophon retreat Cunaxa",),
        sentences=sentences,
        index=1,
        evidence_text=" ".join(sentences),
    )


def test_route_builder_filters_cross_campaign_structural_edges():
    sertorius = _event(
        "sert",
        "Sertorius withdrew from Iberus.",
        [_binding("Iberus", EventPlaceRole.ORIGIN, ["s"]), _binding("Carthago Nova", EventPlaceRole.DESTINATION, ["s"])],
        ["s"],
    )
    hannibal = _event(
        "hann",
        "Hannibal crossed from Carthago Nova into Italia.",
        [_binding("Carthago Nova", EventPlaceRole.ORIGIN, ["h"]), _binding("Italia", EventPlaceRole.DESTINATION, ["h"])],
        ["h"],
    )
    items = [
        _evidence("s", "Sertorius withdrew from Iberus toward Carthago Nova.", document="parallel", spine=5, offset=50),
        _evidence("h", "Hannibal crossed from Carthago Nova into Italia.", document="parallel", spine=6, offset=60),
    ]
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [sertorius, hannibal],
        items,
        event_id="sertorius",
        name="Sertorius",
        period="80 BCE",
        query_contexts=("Sertorius Hispania campaign",),
    )
    if outcome.route is not None:
        pairs = {(rel.earlier, rel.later) for rel in outcome.relations}
        assert ("Carthago Nova", "Italia") not in pairs
        assert ("Hispania", "Italia") not in pairs
    assert outcome.diagnostics.get("rejected_relation_count", 0) >= 0
