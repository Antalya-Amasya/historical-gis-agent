"""G5E-2 sentence-provenance-aware relation admission tests."""
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
from backend.app.routes.event_route_orchestration import OrderingRule, _filter_relations_for_query
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    event_relevance,
    relation_admission_allowed,
    relation_admission_diagnostic,
    same_movement_relation_relevance,
    statement_relation_relevance,
)
from backend.tests.test_g4b_route_components import relation


def _evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


def _place(name: str) -> HistoricalPlace:
    return HistoricalPlace(
        id=name.lower(),
        canonical_name=name,
        latitude=1.0,
        longitude=2.0,
        source="test",
        confidence=0.8,
    )


def _event(
    identifier: str,
    summary: str,
    *,
    source_statements: list[str] | None = None,
    refs: list[str] | None = None,
    bindings: list[HistoricalEventPlaceBinding] | None = None,
) -> HistoricalEvent:
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=summary,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs or ["ev1"],
        source_statements=source_statements or [summary],
        place_bindings=bindings or [],
        temporal_grounding=HistoricalEventTemporalGrounding(
            raw_expression="100",
            normalized_start="100",
            normalized_end="100",
            precision=TemporalPrecision.YEAR,
            evidence_refs=refs or ["ev1"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        ),
    )


def test_same_movement_local_direct_overrides_noisy_chunk():
    movement = (
        "It was a hazardous journey to travel by land from Alpha City to Beta Province."
    )
    noisy_chunk = (
        "Commander Z fought in Province Q. "
        + movement
    )
    event = _event("move", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", noisy_chunk)}
    contexts = ("Subject Alpha retreat route",)

    assert event_relevance(event, evidence_by_id, contexts) is EvidenceRelevance.OTHER_CAMPAIGN
    assert same_movement_relation_relevance(event, evidence_by_id, contexts) in {
        EvidenceRelevance.DIRECT_SUBJECT,
        EvidenceRelevance.DIRECT_CAMPAIGN,
        EvidenceRelevance.DIRECT_EVENT,
        EvidenceRelevance.SAME_CONFLICT_RELEVANT,
    }
    rel = relation("Alpha City", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("move",))
    assert relation_admission_allowed(rel, {"move": event}, evidence_by_id, contexts, rule=rel.rule)


def test_same_movement_local_other_campaign_remains_rejected():
    movement = "Commander Z crossed from Alpha Port into Beta Region."
    event = _event("foreign", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    contexts = ("Subject Alpha Hispania campaign",)
    rel = relation("Alpha Port", "Beta Region", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("foreign",))
    assert same_movement_relation_relevance(event, evidence_by_id, contexts) is EvidenceRelevance.OTHER_CAMPAIGN
    assert not relation_admission_allowed(
        rel, {"foreign": event}, evidence_by_id, contexts, rule=rel.rule,
    )


def test_bounded_continuation_inherits_direct_subject_safely():
    chunk = (
        "Subject Alpha led the retreat after the battle. "
        "From there they marched to Beta Province."
    )
    continuation = "From there they marched to Beta Province."
    event = _event("cont", continuation, source_statements=[continuation], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    contexts = ("Subject Alpha retreat",)
    tag = statement_relation_relevance(
        continuation, contexts, event=event, evidence_by_id=evidence_by_id,
    )
    assert tag in {
        EvidenceRelevance.DIRECT_SUBJECT,
        EvidenceRelevance.DIRECT_EVENT,
        EvidenceRelevance.SAME_CONFLICT_RELEVANT,
    }


def test_unknown_statement_without_support_stays_conservative():
    movement = "From there they marched to Beta Province."
    event = _event("unknown", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    contexts = ("Subject Alpha retreat",)
    rel = relation("Alpha", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("unknown",))
    assert not relation_admission_allowed(
        rel, {"unknown": event}, evidence_by_id, contexts, rule=rel.rule,
    )


def test_structural_remains_strict_on_chunk_relevance():
    first = _event(
        "e1",
        "Subject Alpha advanced from Alpha City toward Beta Port.",
        refs=["a"],
    )
    second = _event(
        "e2",
        "Commander Z crossed from Gamma Port into Delta Region.",
        refs=["b"],
    )
    evidence_by_id = {
        "a": _evidence("a", first.summary),
        "b": _evidence("b", second.summary),
    }
    contexts = ("Subject Alpha campaign",)
    structural = relation(
        "Beta Port",
        "Gamma Port",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=("a", "b"),
        event_ids=("e1", "e2"),
    )
    assert not relation_admission_allowed(
        structural,
        {"e1": first, "e2": second},
        evidence_by_id,
        contexts,
        rule=structural.rule,
    )


def test_same_conflict_not_auto_other_campaign():
    movement = "Commander A and Commander B fought near the same river crossing."
    event = _event("conflict", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    contexts = ("Commander A Second War campaign",)
    tag = same_movement_relation_relevance(event, evidence_by_id, contexts)
    assert tag is not EvidenceRelevance.OTHER_CAMPAIGN


def test_direct_subject_statement_beats_chunk_noise():
    movement = "travel by land from Alpha City to Beta Province"
    chunk = "Unrelated biography of Commander Z. " + movement + "."
    event = _event("pair", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    contexts = ("Subject Alpha Ten Thousand retreat",)
    diagnostic = relation_admission_diagnostic(
        relation("Alpha City", "Beta Province", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("pair",)),
        {"pair": event},
        evidence_by_id,
        contexts,
        rule=OrderingRule.SAME_MOVEMENT_EVENT,
    )
    assert diagnostic["admission"] == "ALLOW"
    assert diagnostic["chunk_relevance"] != diagnostic["final_relation_relevance"] or diagnostic["statement_relevance"]


def test_xenophon_style_noisy_chunk_same_movement_admitted():
    movement = "It was therefore a very hazardous journey to travel by land from Athens to Peloponnesus."
    chunk = (
        "Xenophon and the army deliberated after Cunaxa. "
        + movement
        + " Commander Z later fought elsewhere."
    )
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Athens", role=EventPlaceRole.ORIGIN, evidence_refs=["ev1"]),
            place=_place("Athens"),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(raw_text="Peloponnesus", role=EventPlaceRole.DESTINATION, evidence_refs=["ev1"]),
            place=_place("Peloponnesus"),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=["ev1"],
            resolver_provenance="registry",
        ),
    ]
    event = _event("xen", movement, source_statements=[movement], refs=["ev1"], bindings=bindings)
    evidence_by_id = {"ev1": _evidence("ev1", chunk)}
    contexts = ("Xenophon Ten Thousand Cunaxa retreat route 401 BCE",)
    rel = relation("Athens", "Peloponnesus", OrderingRule.SAME_MOVEMENT_EVENT, event_ids=("xen",))
    kept, rejected = _filter_relations_for_query(
        [rel], {"xen": event}, evidence_by_id, contexts,
    )
    assert kept == [rel]
    assert rejected == []
