"""G4C bounded evidence-grounded relative temporal normalization tests."""
from __future__ import annotations

import json
from pathlib import Path

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
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.temporal import EvidenceTemporalResolver, TemporalResolutionContext

FIXTURE_PATH = Path(__file__).with_name("g4c_relative_temporal_benchmark.json")


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def resolve_sequence(texts: list[str], evidence_ref: str = "e1") -> list[HistoricalEventTemporalGrounding]:
    resolver = EvidenceTemporalResolver()
    context = TemporalResolutionContext()
    readings: list[HistoricalEventTemporalGrounding] = []
    for text in texts:
        batch, _ = resolver.resolve(text, evidence_ref, context=context)
        if batch:
            readings.append(resolver.primary(batch, evidence_ref))
    return readings


def test_explicit_anchor_plus_following_year():
    readings = resolve_sequence([
        "In 218 BCE Hannibal entered Italy.",
        "The following year he marched south.",
    ])
    assert len(readings) == 2
    assert readings[0].normalized_start == "-218"
    assert readings[1].normalized_start == "-217"
    assert readings[1].raw_expression == "The following year"
    assert readings[1].precision is TemporalPrecision.YEAR
    assert readings[1].status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert readings[1].evidence_refs == ["e1"]


def test_explicit_anchor_plus_same_year():
    readings = resolve_sequence([
        "In 218 BCE the army entered Italy.",
        "In the same year the army advanced south.",
    ])
    assert readings[1].normalized_start == "-218"
    assert readings[1].raw_expression == "In the same year"


def test_explicit_anchor_plus_previous_year():
    readings = resolve_sequence([
        "In 218 BCE the army entered Italy.",
        "The previous year supplies had been gathered.",
    ])
    assert readings[1].normalized_start == "-219"


def test_no_explicit_anchor_leaves_following_year_unresolved():
    readings, codes = EvidenceTemporalResolver().resolve("The following year he marched south.", "e1")
    assert readings == [] and codes == ["TEMPORAL_UNRESOLVED"]


def test_anchor_in_previous_evidence_item_does_not_leak():
    resolver = EvidenceTemporalResolver()
    context_a = TemporalResolutionContext()
    resolver.resolve("In 218 BCE Hannibal entered Italy.", "a", context=context_a)
    context_b = TemporalResolutionContext()
    readings, codes = resolver.resolve("The following year he marched south.", "b", context=context_b)
    assert readings == [] and codes == ["TEMPORAL_UNRESOLVED"]


def test_multiple_competing_anchors_use_nearest_preceding_explicit_year():
    readings = resolve_sequence([
        "In 218 BCE Hannibal entered Italy.",
        "In 216 BCE another battle occurred.",
        "The following year he marched south.",
    ])
    assert readings[-1].normalized_start == "-215"


def test_same_year_temporal_order_is_not_created():
    coords = {"Genava": (46.2, 6.1), "Lutetia": (48.9, 2.35)}
    def place(name: str) -> HistoricalPlace:
        lat, lon = coords[name]
        return HistoricalPlace(id=name.lower(), canonical_name=name, latitude=lat, longitude=lon, source="test", confidence=0.8, coordinate_role="exact_site")
    def site(identifier: str, name: str, year: int, raw: str):
        mention = HistoricalEventPlaceMention(raw_text=name, role=EventPlaceRole.EVENT_SITE, evidence_refs=["e1"])
        binding = HistoricalEventPlaceBinding(
            mention=mention, place=place(name), role=EventPlaceRole.EVENT_SITE,
            resolution_status=EventPlaceResolutionStatus.RESOLVED, evidence_refs=["e1"],
        )
        grounding = HistoricalEventTemporalGrounding(
            raw_expression=raw, normalized_start=str(year), normalized_end=str(year),
            precision=TemporalPrecision.YEAR, evidence_refs=["e1"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        )
        return HistoricalEvent(
            id=identifier, name=identifier, summary=raw, event_type=HistoricalEventType.MOVEMENT,
            evidence_refs=["e1"], place_bindings=[binding], temporal_grounding=grounding,
        )
    events = [site("earlier", "Genava", -218, "In 218 BCE"), site("later", "Lutetia", -218, "In the same year")]
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        events, [evidence("e1", "text")], event_id="test", name="Test", period="218 BCE",
    )
    assert outcome.route is None
    assert outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_following_year_temporal_order_is_created():
    coords = {"Genava": (46.2, 6.1), "Lutetia": (48.9, 2.35)}
    def place(name: str) -> HistoricalPlace:
        lat, lon = coords[name]
        return HistoricalPlace(id=name.lower(), canonical_name=name, latitude=lat, longitude=lon, source="test", confidence=0.8, coordinate_role="exact_site")
    def site(identifier: str, name: str, year: int, raw: str):
        mention = HistoricalEventPlaceMention(raw_text=name, role=EventPlaceRole.EVENT_SITE, evidence_refs=["e1"])
        binding = HistoricalEventPlaceBinding(
            mention=mention, place=place(name), role=EventPlaceRole.EVENT_SITE,
            resolution_status=EventPlaceResolutionStatus.RESOLVED, evidence_refs=["e1"],
        )
        grounding = HistoricalEventTemporalGrounding(
            raw_expression=raw, normalized_start=str(year), normalized_end=str(year),
            precision=TemporalPrecision.YEAR, evidence_refs=["e1"],
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        )
        return HistoricalEvent(
            id=identifier, name=identifier, summary=raw, event_type=HistoricalEventType.MOVEMENT,
            evidence_refs=["e1"], place_bindings=[binding], temporal_grounding=grounding,
        )
    events = [site("earlier", "Genava", -218, "In 218 BCE"), site("later", "Lutetia", -217, "The following year")]
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        events, [evidence("e1", "text")], event_id="test", name="Test", period="218 BCE",
    )
    assert [relation.rule for relation in outcome.relations] == [OrderingRule.TEMPORAL_ORDER]


@pytest.mark.parametrize("text", ["three days later", "during the winter", "after this"])
def test_unsupported_relative_phrases_remain_unresolved(text: str):
    context = TemporalResolutionContext()
    EvidenceTemporalResolver().resolve("In 218 BCE the army marched.", "e1", context=context)
    readings, codes = EvidenceTemporalResolver().resolve(text, "e1", context=context)
    assert readings == [] and codes == ["TEMPORAL_UNRESOLVED"]


def test_relative_grounding_preserves_provenance():
    readings = resolve_sequence([
        "In 44 BCE the commander crossed the river.",
        "Next year the senate convened.",
    ])
    grounding = readings[-1]
    assert grounding.raw_expression == "Next year"
    assert grounding.normalized_start == "-43"
    assert grounding.evidence_refs == ["e1"]
    assert grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    assert grounding.precision is TemporalPrecision.YEAR


def test_relative_derived_year_does_not_chain():
    readings = resolve_sequence([
        "In 218 BCE the army entered Italy.",
        "The following year he marched south.",
        "The following year he reached the coast.",
    ])
    assert len(readings) == 2
    assert readings[0].normalized_start == "-218"
    assert readings[1].normalized_start == "-217"


def test_consolidator_keeps_different_relative_years_separate():
    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _ = extractor.extract([
        evidence(
            "a",
            "In 218 BCE the army marched from Place X to Place Y. "
            "The following year the army marched from Place X to Place Y.",
        ),
    ])
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    years = {
        event.temporal_grounding.normalized_start
        for event in events
        if event.temporal_grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    }
    assert years == {"-218", "-217"}
    assert len(events) == 2


def test_bce_ce_boundary_following_year():
    readings = resolve_sequence([
        "In 1 BCE the commander arrived.",
        "The following year the assembly met.",
    ])
    assert readings[1].normalized_start == "1"


def test_bce_ce_boundary_previous_year():
    readings = resolve_sequence([
        "In 1 CE the commander arrived.",
        "The preceding year supplies were gathered.",
    ])
    assert readings[1].normalized_start == "-1"


@pytest.fixture(scope="module")
def benchmark_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_g4c_relative_temporal_benchmark(benchmark_cases: list[dict]):
    resolver = EvidenceTemporalResolver()
    explicit_tp = explicit_fp = explicit_fn = explicit_tn = 0
    relative_tp = relative_fp = relative_fn = 0
    for case in benchmark_cases:
        context = TemporalResolutionContext()
        prior: list[str] = case.get("prior", [])
        for prior_text in prior:
            resolver.resolve(prior_text, case["evidence_ref"], context=context)
        readings, codes = resolver.resolve(case["text"], case["evidence_ref"], context=context)
        primary = resolver.primary(readings, case["evidence_ref"]) if readings else None
        got = primary.normalized_start if primary and primary.status is TemporalGroundingStatus.EVIDENCE_GROUNDED else None
        expected = case.get("expected_year")
        supported_relative = case.get("relative_supported", False)
        has_explicit = case.get("has_explicit", False)
        if has_explicit:
            if got == expected:
                explicit_tp += 1
            elif got is not None and expected is not None:
                explicit_fp += 1
            elif expected is not None:
                explicit_fn += 1
            else:
                explicit_tn += 1
        elif supported_relative:
            if got == expected:
                relative_tp += 1
            elif got is not None:
                relative_fp += 1
            else:
                relative_fn += 1
        else:
            assert got is None, f"{case['id']}: expected unresolved, got {got} ({codes})"
    explicit_total = explicit_tp + explicit_fn
    relative_total = relative_tp + relative_fn
    assert explicit_total == 0 or explicit_tp / explicit_total >= 0.95
    assert relative_total == 0 or relative_tp / relative_total >= 0.95
    assert relative_fp == 0
