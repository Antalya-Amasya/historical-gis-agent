"""G5B cross-campaign movement role generalization tests."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias

FIXTURE = Path(__file__).parent / "fixtures" / "g5b_movement_benchmark.json"

TEST_ALIASES = (
    HistoricalPlaceAlias("A", ("a",)),
    HistoricalPlaceAlias("B", ("b",)),
    HistoricalPlaceAlias("Alps", ("alps",), "audited"),
    HistoricalPlaceAlias("Italy", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Narbo", ("narbo",), "audited"),
    HistoricalPlaceAlias("Greece", ("greece",), "audited"),
)


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _extractor() -> HistoricalPlaceMentionExtractor:
    return HistoricalPlaceMentionExtractor(TEST_ALIASES)


def _production_semantics(text: str):
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    prior_endpoints = ()
    merged_edges = []
    merged_endpoints = []
    is_movement = False
    for sentence in sentences:
        semantics = analyze_sentence(sentence, _extractor().aliases_in(sentence), prior_endpoints=prior_endpoints)
        is_movement = is_movement or semantics.is_movement
        merged_edges.extend(semantics.edges)
        merged_endpoints.extend(semantics.endpoints)
        if semantics.endpoints:
            prior_endpoints = semantics.endpoints
    return is_movement, merged_edges, merged_endpoints


def _evaluate() -> dict[str, float | int]:
    movement_tp = movement_fp = movement_fn = movement_tn = 0
    origin_tp = origin_fp = 0
    dest_tp = dest_fp = 0
    relation_tp = relation_fp = 0

    for case in _cases():
        is_movement, edges, endpoints = _production_semantics(case["text"])
        predicted_movement = is_movement and bool(edges or endpoints)
        expected_movement = case["movement_detected"] == "YES"
        if expected_movement and predicted_movement:
            movement_tp += 1
        elif expected_movement and not predicted_movement:
            movement_fn += 1
        elif not expected_movement and predicted_movement:
            movement_fp += 1
        else:
            movement_tn += 1

        pred_origins = {endpoint.place_name for endpoint in endpoints if endpoint.role == "origin"}
        pred_dests = {endpoint.place_name for endpoint in endpoints if endpoint.role == "destination"}
        pred_origins |= {
            edge.origin.place_name for edge in edges
            if edge.origin and edge.origin.role == "origin"
        }
        pred_dests |= {edge.destination.place_name for edge in edges if edge.destination}

        exp_origin = case.get("origin")
        exp_dest = case.get("destination")
        if exp_origin and exp_origin != "UNKNOWN":
            if exp_origin in pred_origins:
                origin_tp += 1
            elif pred_origins:
                origin_fp += 1
        elif pred_origins:
            origin_fp += len(pred_origins)

        if exp_dest and exp_dest != "UNKNOWN":
            if exp_dest in pred_dests:
                dest_tp += 1
            elif pred_dests:
                dest_fp += 1
        elif pred_dests:
            dest_fp += len(pred_dests)

        should_relation = case.get("should_generate_same_movement") == "YES"
        has_relation = any(
            edge.origin and edge.destination and edge.origin.role == "origin"
            for edge in edges
        )
        if should_relation:
            if has_relation:
                relation_tp += 1
            else:
                relation_fp += 1
        elif has_relation:
            relation_fp += 1

    def ratio(num: int, den: int) -> float:
        return round(num / den, 3) if den else 1.0

    origin_expected = sum(1 for case in _cases() if case.get("origin") not in {None, "UNKNOWN"})
    dest_expected = sum(1 for case in _cases() if case.get("destination") not in {None, "UNKNOWN"})
    relation_expected = sum(1 for case in _cases() if case.get("should_generate_same_movement") == "YES")

    return {
        "samples": len(_cases()),
        "movement_precision": ratio(movement_tp, movement_tp + movement_fp),
        "movement_recall": ratio(movement_tp, movement_tp + movement_fn),
        "origin_precision": ratio(origin_tp, origin_tp + origin_fp),
        "origin_recall": ratio(origin_tp, origin_expected),
        "destination_precision": ratio(dest_tp, dest_tp + dest_fp),
        "destination_recall": ratio(dest_tp, dest_expected),
        "relation_precision": ratio(relation_tp, relation_tp + relation_fp),
        "relation_recall": ratio(relation_tp, relation_expected),
    }


def test_g5b_benchmark_minimum_size_and_mix():
    cases = _cases()
    assert len(cases) >= 40
    hannibal = sum(1 for case in cases if case.get("campaign") == "Hannibal")
    blind = sum(1 for case in cases if case.get("campaign") in {"Xenophon", "Alexander", "Mithridates"})
    assert hannibal / len(cases) <= 0.20
    assert blind >= 20


def test_g5b_precision_thresholds():
    metrics = _evaluate()
    assert metrics["origin_precision"] >= 0.95
    assert metrics["destination_precision"] >= 0.95
    assert metrics["relation_precision"] >= 0.95
    assert metrics["movement_precision"] >= 0.90


@pytest.mark.parametrize(
    "text,origin,destination",
    [
        ("He marched from A to B.", "A", "B"),
        ("He withdrew from A to B.", "A", "B"),
        ("He fled from A to B.", "A", "B"),
        ("He made his way from A to B.", "A", "B"),
    ],
)
def test_g5b_generalized_from_to(text: str, origin: str, destination: str):
    _, edges, endpoints = _production_semantics(text)
    origin_names = {endpoint.place_name for endpoint in endpoints if endpoint.role == "origin"}
    dest_names = {endpoint.place_name for endpoint in endpoints if endpoint.role == "destination"}
    origin_names |= {edge.origin.place_name for edge in edges if edge.origin}
    dest_names |= {edge.destination.place_name for edge in edges if edge.destination}
    assert origin in origin_names
    assert destination in dest_names


def test_g5b_non_spatial_from_guards():
    for text in (
        "The army suffered from disease.",
        "He learned from Caesar.",
        "News from Rome reached the senate.",
        "The camp was four miles from Rome.",
        "His possessions extended as far as the Euphrates.",
    ):
        events, _ = EvidenceGroundedHistoricalEventExtractor().extract(
            [Evidence(id="e1", author="Source", work="Work", locator="1", excerpt=text, text=text)]
        )
        movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
        origins = [
            mention for event in movement for mention in event.place_mentions
            if mention.role is EventPlaceRole.ORIGIN
        ]
        assert origins == [], text


def test_g5b_proceeded_to_alter_not_movement():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        Evidence(
            id="e1", author="Source", work="Work", locator="1",
            excerpt="He proceeded also at once to alter the government, placing thirty rulers in the city.",
            text="He proceeded also at once to alter the government, placing thirty rulers in the city.",
        )
    ])
    assert not any(event.event_type is HistoricalEventType.MOVEMENT for event in events)
