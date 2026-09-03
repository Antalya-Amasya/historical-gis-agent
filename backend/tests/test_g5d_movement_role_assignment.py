"""G5D generalized movement role assignment tests."""
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
    HistoricalPlaceAlias("C", ("c",)),
)


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _extractor_for(text: str) -> HistoricalPlaceMentionExtractor:
    places: set[str] = set()
    for case in _cases():
        for key in ("origin", "destination", "traversal"):
            value = case.get(key)
            if value and value != "UNKNOWN":
                places.add(value)
    aliases = tuple(HistoricalPlaceAlias(place, (place.lower(),), "audited") for place in sorted(places))
    return HistoricalPlaceMentionExtractor(aliases)


def _production_semantics(text: str):
    extractor = _extractor_for(text)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    prior_endpoints = ()
    merged_edges = []
    merged_endpoints = []
    is_movement = False
    for sentence in sentences:
        semantics = analyze_sentence(sentence, extractor.aliases_in(sentence), prior_endpoints=prior_endpoints)
        is_movement = is_movement or semantics.is_movement
        merged_edges.extend(semantics.edges)
        merged_endpoints.extend(semantics.endpoints)
        if semantics.endpoints:
            prior_endpoints = semantics.endpoints
    return is_movement, merged_edges, merged_endpoints


def _role_names(endpoints, edges, role: str) -> set[str]:
    names = {endpoint.place_name for endpoint in endpoints if endpoint.role == role}
    if role == "origin":
        names |= {edge.origin.place_name for edge in edges if edge.origin and edge.origin.role == "origin"}
    if role == "destination":
        names |= {edge.destination.place_name for edge in edges if edge.destination and edge.destination.role == "destination"}
    if role == "traversal":
        names |= {endpoint.place_name for endpoint in endpoints if endpoint.role == "traversal"}
        names |= {edge.traversal.place_name for edge in edges if edge.traversal}
        names |= {edge.origin.place_name for edge in edges if edge.origin and edge.origin.role == "traversal"}
    return names


def _evaluate() -> dict[str, float | int]:
    movement_tp = movement_fp = movement_fn = movement_tn = 0
    origin_tp = origin_fp = 0
    dest_tp = dest_fp = 0
    trav_tp = trav_fp = 0
    pair_tp = pair_fn = pair_fp = 0
    invented = 0

    for case in _cases():
        is_movement, edges, endpoints = _production_semantics(case["text"])
        predicted_movement = is_movement if case["movement_detected"] == "YES" else bool(edges or endpoints)
        expected_movement = case["movement_detected"] == "YES"
        if expected_movement and predicted_movement:
            movement_tp += 1
        elif expected_movement and not predicted_movement:
            movement_fn += 1
        elif not expected_movement and predicted_movement:
            movement_fp += 1
        else:
            movement_tn += 1

        origins = _role_names(endpoints, edges, "origin")
        dests = _role_names(endpoints, edges, "destination")
        travs = _role_names(endpoints, edges, "traversal")

        exp_origin = case.get("origin")
        if exp_origin and exp_origin != "UNKNOWN":
            if exp_origin in origins:
                origin_tp += 1
            elif origins:
                origin_fp += len(origins)
        elif origins:
            origin_fp += len(origins)

        exp_dest = case.get("destination")
        if exp_dest and exp_dest != "UNKNOWN":
            if exp_dest in dests:
                dest_tp += 1
            elif dests:
                dest_fp += len(dests)
        elif dests:
            dest_fp += len(dests)

        exp_trav = case.get("traversal")
        if exp_trav and exp_trav != "UNKNOWN":
            if exp_trav in travs:
                trav_tp += 1
            elif travs:
                trav_fp += len(travs)
        elif travs:
            trav_fp += len(travs)

        has_pair = any(edge.origin and edge.destination for edge in edges)
        if case.get("pairable") == "YES":
            if has_pair:
                pair_tp += 1
            else:
                pair_fn += 1
        elif has_pair and case.get("should_generate_same_movement") == "NO":
            if not (case.get("origin") and case.get("destination")):
                pair_fp += 1

        if exp_origin in (None, "UNKNOWN") and origins:
            invented += len(origins)
        if exp_dest in (None, "UNKNOWN") and dests:
            invented += len(dests)

    def ratio(num: int, den: int) -> float:
        return round(num / den, 3) if den else 1.0

    origin_expected = sum(1 for case in _cases() if case.get("origin") not in {None, "UNKNOWN"})
    dest_expected = sum(1 for case in _cases() if case.get("destination") not in {None, "UNKNOWN"})
    trav_expected = sum(1 for case in _cases() if case.get("traversal") not in {None, "UNKNOWN"})
    pair_expected = sum(1 for case in _cases() if case.get("pairable") == "YES")

    return {
        "samples": len(_cases()),
        "movement_precision": ratio(movement_tp, movement_tp + movement_fp),
        "movement_recall": ratio(movement_tp, movement_tp + movement_fn),
        "origin_precision": ratio(origin_tp, origin_tp + origin_fp),
        "origin_recall": ratio(origin_tp, origin_expected),
        "destination_precision": ratio(dest_tp, dest_tp + dest_fp),
        "destination_recall": ratio(dest_tp, dest_expected),
        "traversal_precision": ratio(trav_tp, trav_tp + trav_fp),
        "traversal_recall": ratio(trav_tp, trav_expected),
        "pair_precision": ratio(pair_tp, pair_tp + pair_fp),
        "pairable_recall": ratio(pair_tp, pair_expected),
        "invented_endpoints": invented,
    }


def test_g5d_benchmark_size_and_mix():
    cases = _cases()
    assert len(cases) >= 80
    hannibal = sum(1 for case in cases if case.get("campaign") == "Hannibal")
    assert hannibal / len(cases) <= 0.10
    blind = sum(
        1 for case in cases
        if case.get("campaign") in {"Xenophon", "Alexander", "Mithridates", "Crassus", "Lucullus", "Sertorius", "Scipio", "Marius", "Antony"}
    )
    assert blind >= 30


def test_g5d_precision_gates():
    metrics = _evaluate()
    assert metrics["movement_precision"] >= 0.97
    assert metrics["origin_precision"] >= 0.95
    assert metrics["destination_precision"] >= 0.95
    assert metrics["traversal_precision"] >= 0.95
    assert metrics["pair_precision"] >= 0.95
    assert metrics["invented_endpoints"] == 0


def test_g5d_pairable_event_recall_gate():
    metrics = _evaluate()
    assert metrics["pairable_recall"] >= 0.80


@pytest.mark.parametrize(
    "text,origin,destination",
    [
        ("Having left Pontus, thence he passed to Asia.", "Pontus", "Asia"),
        ("They were sent from Rome to Gaul.", "Rome", "Gaul"),
        ("He was driven from Italy into Gaul.", "Italy", "Gaul"),
        ("He left A, passed through B, and reached C.", "A", "C"),
    ],
)
def test_g5d_generalized_role_patterns(text: str, origin: str, destination: str):
    _, edges, endpoints = _production_semantics(text)
    assert origin in _role_names(endpoints, edges, "origin")
    assert destination in _role_names(endpoints, edges, "destination")


def test_g5d_person_as_place_suppressed():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([
        Evidence(
            id="e1", author="Source", work="Work", locator="1",
            excerpt="After the death of Philip, he continued in the service of Alexander, with the title of his principal friend.",
            text="After the death of Philip, he continued in the service of Alexander, with the title of his principal friend.",
        )
    ])
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert not movement


def test_g5d_truncated_chunk_no_invented_destination():
    text = (
        "When the civil strife broke out into war Caesar crossed the Adriatic from Brundusium "
        "in the winter, with what forces he had, and opened his"
    )
    extractor = HistoricalPlaceMentionExtractor(
        (HistoricalPlaceAlias("Brundisium", ("brundisium",), "audited"),),
    )
    semantics = analyze_sentence(text, extractor.aliases_in(text))
    origins = {endpoint.place_name for endpoint in semantics.endpoints if endpoint.role == "origin"}
    dests = {endpoint.place_name for endpoint in semantics.endpoints if endpoint.role == "destination"}
    assert "Brundusium" in origins
    assert not dests
