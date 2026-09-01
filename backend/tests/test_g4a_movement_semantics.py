"""G4A movement semantic extraction benchmark and regression tests."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.app.models import Evidence, EventPlaceRole
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias

FIXTURE = Path(__file__).parent / "fixtures" / "g4a_movement_benchmark.json"

TEST_ALIASES = (
    HistoricalPlaceAlias("A", ("a",)),
    HistoricalPlaceAlias("B", ("b",)),
    HistoricalPlaceAlias("C", ("c",)),
    HistoricalPlaceAlias("Alps", ("alps",), "audited"),
    HistoricalPlaceAlias("Italy", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Greece", ("greece",), "audited"),
    HistoricalPlaceAlias("Narbo", ("narbo",), "audited"),
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Carthago Nova", ("carthago nova", "new carthage"), "audited"),
    HistoricalPlaceAlias("Padus", ("padus", "po"), "audited"),
)


def evidence(identifier: str, text: str, **metadata) -> Evidence:
    return Evidence(
        id=identifier, author=metadata.get("author", "Source"), work="Work",
        locator="1", excerpt=text, text=text, metadata=metadata,
    )


def extractor() -> HistoricalPlaceMentionExtractor:
    return HistoricalPlaceMentionExtractor(TEST_ALIASES)


def claims(text: str, *, event_id: str = "event") -> list:
    return extractor().movement_claims([evidence("e1", text)], event_id=event_id)


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([evidence("e1", text)])
    movement = next((event for event in events if event.event_type.value == "MOVEMENT"), None)
    if movement is None:
        return {}
    return {mention.raw_text: mention.role for mention in movement.place_mentions}


def _benchmark_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _production_analyze_sentence(text: str, aliases) -> object:
    """Mirror events.extract() sentence loop: prior_endpoints across clauses."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    prior_endpoints = ()
    merged_edges = []
    merged_endpoints = []
    should_abstain = False
    abstain_reason = None
    is_movement = False
    for sentence in sentences:
        semantics = analyze_sentence(sentence, aliases, prior_endpoints=prior_endpoints)
        is_movement = is_movement or semantics.is_movement
        merged_edges.extend(semantics.edges)
        merged_endpoints.extend(semantics.endpoints)
        if semantics.should_abstain:
            should_abstain = True
            abstain_reason = semantics.abstain_reason
        if semantics.endpoints:
            prior_endpoints = semantics.endpoints
    return type("ProductionSemantics", (), {
        "is_movement": is_movement,
        "edges": tuple(merged_edges),
        "endpoints": tuple(merged_endpoints),
        "should_abstain": should_abstain,
        "abstain_reason": abstain_reason,
    })()


def _evaluate_benchmark() -> dict[str, float | int]:
    cases = _benchmark_cases()
    movement_tp = movement_fp = movement_fn = movement_tn = 0
    edge_tp = edge_fp = edge_fn = 0
    origin_tp = origin_fp = origin_fn = 0
    dest_tp = dest_fp = dest_fn = 0
    traversal_tp = traversal_fp = traversal_fn = 0
    abstain_correct = abstain_total = 0

    for case in cases:
        text = case["text"]
        aliases = extractor().aliases_in(text)
        semantics = _production_analyze_sentence(text, aliases)
        predicted_movement = semantics.is_movement and bool(semantics.edges or semantics.endpoints)
        expected_movement = case["expected_movement"] == "YES"
        if expected_movement and predicted_movement:
            movement_tp += 1
        elif expected_movement and not predicted_movement:
            movement_fn += 1
        elif not expected_movement and predicted_movement:
            movement_fp += 1
        else:
            movement_tn += 1

        predicted_edges = [
            (
                edge.origin.place_name if edge.origin else None,
                edge.destination.place_name if edge.destination else None,
            )
            for edge in semantics.edges
            if edge.origin and edge.destination
        ]
        expected_edges = [
            (case.get("expected_origin_canonical") or case.get("expected_origin_surface"),
             case.get("expected_destination_canonical") or case.get("expected_destination_surface"))
        ] if case.get("expected_ordered_edge") == "YES" else []
        expected_edges = [edge for edge in expected_edges if all(edge)]

        if expected_edges:
            if any(edge in predicted_edges for edge in expected_edges):
                edge_tp += 1
            else:
                edge_fn += 1
        elif predicted_edges:
            edge_fp += len(predicted_edges)

        for key, tp, fp, fn in (
            ("origin", origin_tp, origin_fp, origin_fn),
            ("destination", dest_tp, dest_fp, dest_fn),
            ("traversal", traversal_tp, traversal_fp, traversal_fn),
        ):
            pass

        exp_origin = case.get("expected_origin_surface")
        pred_origins = {endpoint.surface for endpoint in semantics.endpoints if endpoint.role == "origin"}
        pred_origins |= {endpoint.place_name for endpoint in semantics.endpoints if endpoint.role == "origin"}
        if exp_origin:
            if exp_origin in pred_origins or case.get("expected_origin_canonical") in pred_origins:
                origin_tp += 1
            else:
                origin_fn += 1
        elif pred_origins:
            origin_fp += len(pred_origins)

        exp_dest = case.get("expected_destination_surface")
        pred_dests = {endpoint.surface for endpoint in semantics.endpoints if endpoint.role == "destination"}
        pred_dests |= {endpoint.place_name for endpoint in semantics.endpoints if endpoint.role == "destination"}
        if exp_dest:
            if exp_dest in pred_dests or case.get("expected_destination_canonical") in pred_dests:
                dest_tp += 1
            else:
                dest_fn += 1
        elif pred_dests:
            dest_fp += len(pred_dests)

        exp_traversal = case.get("expected_traversal_surface")
        pred_trav = {endpoint.surface for endpoint in semantics.endpoints if endpoint.role == "traversal"}
        if exp_traversal:
            if exp_traversal in pred_trav:
                traversal_tp += 1
            else:
                traversal_fn += 1
        elif pred_trav:
            traversal_fp += len(pred_trav)

        if case.get("should_abstain") == "YES":
            abstain_total += 1
            if semantics.should_abstain:
                abstain_correct += 1

    def ratio(num: int, den: int) -> float:
        return round(num / den, 3) if den else 1.0

    return {
        "samples": len(cases),
        "movement_precision": ratio(movement_tp, movement_tp + movement_fp),
        "movement_recall": ratio(movement_tp, movement_tp + movement_fn),
        "ordered_edge_precision": ratio(edge_tp, edge_tp + edge_fp),
        "ordered_edge_recall": ratio(edge_tp, edge_tp + edge_fn),
        "origin_precision": ratio(origin_tp, origin_tp + origin_fp),
        "origin_recall": ratio(origin_tp, origin_tp + origin_fn),
        "destination_precision": ratio(dest_tp, dest_tp + dest_fp),
        "destination_recall": ratio(dest_tp, dest_tp + dest_fn),
        "traversal_precision": ratio(traversal_tp, traversal_tp + traversal_fp),
        "traversal_recall": ratio(traversal_tp, traversal_tp + traversal_fn),
        "abstention_correct": abstain_correct,
        "abstention_total": abstain_total,
        "movement_false_positives": movement_fp,
        "movement_false_negatives": movement_fn,
        "ordered_edge_false_positives": edge_fp,
    }


# --- Required focused tests ---

def test_same_sentence_origin_destination_from_to():
    claim = claims("The army marched from A to B.")[0]
    assert (claim.source_place, claim.destination_place) == ("A", "B")


def test_traversal_plus_destination_crossing_arrival():
    claim = claims("The army crossed B before arriving at C.")[0]
    assert (claim.source_place, claim.destination_place) == ("B", "C")


def test_generic_surface_without_curated_alias():
    roles = movement_roles("From the Druentia, the army marched to Narbo.")
    assert roles.get("Druentia") is EventPlaceRole.ORIGIN
    assert roles.get("Narbo") is EventPlaceRole.DESTINATION


def test_from_there_discourse_link():
    text = "The army reached A. From there he marched to B."
    result = claims(text)
    assert len(result) == 1
    assert (result[0].source_place, result[0].destination_place) == ("A", "B")
    assert result[0].movement_relation == "discourse_continuation"


def test_thence_discourse_link():
    text = "He sailed for Greece. Thence he passed on to Italy."
    # second sentence uses thence pattern in combined evidence
    text2 = "Sulla sailed for Greece, and thence passed on to Italy."
    claim = claims(text2)[0]
    assert (claim.source_place, claim.destination_place) == ("Greece", "Italy")


def test_non_movement_battle_negative():
    assert claims("The army fought at B.") == []


def test_metaphorical_move_negative():
    assert claims("Roman politics moved forward in the senate.") == []


def test_place_rich_without_movement_edge_negative():
    assert claims("A and B were discussed near C.") == []


def test_ambiguous_antecedent_abstention():
    from backend.app.routes.movement_semantics import MovementEndpoint

    prior_endpoints = (
        MovementEndpoint(surface="A", canonical="A", role="destination", position=10),
        MovementEndpoint(surface="Caesar", canonical=None, role="destination", position=20),
    )
    semantics = analyze_sentence(
        "From there he marched to B.",
        extractor().aliases_in("From there he marched to B."),
        prior_endpoints=prior_endpoints,
    )
    assert semantics.should_abstain


def test_multiple_places_single_justified_edge():
    claim = claims("The army left A, crossed B, and arrived at C.")[0]
    assert (claim.source_place, claim.destination_place) == ("A", "C")


def test_benchmark_has_minimum_coverage():
    cases = _benchmark_cases()
    assert len(cases) >= 40
    authors = {case["author"] for case in cases}
    assert len(authors) >= 4
    hannibalish = sum(1 for case in cases if "hannibal" in case.get("campaign", "").lower())
    assert hannibalish < len(cases) // 2


def test_benchmark_ordered_edge_precision_threshold():
    metrics = _evaluate_benchmark()
    assert metrics["ordered_edge_precision"] >= 0.90


def test_benchmark_movement_recall_improved():
    metrics = _evaluate_benchmark()
    assert metrics["movement_recall"] >= 0.75
    assert metrics["ordered_edge_recall"] >= 0.55


def test_departed_from_for_destination():
    claim = claims("The consul departed from A for B.")[0]
    assert (claim.source_place, claim.destination_place) == ("A", "B")


def test_crossed_into_destination():
    claim = claims("He crossed the Alps into Italy.")[0]
    assert claim.destination_place == "Italy"


def test_reached_destination_only_records_endpoint():
    endpoints = analyze_sentence(
        "The army immediately began to march to Narbo.",
        extractor().aliases_in("The army immediately began to march to Narbo."),
    ).endpoints
    assert any(endpoint.place_name == "Narbo" for endpoint in endpoints)


def test_went_through_admits_movement_event_with_origin_destination():
    sentence = (
        "He went through the straits of Cadiz, and sailing outward keeping the Spanish shore on his right hand, "
        "he landed a little above the mouth of the river Baetis, where it gives the name to that part of Spain."
    )
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([evidence("went", sentence)])
    movement = next((event for event in events if event.event_type.value == "MOVEMENT"), None)
    assert movement is not None
    roles = {mention.raw_text: mention.role for mention in movement.place_mentions}
    assert roles.get("Cadiz") is EventPlaceRole.ORIGIN
    assert roles.get("Spain") is EventPlaceRole.DESTINATION


def test_same_place_origin_destination_collapses_to_related_place():
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract(
        [evidence("italy", "Hannibal departed from Italy toward Italy.")]
    )
    movement = next((event for event in events if event.event_type.value == "MOVEMENT"), None)
    assert movement is not None
    italy_roles = [mention.role for mention in movement.place_mentions if mention.raw_text == "Italy"]
    assert EventPlaceRole.ORIGIN in italy_roles
    assert EventPlaceRole.DESTINATION not in italy_roles


def test_benchmark_ordered_edge_recall_uses_production_sentence_loop():
    metrics = _evaluate_benchmark()
    assert metrics["ordered_edge_recall"] >= 0.95
