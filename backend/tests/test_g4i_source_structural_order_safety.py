"""G4I: SOURCE_STRUCTURAL_ORDER safety — stronger authority wins, recall preserved."""
from __future__ import annotations

from backend.app.routes.event_route_orchestration import (
    AnchorOrderingRelation,
    EventAnchorRouteBuilder,
    OrderingRule,
)
from backend.tests.test_g4b_route_components import assemble, relation


def usable_pairs(relations: list[AnchorOrderingRelation]) -> set[tuple[str, str]]:
    return set(EventAnchorRouteBuilder._assemble(relations).usable)


def test_strong_movement_wins_over_structural_reverse():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT),
        relation("Beta", "Alpha", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert ("Alpha", "Beta") in assembly.usable
    assert ("Beta", "Alpha") not in assembly.usable
    assert assembly.contradictory == ()
    assert len(assembly.suppressed) == 1
    assert assembly.suppressed[0]["reason"] == "CONFLICT_WITH_STRONGER_RELATION"
    assert assembly.suppressed[0]["winning_authority"] == OrderingRule.SAME_MOVEMENT_EVENT.value
    assert assembly.suppressed[0]["suppressed_authority"] == OrderingRule.SOURCE_STRUCTURAL_ORDER.value


def test_temporal_wins_over_structural_reverse():
    relations = [
        relation("Alpha", "Beta", OrderingRule.TEMPORAL_ORDER, refs=("a",), event_ids=("e1", "e2")),
        relation("Beta", "Alpha", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e2", "e1")),
    ]
    assembly = assemble(relations)
    assert ("Alpha", "Beta") in assembly.usable
    assert ("Beta", "Alpha") not in assembly.usable
    assert assembly.suppressed[0]["winning_authority"] == OrderingRule.TEMPORAL_ORDER.value


def test_non_conflicting_structural_edge_preserved():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("a",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert assembly.usable == {("Alpha", "Beta"): relations[0]}
    assert assembly.suppressed == ()
    assert assembly.contradictory == ()


def test_same_direction_dedupes_to_stronger_authority():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("a",), event_ids=("e1", "e2")),
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("b",), event_ids=("e1",)),
    ]
    assembly = assemble(relations)
    assert assembly.usable[("Alpha", "Beta")].rule is OrderingRule.SAME_MOVEMENT_EVENT
    assert assembly.suppressed == ()


def test_branch_structural_edges_preserved_without_contradiction():
    relations = [
        relation("Alpha", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("a",), event_ids=("e1", "e3")),
        relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e2", "e3")),
    ]
    assembly = assemble(relations)
    assert set(assembly.branch_pairs) == {("Alpha", "Gamma"), ("Beta", "Gamma")}
    assert len(assembly.usable) == 2
    assert assembly.suppressed == ()


def test_unknown_subject_structural_edge_not_rejected():
    """Subject continuity is not required; same-document structural order suffices."""
    relations = [
        relation("Alpha", "Beta", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("a",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert ("Alpha", "Beta") in assembly.usable


def test_hannibal_rhodanus_italia_conflict_resolution():
    relations = [
        relation("Rhodanus", "Italia", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("cross",)),
        relation("Italia", "Rhodanus", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("cross", "march")),
    ]
    assembly = assemble(relations)
    assert ("Rhodanus", "Italia") in assembly.usable
    assert ("Italia", "Rhodanus") not in assembly.usable
    assert any(
        item["suppressed_relation"]["earlier"] == "Italia"
        and item["suppressed_relation"]["later"] == "Rhodanus"
        for item in assembly.suppressed
    )


def test_equal_authority_reverse_still_fails_closed():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("e1",)),
        relation("Beta", "Alpha", OrderingRule.SAME_MOVEMENT_EVENT, refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert assembly.usable == {}
    assert assembly.suppressed == ()
    assert set(assembly.contradictory) == {("Alpha", "Beta"), ("Beta", "Alpha")}
