"""G6U: relation junctions require event-occurrence continuity at shared places."""
from __future__ import annotations

from backend.app.routes.event_route_orchestration import (
    AnchorOrderingRelation,
    EventAnchorRouteBuilder,
    OrderingRule,
    _can_chain_relations,
)


def relation(
    earlier: str,
    later: str,
    rule: OrderingRule = OrderingRule.SAME_MOVEMENT_EVENT,
    *,
    refs: tuple[str, ...] = ("ref",),
    event_ids: tuple[str, ...] = ("e1",),
) -> AnchorOrderingRelation:
    return AnchorOrderingRelation(earlier, later, rule, event_ids, refs)


def assemble(relations: list[AnchorOrderingRelation]):
    return EventAnchorRouteBuilder._assemble(relations)


def component_paths(assembly) -> set[tuple[str, ...]]:
    return {tuple(path) for path, _edges in assembly.components}


def test_a_unrelated_event_ids_with_temporal_authority_do_not_chain():
    """GPT-6 F10: shared place + temporal label must not fabricate junction identity."""
    r1 = relation("Alpha", "Beta", event_ids=("E1",))
    r2 = relation("Beta", "Gamma", OrderingRule.TEMPORAL_ORDER, event_ids=("E2",))
    assert _can_chain_relations(r1, r2) is False
    assembly = assemble([r1, r2])
    assert ("Alpha", "Beta", "Gamma") not in component_paths(assembly)


def test_a_unrelated_event_ids_with_structural_authority_do_not_chain():
    r1 = relation("Alpha", "Beta", event_ids=("E1",))
    r2 = relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, event_ids=("E2",))
    assert _can_chain_relations(r1, r2) is False
    assembly = assemble([r1, r2])
    assert ("Alpha", "Beta", "Gamma") not in component_paths(assembly)


def test_b_shared_event_occurrence_chains():
    r1 = relation("Alpha", "Beta", event_ids=("E2",))
    r2 = relation("Beta", "Gamma", event_ids=("E2",))
    assert _can_chain_relations(r1, r2) is True
    assembly = assemble([r1, r2])
    assert component_paths(assembly) == {("Alpha", "Beta", "Gamma")}


def test_c_temporal_authority_with_shared_event_chains():
    r1 = relation("Alpha", "Beta", event_ids=("E2",))
    r2 = relation(
        "Beta",
        "Gamma",
        OrderingRule.TEMPORAL_ORDER,
        event_ids=("E2", "E3"),
    )
    assert _can_chain_relations(r1, r2) is True
    assembly = assemble([r1, r2])
    assert component_paths(assembly) == {("Alpha", "Beta", "Gamma")}


def test_d_structural_authority_with_shared_event_chains():
    r1 = relation("Alpha", "Beta", event_ids=("E2",))
    r2 = relation(
        "Beta",
        "Gamma",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        event_ids=("E2", "E3"),
    )
    assert _can_chain_relations(r1, r2) is True
    assembly = assemble([r1, r2])
    assert component_paths(assembly) == {("Alpha", "Beta", "Gamma")}


def test_e_same_place_different_visits_do_not_chain():
    r1 = relation("Alpha", "Beta", event_ids=("E1",))
    r2 = relation("Beta", "Gamma", event_ids=("E2",))
    assert _can_chain_relations(r1, r2) is False
    assembly = assemble([r1, r2])
    assert ("Alpha", "Beta", "Gamma") not in component_paths(assembly)


def test_f_different_actors_shared_place_do_not_chain():
    r1 = relation("Alpha", "Beta", event_ids=("E1",))
    r2 = relation("Beta", "Gamma", event_ids=("E2",))
    assert _can_chain_relations(r1, r2) is False
    assembly = assemble([r1, r2])
    assert ("Alpha", "Beta", "Gamma") not in component_paths(assembly)


def test_g_invalid_junction_preserves_fragments():
    r1 = relation("Alpha", "Beta", event_ids=("E1",))
    r2 = relation("Beta", "Gamma", OrderingRule.TEMPORAL_ORDER, event_ids=("E2",))
    assembly = assemble([r1, r2])
    assert ("Alpha", "Beta", "Gamma") not in component_paths(assembly)
    assert ("Alpha", "Beta") in component_paths(assembly)
    assert ("Beta", "Gamma") in component_paths(assembly)
    assert len(assembly.components) == 2
