"""G4N: preserve strong SAME_MOVEMENT hub branches as route components."""
from __future__ import annotations

import pytest

from backend.app.agent.tools import AgentToolRegistry
from backend.app.candidate_routes.component_fragment_presentation import ComponentFragmentPresentationAdapter
from backend.app.models import AgentState, HistoricalRouteIntent
from backend.app.route_orchestrator import BarrierCrossingConstraintError, HistoricalRouteOrchestrator
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.tests.test_event_anchor_routes import build, evidence, movement, names, site
from backend.tests.test_g4b_route_components import assemble, relation
from backend.tests.test_g4f_component_fragment_presentation import adapter, component, mountain, river, route, settlement
from backend.tests.test_g4g2_barrier_component_fallback import evidence as g4g2_evidence
from backend.tests.test_g4g2_barrier_component_fallback import registry as g4g2_registry
from backend.tests.test_g4g2_barrier_component_fallback import route as g4g2_route


def component_chains(outcome) -> set[tuple[str, ...]]:
    return {
        tuple(point.historical_place.canonical_name for point in component.ordered_points)
        for component in outcome.route.route_components
    }


def test_single_same_movement_edge_unchanged():
    outcome = build([movement("march", "Alpha", "Beta", ["a"])], [evidence("a")])
    assert names(outcome) == ["Alpha", "Beta"]
    assert component_chains(outcome) == {("Alpha", "Beta")}
    assert outcome.route.branch_relations == []


def test_hub_same_movement_creates_two_components_without_fabricated_order():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Alpha", "Charlie", refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert EventAnchorRouteBuilder._chain(relations)[0] == []
    assert {tuple(path) for path, _ in assembly.components} == {("Alpha", "Beta"), ("Alpha", "Charlie")}
    assert set(assembly.branch_pairs) == {("Alpha", "Beta"), ("Alpha", "Charlie")}
    assert all(len(path) == 2 for path, _ in assembly.components)


def test_three_strong_branches_all_preserved_as_components():
    relations = [
        relation("Rhodanus", "Italia", refs=("a",), event_ids=("cross",)),
        relation("Rhodanus", "Alpes", refs=("b",), event_ids=("march",)),
        relation("Druentia (river)", "Alpes", refs=("c",), event_ids=("druentia",)),
    ]
    assembly = assemble(relations)
    assert {tuple(path) for path, _ in assembly.components} == {
        ("Rhodanus", "Italia"),
        ("Rhodanus", "Alpes"),
        ("Druentia (river)", "Alpes"),
    }
    assert set(assembly.branch_pairs) == {
        ("Rhodanus", "Italia"),
        ("Rhodanus", "Alpes"),
        ("Druentia (river)", "Alpes"),
    }


def test_structural_branch_only_not_auto_promoted():
    relations = [
        relation("Alpha", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("a",), event_ids=("e1", "e3")),
        relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e2", "e3")),
    ]
    assembly = assemble(relations)
    assert assembly.components == ()
    assert set(assembly.branch_pairs) == {("Alpha", "Gamma"), ("Beta", "Gamma")}


def test_existing_linear_component_does_not_duplicate_same_edge():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert len(assembly.components) == 1
    assert assembly.components[0][0] == ("Alpha", "Beta", "Gamma")
    assert sum(1 for path, _ in assembly.components if path == ("Alpha", "Beta")) == 0


def test_suppressed_reverse_same_movement_not_componentized():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("e1",)),
        relation("Beta", "Alpha", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert {tuple(path) for path, _ in assembly.components} == {("Alpha", "Beta")}
    assert ("Beta", "Alpha") not in assembly.branch_pairs


def test_equal_authority_reverse_same_movement_not_componentized():
    relations = [
        relation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, refs=("a",), event_ids=("e1",)),
        relation("Beta", "Alpha", OrderingRule.SAME_MOVEMENT_EVENT, refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert assembly.components == ()
    assert assembly.usable == {}


def test_missing_evidence_refs_fail_closed():
    from backend.app.routes.event_route_orchestration import AnchorOrderingRelation

    relations = [
        AnchorOrderingRelation("Alpha", "Beta", OrderingRule.SAME_MOVEMENT_EVENT, ("e1",), ()),
        relation("Alpha", "Charlie", refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert ("Alpha", "Beta") not in {tuple(path) for path, _ in assembly.components}
    assert ("Alpha", "Charlie") in {tuple(path) for path, _ in assembly.components}


def test_component_provenance_preserved():
    outcome = build(
        [
            movement("ab", "Rhodanus", "Italia", ["a"]),
            movement("ac", "Rhodanus", "Alpes", ["b"]),
        ],
        [evidence("a"), evidence("b")],
    )
    components = {
        tuple(point.historical_place.canonical_name for point in component.ordered_points): component
        for component in outcome.route.route_components
    }
    assert "a" in components[("Rhodanus", "Italia")].evidence_refs
    assert "b" in components[("Rhodanus", "Alpes")].evidence_refs
    claim_refs = {
        (claim.source_place, claim.destination_place): claim.supporting_evidence_ids
        for claim in outcome.route.claims
    }
    assert claim_refs[("Rhodanus", "Italia")] == ["a"]
    assert claim_refs[("Rhodanus", "Alpes")] == ["b"]


def test_branch_relations_preserved_alongside_components():
    outcome = build(
        [
            movement("e1", "Alpha", "Beta", ["a"]),
            movement("e2", "Alpha", "Charlie", ["b"]),
        ],
        [evidence("a"), evidence("b")],
    )
    assert component_chains(outcome) == {("Alpha", "Beta"), ("Alpha", "Charlie")}
    assert {(branch.earlier, branch.later) for branch in outcome.route.branch_relations} == {
        ("Alpha", "Beta"),
        ("Alpha", "Charlie"),
    }


def test_g4f_adapter_accepts_empty_ordered_points_with_strong_components():
    rhone = river("Rhodanus", 4.85, 43.33)
    italia = settlement("Italia", 12.5, 42.5, sequence=2)
    alpes = mountain("Alpes", 6.8, 45.8, sequence=2)
    druentia = river("Druentia (river)", 6.1, 44.7)
    historical_route = route(
        ordered_points=[],
        components=[
            component("rh-it", [rhone, italia]),
            component("rh-al", [rhone, alpes]),
            component("dr-al", [druentia, alpes]),
        ],
    )
    result = adapter().present(
        HistoricalRouteIntent(campaign_id="g4n", entity="generic", route_type="movement"),
        historical_route,
        [],
    )
    assert result.presentation is not None
    assert result.fragments


def test_g4g2_barrier_fallback_sees_strong_components():
    global_chain = [river("Druentia (river)", 5.0, 44.2), mountain("Alpes", 7.0, 44.0)]
    historical_route = g4g2_route(
        ordered_points=global_chain,
        components=[
            component("global-dup", global_chain),
            component("rh-it", [river("Rhodanus", 4.85, 43.33), settlement("Italia", 12.5, 42.5, sequence=2)]),
        ],
    )
    state = AgentState(session_id="g4n-g4g2", requested_output="historical_route")
    state.historical_evidence = g4g2_evidence()
    reconstruction = g4g2_registry()._reconstruct_candidate_route(historical_route, state)
    assert reconstruction["presentation"] is not None
    assert reconstruction["diagnostics"]["component_fallback"]["fragments_presented"] >= 1
    with pytest.raises(BarrierCrossingConstraintError):
        HistoricalRouteOrchestrator(cell_size_m=25_000).present(
            HistoricalRouteIntent(campaign_id="generic", entity="generic", route_type="movement"),
            historical_route,
            g4g2_evidence(),
        )


def test_global_linear_regression_unchanged():
    events = [
        movement("helvetii", "Helvetii", "Bibracte", ["a"]),
        site("bridge", "Bibracte", ["b"]),
        site("bridge2", "Bituriges Cubi", ["b"]),
        movement("bituriges", "Bituriges Cubi", "Gergovia", ["c"]),
    ]
    items = [
        evidence("a", document="caesar", spine=1, offset=100),
        evidence("b", document="caesar", spine=1, offset=200),
        evidence("c", document="caesar", spine=1, offset=300),
    ]
    outcome = build(events, items)
    assert names(outcome) == ["Helvetii", "Bibracte", "Bituriges Cubi", "Gergovia"]
    assert len(outcome.route.route_components) == 1


def test_hannibal_live_shape_hub_components():
    outcome = build(
        [
            movement("cross", "Rhodanus", "Italia", ["a"]),
            movement("march", "Rhodanus", "Alpes", ["b"]),
            movement("druentia", "Druentia (river)", "Alpes", ["c"]),
            movement("iberus", "Iberus", "Carthago Nova", ["d"]),
        ],
        [evidence("a"), evidence("b"), evidence("c"), evidence("d")],
    )
    chains = component_chains(outcome)
    assert ("Rhodanus", "Italia") in chains
    assert ("Rhodanus", "Alpes") in chains
    assert ("Druentia (river)", "Alpes") in chains
    represented = {
        point.historical_place.canonical_name
        for component in outcome.route.route_components
        for point in component.ordered_points
    }
    assert {"Rhodanus", "Italia", "Alpes", "Druentia (river)"} <= represented
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]
