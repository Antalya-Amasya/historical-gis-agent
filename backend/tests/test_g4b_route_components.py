"""G4B evidence-safe multi-component route assembly tests."""
from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState
from backend.app.routes.event_route_orchestration import (
    AnchorOrderingRelation,
    EventAnchorRouteBuilder,
    OrderingRule,
)
from backend.tests.test_event_anchor_routes import (
    build,
    evidence,
    movement,
    names,
    site,
)


def relation(
    earlier: str,
    later: str,
    rule: OrderingRule = OrderingRule.SAME_MOVEMENT_EVENT,
    *,
    refs: tuple[str, ...] = ("a",),
    event_ids: tuple[str, ...] = ("e1",),
) -> AnchorOrderingRelation:
    return AnchorOrderingRelation(earlier, later, rule, event_ids, refs)


def assemble(relations: list[AnchorOrderingRelation]):
    return EventAnchorRouteBuilder._assemble(relations)


def retained_pairs(outcome) -> set[tuple[str, str]]:
    pairs = {(rel.earlier, rel.later) for rel in outcome.relations}
    for component in outcome.route.route_components:
        points = [point.historical_place.canonical_name for point in component.ordered_points]
        pairs.update((points[index], points[index + 1]) for index in range(len(points) - 1))
    pairs.update((branch.earlier, branch.later) for branch in outcome.route.branch_relations)
    return pairs


def structural_pairs(outcome) -> list[tuple[str, str]]:
    return [
        (relation.earlier, relation.later)
        for relation in outcome.relations
        if relation.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER
    ]


def test_disconnected_proven_components_are_preserved():
    events = [
        movement("ab", "Alpha", "Beta", ["a"]),
        movement("cd", "Charlie", "Delta", ["b"]),
    ]
    outcome = build(events, [evidence("a", document="doc-a"), evidence("b", document="doc-b")])
    assert outcome.route is not None
    assert len(outcome.route.route_components) == 2
    component_chains = [
        [point.historical_place.canonical_name for point in component.ordered_points]
        for component in outcome.route.route_components
    ]
    assert ["Alpha", "Beta"] in component_chains
    assert ["Charlie", "Delta"] in component_chains
    assert retained_pairs(outcome) == {("Alpha", "Beta"), ("Charlie", "Delta")}
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]


def test_equal_length_components_are_preserved_without_tie_break():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e1", "e2")),
        relation("Charlie", "Delta", refs=("c",), event_ids=("e3",)),
        relation("Delta", "Genava", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("d",), event_ids=("e3", "e4")),
    ]
    assembly = assemble(relations)
    assert len(assembly.components) == 2
    assert assembly.branch_pairs == ()
    assert {tuple(path) for path, _edges in assembly.components} == {
        ("Alpha", "Beta", "Gamma"),
        ("Charlie", "Delta", "Genava"),
    }
    assert EventAnchorRouteBuilder._chain(relations)[0] == []


def test_offset_only_adjacency_does_not_extend_linear_component():
    events = [
        movement("one", "Alpha", "Beta", ["a"]),
        site("bridge", "Beta", ["b"]),
        site("bridge2", "Gamma", ["b"]),
    ]
    items = [
        evidence("a", document="doc", spine=1, offset=100),
        evidence("b", document="doc", spine=1, offset=200),
    ]
    outcome = build(events, items)
    assert structural_pairs(outcome) == []
    assert names(outcome) == ["Alpha", "Beta"]
    assert "Gamma" not in names(outcome)
    assert len(outcome.route.route_components) == 1
    assert outcome.route.branch_relations == []
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]


def test_hub_incoming_relations_are_retained_as_branches():
    relations = [
        relation("Alpha", "Gamma", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert set(assembly.branch_pairs) == {("Alpha", "Gamma"), ("Beta", "Gamma")}
    assert {tuple(path) for path, _edges in assembly.components} == {
        ("Alpha", "Gamma"),
        ("Beta", "Gamma"),
    }
    chain, _edges = EventAnchorRouteBuilder._chain(relations)
    assert chain == []


def test_branching_outgoing_relations_are_retained():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", refs=("b",), event_ids=("e2",)),
        relation("Beta", "Delta", refs=("c",), event_ids=("e3",)),
    ]
    assembly = assemble(relations)
    assert len(assembly.components) == 3
    assert {tuple(path) for path, _edges in assembly.components} == {
        ("Alpha", "Beta"),
        ("Beta", "Gamma"),
        ("Beta", "Delta"),
    }
    assert set(assembly.branch_pairs) == {("Beta", "Gamma"), ("Beta", "Delta")}


def test_contradictory_relations_fail_closed():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Alpha", refs=("b",), event_ids=("e2",)),
    ]
    assembly = assemble(relations)
    assert assembly.components == ()
    assert assembly.branch_pairs == ()
    assert assembly.usable == {}
    assert assembly.contradictory == (("Alpha", "Beta"), ("Beta", "Alpha"))


def test_cross_document_without_chronology_does_not_synthesize_global_chain():
    events = [
        movement("doc1", "Alpha", "Beta", ["a"]),
        movement("doc2", "Beta", "Gamma", ["b"]),
    ]
    outcome = build(events, [evidence("a", document="polybius"), evidence("b", document="livy")])
    assert outcome.route is not None
    assert names(outcome) == []
    assert len(outcome.route.route_components) == 2
    assert retained_pairs(outcome) == {("Alpha", "Beta"), ("Beta", "Gamma")}
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]


def test_same_document_offsets_do_not_infer_structural_chaining():
    events = [site("later", "Lutetia", ["b"]), site("earlier", "Genava", ["a"])]
    outcome = build(events, [evidence("b", spine=1, offset=200), evidence("a", spine=1, offset=100)])
    assert outcome.route is None
    assert structural_pairs(outcome) == []
    assert outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_offset_only_junction_does_not_merge_provenance():
    outcome = build(
        [
            movement("march", "Alpha", "Beta", ["a"]),
            site("bridge", "Beta", ["b"]),
            site("bridge2", "Gamma", ["b"]),
        ],
        [evidence("a", document="doc", spine=1, offset=100), evidence("b", document="doc", spine=1, offset=200)],
    )
    assert structural_pairs(outcome) == []
    beta_point = next(point for point in outcome.route.ordered_points if point.historical_place.canonical_name == "Beta")
    assert set(beta_point.evidence_refs) == {"a"}


def test_junction_provenance_merges_when_structural_relation_is_explicitly_authorized():
    relations = [
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", OrderingRule.SOURCE_STRUCTURAL_ORDER, refs=("b",), event_ids=("e1", "e2")),
    ]
    assembly = assemble(relations)
    assert {tuple(path) for path, _edges in assembly.components} == {("Alpha", "Beta", "Gamma")}
    assert EventAnchorRouteBuilder._chain(relations)[0] == ["Alpha", "Beta", "Gamma"]


def test_component_assembly_is_deterministic_under_relation_shuffle():
    relations = [
        relation("Charlie", "Delta", refs=("c",), event_ids=("e3",)),
        relation("Alpha", "Beta", refs=("a",), event_ids=("e1",)),
        relation("Beta", "Gamma", refs=("b",), event_ids=("e2",)),
    ]
    first = assemble(relations)
    second = assemble(list(reversed(relations)))
    assert first.components == second.components
    assert first.branch_pairs == second.branch_pairs


def test_offset_only_bridge_does_not_synthesize_middle_edge():
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
    assert structural_pairs(outcome) == []
    assert len(outcome.relations) == 2
    assert names(outcome) == []
    assert len(outcome.route.route_components) == 2
    component_chains = [
        [point.historical_place.canonical_name for point in component.ordered_points]
        for component in outcome.route.route_components
    ]
    assert ["Helvetii", "Bibracte"] in component_chains
    assert ["Bituriges Cubi", "Gergovia"] in component_chains
    assert retained_pairs(outcome) == {("Helvetii", "Bibracte"), ("Bituriges Cubi", "Gergovia")}
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]


class Geography:
    values = {"Melodunum": (48.5, 2.7), "Lutetia": (48.9, 2.35)}

    def call(self, tool, arguments):
        name = arguments["name"]
        if name not in self.values:
            return {"found": False}
        lat, lon = self.values[name]
        return {
            "found": True, "id": name.lower(), "canonical_name": name,
            "latitude": lat, "longitude": lon, "source": "test registry",
            "confidence": 0.8, "coordinate_role": "exact_site",
        }


class Retriever:
    def retrieve(self, query, top_k=5, filters=None):
        return []


def test_event_first_partial_components_preferred_over_legacy_fallback():
    state = AgentState(session_id="partial-components")
    state.historical_evidence = [
        evidence("a", document="doc-a"),
        evidence("b", document="doc-b"),
    ]
    state.historical_events = [
        movement("ab", "Alpha", "Beta", ["a"]),
        movement("cd", "Charlie", "Delta", ["b"]),
    ]
    result, _ = AgentToolRegistry(Retriever(), Geography()).execute(
        "build_historical_route",
        {"event_id": "campaign", "name": "Campaign", "period": "218 BCE"},
        state,
    )
    assert result["result"]["route"] is not None
    assert state.historical_route_diagnostics["route_source"] == "event_anchor"
    assert len(state.historical_route.route_components) == 2
    assert state.historical_route.ordered_points == []
