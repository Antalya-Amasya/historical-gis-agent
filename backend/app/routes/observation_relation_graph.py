"""Shared observation-relation graph normalization for route authority."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
)

OBS_RULE_PRIORITY = {
    ObservationOrderingAuthority.BEFORE_SUBORDINATE: 0,
    ObservationOrderingAuthority.AFTER_SUBORDINATE: 0,
    ObservationOrderingAuthority.BEFORE_POSTPOSED: 0,
    ObservationOrderingAuthority.AFTER_POSTPOSED: 0,
    ObservationOrderingAuthority.FIRST_THEN: 0,
    ObservationOrderingAuthority.TEMPORAL_ORDER: 1,
    ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER: 2,
}

LinkPredicate = Callable[[ObservationOrderingRelation, ObservationOrderingRelation], bool]


@dataclass(frozen=True)
class NormalizedObservationGraph:
    usable: dict[tuple[str, str], ObservationOrderingRelation]
    contradictory_edges: tuple[tuple[str, str], ...]
    branch_edges: tuple[tuple[str, str], ...]
    cyclic_nodes: frozenset[str]

    def linear_edges(self) -> dict[tuple[str, str], ObservationOrderingRelation]:
        return {
            edge: relation
            for edge, relation in self.usable.items()
            if edge not in self.branch_edges
            and edge[0] not in self.cyclic_nodes
            and edge[1] not in self.cyclic_nodes
        }

    def weak_components(self) -> list[set[str]]:
        return weakly_connected_components(self.usable)


def normalize_duplicate_edges(
    relations: list[ObservationOrderingRelation],
    *,
    merge_relations: Callable[[ObservationOrderingRelation, ObservationOrderingRelation], ObservationOrderingRelation],
) -> tuple[dict[tuple[str, str], ObservationOrderingRelation], tuple[dict[str, object], ...]]:
    best: dict[tuple[str, str], ObservationOrderingRelation] = {}
    rejected: list[dict[str, object]] = []
    for relation in relations:
        key = (relation.earlier_observation_id, relation.later_observation_id)
        if key not in best:
            best[key] = relation
            continue
        current = best[key]
        left_priority = OBS_RULE_PRIORITY[relation.ordering_rule]
        right_priority = OBS_RULE_PRIORITY[current.ordering_rule]
        if left_priority < right_priority:
            best[key] = relation
        elif left_priority == right_priority and relation.authority == current.authority:
            best[key] = merge_relations(current, relation)
        else:
            rejected.append({
                "edge": key,
                "reason": "CONFLICTING_ORDERING_AUTHORITY",
                "authorities": [current.authority, relation.authority],
            })
    return best, tuple(rejected)


def resolve_reverse_conflicts(
    best: dict[tuple[str, str], ObservationOrderingRelation],
) -> tuple[dict[tuple[str, str], ObservationOrderingRelation], tuple[tuple[str, str], ...], tuple[dict[str, object], ...]]:
    usable = dict(best)
    contradictory: list[tuple[str, str]] = []
    rejected: list[dict[str, object]] = []
    resolved: set[tuple[str, str]] = set()
    for pair in sorted(best):
        if pair in resolved:
            continue
        reverse = (pair[1], pair[0])
        if reverse not in best:
            continue
        forward = best[pair]
        backward = best[reverse]
        resolved.update({pair, reverse})
        forward_priority = OBS_RULE_PRIORITY[forward.ordering_rule]
        backward_priority = OBS_RULE_PRIORITY[backward.ordering_rule]
        if forward_priority < backward_priority:
            usable.pop(reverse, None)
            rejected.append({
                "edge": reverse,
                "reason": "CONFLICT_WITH_STRONGER_RELATION",
                "winning_authority": forward.authority,
            })
        elif backward_priority < forward_priority:
            usable.pop(pair, None)
            rejected.append({
                "edge": pair,
                "reason": "CONFLICT_WITH_STRONGER_RELATION",
                "winning_authority": backward.authority,
            })
        else:
            usable.pop(pair, None)
            usable.pop(reverse, None)
            contradictory.extend([pair, reverse])
    return usable, tuple(sorted(set(contradictory))), tuple(rejected)


def normalize_observation_graph(
    relations: list[ObservationOrderingRelation],
    *,
    merge_relations: Callable[[ObservationOrderingRelation, ObservationOrderingRelation], ObservationOrderingRelation],
) -> tuple[NormalizedObservationGraph, tuple[dict[str, object], ...]]:
    best, duplicate_rejected = normalize_duplicate_edges(relations, merge_relations=merge_relations)
    usable, contradictory, conflict_rejected = resolve_reverse_conflicts(best)
    branch_edges, cyclic_nodes = classify_branch_and_cycle_nodes(usable)
    graph = NormalizedObservationGraph(
        usable=usable,
        contradictory_edges=contradictory,
        branch_edges=branch_edges,
        cyclic_nodes=cyclic_nodes,
    )
    return graph, duplicate_rejected + conflict_rejected


def weakly_connected_components(
    usable: dict[tuple[str, str], ObservationOrderingRelation],
) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for earlier, later in usable:
        union(earlier, later)
    groups: dict[str, set[str]] = defaultdict(set)
    for earlier, later in usable:
        groups[find(earlier)].update({earlier, later})
    return sorted(groups.values(), key=lambda nodes: sorted(nodes))


def edge_degrees(
    usable: dict[tuple[str, str], ObservationOrderingRelation],
) -> tuple[dict[str, int], dict[str, int]]:
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for earlier, later in usable:
        outgoing[earlier] += 1
        incoming[later] += 1
    return incoming, outgoing


def is_branch_edge(
    pair: tuple[str, str],
    incoming: dict[str, int],
    outgoing: dict[str, int],
) -> bool:
    earlier, later = pair
    return outgoing.get(earlier, 0) > 1 or incoming.get(later, 0) > 1


def classify_branch_and_cycle_nodes(
    usable: dict[tuple[str, str], ObservationOrderingRelation],
) -> tuple[tuple[tuple[str, str], ...], frozenset[str]]:
    branch_edges: list[tuple[str, str]] = []
    cyclic_nodes: set[str] = set()
    for node_group in weakly_connected_components(usable):
        subgraph = {
            pair: relation
            for pair, relation in usable.items()
            if pair[0] in node_group and pair[1] in node_group
        }
        incoming, outgoing = edge_degrees(subgraph)
        branch_edges.extend(
            sorted(pair for pair in subgraph if is_branch_edge(pair, incoming, outgoing))
        )
        cyclic_nodes.update(_directed_cycle_nodes(subgraph))
    return tuple(sorted(set(branch_edges))), frozenset(cyclic_nodes)


def _directed_cycle_nodes(
    edges: dict[tuple[str, str], ObservationOrderingRelation],
) -> set[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for earlier, later in edges:
        adjacency[earlier].append(later)
        nodes.update({earlier, later})
    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    cyclic: set[str] = set()

    def strongconnect(node: str) -> None:
        nonlocal index_counter
        indices[node] = index_counter
        lowlink[node] = index_counter
        index_counter += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in adjacency.get(node, []):
            if nxt not in indices:
                strongconnect(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif nxt in on_stack:
                lowlink[node] = min(lowlink[node], indices[nxt])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or (len(component) == 1 and node in adjacency.get(node, [])):
                cyclic.update(component)

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return cyclic


def extract_linear_chains(
    linear_edges: dict[tuple[str, str], ObservationOrderingRelation],
    node_group: set[str],
    *,
    can_link: LinkPredicate | None = None,
) -> list[list[tuple[str, str]]]:
    subgraph = {
        edge: relation
        for edge, relation in linear_edges.items()
        if edge[0] in node_group and edge[1] in node_group
    }
    if not subgraph:
        return []
    link = can_link or _default_can_link
    used_edges: set[tuple[str, str]] = set()
    chains: list[list[tuple[str, str]]] = []
    for start_pair in sorted(subgraph):
        if start_pair in used_edges:
            continue
        chain_edges = [start_pair]
        used_edges.add(start_pair)
        while True:
            right_candidates = sorted(
                pair for pair in subgraph
                if pair not in used_edges
                and link(subgraph[chain_edges[-1]], subgraph[pair])
            )
            if len(right_candidates) != 1:
                break
            chain_edges.append(right_candidates[0])
            used_edges.add(right_candidates[0])
        while True:
            left_candidates = sorted(
                pair for pair in subgraph
                if pair not in used_edges
                and link(subgraph[pair], subgraph[chain_edges[0]])
            )
            if len(left_candidates) != 1:
                break
            chain_edges.insert(0, left_candidates[0])
            used_edges.add(left_candidates[0])
        chains.append(chain_edges)
    return chains


def _default_can_link(
    left: ObservationOrderingRelation,
    right: ObservationOrderingRelation,
) -> bool:
    return left.later_observation_id == right.earlier_observation_id
