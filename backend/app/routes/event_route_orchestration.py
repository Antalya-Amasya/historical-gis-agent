"""EventAnchor ordering validation and ordered HistoricalRoute assembly.

Historical order must be proven by an approved ordering authority.  Retrieval
rank, list order, narrative order, coordinates, roads, and terrain never
establish it.  The resulting ordered waypoints constrain later GIS
reconstruction; their schematic connections are not documentary path proof.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from backend.app.models import (
    Evidence,
    EventPlaceRole,
    GeoJsonLineString,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalRoute,
    HistoricalRouteBranchRelation,
    HistoricalRouteComponent,
    HistoricalRoutePoint,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_anchors import EventAnchor, project_event_anchors
from backend.app.routes.evidence_relevance import EvidenceRelevance, event_relevance
from backend.app.routes.extractor import evidence_structural_key


class OrderingRule(str, Enum):
    SAME_MOVEMENT_EVENT = "SAME_MOVEMENT_EVENT"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    SOURCE_STRUCTURAL_ORDER = "SOURCE_STRUCTURAL_ORDER"


_RULE_PRIORITY = {
    OrderingRule.SAME_MOVEMENT_EVENT: 0,
    OrderingRule.TEMPORAL_ORDER: 1,
    OrderingRule.SOURCE_STRUCTURAL_ORDER: 2,
}
_RULE_CONFIDENCE = {
    OrderingRule.SAME_MOVEMENT_EVENT: 0.9,
    OrderingRule.TEMPORAL_ORDER: 0.8,
    OrderingRule.SOURCE_STRUCTURAL_ORDER: 0.7,
}
_RESOLUTION_FAILURES = {"UNRESOLVED_PLACE", "AMBIGUOUS_PLACE", "MISSING_COORDINATE"}
_COMPARABLE_PRECISION = {TemporalPrecision.DAY, TemporalPrecision.MONTH, TemporalPrecision.YEAR, TemporalPrecision.YEAR_RANGE}
_ROUTE_ADMISSIBLE = frozenset({
    EvidenceRelevance.DIRECT_SUBJECT,
    EvidenceRelevance.DIRECT_CAMPAIGN,
    EvidenceRelevance.DIRECT_EVENT,
    EvidenceRelevance.SAME_CONFLICT_RELEVANT,
})
_STRUCTURAL_ADMISSIBLE = _ROUTE_ADMISSIBLE


def _relation_admission_allowed(
    relation: AnchorOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...] | None,
) -> bool:
    if not query_contexts:
        return True
    relevances = [
        event_relevance(events_by_id[event_id], evidence_by_id, query_contexts)
        for event_id in relation.event_ids
        if event_id in events_by_id
    ]
    if any(tag is EvidenceRelevance.OTHER_CAMPAIGN for tag in relevances):
        return False
    if relation.rule is OrderingRule.SAME_MOVEMENT_EVENT:
        if not relevances:
            return True
        return all(tag in _ROUTE_ADMISSIBLE or tag is EvidenceRelevance.UNKNOWN for tag in relevances) and any(
            tag in _ROUTE_ADMISSIBLE for tag in relevances
        )
    if relation.rule is OrderingRule.TEMPORAL_ORDER:
        return all(tag in _ROUTE_ADMISSIBLE or tag is EvidenceRelevance.UNKNOWN for tag in relevances)
    if relation.rule is OrderingRule.SOURCE_STRUCTURAL_ORDER:
        if len(relevances) < 2:
            return relevances[0] in _STRUCTURAL_ADMISSIBLE if relevances else False
        return all(tag in _STRUCTURAL_ADMISSIBLE for tag in relevances)
    return True


def _filter_relations_for_query(
    relations: list[AnchorOrderingRelation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...] | None,
) -> tuple[list[AnchorOrderingRelation], list[dict[str, object]]]:
    if not query_contexts:
        return relations, []
    kept: list[AnchorOrderingRelation] = []
    rejected: list[dict[str, object]] = []
    for relation in relations:
        if _relation_admission_allowed(relation, events_by_id, evidence_by_id, query_contexts):
            kept.append(relation)
            continue
        rejected.append({
            "reason": "CAMPAIGN_RELEVANCE_REJECTED",
            "rule": relation.rule.value,
            "earlier": relation.earlier,
            "later": relation.later,
            "event_ids": list(relation.event_ids),
            "evidence_refs": list(relation.evidence_refs),
        })
    return kept, rejected


@dataclass(frozen=True)
class AnchorOrderingRelation:
    """One proven ordering step between two canonical historical places."""

    earlier: str
    later: str
    rule: OrderingRule
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def as_provenance(self) -> dict[str, object]:
        direct_movement = self.rule is OrderingRule.SAME_MOVEMENT_EVENT
        return {
            "earlier": self.earlier,
            "later": self.later,
            "rule": self.rule.value,
            "event_ids": list(self.event_ids),
            "evidence_refs": list(self.evidence_refs),
            "historical_authority": "ATTESTED_MOVEMENT_ORDERING" if direct_movement else "EVIDENCE_GROUNDED_WAYPOINT_ORDERING",
            "connection_semantics": "ALGORITHMIC_GIS_RECONSTRUCTION_REQUIRED",
        }


@dataclass(frozen=True)
class RouteAssembly:
  """Deterministic decomposition of conflict-safe relations into linear components and branches."""

  components: tuple[tuple[str, ...], tuple[tuple[str, str], ...], ...]
  branch_pairs: tuple[tuple[str, str], ...]
  usable: dict[tuple[str, str], AnchorOrderingRelation]
  contradictory: tuple[tuple[str, str], ...]
  suppressed: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class EventRouteOutcome:
    route: HistoricalRoute | None
    relations: tuple[AnchorOrderingRelation, ...]
    diagnostics: dict[str, object]


def _sole(anchors: list[EventAnchor], role: EventPlaceRole) -> EventAnchor | None:
    matches = [anchor for anchor in anchors if anchor.role is role]
    return matches[0] if len(matches) == 1 else None


def _initial(anchors: list[EventAnchor]) -> EventAnchor | None:
    return _sole(anchors, EventPlaceRole.ORIGIN) or _sole(anchors, EventPlaceRole.EVENT_SITE)


def _terminal(anchors: list[EventAnchor]) -> EventAnchor | None:
    return _sole(anchors, EventPlaceRole.DESTINATION) or _sole(anchors, EventPlaceRole.EVENT_SITE)


def _temporal_interval(event: HistoricalEvent | None) -> tuple[int, int] | None:
    """Evidence-grounded, comparable, sufficiently precise historical years only."""
    if event is None:
        return None
    grounding = event.temporal_grounding
    if grounding.status is not TemporalGroundingStatus.EVIDENCE_GROUNDED or grounding.precision not in _COMPARABLE_PRECISION:
        return None
    try:
        start, end = int(grounding.normalized_start), int(grounding.normalized_end)
    except (TypeError, ValueError):
        return None
    return (start, end) if start <= end else (end, start)


def _structural_span(anchors: list[EventAnchor], evidence_by_id: dict[str, Evidence]) -> tuple[tuple[str, int, int], tuple[str, int, int]] | None:
    keys: list[tuple[str, int, int]] = []
    for anchor in anchors:
        for ref in anchor.evidence_refs:
            item = evidence_by_id.get(ref)
            key = evidence_structural_key(item) if item is not None else None
            if key is None:
                return None
            keys.append(key)
    if not keys or len({key[0] for key in keys}) != 1:
        return None
    return min(keys), max(keys)


def _relation_provenance_record(relation: AnchorOrderingRelation) -> dict[str, object]:
    return {
        "earlier": relation.earlier,
        "later": relation.later,
        "rule": relation.rule.value,
        "event_ids": list(relation.event_ids),
        "evidence_refs": list(relation.evidence_refs),
    }


def _resolve_authority_conflicts(
    best: dict[tuple[str, str], AnchorOrderingRelation],
) -> tuple[dict[tuple[str, str], AnchorOrderingRelation], tuple[tuple[str, str], ...], tuple[dict[str, object], ...]]:
    """Prefer stronger ordering authority when reverse edges disagree."""
    usable = dict(best)
    unresolved_contradictory: list[tuple[str, str]] = []
    suppressed: list[dict[str, object]] = []
    resolved: set[tuple[str, str]] = set()

    for pair in sorted(best):
        if pair in resolved:
            continue
        reverse = (pair[1], pair[0])
        if reverse not in best:
            continue
        forward = best[pair]
        backward = best[reverse]
        resolved.add(pair)
        resolved.add(reverse)

        forward_priority = _RULE_PRIORITY[forward.rule]
        backward_priority = _RULE_PRIORITY[backward.rule]
        if forward_priority < backward_priority:
            usable.pop(reverse, None)
            suppressed.append({
                "suppressed_relation": _relation_provenance_record(backward),
                "reason": "CONFLICT_WITH_STRONGER_RELATION",
                "winning_relation": _relation_provenance_record(forward),
                "winning_authority": forward.rule.value,
                "suppressed_authority": backward.rule.value,
            })
        elif backward_priority < forward_priority:
            usable.pop(pair, None)
            suppressed.append({
                "suppressed_relation": _relation_provenance_record(forward),
                "reason": "CONFLICT_WITH_STRONGER_RELATION",
                "winning_relation": _relation_provenance_record(backward),
                "winning_authority": backward.rule.value,
                "suppressed_authority": forward.rule.value,
            })
        else:
            usable.pop(pair, None)
            usable.pop(reverse, None)
            unresolved_contradictory.extend([pair, reverse])

    return usable, tuple(sorted(set(unresolved_contradictory))), tuple(suppressed)


def _weakly_connected_components(usable: dict[tuple[str, str], AnchorOrderingRelation]) -> list[set[str]]:
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
        root = find(earlier)
        groups[root].update({earlier, later})
    return sorted(groups.values(), key=lambda nodes: sorted(nodes))


def _can_chain_relations(left: AnchorOrderingRelation, right: AnchorOrderingRelation) -> bool:
    """Allow adjacent chaining only when ordering authority supports the junction."""
    if left.later != right.earlier:
        return False
    if set(left.event_ids) & set(right.event_ids):
        return True
  # Cross-event chaining requires an explicit temporal or structural bridge relation.
    if right.rule in {OrderingRule.TEMPORAL_ORDER, OrderingRule.SOURCE_STRUCTURAL_ORDER}:
        return True
    if left.rule in {OrderingRule.TEMPORAL_ORDER, OrderingRule.SOURCE_STRUCTURAL_ORDER}:
        return True
    return False


def _path_from_edges(
    edges: list[tuple[str, str]],
    usable: dict[tuple[str, str], AnchorOrderingRelation],
) -> list[str]:
    if not edges:
        return []
    path = [edges[0][0], edges[0][1]]
    for earlier, later in edges[1:]:
        if path[-1] == earlier:
            path.append(later)
        elif path[0] == later:
            path.insert(0, earlier)
        else:
            path.extend([earlier, later])
    return path


def _edge_degrees(usable: dict[tuple[str, str], AnchorOrderingRelation]) -> tuple[dict[str, int], dict[str, int]]:
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)
    for earlier, later in usable:
        outgoing[earlier] += 1
        incoming[later] += 1
    return incoming, outgoing


def _is_branch_edge(
    pair: tuple[str, str],
    incoming: dict[str, int],
    outgoing: dict[str, int],
) -> bool:
    earlier, later = pair
    return outgoing.get(earlier, 0) > 1 or incoming.get(later, 0) > 1


def _branch_kind(
    pair: tuple[str, str],
    usable: dict[tuple[str, str], AnchorOrderingRelation],
) -> str:
    earlier, later = pair
    incoming = sum(1 for left, right in usable if right == later)
    outgoing = sum(1 for left, right in usable if left == earlier)
    if incoming > 1:
        return "incoming_hub"
    if outgoing > 1:
        return "outgoing_branch"
    return "isolated"


def _component_covered_edges(
    components: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]],
) -> set[tuple[str, str]]:
    covered: set[tuple[str, str]] = set()
    for path, _edges in components:
        for index in range(len(path) - 1):
            covered.add((path[index], path[index + 1]))
    return covered


def _eligible_same_movement_component(
    relation: AnchorOrderingRelation,
    pair: tuple[str, str],
) -> bool:
    return (
        relation.rule is OrderingRule.SAME_MOVEMENT_EVENT
        and relation.earlier != relation.later
        and bool(relation.evidence_refs)
    )


def _materialize_same_movement_branch_components(
    components: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]],
    branch_pairs: list[tuple[str, str]],
    subgraph: dict[tuple[str, str], AnchorOrderingRelation],
) -> None:
    """Promote proven SAME_MOVEMENT hub branches into drawable 2-point components."""
    covered = _component_covered_edges(components)
    for pair in sorted(branch_pairs):
        if pair in covered:
            continue
        relation = subgraph.get(pair)
        if relation is None or not _eligible_same_movement_component(relation, pair):
            continue
        components.append(((pair[0], pair[1]), (pair,)))
        covered.add(pair)


class EventAnchorRouteBuilder:
    """Build a HistoricalRoute only from anchors whose order is independently proven."""

    def build_with_diagnostics(
        self,
        events: list[HistoricalEvent],
        evidence: list[Evidence],
        *,
        event_id: str,
        name: str,
        period: str,
        allow_contextual_related_places: bool = False,
        query_contexts: tuple[str, ...] | None = None,
    ) -> EventRouteOutcome:
        anchors, projection = project_event_anchors(
            events, evidence, allow_contextual_related_places=allow_contextual_related_places,
        )
        diagnostics: dict[str, object] = {
            "route_source": "event_anchor",
            "anchor_count": len(anchors),
            "distinct_place_count": 0,
            "ordering_relation_count": 0,
            "ordered_place_count": 0,
            "route_point_count": 0,
            "component_count": 0,
            "branch_relation_count": 0,
            "retained_relation_count": 0,
            "contradictory_relation_count": 0,
            "strong_anchor_count": sum(anchor.admission_type != "CONTEXTUAL_WAYPOINT" for anchor in anchors),
            "contextual_anchor_count": sum(anchor.admission_type == "CONTEXTUAL_WAYPOINT" for anchor in anchors),
            "contextual_anchor_keys": [f"{anchor.event_id}|{anchor.canonical_name}" for anchor in anchors if anchor.admission_type == "CONTEXTUAL_WAYPOINT"],
            "projection_diagnostics": list(projection),
            "ordering_provenance": [],
            "reason_codes": [],
        }
        if not anchors:
            unresolved = any(code.split(":")[0] in _RESOLUTION_FAILURES for code in projection)
            diagnostics["reason_codes"] = ["PLACE_RESOLUTION_FAILED"] if unresolved else ["NO_MOVEMENT_EVENTS"]
            return EventRouteOutcome(None, (), diagnostics)
        places: dict[str, list[EventAnchor]] = {}
        for anchor in anchors:
            places.setdefault(anchor.canonical_name, []).append(anchor)
        if any(len({(anchor.latitude, anchor.longitude) for anchor in group}) > 1 for group in places.values()):
            diagnostics["reason_codes"] = ["PLACE_RESOLUTION_FAILED"]
            return EventRouteOutcome(None, (), diagnostics)
        diagnostics["distinct_place_count"] = len(places)
        if len(places) < 2:
            diagnostics["reason_codes"] = ["INSUFFICIENT_PLACES"]
            return EventRouteOutcome(None, (), diagnostics)
        relations = self._relations(events, anchors, {item.id: item for item in evidence})
        events_by_id = {event.id: event for event in events}
        evidence_by_id = {item.id: item for item in evidence}
        relations, rejected_relations = _filter_relations_for_query(
            relations, events_by_id, evidence_by_id, query_contexts,
        )
        diagnostics["rejected_relation_count"] = len(rejected_relations)
        diagnostics["rejected_relations"] = rejected_relations
        diagnostics["ordering_relation_count"] = len(relations)
        if not relations:
            diagnostics["reason_codes"] = ["INSUFFICIENT_ORDERING"]
            return EventRouteOutcome(None, (), diagnostics)
        assembly = self._assemble(relations)
        route, retained, main_chain = self._route_from_assembly(
            assembly, places, events_by_id, evidence_by_id,
            event_id=event_id, name=name, period=period,
        )
        diagnostics["retained_relation_count"] = len(retained)
        diagnostics["contradictory_relation_count"] = len(assembly.contradictory)
        diagnostics["suppressed_relation_count"] = len(assembly.suppressed)
        diagnostics["suppressed_relations"] = list(assembly.suppressed)
        diagnostics["component_count"] = len(route.route_components)
        diagnostics["branch_relation_count"] = len(route.branch_relations)
        diagnostics["ordering_provenance"] = [relation.as_provenance() for relation in retained]
        diagnostics["ordered_place_count"] = len(main_chain)
        diagnostics["route_point_count"] = len(route.ordered_points)
        represented_places = {
            point.historical_place.canonical_name
            for component in route.route_components
            for point in component.ordered_points
        }
        represented_places.update({branch.earlier for branch in route.branch_relations})
        represented_places.update({branch.later for branch in route.branch_relations})
        if (
            len(route.route_components) > 1
            or route.branch_relations
            or len(represented_places) < len(places)
            or not route.ordered_points
        ):
            diagnostics["reason_codes"] = ["PARTIAL_ROUTE"]
        return EventRouteOutcome(route, tuple(retained), diagnostics)

    def _relations(self, events: list[HistoricalEvent], anchors: list[EventAnchor], evidence_by_id: dict[str, Evidence]) -> list[AnchorOrderingRelation]:
        events_by_id = {event.id: event for event in events}
        by_event: dict[str, list[EventAnchor]] = {}
        for anchor in anchors:
            by_event.setdefault(anchor.event_id, []).append(anchor)
        found: list[AnchorOrderingRelation] = []
        for identifier, group in by_event.items():
            origin, destination = _sole(group, EventPlaceRole.ORIGIN), _sole(group, EventPlaceRole.DESTINATION)
            if origin is not None and destination is not None and origin.canonical_name != destination.canonical_name:
                found.append(AnchorOrderingRelation(origin.canonical_name, destination.canonical_name, OrderingRule.SAME_MOVEMENT_EVENT, (identifier,), tuple(sorted(set(origin.evidence_refs) | set(destination.evidence_refs)))))
        identifiers = list(by_event)
        for position, first in enumerate(identifiers):
            for second in identifiers[position + 1:]:
                ordered = self._inter_event_order(first, second, by_event, events_by_id, evidence_by_id)
                if ordered is None:
                    continue
                earlier_id, later_id, rule = ordered
                tail, head = _terminal(by_event[earlier_id]), _initial(by_event[later_id])
                if tail is None or head is None or tail.canonical_name == head.canonical_name:
                    continue
                found.append(AnchorOrderingRelation(tail.canonical_name, head.canonical_name, rule, (earlier_id, later_id), tuple(sorted(set(tail.evidence_refs) | set(head.evidence_refs)))))
        return found

    @staticmethod
    def _inter_event_order(first: str, second: str, by_event: dict[str, list[EventAnchor]], events_by_id: dict[str, HistoricalEvent], evidence_by_id: dict[str, Evidence]) -> tuple[str, str, OrderingRule] | None:
        first_time, second_time = _temporal_interval(events_by_id.get(first)), _temporal_interval(events_by_id.get(second))
        if first_time is not None and second_time is not None:
            if first_time[1] < second_time[0]:
                return first, second, OrderingRule.TEMPORAL_ORDER
            if second_time[1] < first_time[0]:
                return second, first, OrderingRule.TEMPORAL_ORDER
            return None
        first_span, second_span = _structural_span(by_event[first], evidence_by_id), _structural_span(by_event[second], evidence_by_id)
        if first_span is None or second_span is None or first_span[0][0] != second_span[0][0]:
            return None
        if first_span[1] < second_span[0]:
            return first, second, OrderingRule.SOURCE_STRUCTURAL_ORDER
        if second_span[1] < first_span[0]:
            return second, first, OrderingRule.SOURCE_STRUCTURAL_ORDER
        return None

    @staticmethod
    def _assemble(relations: list[AnchorOrderingRelation]) -> RouteAssembly:
        best: dict[tuple[str, str], AnchorOrderingRelation] = {}
        for relation in relations:
            key = (relation.earlier, relation.later)
            if key not in best or _RULE_PRIORITY[relation.rule] < _RULE_PRIORITY[best[key].rule]:
                best[key] = relation
        usable, contradictory, suppressed = _resolve_authority_conflicts(best)
        components: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = []
        branch_pairs: list[tuple[str, str]] = []
        for node_group in _weakly_connected_components(usable):
            subgraph = {
                pair: relation
                for pair, relation in usable.items()
                if pair[0] in node_group and pair[1] in node_group
            }
            incoming, outgoing = _edge_degrees(subgraph)
            subgraph_branches = sorted(pair for pair in subgraph if _is_branch_edge(pair, incoming, outgoing))
            branch_pairs.extend(subgraph_branches)
            linear_edges = {
                pair: relation
                for pair, relation in subgraph.items()
                if pair not in subgraph_branches
            }
            used_edges: set[tuple[str, str]] = set()
            for start_pair in sorted(linear_edges):
                if start_pair in used_edges:
                    continue
                chain_edges = [start_pair]
                used_edges.add(start_pair)
                while True:
                    right_candidates = sorted(
                        pair for pair in linear_edges
                        if pair not in used_edges and _can_chain_relations(linear_edges[chain_edges[-1]], linear_edges[pair])
                    )
                    if len(right_candidates) != 1:
                        break
                    chain_edges.append(right_candidates[0])
                    used_edges.add(right_candidates[0])
                while True:
                    left_candidates = sorted(
                        pair for pair in linear_edges
                        if pair not in used_edges and _can_chain_relations(linear_edges[pair], linear_edges[chain_edges[0]])
                    )
                    if len(left_candidates) != 1:
                        break
                    chain_edges.insert(0, left_candidates[0])
                    used_edges.add(left_candidates[0])
                path = _path_from_edges(chain_edges, linear_edges)
                if len(path) >= 2:
                    components.append((tuple(path), tuple(chain_edges)))
            for pair in sorted(subgraph):
                if pair in used_edges or pair in subgraph_branches:
                    continue
                relation = subgraph[pair]
                if not _eligible_same_movement_component(relation, pair):
                    continue
                components.append(((pair[0], pair[1]), (pair,)))
                used_edges.add(pair)
            _materialize_same_movement_branch_components(components, subgraph_branches, subgraph)
        branch_pairs = sorted(set(branch_pairs))
        components.sort(key=lambda item: (-len(item[0]), item[0]))
        return RouteAssembly(tuple(components), tuple(branch_pairs), usable, contradictory, suppressed)

    @staticmethod
    def _chain(relations: list[AnchorOrderingRelation]) -> tuple[list[str], dict[tuple[str, str], AnchorOrderingRelation]]:
        """Backward-compatible view of the unique main linear chain, if one exists."""
        assembly = EventAnchorRouteBuilder._assemble(relations)
        if len(assembly.components) == 1 and not assembly.branch_pairs:
            path, edges = assembly.components[0]
            return list(path), {edge: assembly.usable[edge] for edge in edges}
        lengths = [len(path) for path, _ in assembly.components]
        if lengths:
            max_len = max(lengths)
            longest = [path for path, _ in assembly.components if len(path) == max_len]
            if len(longest) == 1:
                path = longest[0]
                edge_map = {
                    (path[index], path[index + 1]): assembly.usable[(path[index], path[index + 1])]
                    for index in range(len(path) - 1)
                    if (path[index], path[index + 1]) in assembly.usable
                }
                if len(edge_map) == len(path) - 1:
                    return list(path), edge_map
        return [], dict(assembly.usable)

    def _route_from_assembly(
        self,
        assembly: RouteAssembly,
        places: dict[str, list[EventAnchor]],
        events_by_id: dict[str, HistoricalEvent],
        evidence_by_id: dict[str, Evidence],
        *,
        event_id: str,
        name: str,
        period: str,
    ) -> tuple[HistoricalRoute, list[AnchorOrderingRelation], list[str]]:
        retained: list[AnchorOrderingRelation] = []
        claims: list[HistoricalClaim] = []
        claim_index = 0
        component_models: list[HistoricalRouteComponent] = []
        for component_number, (path, edges) in enumerate(assembly.components, start=1):
            component_relations = [assembly.usable[edge] for edge in edges]
            retained.extend(component_relations)
            component_claim_ids: list[str] = []
            for relation in component_relations:
                claim_index += 1
                claim = self._claim_for_relation(relation, event_id=event_id, index=claim_index, evidence_by_id=evidence_by_id)
                claims.append(claim)
                component_claim_ids.append(claim.id)
            points = self._points_for_path(
                list(path), places, component_relations, claims, events_by_id, evidence_by_id, period=period,
            )
            component_models.append(HistoricalRouteComponent(
                component_id=f"{event_id}-component-{component_number}",
                ordered_points=points,
                relation_claim_ids=component_claim_ids,
                evidence_refs=sorted({ref for point in points for ref in point.evidence_refs}),
                status="PROVEN_LINEAR",
            ))

        branch_models: list[HistoricalRouteBranchRelation] = []
        for pair in assembly.branch_pairs:
            relation = assembly.usable[pair]
            retained.append(relation)
            claim_index += 1
            claim = self._claim_for_relation(relation, event_id=event_id, index=claim_index, evidence_by_id=evidence_by_id)
            claims.append(claim)
            branch_models.append(HistoricalRouteBranchRelation(
                earlier=relation.earlier,
                later=relation.later,
                rule=relation.rule.value,
                event_ids=list(relation.event_ids),
                evidence_refs=list(relation.evidence_refs),
                branch_kind=_branch_kind(pair, assembly.usable),
            ))

        main_chain = self._main_chain_places(component_models)
        main_points = (
            self._points_for_path(
                main_chain, places,
                [assembly.usable[(main_chain[index], main_chain[index + 1])] for index in range(len(main_chain) - 1)],
                claims, events_by_id, evidence_by_id, period=period,
            )
            if len(main_chain) >= 2 else []
        )
        limitations = {
            "Historical reconstruction only; not an exact march track or road route.",
            "Geometry is a schematic connection between ordered historical anchors, not path evidence.",
        }
        if len(component_models) > 1:
            limitations.add("Multiple evidence-backed route components are present; component order is not asserted.")
        if branch_models:
            limitations.add("Some proven relations form branches or hubs and are not placed in a unique linear traversal.")
        all_points = main_points or [point for component in component_models for point in component.ordered_points]
        confidence = round(sum(point.confidence for point in all_points) / len(all_points), 2) if all_points else 0.0
        route = HistoricalRoute(
            id=f"{event_id}-event-anchor-route",
            event_id=event_id,
            name=name,
            period=period,
            ordered_points=main_points,
            geometry=GeoJsonLineString(coordinates=[(point.historical_place.longitude, point.historical_place.latitude) for point in main_points]),
            evidence_refs=sorted({ref for point in all_points for ref in point.evidence_refs}),
            assumptions=[
                "Anchor order is taken only from proven historical ordering relations, never from geography or retrieval order.",
                "Connections between consecutive waypoints are inputs to later algorithmic GIS reconstruction and do not by themselves assert direct historical movement.",
            ],
            limitations=sorted(limitations),
            historical_confidence=confidence,
            claims=claims,
            route_components=component_models,
            branch_relations=branch_models,
        )
        return route, retained, main_chain

    @staticmethod
    def _main_chain_places(components: list[HistoricalRouteComponent]) -> list[str]:
        if len(components) == 1 and components[0].ordered_points:
            return [point.historical_place.canonical_name for point in components[0].ordered_points]
        lengths = [len(component.ordered_points) for component in components if component.ordered_points]
        if not lengths:
            return []
        max_len = max(lengths)
        longest = [
            component for component in components
            if len(component.ordered_points) == max_len
        ]
        if len(longest) != 1:
            return []
        return [point.historical_place.canonical_name for point in longest[0].ordered_points]

    @staticmethod
    def _claim_for_relation(
        relation: AnchorOrderingRelation,
        *,
        event_id: str,
        index: int,
        evidence_by_id: dict[str, Evidence],
    ) -> HistoricalClaim:
        direct_movement = relation.rule is OrderingRule.SAME_MOVEMENT_EVENT
        text = (
            f"{relation.earlier} precedes {relation.later} within the same attested movement event."
            if direct_movement
            else f"{relation.earlier} is an evidence-grounded waypoint before {relation.later} by {relation.rule.value}; no direct movement is asserted."
        )
        return HistoricalClaim(
            id=f"{event_id}-ordering-{index}",
            claim_type="ORDERING" if direct_movement else "WAYPOINT_ORDERING",
            text=text,
            textual_basis=relation.rule.value,
            source_place=relation.earlier,
            destination_place=relation.later,
            movement_relation=relation.rule.value if direct_movement else None,
            sequence_status="explicit",
            supporting_evidence_ids=list(relation.evidence_refs),
            source_documents=sorted({
                str(evidence_by_id[ref].metadata.get("document_id") or evidence_by_id[ref].source_file or evidence_by_id[ref].author)
                for ref in relation.evidence_refs if ref in evidence_by_id
            }),
            confidence=_RULE_CONFIDENCE[relation.rule],
        )

    @staticmethod
    def _points_for_path(
        chain: list[str],
        places: dict[str, list[EventAnchor]],
        path_relations: list[AnchorOrderingRelation],
        claims: list[HistoricalClaim],
        events_by_id: dict[str, HistoricalEvent],
        evidence_by_id: dict[str, Evidence],
        *,
        period: str,
    ) -> list[HistoricalRoutePoint]:
        relation_evidence: dict[str, set[str]] = defaultdict(set)
        for relation in path_relations:
            relation_evidence[relation.earlier].update(relation.evidence_refs)
            relation_evidence[relation.later].update(relation.evidence_refs)
        points: list[HistoricalRoutePoint] = []
        for position, place_name in enumerate(chain, start=1):
            group = places[place_name]
            anchor = group[0]
            refs = sorted({ref for item in group for ref in item.evidence_refs} | relation_evidence.get(place_name, set()))
            source_events = [events_by_id[item.event_id] for item in group if item.event_id in events_by_id]
            points.append(HistoricalRoutePoint(
                sequence=position,
                historical_place=anchor.place,
                event_summary=source_events[0].summary if source_events else place_name,
                date_or_period=anchor.period or period,
                evidence_refs=refs,
                confidence=anchor.place.confidence,
                coordinate_role=anchor.coordinate_role,
                source_support=sorted({evidence_by_id[ref].author for ref in refs if ref in evidence_by_id}),
                claim_ids=[claim.id for claim in claims if place_name in (claim.source_place, claim.destination_place)],
            ))
        return points
