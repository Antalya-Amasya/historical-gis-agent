"""Adapt canonical observation components into HistoricalRoute."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.app.models import EventPlaceResolutionStatus, EventPlaceRole, HistoricalEvent
from backend.app.routes.event_anchors import EventAnchor
from backend.app.routes.event_route_orchestration import (
    AnchorOrderingRelation,
    EventAnchorRouteBuilder,
    OrderingRule,
    RouteAssembly,
)
from backend.app.routes.observation_components import ObservationComponentAssembly
from backend.app.routes.query_route_admission import QueryRouteScope, parse_query_route_scope
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
    RouteObservationKind,
)


class CanonicalRouteCompleteness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class CanonicalRoutePlan:
    completeness: CanonicalRouteCompleteness
    components: tuple
    branch_edges: tuple[tuple[str, str], ...]
    place_edges: dict[tuple[str, str], AnchorOrderingRelation]
    observation_provenance: tuple[dict[str, object], ...]


def _ordering_rule_for_relation(relation: ObservationOrderingRelation) -> OrderingRule:
    if len(relation.event_ids) == 1:
        return OrderingRule.SAME_MOVEMENT_EVENT
    if relation.ordering_rule is ObservationOrderingAuthority.TEMPORAL_ORDER:
        return OrderingRule.TEMPORAL_ORDER
    return OrderingRule.SOURCE_STRUCTURAL_ORDER


def _anchor_relation_for_observation(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
) -> AnchorOrderingRelation | None:
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    if earlier is None or later is None:
        return None
    if earlier.kind is not RouteObservationKind.PLACE or later.kind is not RouteObservationKind.PLACE:
        return None
    return AnchorOrderingRelation(
        earlier=earlier.label,
        later=later.label,
        rule=_ordering_rule_for_relation(relation),
        event_ids=relation.event_ids,
        evidence_refs=relation.evidence_refs,
    )


def _component_place_path(
    component,
    observations_by_id: dict[str, RouteObservation],
) -> list[str]:
    labels: list[str] = []
    for observation_id in component.observation_ids:
        observation = observations_by_id.get(observation_id)
        if observation is None or observation.kind is not RouteObservationKind.PLACE:
            continue
        if labels and labels[-1] == observation.label:
            continue
        labels.append(observation.label)
    return labels


def _places_with_resolved_observation_bindings(
    places: dict[str, list[EventAnchor]],
    observations: list[RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
) -> dict[str, list[EventAnchor]]:
    enriched: dict[str, list[EventAnchor]] = {name: list(items) for name, items in places.items()}
    for observation in observations:
        if observation.kind is not RouteObservationKind.PLACE or observation.place_role is None:
            continue
        event = events_by_id.get(observation.event_id)
        if event is None:
            continue
        binding = next(
            (
                item
                for item in event.place_bindings
                if item.role is observation.place_role and item.place is not None
            ),
            None,
        )
        if binding is None or binding.resolution_status is not EventPlaceResolutionStatus.RESOLVED:
            continue
        place = binding.place
        if place is None or place.latitude is None or place.longitude is None:
            continue
        group = enriched.setdefault(observation.label, [])
        if any(
            anchor.event_id == observation.event_id and anchor.role is observation.place_role
            for anchor in group
        ):
            continue
        refs = tuple(sorted(set(observation.evidence_refs) | set(binding.evidence_refs)))
        admission_type = (
            "MOVEMENT_WAYPOINT"
            if observation.place_role in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}
            else "EVENT_SITE_WAYPOINT"
        )
        group.append(EventAnchor(
            observation.event_id,
            event.event_type.value,
            place.canonical_name,
            observation.place_role,
            place.latitude,
            place.longitude,
            refs,
            binding.resolver_provenance,
            place.coordinate_role,
            tuple(binding.limitations),
            event.period,
            place,
            admission_type,
        ))
    return enriched


def _classify_completeness(
    components: tuple,
    branch_edges: tuple[tuple[str, str], ...],
    scope: QueryRouteScope | None,
    observations_by_id: dict[str, RouteObservation],
) -> CanonicalRouteCompleteness:
    if not components:
        return CanonicalRouteCompleteness.ABSENT
    if branch_edges or len(components) > 1:
        return CanonicalRouteCompleteness.PARTIAL
    if scope is None or not scope.has_endpoint_constraint or not scope.origin or not scope.destination:
        return CanonicalRouteCompleteness.COMPLETE
    labels = {
        observations_by_id[observation_id].label
        for component in components
        for observation_id in component.observation_ids
        if observation_id in observations_by_id
        and observations_by_id[observation_id].kind is RouteObservationKind.PLACE
    }
    origin = scope.origin.casefold()
    destination = scope.destination.casefold()
    if any(label.casefold() == origin for label in labels) and any(label.casefold() == destination for label in labels):
        return CanonicalRouteCompleteness.COMPLETE
    return CanonicalRouteCompleteness.PARTIAL


def build_canonical_route_plan(
    observation_assembly: ObservationComponentAssembly,
    observations: list[RouteObservation],
    observation_relations: list[ObservationOrderingRelation],
    *,
    query_contexts: tuple[str, ...] | None,
) -> CanonicalRoutePlan:
    observations_by_id = {item.observation_id: item for item in observations}
    relations_by_edge = {
        (relation.earlier_observation_id, relation.later_observation_id): relation
        for relation in observation_relations
    }
    place_edges: dict[tuple[str, str], AnchorOrderingRelation] = {}
    provenance: list[dict[str, object]] = []
    branch_edges: list[tuple[str, str]] = []
    for edge in observation_assembly.branch_edges:
        relation = relations_by_edge.get(edge)
        if relation is None:
            continue
        anchor_relation = _anchor_relation_for_observation(relation, observations_by_id)
        if anchor_relation is None:
            continue
        place_edge = (anchor_relation.earlier, anchor_relation.later)
        place_edges[place_edge] = anchor_relation
        branch_edges.append(place_edge)
    for component in observation_assembly.components:
        for edge in component.relation_ids:
            relation = relations_by_edge.get(edge)
            if relation is None:
                continue
            anchor_relation = _anchor_relation_for_observation(relation, observations_by_id)
            if anchor_relation is None:
                continue
            place_edges[(anchor_relation.earlier, anchor_relation.later)] = anchor_relation
            provenance.append({
                "observation_edge": list(edge),
                "place_edge": [anchor_relation.earlier, anchor_relation.later],
                "rule": anchor_relation.rule.value,
                "event_ids": list(anchor_relation.event_ids),
                "evidence_refs": list(anchor_relation.evidence_refs),
            })
    scope = parse_query_route_scope(query_contexts) if query_contexts else None
    completeness = _classify_completeness(
        observation_assembly.components,
        tuple(branch_edges),
        scope,
        observations_by_id,
    )
    return CanonicalRoutePlan(
        completeness=completeness,
        components=observation_assembly.components,
        branch_edges=tuple(branch_edges),
        place_edges=place_edges,
        observation_provenance=tuple(provenance),
    )


def historical_route_from_canonical_plan(
    plan: CanonicalRoutePlan,
    observations: list[RouteObservation],
    observation_relations: list[ObservationOrderingRelation],
    places: dict[str, list[EventAnchor]],
    events_by_id: dict[str, object],
    evidence_by_id: dict[str, object],
    *,
    builder: EventAnchorRouteBuilder,
    event_id: str,
    name: str,
    period: str,
) -> tuple[HistoricalRoute | None, tuple[AnchorOrderingRelation, ...], list[str], dict[str, object]]:
    if plan.completeness is CanonicalRouteCompleteness.ABSENT or not plan.components:
        return None, (), [], {"canonical_completeness": plan.completeness.value}
    observations_by_id = {item.observation_id: item for item in observations}
    relations_by_edge = {
        (relation.earlier_observation_id, relation.later_observation_id): relation
        for relation in observation_relations
    }
    components: list[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = []
    for component in plan.components:
        path = _component_place_path(component, observations_by_id)
        edges: list[tuple[str, str]] = []
        for edge in component.relation_ids:
            relation = relations_by_edge.get(edge)
            if relation is None:
                continue
            anchor_relation = _anchor_relation_for_observation(relation, observations_by_id)
            if anchor_relation is None:
                continue
            place_edge = (anchor_relation.earlier, anchor_relation.later)
            if place_edge not in edges:
                edges.append(place_edge)
        if len(path) >= 2 and edges:
            components.append((tuple(path), tuple(edges)))
    assembly = RouteAssembly(
        components=tuple(components),
        branch_pairs=plan.branch_edges,
        usable=plan.place_edges,
        contradictory=(),
        suppressed=(),
    )
    route, retained, main_chain = builder._route_from_assembly(
        assembly,
        places,
        events_by_id,
        evidence_by_id,
        event_id=event_id,
        name=name,
        period=period,
    )
    route = route.model_copy(update={"id": f"{event_id}-canonical-route"})
    adapter_diagnostics = {
        "canonical_completeness": plan.completeness.value,
        "canonical_observation_provenance": list(plan.observation_provenance),
    }
    if plan.completeness is CanonicalRouteCompleteness.PARTIAL:
        adapter_diagnostics["reason_codes"] = ["PARTIAL_ROUTE"]
    return route, tuple(retained), main_chain, adapter_diagnostics
