"""Route observation identity and typed event-ordering projection."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    EventRouteOrdering,
    EventRouteOrderingAuthority,
    EventRouteOrderingEndpointKind,
    EventRouteOrderingRef,
    GeographicFeatureKind,
    HistoricalEvent,
    HistoricalEventType,
    TransitionConstraint,
    Evidence,
)
from backend.app.routes.event_anchors import EventAnchor
from backend.app.routes.event_route_orchestration import (
    _adjacent_movement_statements,
    _authorized_source_structural_order,
    _connector_governs_later_statement,
    _explicit_structural_same_actor,
    _inter_event_actor_compatible,
    _independent_relation_episode_compatible,
    _movement_statement,
    _positive_asserted_structural_order_pair,
    _temporal_interval,
)
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor


class RouteObservationKind(str, Enum):
    """PLACE does not imply exact-site authority; coordinate_role carries that separately."""

    PLACE = "PLACE"
    TRANSITION = "TRANSITION"


class ObservationOrderingAuthority(str, Enum):
    BEFORE_SUBORDINATE = "BEFORE_SUBORDINATE"
    AFTER_SUBORDINATE = "AFTER_SUBORDINATE"
    BEFORE_POSTPOSED = "BEFORE_POSTPOSED"
    AFTER_POSTPOSED = "AFTER_POSTPOSED"
    FIRST_THEN = "FIRST_THEN"
    TEMPORAL_ORDER = "TEMPORAL_ORDER"
    SOURCE_STRUCTURAL_ORDER = "SOURCE_STRUCTURAL_ORDER"


_AUTHORITY_MAP = {
    EventRouteOrderingAuthority.BEFORE_SUBORDINATE: ObservationOrderingAuthority.BEFORE_SUBORDINATE,
    EventRouteOrderingAuthority.AFTER_SUBORDINATE: ObservationOrderingAuthority.AFTER_SUBORDINATE,
    EventRouteOrderingAuthority.BEFORE_POSTPOSED: ObservationOrderingAuthority.BEFORE_POSTPOSED,
    EventRouteOrderingAuthority.AFTER_POSTPOSED: ObservationOrderingAuthority.AFTER_POSTPOSED,
    EventRouteOrderingAuthority.FIRST_THEN: ObservationOrderingAuthority.FIRST_THEN,
}


@dataclass(frozen=True)
class RouteObservation:
    observation_id: str
    kind: RouteObservationKind
    event_id: str
    label: str
    actor_text: str | None
    actor_status: EventActorStatus
    evidence_refs: tuple[str, ...]
    place_role: EventPlaceRole | None = None
    constraint_id: str | None = None
    feature_kind: GeographicFeatureKind | None = None
    coordinate_role: str | None = None


@dataclass(frozen=True)
class ObservationOrderingRelation:
    earlier_observation_id: str
    later_observation_id: str
    ordering_rule: ObservationOrderingAuthority
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    authority: str


_PLACE_ROLES = {
    EventPlaceRole.ORIGIN,
    EventPlaceRole.DESTINATION,
    EventPlaceRole.EVENT_SITE,
}


def _observation_id_for_place(event_id: str, label: str) -> str:
    return f"place:{event_id}:{label.casefold()}"


def _observation_id_for_transition(constraint_id: str) -> str:
    return f"transition:{constraint_id}"


def _structural_span_for_event(
    event: HistoricalEvent,
    evidence_by_id: dict[str, Evidence],
) -> tuple[tuple[str, int, int], tuple[str, int, int]] | None:
    from backend.app.routes.extractor import evidence_structural_key

    keys: list[tuple[str, int, int]] = []
    for ref in event.evidence_refs:
        item = evidence_by_id.get(ref)
        key = evidence_structural_key(item) if item is not None else None
        if key is None:
            return None
        keys.append(key)
    if not keys or len({key[0] for key in keys}) != 1:
        return None
    return min(keys), max(keys)


def _relation(
    earlier: RouteObservation,
    later: RouteObservation,
    event: HistoricalEvent,
    authority: ObservationOrderingAuthority,
    *,
    provenance: str,
) -> ObservationOrderingRelation:
    refs = tuple(sorted(set(earlier.evidence_refs) | set(later.evidence_refs) | set(event.evidence_refs)))
    return ObservationOrderingRelation(
        earlier_observation_id=earlier.observation_id,
        later_observation_id=later.observation_id,
        ordering_rule=authority,
        event_ids=(event.id,),
        evidence_refs=refs,
        authority=provenance,
    )


def _place_role_for_ref(ref: EventRouteOrderingRef) -> EventPlaceRole | None:
    if ref.endpoint_kind is EventRouteOrderingEndpointKind.ORIGIN:
        return EventPlaceRole.ORIGIN
    if ref.endpoint_kind is EventRouteOrderingEndpointKind.DESTINATION:
        return EventPlaceRole.DESTINATION
    return EventPlaceRole.RELATED_PLACE


def _ref_matches_label(ref: EventRouteOrderingRef, label: str, raw_text: str | None = None) -> bool:
    ref_labels = {ref.surface.casefold()}
    if ref.canonical:
        ref_labels.add(ref.canonical.casefold())
    target_labels = {label.casefold()}
    if raw_text:
        target_labels.add(raw_text.casefold())
    return bool(ref_labels & target_labels)


def _observation_for_ref(
    ref: EventRouteOrderingRef,
    event: HistoricalEvent,
    observations: list[RouteObservation],
    constraints: list[TransitionConstraint],
) -> RouteObservation | None:
    if ref.endpoint_kind is EventRouteOrderingEndpointKind.TRAVERSAL:
        event_constraints = [item for item in constraints if item.event_id == event.id]
        for constraint in event_constraints:
            if _ref_matches_label(ref, constraint.feature_canonical or constraint.feature_surface, constraint.feature_surface):
                observation_id = _observation_id_for_transition(constraint.id)
                matches = [item for item in observations if item.observation_id == observation_id]
                if matches:
                    return matches[0]
        for item in observations:
            if item.event_id != event.id or item.kind is not RouteObservationKind.TRANSITION:
                continue
            if _ref_matches_label(ref, item.label):
                return item
        return None

    role = _place_role_for_ref(ref)
    for binding in event.place_bindings:
        if binding.role is not role or binding.resolution_status is not EventPlaceResolutionStatus.RESOLVED:
            continue
        label = binding.place.canonical_name if binding.place is not None else binding.mention.raw_text
        if _ref_matches_label(ref, label, binding.mention.raw_text):
            observation_id = _observation_id_for_place(event.id, label)
            matches = [item for item in observations if item.observation_id == observation_id]
            if matches:
                return matches[0]
    for item in observations:
        if item.event_id != event.id or item.kind is not RouteObservationKind.PLACE:
            continue
        if item.place_role is role and _ref_matches_label(ref, item.label):
            return item
    return None


def _typed_same_event_relations(
    event: HistoricalEvent,
    observations: list[RouteObservation],
    constraints: list[TransitionConstraint],
) -> list[ObservationOrderingRelation]:
    if event.event_type is not HistoricalEventType.MOVEMENT or not event.source_statements:
        return []
    if not EvidenceGroundedHistoricalEventExtractor._has_positive_movement_assertion(_movement_statement(event)):
        return []
    if event.actor.actor_status is not EventActorStatus.EXPLICIT:
        return []

    found: list[ObservationOrderingRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for ordering in event.route_orderings:
        earlier = _observation_for_ref(ordering.earlier, event, observations, constraints)
        later = _observation_for_ref(ordering.later, event, observations, constraints)
        if earlier is None or later is None or earlier.observation_id == later.observation_id:
            continue
        authority = _AUTHORITY_MAP.get(ordering.authority)
        if authority is None:
            continue
        key = (earlier.observation_id, later.observation_id, authority.value)
        if key in seen:
            continue
        seen.add(key)
        found.append(_relation(earlier, later, event, authority, provenance=ordering.authority.value))
    if found:
        return found
    origin_obs = None
    destination_obs = None
    for item in observations:
        if item.event_id != event.id or item.kind is not RouteObservationKind.PLACE:
            continue
        if item.place_role is EventPlaceRole.ORIGIN:
            origin_obs = item
        elif item.place_role is EventPlaceRole.DESTINATION:
            destination_obs = item
    if (
        origin_obs is not None
        and destination_obs is not None
        and origin_obs.observation_id != destination_obs.observation_id
    ):
        key = (
            origin_obs.observation_id,
            destination_obs.observation_id,
            ObservationOrderingAuthority.AFTER_SUBORDINATE.value,
        )
        if key not in seen:
            found.append(_relation(
                origin_obs,
                destination_obs,
                event,
                ObservationOrderingAuthority.AFTER_SUBORDINATE,
                provenance="SAME_MOVEMENT_EVENT",
            ))
    return found


def project_route_observations(
    events: list[HistoricalEvent],
    anchors: list[EventAnchor],
    constraints: list[TransitionConstraint],
) -> list[RouteObservation]:
    del anchors
    observations: list[RouteObservation] = []
    for event in events:
        actor = event.actor
        for binding in event.place_bindings:
            if binding.role not in _PLACE_ROLES:
                continue
            if binding.resolution_status is not EventPlaceResolutionStatus.RESOLVED:
                continue
            label = binding.place.canonical_name if binding.place is not None else binding.mention.raw_text
            refs = tuple(sorted(set(binding.evidence_refs or event.evidence_refs)))
            observations.append(RouteObservation(
                observation_id=_observation_id_for_place(event.id, label),
                kind=RouteObservationKind.PLACE,
                event_id=event.id,
                label=label,
                actor_text=actor.actor_text,
                actor_status=actor.actor_status,
                evidence_refs=refs,
                place_role=binding.role,
                coordinate_role=binding.place.coordinate_role if binding.place is not None else None,
            ))
        for constraint in constraints:
            if constraint.event_id != event.id:
                continue
            observations.append(RouteObservation(
                observation_id=_observation_id_for_transition(constraint.id),
                kind=RouteObservationKind.TRANSITION,
                event_id=event.id,
                label=constraint.feature_canonical or constraint.feature_surface,
                actor_text=constraint.actor_text,
                actor_status=constraint.actor_status,
                evidence_refs=tuple(constraint.evidence_refs),
                constraint_id=constraint.id,
                feature_kind=constraint.feature_kind,
                coordinate_role=constraint.feature_coordinate_role,
            ))
    return observations


def _observation_chain_endpoints(
    event_id: str,
    relations: list[ObservationOrderingRelation],
    observations: list[RouteObservation],
) -> tuple[str | None, str | None]:
    relevant = [relation for relation in relations if event_id in relation.event_ids]
    if relevant:
        earlier_nodes = {relation.earlier_observation_id for relation in relevant}
        later_nodes = {relation.later_observation_id for relation in relevant}
        heads = earlier_nodes - later_nodes
        tails = later_nodes - earlier_nodes
        if len(heads) == 1 and len(tails) == 1:
            return next(iter(heads)), next(iter(tails))
    event_observations = [item for item in observations if item.event_id == event_id]
    if len(event_observations) == 1:
        only = event_observations[0].observation_id
        return only, only
    return None, None


def _inter_event_observation_order(
    first: str,
    second: str,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> tuple[str, str, ObservationOrderingAuthority] | None:
    first_event = events_by_id.get(first)
    second_event = events_by_id.get(second)
    if first_event is None or second_event is None:
        return None
    first_time, second_time = _temporal_interval(first_event), _temporal_interval(second_event)
    if first_time is not None and second_time is not None:
        if first_time[1] < second_time[0]:
            return first, second, ObservationOrderingAuthority.TEMPORAL_ORDER
        if second_time[1] < first_time[0]:
            return second, first, ObservationOrderingAuthority.TEMPORAL_ORDER
    first_span = _structural_span_for_event(first_event, evidence_by_id)
    second_span = _structural_span_for_event(second_event, evidence_by_id)
    if first_span is None or second_span is None or first_span[0][0] != second_span[0][0]:
        return None
    if first_span[1] < second_span[0]:
        if _authorized_source_structural_order(first_event, second_event, evidence_by_id):
            return first, second, ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER
    elif second_span[1] < first_span[0]:
        if _authorized_source_structural_order(second_event, first_event, evidence_by_id):
            return second, first, ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER
    return None


def _inter_event_observation_relation_allowed(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> bool:
    if relation.ordering_rule not in {
        ObservationOrderingAuthority.TEMPORAL_ORDER,
        ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER,
    }:
        return True
    if len(relation.event_ids) < 2:
        return False
    earlier_event = events_by_id.get(relation.event_ids[0])
    later_event = events_by_id.get(relation.event_ids[1])
    if earlier_event is None or later_event is None:
        return False
    if not _inter_event_actor_compatible(earlier_event, later_event):
        return False
    adapter = type("RelationAdapter", (), {
        "rule": type("Rule", (), {"value": relation.ordering_rule.value})(),
        "event_ids": relation.event_ids,
        "earlier": relation.earlier_observation_id,
        "later": relation.later_observation_id,
        "evidence_refs": relation.evidence_refs,
    })()
    if not _independent_relation_episode_compatible(adapter, events_by_id, evidence_by_id):
        return False
    if relation.ordering_rule is ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER:
        if not _positive_asserted_structural_order_pair(earlier_event, later_event):
            return False
        if not _explicit_structural_same_actor(earlier_event, later_event):
            return False
        if not _connector_governs_later_statement(later_event):
            return False
        if not _adjacent_movement_statements(earlier_event, later_event, evidence_by_id):
            return False
    return True


def project_observation_ordering(
    events: list[HistoricalEvent],
    anchors: list[EventAnchor],
    constraints: list[TransitionConstraint],
    evidence: list[Evidence],
) -> tuple[list[RouteObservation], list[ObservationOrderingRelation], list[str]]:
    del anchors
    evidence_by_id = {item.id: item for item in evidence}
    observations = project_route_observations(events, [], constraints)
    events_by_id = {event.id: event for event in events}
    same_event_relations: list[ObservationOrderingRelation] = []
    for event in events:
        same_event_relations.extend(_typed_same_event_relations(event, observations, constraints))
    found = list(same_event_relations)
    identifiers = list(events_by_id)
    observations_by_id = {item.observation_id: item for item in observations}
    for position, first in enumerate(identifiers):
        for second in identifiers[position + 1:]:
            ordered = _inter_event_observation_order(
                first, second, events_by_id, evidence_by_id,
            )
            if ordered is None:
                continue
            earlier_event_id, later_event_id, authority = ordered
            earlier_tail = _observation_chain_endpoints(earlier_event_id, same_event_relations, observations)[1]
            later_head = _observation_chain_endpoints(later_event_id, same_event_relations, observations)[0]
            if earlier_tail is None or later_head is None or earlier_tail == later_head:
                continue
            tail = observations_by_id.get(earlier_tail)
            head = observations_by_id.get(later_head)
            if tail is None or head is None:
                continue
            relation = ObservationOrderingRelation(
                earlier_observation_id=earlier_tail,
                later_observation_id=later_head,
                ordering_rule=authority,
                event_ids=(earlier_event_id, later_event_id),
                evidence_refs=tuple(sorted(set(tail.evidence_refs) | set(head.evidence_refs))),
                authority=authority.value,
            )
            if not _inter_event_observation_relation_allowed(relation, events_by_id, evidence_by_id):
                continue
            found.append(relation)
    diagnostics = [] if found else ["NO_OBSERVATION_ORDERING"]
    return observations, found, diagnostics
