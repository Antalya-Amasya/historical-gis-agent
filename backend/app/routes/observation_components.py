"""Evidence-grounded semantic route components from observation ordering relations."""
from __future__ import annotations

from dataclasses import dataclass

from backend.app.models import EventActorStatus, Evidence, HistoricalEvent
from backend.app.routes.event_route_orchestration import _inter_event_actor_compatible
from backend.app.routes.observation_relation_graph import (
    extract_linear_chains,
    normalize_observation_graph,
)
from backend.app.routes.query_route_admission import (
    classify_observation_relation_admission,
    primary_rejection_reason,
)
from backend.app.routes.soft_phase_membership import build_soft_phase_membership_index
from backend.app.routes.route_observations import (
    ObservationOrderingRelation,
    RouteObservation,
)


@dataclass(frozen=True)
class ObservationRouteComponent:
    component_id: str
    actor_text: str
    actor_status: EventActorStatus
    observation_ids: tuple[str, ...]
    relation_ids: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]
    ordering_authorities: tuple[str, ...]
    status: str = "PROVEN_LINEAR"
    route_complete: bool = False


@dataclass(frozen=True)
class ObservationComponentAssembly:
    components: tuple[ObservationRouteComponent, ...]
    contradictory_edges: tuple[tuple[str, str], ...]
    rejected_edges: tuple[dict[str, object], ...]
    branch_edges: tuple[tuple[str, str], ...]
    diagnostics: tuple[str, ...]


def _observation_actor_admissible(observation: RouteObservation) -> bool:
    return observation.actor_status is EventActorStatus.EXPLICIT and bool(observation.actor_text)


def _relation_actor_compatible(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
) -> bool:
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    if earlier is None or later is None:
        return False
    if not (_observation_actor_admissible(earlier) and _observation_actor_admissible(later)):
        return False
    return earlier.actor_text.casefold() == later.actor_text.casefold()


def _relation_query_route_admitted(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...],
    *,
    soft_phase_membership_index=None,
):
    return classify_observation_relation_admission(
        relation,
        observations_by_id,
        events_by_id,
        evidence_by_id,
        query_contexts,
        soft_phase_membership_index=soft_phase_membership_index,
    )


def _relation_episode_compatible(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...] | None = None,
    *,
    soft_phase_membership_index=None,
) -> tuple[bool, str | None]:
    admission = _relation_query_route_admitted(
        relation,
        observations_by_id,
        events_by_id,
        evidence_by_id,
        query_contexts or (),
        soft_phase_membership_index=soft_phase_membership_index,
    )
    if admission.admitted:
        return True, None
    return False, primary_rejection_reason(admission)


def _can_chain_observation_relations(
    left: ObservationOrderingRelation,
    right: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...] | None = None,
    *,
    soft_phase_membership_index=None,
) -> bool:
    if left.later_observation_id != right.earlier_observation_id:
        return False
    if set(left.event_ids) & set(right.event_ids):
        return True
    if len(left.event_ids) == 1 and len(right.event_ids) == 1:
        earlier_event = events_by_id.get(left.event_ids[0])
        later_event = events_by_id.get(right.event_ids[0])
        if earlier_event is None or later_event is None:
            return False
        if not _inter_event_actor_compatible(earlier_event, later_event):
            return False
    combined = ObservationOrderingRelation(
        earlier_observation_id=left.earlier_observation_id,
        later_observation_id=right.later_observation_id,
        ordering_rule=left.ordering_rule,
        event_ids=tuple(dict.fromkeys(left.event_ids + right.event_ids)),
        evidence_refs=tuple(sorted(set(left.evidence_refs) | set(right.evidence_refs))),
        authority=left.authority,
    )
    admitted, _ = _relation_episode_compatible(
        combined,
        observations_by_id,
        events_by_id,
        evidence_by_id,
        query_contexts=query_contexts,
        soft_phase_membership_index=soft_phase_membership_index,
    )
    return admitted


def _merge_observation_relations(
    left: ObservationOrderingRelation,
    right: ObservationOrderingRelation,
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=left.earlier_observation_id,
        later_observation_id=left.later_observation_id,
        ordering_rule=left.ordering_rule,
        event_ids=tuple(dict.fromkeys(left.event_ids + right.event_ids)),
        evidence_refs=tuple(sorted(set(left.evidence_refs) | set(right.evidence_refs))),
        authority=left.authority,
    )


def _path_from_edges(edges: list[tuple[str, str]]) -> list[str]:
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


def _component_from_chain(
    chain_edges: list[tuple[str, str]],
    usable: dict[tuple[str, str], ObservationOrderingRelation],
    observations_by_id: dict[str, RouteObservation],
    *,
    component_index: int,
) -> ObservationRouteComponent | None:
    path = _path_from_edges(chain_edges)
    if len(path) < 2:
        return None
    actor_text = observations_by_id[path[0]].actor_text or ""
    evidence_refs = sorted({
        ref
        for edge in chain_edges
        for ref in usable[edge].evidence_refs
    })
    authorities = tuple(dict.fromkeys(usable[edge].authority for edge in chain_edges))
    return ObservationRouteComponent(
        component_id=f"observation-component-{component_index}",
        actor_text=actor_text,
        actor_status=EventActorStatus.EXPLICIT,
        observation_ids=tuple(path),
        relation_ids=tuple(chain_edges),
        evidence_refs=tuple(evidence_refs),
        ordering_authorities=authorities,
        status="PROVEN_LINEAR" if len(path) > 2 else "ISOLATED_PAIR",
        route_complete=False,
    )


def assemble_observation_components(
    observations: list[RouteObservation],
    relations: list[ObservationOrderingRelation],
    events: list[HistoricalEvent],
    evidence: list[Evidence],
    *,
    query_contexts: tuple[str, ...] | None = None,
) -> ObservationComponentAssembly:
    observations_by_id = {item.observation_id: item for item in observations}
    events_by_id = {event.id: event for event in events}
    evidence_by_id = {item.id: item for item in evidence}
    soft_phase_membership_index = (
        build_soft_phase_membership_index(
            relations,
            observations_by_id,
            events_by_id,
            evidence_by_id,
            query_contexts,
        )
        if query_contexts
        else None
    )
    rejected: list[dict[str, object]] = []
    admitted: list[ObservationOrderingRelation] = []
    for relation in relations:
        if query_contexts:
            compatible, rejection_reason = _relation_episode_compatible(
                relation,
                observations_by_id,
                events_by_id,
                evidence_by_id,
                query_contexts=query_contexts,
                soft_phase_membership_index=soft_phase_membership_index,
            )
            if not compatible:
                rejected.append({
                    "edge": (relation.earlier_observation_id, relation.later_observation_id),
                    "reason": rejection_reason or "QUERY_ROUTE_ADMISSION_REJECTED",
                    "authority": relation.authority,
                })
                continue
            admitted.append(relation)
            continue
        if not _relation_actor_compatible(relation, observations_by_id):
            rejected.append({
                "edge": (relation.earlier_observation_id, relation.later_observation_id),
                "reason": "ACTOR_AUTHORITY_REJECTED",
                "authority": relation.authority,
            })
            continue
        compatible, rejection_reason = _relation_episode_compatible(
            relation,
            observations_by_id,
            events_by_id,
            evidence_by_id,
            query_contexts=query_contexts,
            soft_phase_membership_index=soft_phase_membership_index,
        )
        if not compatible:
            rejected.append({
                "edge": (relation.earlier_observation_id, relation.later_observation_id),
                "reason": rejection_reason or "EPISODE_AUTHORITY_REJECTED",
                "authority": relation.authority,
            })
            continue
        admitted.append(relation)

    graph, graph_rejected = normalize_observation_graph(
        admitted,
        merge_relations=_merge_observation_relations,
    )
    rejected.extend(graph_rejected)
    usable = graph.usable
    contradictory = graph.contradictory_edges
    branch_edges = list(graph.branch_edges)

    components: list[ObservationRouteComponent] = []
    component_index = 0
    linear_edges = graph.linear_edges()

    def _can_link(left: ObservationOrderingRelation, right: ObservationOrderingRelation) -> bool:
        return _can_chain_observation_relations(
            left,
            right,
            observations_by_id,
            events_by_id,
            evidence_by_id,
            query_contexts=query_contexts,
            soft_phase_membership_index=soft_phase_membership_index,
        )

    for node_group in graph.weak_components():
        subgraph = {
            pair: relation
            for pair, relation in usable.items()
            if pair[0] in node_group and pair[1] in node_group
        }
        used_edges: set[tuple[str, str]] = set()
        for chain_edges in extract_linear_chains(
            linear_edges,
            node_group,
            can_link=_can_link,
        ):
            used_edges.update(chain_edges)
            component = _component_from_chain(
                chain_edges,
                linear_edges,
                observations_by_id,
                component_index=component_index,
            )
            if component is not None:
                components.append(component)
                component_index += 1
        for pair in sorted(subgraph):
            if pair in used_edges or pair in graph.branch_edges:
                continue
            if pair[0] in graph.cyclic_nodes or pair[1] in graph.cyclic_nodes:
                continue
            component = _component_from_chain([pair], subgraph, observations_by_id, component_index=component_index)
            if component is not None:
                components.append(component)
                component_index += 1

    diagnostics: list[str] = []
    if contradictory:
        diagnostics.append("OBSERVATION_ORDERING_CYCLE")
    if graph.cyclic_nodes:
        diagnostics.append("OBSERVATION_ORDERING_CYCLE")
    if branch_edges:
        diagnostics.append("OBSERVATION_ORDERING_BRANCH")
    if not components and relations:
        diagnostics.append("NO_OBSERVATION_COMPONENT")
    return ObservationComponentAssembly(
        components=tuple(components),
        contradictory_edges=contradictory,
        rejected_edges=tuple(rejected),
        branch_edges=tuple(sorted(set(branch_edges))),
        diagnostics=tuple(diagnostics),
    )


def serialize_observation_component(component: ObservationRouteComponent) -> dict[str, object]:
    return {
        "component_id": component.component_id,
        "actor_text": component.actor_text,
        "actor_status": component.actor_status.value,
        "observation_ids": list(component.observation_ids),
        "relation_ids": [list(edge) for edge in component.relation_ids],
        "evidence_refs": list(component.evidence_refs),
        "ordering_authorities": list(component.ordering_authorities),
        "status": component.status,
        "route_complete": component.route_complete,
    }
