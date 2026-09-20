"""Pre-admission soft-direction phase membership from structured observation relations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from backend.app.models import Evidence, HistoricalEvent
from backend.app.routes.observation_relation_graph import (
    extract_linear_chains,
    normalize_observation_graph,
)
from backend.app.routes.query_route_admission import (
    QueryRouteScope,
    _place_identity_matches,
    parse_query_route_scope,
    relation_non_phase_eligible,
)
from backend.app.routes.route_observations import ObservationOrderingRelation, RouteObservation

_BUILDER_TOKEN = object()
_UNTRUSTED_INDEX_TOKEN = object()


class SoftPhaseAnchorType(str, Enum):
    ORIGIN_PREFIX = "ORIGIN_PREFIX"
    DESTINATION_SUFFIX = "DESTINATION_SUFFIX"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class SoftPhaseMembershipProof:
    earlier_observation_id: str
    later_observation_id: str
    query_origin: str
    query_destination: str
    anchor_type: SoftPhaseAnchorType


@dataclass(frozen=True)
class SoftPhaseMembershipIndex:
    query_origin: str
    query_destination: str
    _memberships: tuple[tuple[tuple[str, str], SoftPhaseAnchorType], ...]
    _builder_token: object = field(default=_UNTRUSTED_INDEX_TOKEN, repr=False, compare=False, hash=False)

    def proof_for(
        self,
        relation: ObservationOrderingRelation,
        scope: QueryRouteScope,
    ) -> SoftPhaseMembershipProof | None:
        if self._builder_token is not _BUILDER_TOKEN:
            return None
        if not _index_scope_matches(self, scope):
            return None
        edge = (relation.earlier_observation_id, relation.later_observation_id)
        memberships = dict(self._memberships)
        anchor = memberships.get(edge)
        if anchor is None:
            return None
        return SoftPhaseMembershipProof(
            earlier_observation_id=relation.earlier_observation_id,
            later_observation_id=relation.later_observation_id,
            query_origin=self.query_origin,
            query_destination=self.query_destination,
            anchor_type=anchor,
        )

    def anchor_for_edge(self, edge: tuple[str, str]) -> SoftPhaseAnchorType | None:
        return dict(self._memberships).get(edge)

    @classmethod
    def _from_builder(
        cls,
        scope: QueryRouteScope,
        memberships: dict[tuple[str, str], SoftPhaseAnchorType],
    ) -> SoftPhaseMembershipIndex:
        return cls(
            query_origin=scope.origin or "",
            query_destination=scope.destination or "",
            _memberships=tuple(sorted(memberships.items())),
            _builder_token=_BUILDER_TOKEN,
        )


def build_soft_phase_membership_index(
    relations: list[ObservationOrderingRelation],
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...],
) -> SoftPhaseMembershipIndex:
    scope = parse_query_route_scope(query_contexts)
    if not _soft_membership_active(scope):
        return SoftPhaseMembershipIndex._from_builder(scope, {})
    eligible = [
        relation
        for relation in relations
        if relation_non_phase_eligible(
            relation,
            observations_by_id,
            events_by_id,
            evidence_by_id,
            query_contexts,
            scope,
        )
    ]
    graph, _ = normalize_observation_graph(eligible, merge_relations=_passthrough_merge)
    if graph.contradictory_edges or graph.cyclic_nodes:
        return SoftPhaseMembershipIndex._from_builder(scope, {})
    index: dict[tuple[str, str], SoftPhaseAnchorType] = {}
    linear_edges = graph.linear_edges()
    for node_group in graph.weak_components():
        for chain in extract_linear_chains(linear_edges, node_group):
            anchor = _chain_anchor_type(chain, observations_by_id, scope)
            if anchor is None:
                continue
            for chain_edge in chain:
                if _is_direct_endpoint_touch_edge(chain_edge, observations_by_id, scope):
                    continue
                index[chain_edge] = anchor
    return SoftPhaseMembershipIndex._from_builder(scope, index)


def _index_scope_matches(index: SoftPhaseMembershipIndex, scope: QueryRouteScope) -> bool:
    return (
        _place_identity_matches(index.query_origin, scope.origin)
        and _place_identity_matches(index.query_destination, scope.destination)
    )


def _passthrough_merge(
    left: ObservationOrderingRelation,
    right: ObservationOrderingRelation,
) -> ObservationOrderingRelation:
    return left


def _soft_membership_active(scope: QueryRouteScope) -> bool:
    return bool(
        scope.has_endpoint_constraint
        and not scope.endpoint_strict
        and scope.origin
        and scope.destination
    )


def _is_direct_endpoint_touch_edge(
    edge: tuple[str, str],
    observations_by_id: dict[str, RouteObservation],
    scope: QueryRouteScope,
) -> bool:
    earlier_label = observations_by_id[edge[0]].label
    later_label = observations_by_id[edge[1]].label
    return (
        _place_identity_matches(earlier_label, scope.origin)
        or _place_identity_matches(later_label, scope.destination)
    )


def _chain_anchor_type(
    chain: list[tuple[str, str]],
    observations_by_id: dict[str, RouteObservation],
    scope: QueryRouteScope,
) -> SoftPhaseAnchorType | None:
    start_id = chain[0][0]
    end_id = chain[-1][1]
    start_label = observations_by_id[start_id].label
    end_label = observations_by_id[end_id].label
    origin_match = _place_identity_matches(start_label, scope.origin)
    destination_match = _place_identity_matches(end_label, scope.destination)
    if origin_match and destination_match:
        return SoftPhaseAnchorType.COMPLETE
    if origin_match:
        return SoftPhaseAnchorType.ORIGIN_PREFIX
    if destination_match:
        return SoftPhaseAnchorType.DESTINATION_SUFFIX
    return None
