"""R2-SD2 / R2-SD2R / R2-SD2R1: pre-admission soft-direction phase membership seam."""
from __future__ import annotations

import pytest

from backend.app.models import EventActorStatus, EventPlaceRole, Evidence
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.observation_relation_graph import extract_linear_chains, normalize_observation_graph
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.app.routes.soft_phase_membership import (
    SoftPhaseAnchorType,
    SoftPhaseMembershipIndex,
    SoftPhaseMembershipProof,
    build_soft_phase_membership_index,
)
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
)

def _fixture_evidence(text: str, *, eid: str = "ev", metadata: dict | None = None) -> Evidence:
    if metadata is None:
        return _evidence(text, eid=eid)
    return Evidence(
        id=eid,
        author="Fixture",
        work="Test",
        locator="1",
        excerpt=text,
        text=text,
        metadata=metadata,
    )

SOFT_QUERY = "Trace Commander Delta from Port A toward City D."
SOFT_EPISODE_QUERY = "Trace Commander Delta from Harbor North toward Fort VII during Campaign Gold."
GROK_QUERY = SOFT_EPISODE_QUERY


def _obs_id(label: str) -> str:
    return f"obs-{label.lower().replace(' ', '-')}"


def _event_id(left: str, right: str) -> str:
    return f"event-{left.lower().replace(' ', '-')}-{right.lower().replace(' ', '-')}"


def _edge(
    left: str,
    right: str,
    *,
    actor: str = "Commander Delta",
    episode_prefix: str = "",
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
) -> tuple[list, list, list, ObservationOrderingRelation]:
    earlier = _obs_id(left)
    later = _obs_id(right)
    evidence_id = f"ev-{_event_id(left, right)}"
    statement = f"{episode_prefix}Commander Delta marched from {left} to {right}."
    ev = _evidence(statement, eid=evidence_id)
    event = _event(_event_id(left, right), statement, origin=left, destination=right, evidence_id=evidence_id)
    observations = [
        _observation(earlier, label=left, event_id=event.id, evidence_id=ev.id, place_role=EventPlaceRole.ORIGIN, actor_text=actor),
        _observation(later, label=right, event_id=event.id, evidence_id=ev.id, place_role=EventPlaceRole.DESTINATION, actor_text=actor),
    ]
    relation = _relation(
        earlier,
        later,
        event_ids=(event.id,),
        evidence_refs=(ev.id,),
        authority=authority,
    )
    return observations, [event], [ev], relation


def _prod_event_id(left: str, right: str) -> str:
    return f"event-{left.lower().replace(' ', '-')}-to-{right.lower().replace(' ', '-')}"


def _prod_obs_id(event_id: str, role: str) -> str:
    return f"{event_id}-{role}"


def _production_equivalent_chain(
    segments: list[tuple[str, str]],
    *,
    actor: str = "Commander Delta",
) -> tuple[list, list, list, list]:
    from dataclasses import replace

    from backend.app.models import HistoricalEventActorGrounding
    from backend.app.routes.route_observations import project_observation_ordering
    from backend.tests.test_g6dq_explicit_connector_source_chronology import explicit_actor

    passage_statements: list[str] = []
    events: list = []
    evidence: list = []
    for index, (left, right) in enumerate(segments):
        statement = (
            f"During Campaign Neutral, {actor} marched from {left} to {right}."
            if index == 0
            else f"Then during Campaign Neutral, {actor} marched from {left} to {right}."
        )
        passage_statements.append(statement)
    passage = " ".join(passage_statements)
    for index, (left, right) in enumerate(segments):
        statement = passage_statements[index]
        event_id = _prod_event_id(left, right)
        evidence_id = f"ev-{event_id}"
        ev = _fixture_evidence(
            passage,
            eid=evidence_id,
            metadata={"document_id": "doc-production-chain", "spine_index": 1, "start_offset": 100 + index * 150},
        )
        event = _event(event_id, statement, origin=left, destination=right, evidence_id=evidence_id).model_copy(
            update={"actor": explicit_actor(actor)},
        )
        events.append(event)
        evidence.append(ev)
    observations, _, _ = project_observation_ordering(events, [], [], evidence)
    observations = [
        replace(observation, actor_text=actor, actor_status=EventActorStatus.EXPLICIT)
        for observation in observations
    ]
    obs_by_event: dict[str, dict[str, str]] = {}
    for observation in observations:
        obs_by_event.setdefault(observation.event_id, {})[observation.label.casefold()] = observation.observation_id
    relations: list = []
    for index, (left, right) in enumerate(segments):
        event = events[index]
        ev = evidence[index]
        origin_obs = obs_by_event[event.id][left.casefold()]
        dest_obs = obs_by_event[event.id][right.casefold()]
        relations.append(_relation(
            origin_obs,
            dest_obs,
            event_ids=(event.id,),
            evidence_refs=(ev.id,),
            authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
        ))
        if index == 0:
            continue
        prior_event = events[index - 1]
        prior_left, prior_right = segments[index - 1]
        prior_dest = obs_by_event[prior_event.id][prior_right.casefold()]
        bridge = _relation(
            prior_dest,
            origin_obs,
            event_ids=(prior_event.id, event.id),
            evidence_refs=tuple(sorted({evidence[index - 1].id, ev.id})),
            authority=ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER,
        )
        relations.append(bridge)
    return observations, events, evidence, relations


def _graph_admission(
    query: str,
    edges: list[tuple[str, str]],
    candidate: tuple[str, str],
    *,
    actor: str = "Commander Delta",
    episode_prefix: str = "",
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
    relations_override: list | None = None,
    observations_override: list | None = None,
    events_override: list | None = None,
    evidence_override: list | None = None,
):
    if relations_override is not None:
        observations = observations_override or []
        events = events_override or []
        evidence = evidence_override or []
        relations = relations_override
    else:
        observations = []
        events = []
        evidence = []
        relations = []
        for left, right in edges:
            obs, evs, evd, rel = _edge(left, right, actor=actor, episode_prefix=episode_prefix, authority=authority)
            observations.extend(obs)
            events.extend(evs)
            evidence.extend(evd)
            relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (query,),
    )
    candidate_rel = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == candidate[0]
        and obs_by[relation.later_observation_id].label == candidate[1]
    )
    admission = classify_observation_relation_admission(
        candidate_rel,
        obs_by,
        events_by,
        evidence_by,
        (query,),
        soft_phase_membership_index=membership,
    )
    assembly = assemble_observation_components(
        observations, relations, events, evidence, query_contexts=(query,),
    )
    edge = (candidate_rel.earlier_observation_id, candidate_rel.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    member = membership.anchor_for_edge(edge)
    return admission, component_admitted, member, relations, assembly, membership


def test_grok_unrelated_route_with_gold_episode_rejects():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        GROK_QUERY,
        [("Station 12", "Gate Q")],
        ("Station 12", "Gate Q"),
        episode_prefix="During Campaign Gold, ",
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert member is None
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_unrelated_component_same_episode_rejects():
    admission, component_admitted, _, _, _, _ = _graph_admission(
        SOFT_EPISODE_QUERY,
        [("Station 12", "Gate Q")],
        ("Station 12", "Gate Q"),
        episode_prefix="During Campaign Gold, ",
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_complete_graph_internal_edge_matches():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("Port A", "River B"), ("River B", "Island C"), ("Island C", "City D")],
        ("River B", "Island C"),
    )
    assert member is SoftPhaseAnchorType.COMPLETE
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_origin_prefix_internal_edge_matches():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("Port A", "River B"), ("River B", "Island C")],
        ("River B", "Island C"),
    )
    assert member is SoftPhaseAnchorType.ORIGIN_PREFIX
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_destination_suffix_internal_edge_matches():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("River B", "Island C"), ("Island C", "City D")],
        ("River B", "Island C"),
    )
    assert member is SoftPhaseAnchorType.DESTINATION_SUFFIX
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_unrelated_component_graph_rejects():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("Station 12", "Gate Q"), ("Gate Q", "Fort Z")],
        ("Station 12", "Gate Q"),
    )
    assert member is None
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_disconnected_components_keep_separate_anchoring():
    query = SOFT_QUERY
    left_obs, left_events, left_evidence, rel_ab = _edge("Port A", "River B")
    left_obs2, left_events2, left_evidence2, rel_bc = _edge("River B", "Island C")
    right_obs, right_events, right_evidence, rel_yz = _edge("Fort X", "Gate Q")
    right_obs2, right_events2, right_evidence2, rel_qd = _edge("Gate Q", "City D")
    mid_obs, mid_events, mid_evidence, rel_unrel = _edge("Station 12", "Harbor Z")
    observations = left_obs + left_obs2 + right_obs + right_obs2 + mid_obs
    events = left_events + left_events2 + right_events + right_events2 + mid_events
    evidence = left_evidence + left_evidence2 + right_evidence + right_evidence2 + mid_evidence
    relations = [rel_ab, rel_bc, rel_yz, rel_qd, rel_unrel]
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (query,),
    )
    assert membership.anchor_for_edge((rel_bc.earlier_observation_id, rel_bc.later_observation_id)) is SoftPhaseAnchorType.ORIGIN_PREFIX
    assert membership.anchor_for_edge((rel_yz.earlier_observation_id, rel_yz.later_observation_id)) is SoftPhaseAnchorType.DESTINATION_SUFFIX
    assert membership.anchor_for_edge((rel_unrel.earlier_observation_id, rel_unrel.later_observation_id)) is None


def test_branch_graph_does_not_grant_membership():
    observations: list = []
    events: list = []
    evidence: list = []
    relations: list = []
    for left, right in (("Port A", "River B"), ("Port A", "Island C"), ("River B", "City D")):
        obs, evs, evd, rel = _edge(left, right)
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evd)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    assert all(membership.proof_for(relation, scope) is None for relation in relations)


def test_cycle_graph_does_not_grant_membership():
    observations: list = []
    events: list = []
    evidence: list = []
    relations: list = []
    for left, right in (("Port A", "River B"), ("River B", "Island C"), ("Island C", "River B")):
        obs, evs, evd, rel = _edge(left, right)
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evd)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    assert all(membership.proof_for(relation, scope) is None for relation in relations)


def test_direct_touch_preservation_without_membership():
    query = SOFT_QUERY
    for left, right in (("Port A", "River B"), ("River B", "City D"), ("Port A", "City D")):
        admission, _, member, _, _, _ = _graph_admission(query, [(left, right)], (left, right))
        assert member is None
        assert admission.route_phase_match is AuthorityState.MATCH
        assert admission.admitted
    reverse_admission, _, _, _, _, _ = _graph_admission(query, [("City D", "Port A")], ("City D", "Port A"))
    assert reverse_admission.route_phase_match is AuthorityState.WRONG
    assert not reverse_admission.admitted


def test_wrong_subject_on_proven_graph_rejects():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("Port A", "River B"), ("River B", "Island C"), ("Island C", "City D")],
        ("River B", "Island C"),
        actor="Commander Sigma",
    )
    assert member is None
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_isolated_rhodanus_fixture_remains_rejected():
    from backend.tests.test_v1_1g9gr2r_canonical_route_admission import HANNIBAL_QUERY

    admission, component_admitted, member, _, _, _ = _graph_admission(
        HANNIBAL_QUERY,
        [("Rhodanus", "Island")],
        ("Rhodanus", "Island"),
        actor="Hannibal",
        episode_prefix="During the Second Punic War, ",
    )
    assert member is None
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_production_equivalent_internal_positive():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [],
        ("River B", "Island C"),
        relations_override=relations,
        observations_override=observations,
        events_override=events,
        evidence_override=evidence,
    )
    assert member is SoftPhaseAnchorType.COMPLETE
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted == component_admitted


def test_identity_collision_without_bridge_rejects_membership():
    event_a = _event("event-a", "from River B to X", origin="River B", destination="Harbor X", evidence_id="ev-a")
    event_b = _event("event-b", "from River B to Y", origin="River B", destination="Harbor Y", evidence_id="ev-b")
    ev_a = _evidence("from River B to X", eid="ev-a")
    ev_b = _evidence("from River B to Y", eid="ev-b")
    obs_a_origin = "event-a-origin"
    obs_a_dest = "event-a-destination"
    obs_b_origin = "event-b-origin"
    obs_b_dest = "event-b-destination"
    observations = [
        _observation(obs_a_origin, label="Port A", event_id=event_a.id, evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation(obs_a_dest, label="River B", event_id=event_a.id, evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
        _observation(obs_b_origin, label="River B", event_id=event_b.id, evidence_id="ev-b", place_role=EventPlaceRole.ORIGIN),
        _observation(obs_b_dest, label="City D", event_id=event_b.id, evidence_id="ev-b", place_role=EventPlaceRole.DESTINATION),
    ]
    relations = [
        _relation(obs_a_origin, obs_a_dest, event_ids=(event_a.id,), evidence_refs=("ev-a",)),
        _relation(obs_b_origin, obs_b_dest, event_ids=(event_b.id,), evidence_refs=("ev-b",)),
    ]
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in [event_a, event_b]}
    evidence_by = {item.id: item for item in [ev_a, ev_b]}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    assert membership.proof_for(relations[0], parse_query_route_scope((SOFT_QUERY,))) is None
    assert membership.proof_for(relations[1], parse_query_route_scope((SOFT_QUERY,))) is None


def test_equal_priority_reverse_conflict_rejects_internal_membership():
    observations: list = []
    events: list = []
    evidence: list = []
    relations: list = []
    for left, right in (
        ("Port A", "Node B"),
        ("Node B", "Node C"),
        ("Node C", "Node B"),
        ("Node C", "City D"),
    ):
        obs, evs, evd, rel = _edge(left, right)
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evd)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    internal = next(
        (
            relation
            for relation in relations
            if obs_by[relation.earlier_observation_id].label == "Node B"
            and obs_by[relation.later_observation_id].label == "Node C"
        ),
        None,
    )
    assert internal is not None
    assert membership.proof_for(internal, scope) is None


def test_three_node_cycle_rejects_all_membership():
    observations: list = []
    events: list = []
    evidence: list = []
    relations: list = []
    for left, right in (("Node B", "Node C"), ("Node C", "Port A"), ("Port A", "Node B")):
        obs, evs, evd, rel = _edge(left, right)
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evd)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    query = "Trace Commander Delta from Node B toward City D."
    membership = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (query,),
    )
    scope = parse_query_route_scope((query,))
    assert all(membership.proof_for(relation, scope) is None for relation in relations)


def test_terra_forged_exact_key_membership_cannot_authorize():
    observations: list = []
    events: list = []
    evidence: list = []
    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    observations.extend(obs)
    events.extend(evs)
    evidence.extend(evd)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    membership = build_soft_phase_membership_index(
        [candidate], obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    assert membership.proof_for(candidate, scope) is None
    forged_index = SoftPhaseMembershipIndex(
        query_origin=scope.origin or "",
        query_destination=scope.destination or "",
        _memberships=(((
            candidate.earlier_observation_id,
            candidate.later_observation_id,
        ), SoftPhaseAnchorType.COMPLETE),),
    )
    forged_admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (SOFT_QUERY,),
        soft_phase_membership_index=forged_index,
    )
    real_admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (SOFT_QUERY,),
        soft_phase_membership_index=membership,
    )
    assembly = assemble_observation_components(
        observations, [candidate], events, evidence, query_contexts=(SOFT_QUERY,),
    )
    edge = (candidate.earlier_observation_id, candidate.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    assert forged_admission.route_phase_match is not AuthorityState.MATCH
    assert not forged_admission.admitted
    assert real_admission.route_phase_match is not AuthorityState.MATCH
    assert not real_admission.admitted
    assert not component_admitted
    assert real_admission.admitted == component_admitted


def test_wrong_relation_proof_cannot_authorize():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    valid_index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    unrelated_obs, unrelated_events, unrelated_evidence, unrelated = _edge("Station 12", "Gate Q")
    unrelated_by = {item.observation_id: item for item in unrelated_obs}
    scope = parse_query_route_scope((SOFT_QUERY,))
    assert valid_index.proof_for(unrelated, scope) is None
    unrelated_admission = classify_observation_relation_admission(
        unrelated,
        unrelated_by,
        {item.id: item for item in unrelated_events},
        {item.id: item for item in unrelated_evidence},
        (SOFT_QUERY,),
        soft_phase_membership_index=valid_index,
    )
    assert unrelated_admission.route_phase_match is not AuthorityState.MATCH
    assert not unrelated_admission.admitted


def test_wrong_query_proof_cannot_authorize():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    candidate = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == "River B"
        and obs_by[relation.later_observation_id].label == "Island C"
        and len(relation.event_ids) == 1
    )
    valid_index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    wrong_query = "Trace Commander Delta from Station X toward Gate Y."
    wrong_scope = parse_query_route_scope((wrong_query,))
    assert valid_index.proof_for(candidate, wrong_scope) is None
    wrong_query_admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (wrong_query,),
        soft_phase_membership_index=valid_index,
    )
    assert wrong_query_admission.route_phase_match is not AuthorityState.MATCH
    assert not wrong_query_admission.admitted


def test_stale_membership_index_cannot_authorize_new_candidate():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    stale_index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    unrelated_obs, unrelated_events, unrelated_evidence, unrelated_rel = _edge("Station 12", "Gate Q")
    unrelated_by = {item.observation_id: item for item in unrelated_obs}
    stale_admission = classify_observation_relation_admission(
        unrelated_rel,
        unrelated_by,
        {item.id: item for item in unrelated_events},
        {item.id: item for item in unrelated_evidence},
        (SOFT_QUERY,),
        soft_phase_membership_index=stale_index,
    )
    assert stale_index.proof_for(unrelated_rel, parse_query_route_scope((SOFT_QUERY,))) is None
    assert stale_admission.route_phase_match is not AuthorityState.MATCH
    assert not stale_admission.admitted


def test_membership_authority_cannot_be_forged_by_mismatched_key():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    candidate = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == "River B"
        and obs_by[relation.later_observation_id].label == "Island C"
        and len(relation.event_ids) == 1
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    valid_index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    assert valid_index.proof_for(candidate, scope) is not None
    valid = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (SOFT_QUERY,),
        soft_phase_membership_index=valid_index,
    )
    assert valid.route_phase_match is AuthorityState.MATCH
    assert valid.admitted


def test_pre_admission_and_final_component_graph_agreement():
    from backend.app.routes.query_route_admission import parse_query_route_scope, relation_non_phase_eligible

    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    scope = parse_query_route_scope((SOFT_QUERY,))
    eligible = [
        relation for relation in relations
        if relation_non_phase_eligible(
            relation, obs_by, events_by, evidence_by, (SOFT_QUERY,), scope,
        )
    ]
    pre_graph, _ = normalize_observation_graph(eligible, merge_relations=lambda left, right: left)
    assembly = assemble_observation_components(
        observations, relations, events, evidence, query_contexts=(SOFT_QUERY,),
    )
    rejected = {
        tuple(item["edge"])
        for item in assembly.rejected_edges
    }
    admitted = [
        relation for relation in relations
        if (relation.earlier_observation_id, relation.later_observation_id) not in rejected
    ]
    post_graph, _ = normalize_observation_graph(admitted, merge_relations=lambda left, right: left)
    assert set(pre_graph.usable) == set(post_graph.usable)
    assert pre_graph.contradictory_edges == post_graph.contradictory_edges
    assert pre_graph.branch_edges == post_graph.branch_edges
    assert pre_graph.cyclic_nodes == post_graph.cyclic_nodes
    pre_chains = [
        tuple(chain)
        for node_group in pre_graph.weak_components()
        for chain in extract_linear_chains(pre_graph.linear_edges(), node_group)
    ]
    post_chains = [
        tuple(chain)
        for node_group in post_graph.weak_components()
        for chain in extract_linear_chains(post_graph.linear_edges(), node_group)
    ]
    assert pre_chains == post_chains


def test_forged_proof_cannot_be_passed_as_route_phase_authority():
    import inspect

    from backend.app.routes.query_route_admission import (
        _classify_route_phase_match,
        _observation_relation_adapter,
        _observation_relation_statement,
    )

    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    obs_by = {item.observation_id: item for item in obs}
    events_by = {item.id: item for item in evs}
    scope = parse_query_route_scope((SOFT_QUERY,))
    forged_proof = SoftPhaseMembershipProof(
        earlier_observation_id=candidate.earlier_observation_id,
        later_observation_id=candidate.later_observation_id,
        query_origin=scope.origin or "",
        query_destination=scope.destination or "",
        anchor_type=SoftPhaseAnchorType.COMPLETE,
    )
    classifier_params = inspect.signature(_classify_route_phase_match).parameters
    assert "soft_phase_membership_index" in classifier_params
    assert "soft_phase_membership" not in classifier_params
    admission_params = inspect.signature(classify_observation_relation_admission).parameters
    assert "soft_phase_membership_index" in admission_params
    assert "soft_phase_membership" not in admission_params
    adapter = _observation_relation_adapter(candidate, obs_by)
    statement = _observation_relation_statement(candidate, events_by)
    route_phase = _classify_route_phase_match(
        adapter.earlier,
        adapter.later,
        scope,
        statement,
        relation=candidate,
        soft_phase_membership_index=None,
    )
    assert route_phase is not AuthorityState.MATCH
    admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        {item.id: item for item in evd},
        (SOFT_QUERY,),
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert forged_proof.anchor_type is SoftPhaseAnchorType.COMPLETE


def test_classifier_api_accepts_index_not_proof():
    import inspect

    from backend.app.routes.query_route_admission import _classify_route_phase_match

    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    candidate = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == "River B"
        and obs_by[relation.later_observation_id].label == "Island C"
        and len(relation.event_ids) == 1
    )
    index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    scope = parse_query_route_scope((SOFT_QUERY,))
    param = inspect.signature(_classify_route_phase_match).parameters["soft_phase_membership_index"]
    assert param.annotation.endswith("SoftPhaseMembershipIndex") or "SoftPhaseMembershipIndex" in str(param.annotation)
    from backend.app.routes.query_route_admission import (
        _classify_route_phase_match,
        _observation_relation_adapter,
        _observation_relation_statement,
    )

    adapter = _observation_relation_adapter(candidate, obs_by)
    statement = _observation_relation_statement(candidate, events_by)
    route_phase = _classify_route_phase_match(
        adapter.earlier,
        adapter.later,
        scope,
        statement,
        relation=candidate,
        soft_phase_membership_index=index,
    )
    assert route_phase is AuthorityState.MATCH


def test_direct_touch_works_without_membership_index():
    admission, component_admitted, member, _, _, _ = _graph_admission(
        SOFT_QUERY,
        [("Port A", "River B")],
        ("Port A", "River B"),
    )
    assert member is None
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_ambiguous_endpoint_index_cannot_force_match():
    query = "Trace Commander Delta from Port A toward City D, then from Harbor X toward Fort Y."
    scope = parse_query_route_scope((query,))
    assert scope.has_endpoint_constraint is True
    assert scope.origin is None
    assert scope.destination is None
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (query,),
    )
    candidate = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == "River B"
        and obs_by[relation.later_observation_id].label == "Island C"
        and len(relation.event_ids) == 1
    )
    assert index.proof_for(candidate, scope) is None
    admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (query,),
        soft_phase_membership_index=index,
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted


def _admission_with_forged_index(
    query: str,
    candidate: ObservationOrderingRelation,
    observations: list,
    events: list,
    evidence: list,
    forged_index: object,
):
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (query,),
        soft_phase_membership_index=forged_index,
    )
    assembly = assemble_observation_components(
        observations,
        [candidate],
        events,
        evidence,
        query_contexts=(query,),
    )
    edge = (candidate.earlier_observation_id, candidate.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    return admission, component_admitted


def test_duck_typed_forged_index_cannot_authorize():
    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    scope = parse_query_route_scope((SOFT_QUERY,))

    class ForgedIndex:
        def proof_for(self, relation, query_scope):
            return SoftPhaseMembershipProof(
                earlier_observation_id=relation.earlier_observation_id,
                later_observation_id=relation.later_observation_id,
                query_origin=scope.origin or "",
                query_destination=scope.destination or "",
                anchor_type=SoftPhaseAnchorType.COMPLETE,
            )

    admission, component_admitted = _admission_with_forged_index(
        SOFT_QUERY, candidate, obs, evs, evd, ForgedIndex(),
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted


def test_subclass_forged_index_cannot_authorize():
    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    scope = parse_query_route_scope((SOFT_QUERY,))

    class ForgedSubclass(SoftPhaseMembershipIndex):
        def proof_for(self, relation, query_scope):
            return SoftPhaseMembershipProof(
                earlier_observation_id=relation.earlier_observation_id,
                later_observation_id=relation.later_observation_id,
                query_origin=scope.origin or "",
                query_destination=scope.destination or "",
                anchor_type=SoftPhaseAnchorType.COMPLETE,
            )

    admission, component_admitted = _admission_with_forged_index(
        SOFT_QUERY,
        candidate,
        obs,
        evs,
        evd,
        ForgedSubclass(
            query_origin=scope.origin or "",
            query_destination=scope.destination or "",
            _memberships=(((
                candidate.earlier_observation_id,
                candidate.later_observation_id,
            ), SoftPhaseAnchorType.COMPLETE),),
        ),
    )
    assert type(ForgedSubclass(
        query_origin="",
        query_destination="",
        _memberships=(),
    )) is not SoftPhaseMembershipIndex
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted


def test_raw_dict_cannot_act_as_membership_index():
    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    forged_dict = {
        (candidate.earlier_observation_id, candidate.later_observation_id): SoftPhaseAnchorType.COMPLETE,
    }
    admission, component_admitted = _admission_with_forged_index(
        SOFT_QUERY, candidate, obs, evs, evd, forged_dict,
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted


def test_structural_lookalike_cannot_act_as_membership_index():
    from dataclasses import dataclass

    obs, evs, evd, candidate = _edge("Station 12", "Gate Q")
    scope = parse_query_route_scope((SOFT_QUERY,))

    @dataclass
    class StructuralLookalike:
        query_origin: str
        query_destination: str

        def proof_for(self, relation, query_scope):
            return SoftPhaseMembershipProof(
                earlier_observation_id=relation.earlier_observation_id,
                later_observation_id=relation.later_observation_id,
                query_origin=self.query_origin,
                query_destination=self.query_destination,
                anchor_type=SoftPhaseAnchorType.COMPLETE,
            )

    lookalike = StructuralLookalike(
        query_origin=scope.origin or "",
        query_destination=scope.destination or "",
    )
    admission, component_admitted = _admission_with_forged_index(
        SOFT_QUERY, candidate, obs, evs, evd, lookalike,
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted


def test_builder_index_is_exact_trusted_type():
    observations, events, evidence, relations = _production_equivalent_chain([
        ("Port A", "River B"),
        ("River B", "Island C"),
        ("Island C", "City D"),
    ])
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (SOFT_QUERY,),
    )
    assert type(index) is SoftPhaseMembershipIndex
    candidate = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == "River B"
        and obs_by[relation.later_observation_id].label == "Island C"
        and len(relation.event_ids) == 1
    )
    admission = classify_observation_relation_admission(
        candidate,
        obs_by,
        events_by,
        evidence_by,
        (SOFT_QUERY,),
        soft_phase_membership_index=index,
    )
    assembly = assemble_observation_components(
        observations, relations, events, evidence, query_contexts=(SOFT_QUERY,),
    )
    edge = (candidate.earlier_observation_id, candidate.later_observation_id)
    component_admitted = any(
        edge in component.relation_ids for component in assembly.components
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert component_admitted
    assert admission.admitted == component_admitted


def test_same_episode_unrelated_rejects_duck_typed_forged_index():
    scope = parse_query_route_scope((SOFT_EPISODE_QUERY,))

    class ForgedIndex:
        def proof_for(self, relation, query_scope):
            return SoftPhaseMembershipProof(
                earlier_observation_id=relation.earlier_observation_id,
                later_observation_id=relation.later_observation_id,
                query_origin=scope.origin or "",
                query_destination=scope.destination or "",
                anchor_type=SoftPhaseAnchorType.COMPLETE,
            )

    obs, evs, evd, candidate = _edge(
        "Station 12",
        "Gate Q",
        episode_prefix="During Campaign Gold, ",
    )
    admission, component_admitted = _admission_with_forged_index(
        SOFT_EPISODE_QUERY, candidate, obs, evs, evd, ForgedIndex(),
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
