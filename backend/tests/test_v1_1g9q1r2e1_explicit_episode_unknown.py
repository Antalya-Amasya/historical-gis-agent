"""R2-E1: explicit episode constraint requires positive evidence authority."""
from __future__ import annotations

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.app.routes.soft_phase_membership import build_soft_phase_membership_index
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
)
from backend.tests.test_v1_1g9q1r2sd2_soft_phase_membership import _edge

SOFT_EPISODE_QUERY = "Trace Commander Delta from Harbor North toward Fort VII during Campaign Gold."
SOFT_QUERY_NO_EPISODE = "Trace Commander Delta from Harbor North toward Fort VII."
STRICT_EPISODE_QUERY = "Trace Commander Delta from Harbor North to Fort VII during Campaign Gold."


def _chain_admission(
    query: str,
    segments: list[tuple[str, str]],
    candidate: tuple[str, str],
    *,
    episode_prefix: str = "",
):
    observations: list = []
    events: list = []
    evidence: list = []
    relations: list = []
    for left, right in segments:
        obs, evs, evd, rel = _edge(left, right, episode_prefix=episode_prefix)
        observations.extend(obs)
        events.extend(evs)
        evidence.extend(evd)
        relations.append(rel)
    obs_by = {item.observation_id: item for item in observations}
    events_by = {item.id: item for item in events}
    evidence_by = {item.id: item for item in evidence}
    candidate_rel = next(
        relation for relation in relations
        if obs_by[relation.earlier_observation_id].label == candidate[0]
        and obs_by[relation.later_observation_id].label == candidate[1]
    )
    index = build_soft_phase_membership_index(
        relations, obs_by, events_by, evidence_by, (query,),
    )
    admission = classify_observation_relation_admission(
        candidate_rel,
        obs_by,
        events_by,
        evidence_by,
        (query,),
        soft_phase_membership_index=index,
    )
    assembly = assemble_observation_components(
        observations, relations, events, evidence, query_contexts=(query,),
    )
    edge = (candidate_rel.earlier_observation_id, candidate_rel.later_observation_id)
    component_admitted = any(
        edge in component.relation_ids for component in assembly.components
    )
    return admission, component_admitted


def _single_edge_admission(
    query: str,
    text: str,
    *,
    origin: str,
    destination: str,
):
    ev = _evidence(text, eid="ev-a")
    event = _event("event-a", text, origin=origin, destination=destination, evidence_id="ev-a")
    observations = [
        _observation(
            "a1",
            label=origin,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
            actor_text="Commander Delta",
        ),
        _observation(
            "a2",
            label=destination,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.DESTINATION,
            actor_text="Commander Delta",
        ),
    ]
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation, obs_by, {event.id: event}, {ev.id: ev}, (query,),
    )
    assembly = assemble_observation_components(
        observations, [relation], [event], [ev], query_contexts=(query,),
    )
    edge = (relation.earlier_observation_id, relation.later_observation_id)
    component_admitted = any(
        edge in component.relation_ids for component in assembly.components
    )
    return admission, component_admitted


def test_missing_episode_evidence_is_unknown_not_match():
    admission, component_admitted = _chain_admission(
        SOFT_EPISODE_QUERY,
        [
            ("Harbor North", "River B"),
            ("River B", "Island C"),
            ("Island C", "Fort VII"),
        ],
        ("Harbor North", "River B"),
    )
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.movement_assertion is AuthorityState.MATCH
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
    assert "QUERY_EPISODE_UNKNOWN" in admission.reason_codes


def test_matching_episode_evidence_still_admits():
    admission, component_admitted = _chain_admission(
        SOFT_EPISODE_QUERY,
        [
            ("Harbor North", "River B"),
            ("River B", "Island C"),
            ("Island C", "Fort VII"),
        ],
        ("River B", "Island C"),
        episode_prefix="During Campaign Gold, ",
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert component_admitted
    assert admission.admitted == component_admitted


def test_wrong_episode_evidence_rejects():
    admission, component_admitted = _chain_admission(
        SOFT_EPISODE_QUERY,
        [
            ("Harbor North", "River B"),
            ("River B", "Island C"),
            ("Island C", "Fort VII"),
        ],
        ("River B", "Island C"),
        episode_prefix="During Campaign Silver, ",
    )
    assert admission.episode_match is AuthorityState.WRONG
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
    assert "QUERY_EPISODE_REJECTED" in admission.reason_codes


def test_conflicting_episode_provenance_rejects():
    text = (
        "During Campaign Gold, Commander Delta marched from Harbor North to River B. "
        "During Campaign Silver, Commander Delta marched from River B to Island C."
    )
    ev = _evidence(text, eid="ev-a")
    event = _event("event-a", text, origin="Harbor North", destination="River B", evidence_id="ev-a")
    observations = [
        _observation(
            "a1",
            label="Harbor North",
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
            actor_text="Commander Delta",
        ),
        _observation(
            "a2",
            label="River B",
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.DESTINATION,
            actor_text="Commander Delta",
        ),
    ]
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation, obs_by, {event.id: event}, {ev.id: ev}, (SOFT_EPISODE_QUERY,),
    )
    assert admission.episode_match is AuthorityState.WRONG
    assert not admission.admitted


def test_no_episode_query_does_not_newly_reject_missing_campaign_terms():
    admission, component_admitted = _chain_admission(
        SOFT_QUERY_NO_EPISODE,
        [
            ("Harbor North", "River B"),
            ("River B", "Island C"),
            ("Island C", "Fort VII"),
        ],
        ("River B", "Island C"),
    )
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert component_admitted
    assert admission.admitted == component_admitted


def test_strict_endpoint_missing_episode_evidence_is_unknown():
    admission, component_admitted = _single_edge_admission(
        STRICT_EPISODE_QUERY,
        "Commander Delta marched from Harbor North to Fort VII.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert parse_query_route_scope((STRICT_EPISODE_QUERY,)).endpoint_strict is True
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted


def test_soft_endpoint_missing_episode_evidence_is_unknown():
    admission, component_admitted = _chain_admission(
        SOFT_EPISODE_QUERY,
        [
            ("Harbor North", "River B"),
            ("River B", "Island C"),
            ("Island C", "Fort VII"),
        ],
        ("River B", "Island C"),
    )
    assert parse_query_route_scope((SOFT_EPISODE_QUERY,)).endpoint_strict is False
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
