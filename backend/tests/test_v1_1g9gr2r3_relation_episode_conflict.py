"""V1.1G9G-R2R3: relation-local episode conflict closure."""
from __future__ import annotations

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    BROAD_QUERY,
    EXPLICIT_RED_QUERY,
    _canonical_and_component,
    _event,
    _evidence,
    _inter_event_case,
    _observation,
    _relation,
)

EXPLICIT_BLUE_QUERY = "Trace Commander Alpha during Campaign Blue."
MIXED_RED = "During Campaign Red, Commander Alpha crossed River X."
MIXED_BLUE = "Years later Commander Alpha marched to City B during Campaign Blue."
CONFLICTING = (
    "During Campaign Red, Commander Alpha advanced toward City B; "
    "during Campaign Blue, he later marched elsewhere."
)


def _same_event_relation(text: str, *, origin: str = "Port A", destination: str = "City B"):
    ev = _evidence(text)
    event = _event("event-a", text, origin=origin, destination=destination, evidence_id="ev-a")
    observations = [
        _observation("a1", label=origin, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label=destination, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation("a1", "a2", event_ids=("event-a",), evidence_refs=("ev-a",))
    return observations, relation, [event], [ev]


def test_broad_same_event_red_plus_blue_rejects():
    observations, relation, events, evidence = _same_event_relation(CONFLICTING)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (BROAD_QUERY,),
    )
    assert admission.relation_episode_identity is AuthorityState.WRONG
    assert admission.event_episode_compatibility is AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.WRONG
    assert not admission.admitted
    assert "RELATION_EPISODE_CONFLICT" in admission.reason_codes


def test_broad_same_event_red_plus_blue_component_agrees():
    observations, relation, events, evidence = _same_event_relation(CONFLICTING)
    admission, assembly, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
    assert any(item["reason"] == "RELATION_EPISODE_CONFLICT" for item in assembly.rejected_edges)


def test_red_query_same_event_red_plus_blue_rejects():
    observations, relation, events, evidence = _same_event_relation(CONFLICTING)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (EXPLICIT_RED_QUERY,),
    )
    assert admission.relation_episode_identity is AuthorityState.WRONG
    assert admission.episode_match is AuthorityState.WRONG
    assert not admission.admitted


def test_blue_query_same_event_red_plus_blue_rejects():
    observations, relation, events, evidence = _same_event_relation(CONFLICTING)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (EXPLICIT_BLUE_QUERY,),
    )
    assert admission.relation_episode_identity is AuthorityState.WRONG
    assert not admission.admitted


def test_broad_same_event_red_only_may_admit():
    text = "During Campaign Red, Commander Alpha marched from Port A to City B."
    observations, relation, events, evidence = _same_event_relation(text)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (BROAD_QUERY,),
    )
    assert admission.relation_episode_identity is AuthorityState.MATCH
    assert admission.admitted


def test_red_query_same_event_red_only_admits():
    text = "During Campaign Red, Commander Alpha marched from Port A to City B."
    observations, relation, events, evidence = _same_event_relation(text)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (EXPLICIT_RED_QUERY,),
    )
    assert admission.relation_episode_identity is AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.admitted


def test_red_query_same_event_blue_only_rejects():
    text = "During Campaign Blue, Commander Alpha marched from Port A to City B."
    observations, relation, events, evidence = _same_event_relation(text)
    obs_by = {item.observation_id: item for item in observations}
    admission = classify_observation_relation_admission(
        relation,
        obs_by,
        {events[0].id: events[0]},
        {evidence[0].id: evidence[0]},
        (EXPLICIT_RED_QUERY,),
    )
    assert not admission.admitted
    assert admission.episode_match is AuthorityState.WRONG


def test_mixed_window_separate_relations_classify_independently():
    observations_red, relation_red, events, evidence, obs_by, events_by, evidence_by = _inter_event_case(
        text_a=MIXED_RED,
        text_b=MIXED_BLUE,
        origin_a="River X",
        dest_a="River X",
        origin_b="City B",
        dest_b="City B",
    )
    event_red = _event("event-red", MIXED_RED, origin="River X", destination="River X", evidence_id="ev-red")
    event_blue = _event("event-blue", MIXED_BLUE, origin="City B", destination="City B", evidence_id="ev-blue")
    ev_red = _evidence(MIXED_RED, eid="ev-red")
    ev_blue = _evidence(MIXED_BLUE, eid="ev-blue")
    obs_red = _observation("r1", label="River X", event_id="event-red", evidence_id="ev-red", place_role=EventPlaceRole.ORIGIN)
    obs_red2 = _observation("r2", label="River X", event_id="event-red", evidence_id="ev-red", place_role=EventPlaceRole.DESTINATION)
    obs_blue = _observation("b1", label="City B", event_id="event-blue", evidence_id="ev-blue", place_role=EventPlaceRole.ORIGIN)
    obs_blue2 = _observation("b2", label="City B", event_id="event-blue", evidence_id="ev-blue", place_role=EventPlaceRole.DESTINATION)
    rel_red = _relation("r1", "r2", event_ids=("event-red",), evidence_refs=("ev-red",))
    rel_blue = _relation("b1", "b2", event_ids=("event-blue",), evidence_refs=("ev-blue",))
    events_by = {event_red.id: event_red, event_blue.id: event_blue}
    evidence_by = {ev_red.id: ev_red, ev_blue.id: ev_blue}
    red_adm = classify_observation_relation_admission(
        rel_red,
        {obs_red.observation_id: obs_red, obs_red2.observation_id: obs_red2},
        events_by,
        evidence_by,
        (BROAD_QUERY,),
    )
    blue_adm = classify_observation_relation_admission(
        rel_blue,
        {obs_blue.observation_id: obs_blue, obs_blue2.observation_id: obs_blue2},
        events_by,
        evidence_by,
        (BROAD_QUERY,),
    )
    assert red_adm.relation_episode_identity is AuthorityState.MATCH
    assert blue_adm.relation_episode_identity is AuthorityState.MATCH
    assert red_adm.admitted
    assert blue_adm.admitted


def test_terra_reproduction_agreement():
    observations, relation, events, evidence = _same_event_relation(CONFLICTING)
    admission, _, component_admitted = _canonical_and_component(
        BROAD_QUERY, observations, relation, events, evidence,
    )
    assert not admission.admitted
    assert not component_admitted
    assert admission.admitted == component_admitted
