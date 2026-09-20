"""V1.1G9G-R2R4: query subject identity closure."""
from __future__ import annotations

import pytest

from backend.app.models import EventActorStatus, EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    BROAD_QUERY,
    _evidence,
    _event,
    _observation,
    _relation,
)

ALPHA_II_QUERY = "Show Commander Alpha II's historical route."


def _subject_admission(actor_text: str, *, query: str = BROAD_QUERY, actor_status=EventActorStatus.EXPLICIT):
    text = f"During Campaign Red, {actor_text} marched from Port A to City B."
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="City B", evidence_id="ev-a")
    observations = [
        _observation(
            "a1",
            label="Port A",
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
            actor_text=actor_text,
            actor_status=actor_status,
        ),
        _observation(
            "a2",
            label="City B",
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.DESTINATION,
            actor_text=actor_text,
            actor_status=actor_status,
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
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    return admission, assembly, component_admitted


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("Commander Alpha", AuthorityState.MATCH),
        (" commander alpha ", AuthorityState.MATCH),
        ("Commander Alpha II", AuthorityState.WRONG),
        ("Commander Alpha Junior", AuthorityState.WRONG),
        ("Commander Alpha the Younger", AuthorityState.WRONG),
        ("General Alpha", AuthorityState.WRONG),
        ("Alpha's army", AuthorityState.WRONG),
        ("Commander Beta", AuthorityState.WRONG),
    ],
)
def test_generic_subject_matrix(actor, expected):
    admission, _, component_admitted = _subject_admission(actor)
    assert admission.subject_match is expected
    if expected is AuthorityState.MATCH:
        assert admission.admitted
        assert component_admitted
    else:
        assert not admission.admitted
        assert not component_admitted
    assert admission.admitted == component_admitted


def test_unknown_actor_is_unknown_not_wrong():
    admission, _, _ = _subject_admission("Commander Alpha", actor_status=EventActorStatus.UNKNOWN)
    assert admission.subject_match is AuthorityState.UNKNOWN
    assert not admission.admitted


def test_prefix_collision_alpha_ii_query_rejects_alpha_actor():
    admission, _, component_admitted = _subject_admission("Commander Alpha", query=ALPHA_II_QUERY)
    assert admission.subject_match is AuthorityState.WRONG
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_suffix_collision_alpha_query_rejects_alpha_ii_actor():
    admission, assembly, component_admitted = _subject_admission("Commander Alpha II")
    assert admission.subject_match is AuthorityState.WRONG
    assert not admission.admitted
    assert "QUERY_SUBJECT_REJECTED" in admission.reason_codes
    assert admission.admitted == component_admitted
    assert not any(item["reason"] == "ACTOR_AUTHORITY_REJECTED" for item in assembly.rejected_edges)


def test_relation_actor_integrity_still_requires_matching_endpoints():
    text = "During Campaign Red, Commander Alpha II marched from Port A to City B."
    ev = _evidence(text)
    event = _event("event-a", text, origin="Port A", destination="City B", evidence_id="ev-a")
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN, actor_text="Commander Alpha II"),
        _observation("a2", label="City B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION, actor_text="Commander Alpha"),
    ]
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    admission = classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event},
        {ev.id: ev},
        (BROAD_QUERY,),
    )
    assert admission.subject_match is AuthorityState.UNKNOWN


def test_subjectless_query_does_not_fabricate_subject():
    scope = parse_query_route_scope(("Show historical military routes.",))
    assert scope.subject is None
    admission, _, component_admitted = _subject_admission("Commander Alpha", query="Show historical military routes.")
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.admitted == component_admitted
