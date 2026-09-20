"""G9Q1-R5: preserve unresolved endpoint-constraint intent on QueryRouteScope."""
from __future__ import annotations

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
)

TERRA_QUERY = (
    "Trace Commander Delta from Port A to City B followed by Commander Delta moving from "
    "City C to Fort D."
)


def _scope(query: str):
    return parse_query_route_scope((query,))


def _admission(query: str, text: str, *, origin: str, destination: str, actor: str = "Commander Delta"):
    ev = _evidence(text, eid="ev-a")
    event = _event("event-a", text, origin=origin, destination=destination, evidence_id="ev-a")
    observations = [
        _observation(
            "a1",
            label=origin,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
            actor_text=actor,
        ),
        _observation(
            "a2",
            label=destination,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.DESTINATION,
            actor_text=actor,
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
    return admission, component_admitted


def test_absent_endpoint_scope_remains_broad_subject_query():
    scope = _scope("Show Commander Delta's route.")
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is False
    admission, component_admitted = _admission(
        "Show Commander Delta's route.",
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_resolved_endpoint_scope_requires_phase():
    query = "Trace Commander Delta from Port A to City B."
    scope = _scope(query)
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.has_endpoint_constraint is True
    assert scope.endpoint_strict is True
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_episode_symmetry_for_endpoint_constraint_classes():
    absent_ep = _scope("Trace Commander Delta from Port A to City B.")
    assert absent_ep.episode is None
    assert absent_ep.has_episode_constraint is False
    ambiguous_ep = _scope(
        "Trace Commander Delta from Port A to City B during Campaign Gold or Campaign Silver."
    )
    assert ambiguous_ep.episode is None
    assert ambiguous_ep.has_episode_constraint is True
    absent_endpoints = _scope("Show Commander Delta's route.")
    assert absent_endpoints.origin is None
    assert absent_endpoints.destination is None
    assert absent_endpoints.has_endpoint_constraint is False
    ambiguous_endpoints = _scope(TERRA_QUERY)
    assert ambiguous_endpoints.origin is None
    assert ambiguous_endpoints.destination is None
    assert ambiguous_endpoints.has_endpoint_constraint is True


def test_terra_competing_frames_do_not_degrade_to_subject_only_admit():
    scope = _scope(TERRA_QUERY)
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True
    admission, component_admitted = _admission(
        TERRA_QUERY,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert admission.admitted == component_admitted
    assert not component_admitted


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B and from City C to Fort D.",
        "Trace Commander Delta from Port A to City B or from City C to Fort D.",
        "Trace Commander Delta from Port A to City B then from City C to Fort D.",
        "Trace Commander Delta from Port A to City B followed by movement from City C to Fort D.",
        "Trace Commander Delta from Port A to City B and Commander Sigma from City C to Fort D.",
        "Trace Commander Delta from Port A to City B then toward Fort C.",
    ],
)
def test_ambiguous_explicit_endpoint_scope_stays_constrained(query: str):
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_toward_resolved_pair_keeps_non_strict_endpoint_flag():
    query = "Trace Commander Delta from Port A toward City B."
    scope = _scope(query)
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.endpoint_strict is False
    assert scope.has_endpoint_constraint is True


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A.",
        "Trace movement to City B.",
        "Trace movement into City B.",
    ],
)
def test_incomplete_endpoint_fragments_are_not_resolved_phases(query: str):
    scope = _scope(query)
    assert scope.origin is None or scope.destination is None
    assert not (scope.origin and scope.destination)
    assert scope.has_endpoint_constraint is False
