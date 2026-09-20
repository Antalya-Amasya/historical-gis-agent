"""G9Q1-R5.1: QueryRouteScope endpoint-constraint state consistency."""
from __future__ import annotations

import copy
import dataclasses
import inspect

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    QueryRouteScope,
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


def _scope(query: str) -> QueryRouteScope:
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


def test_has_endpoint_constraint_field_declared_once():
    names = [field.name for field in dataclasses.fields(QueryRouteScope)]
    assert names.count("has_endpoint_constraint") == 1
    attr = QueryRouteScope.__dict__.get("has_endpoint_constraint")
    assert not isinstance(attr, property)


def test_constructor_signature_has_endpoint_constraint_once():
    params = list(inspect.signature(QueryRouteScope).parameters)
    assert params.count("has_endpoint_constraint") == 1


def test_serialization_exposes_single_endpoint_constraint_state():
    scope = QueryRouteScope(
        subject="Commander Delta",
        origin=None,
        destination=None,
        episode=None,
        has_endpoint_constraint=True,
    )
    payload = dataclasses.asdict(scope)
    assert list(payload.keys()).count("has_endpoint_constraint") == 1
    assert payload["has_endpoint_constraint"] is True
    assert "has_endpoint_constraint=True" in repr(scope)
    assert repr(scope).count("has_endpoint_constraint=") == 1


def test_absent_endpoint_state():
    scope = _scope("Show Commander Delta's route.")
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is False


def test_resolved_endpoint_state():
    scope = _scope("Trace Commander Delta from Port A to City B.")
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.has_endpoint_constraint is True
    assert scope.endpoint_strict is True


def test_ambiguous_endpoint_state():
    scope = _scope(TERRA_QUERY)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


def test_serialize_reconstruct_preserves_ambiguous_endpoint_constraint():
    scope = _scope(TERRA_QUERY)
    reconstructed = QueryRouteScope(**dataclasses.asdict(scope))
    assert reconstructed.has_endpoint_constraint is True


def test_copy_paths_preserve_endpoint_constraint():
    scope = _scope(TERRA_QUERY)
    assert dataclasses.replace(scope).has_endpoint_constraint is True
    assert copy.copy(scope).has_endpoint_constraint is True
    assert copy.deepcopy(scope).has_endpoint_constraint is True


def test_positional_construction_preserves_intended_endpoint_constraint():
    scope = QueryRouteScope("Delta", None, None, None, False, True, True)
    assert scope.subject == "Delta"
    assert scope.has_episode_constraint is False
    assert scope.endpoint_strict is True
    assert scope.has_endpoint_constraint is True


def test_endpoint_strict_remains_independent_from_has_endpoint_constraint():
    toward = _scope("Trace Commander Delta from Port A toward City B.")
    assert toward.origin == "Port A"
    assert toward.destination == "City B"
    assert toward.endpoint_strict is False
    assert toward.has_endpoint_constraint is True

    ambiguous = _scope(TERRA_QUERY)
    assert ambiguous.has_endpoint_constraint is True
    assert ambiguous.endpoint_strict is True


def test_episode_symmetry_for_endpoint_constraint_classes():
    absent_episode = _scope("Trace Commander Delta from Port A to City B.")
    assert absent_episode.episode is None
    assert absent_episode.has_episode_constraint is False

    ambiguous_episode = _scope(
        "Trace Commander Delta from Port A to City B during Campaign Gold or Campaign Silver."
    )
    assert ambiguous_episode.episode is None
    assert ambiguous_episode.has_episode_constraint is True

    absent_endpoints = _scope("Show Commander Delta's route.")
    assert absent_endpoints.origin is None
    assert absent_endpoints.destination is None
    assert absent_endpoints.has_endpoint_constraint is False

    ambiguous_endpoints = _scope(TERRA_QUERY)
    assert ambiguous_endpoints.origin is None
    assert ambiguous_endpoints.destination is None
    assert ambiguous_endpoints.has_endpoint_constraint is True


def test_ambiguous_endpoint_scope_with_unknown_route_phase_rejects():
    admission, component_admitted = _admission(
        TERRA_QUERY,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert admission.admitted == component_admitted


@pytest.mark.parametrize(
    ("query", "origin", "destination", "expects_admit"),
    [
        ("Show Commander Delta's route.", "Port A", "City B", True),
        ("Trace Commander Delta from Port A to City B.", "Port A", "City B", True),
        (TERRA_QUERY, "Port A", "City B", False),
    ],
)
def test_canonical_and_component_agreement(query, origin, destination, expects_admit):
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin=origin,
        destination=destination,
    )
    assert admission.admitted is expects_admit
    assert admission.admitted == component_admitted
