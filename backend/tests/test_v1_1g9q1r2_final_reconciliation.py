"""G9Q/R2 final bounded reconciliation: episode punctuation, toward constraint, scope authority."""
from __future__ import annotations

import dataclasses
import inspect
import itertools

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.query_route_admission import (
    AuthorityState,
    QueryRouteScope,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.query_scope_parser import parse_query_scope
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
)

RIDGE_GOLD_QUERY = (
    "Trace Commander Delta from Harbor North through Campaign Ridge to Fort VII "
    "during Campaign Gold."
)
CONTEXT_SILVER_GOLD = (
    "During Campaign Silver, trace Commander Delta from Harbor North to Fort VII "
    "while Commander Sigma operated during Campaign Gold."
)


def _scope(query: str) -> QueryRouteScope:
    return parse_query_route_scope((query,))


def _admission(
    query: str,
    text: str,
    *,
    origin: str,
    destination: str,
    actor: str = "Commander Delta",
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


@pytest.mark.parametrize(
    "connector",
    [
        "subsequently",
        "and",
        "or",
        "before",
        "after",
        "then",
        "followed by",
    ],
)
def test_punctuated_multi_episode_sequences_fail_closed(connector: str):
    query = (
        f"Trace Commander Delta from Port A to City B "
        f"during Campaign Gold, {connector} Campaign Silver."
    )
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True


@pytest.mark.parametrize(
    "connector",
    ["and", "or", "before", "after", "then", "followed by", "subsequently"],
)
def test_semicolon_multi_episode_sequences_fail_closed(connector: str):
    query = (
        f"Trace Commander Delta from Port A to City B "
        f"during Campaign Gold; {connector} Campaign Silver."
    )
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True


def test_gold_subsequently_silver_query_rejects_gold_only_evidence():
    query = (
        "Trace Commander Delta from Port A to City B "
        "during Campaign Gold, subsequently Campaign Silver."
    )
    admission, component_admitted = _admission(
        query,
        "During Campaign Gold, Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_gold_then_silver_semicolon_query_rejects_gold_only_evidence():
    query = (
        "Trace Commander Delta from Port A to City B "
        "during Campaign Gold; then Campaign Silver."
    )
    admission, component_admitted = _admission(
        query,
        "During Campaign Gold, Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


@pytest.mark.parametrize("direction", ["toward", "towards"])
def test_toward_preserves_explicit_endpoint_constraint_intent(direction: str):
    query = f"Trace Commander Delta from Port A {direction} City B."
    scope = _scope(query)
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.endpoint_strict is False
    assert scope.has_endpoint_constraint is True


@pytest.mark.parametrize("direction", ["toward", "towards"])
def test_toward_query_rejects_unrelated_evidence(direction: str):
    query = f"Trace Commander Delta from Port A {direction} City B."
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from City X to Fort Y.",
        origin="City X",
        destination="Fort Y",
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_ridge_only_evidence_must_not_match_gold_episode_query():
    admission, component_admitted = _admission(
        RIDGE_GOLD_QUERY,
        "Commander Delta traversed Campaign Ridge on the march.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_single_episode_punctuation_control_preserved():
    scope = _scope("During Campaign Gold, trace Commander Delta from Port A to City B.")
    assert scope.episode == "Campaign Gold"
    assert scope.has_episode_constraint is True


def test_context_episode_control_preserved():
    scope = _scope(CONTEXT_SILVER_GOLD)
    assert scope.episode == "Campaign Silver"
    assert scope.has_episode_constraint is True


def test_strict_endpoint_state_model():
    scope = _scope("Trace Commander Delta from Port A to City B.")
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.has_endpoint_constraint is True
    assert scope.endpoint_strict is True


def test_absent_endpoint_state_model():
    scope = _scope("Show Commander Delta's route.")
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is False


def test_ambiguous_endpoint_state_model():
    query = (
        "Trace Commander Delta from Port A to City B followed by Commander Delta moving from "
        "City C to Fort D."
    )
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


def test_runtime_query_route_scope_remains_healthy():
    assert [field.name for field in dataclasses.fields(QueryRouteScope)].count("has_endpoint_constraint") == 1
    assert list(inspect.signature(QueryRouteScope).parameters).count("has_endpoint_constraint") == 1


def test_gold_only_query_still_matches_gold_evidence():
    query = "Trace Commander Delta from Port A to City B during Campaign Gold."
    admission, component_admitted = _admission(
        query,
        "During Campaign Gold, Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_strict_route_query_still_matches_resolved_evidence():
    query = "Trace Commander Delta from Port A to City B."
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_broad_subject_query_still_admits_matching_evidence():
    query = "Show Commander Delta's route."
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.admitted
    assert admission.admitted == component_admitted


MATRIX = {
    "punctuation": {"none": "", "comma": ", ", "semicolon": "; "},
    "sequence": {
        "and": "and Campaign Silver",
        "or": "or Campaign Silver",
        "before": "before Campaign Silver",
        "after": "after Campaign Silver",
        "then": "then Campaign Silver",
        "followed_by": "followed by Campaign Silver",
        "subsequently": "subsequently Campaign Silver",
    },
    "endpoint_mode": {
        "to": "to City B",
        "into": "into City B",
        "toward": "toward City B",
        "towards": "towards City B",
    },
}


def _matrix_query(*, punctuation: str, sequence: str, endpoint_mode: str) -> str:
    dest = MATRIX["endpoint_mode"][endpoint_mode]
    seq = MATRIX["sequence"][sequence]
    return (
        f"Trace Commander Delta from Port A {dest} during Campaign Gold"
        f"{punctuation}{seq}."
    )


@pytest.mark.parametrize(
    ("punctuation", "sequence"),
    list(itertools.product(("comma", "semicolon"), MATRIX["sequence"].keys())),
)
def test_generated_episode_punctuation_matrix_is_ambiguous(punctuation: str, sequence: str):
    scope = _scope(_matrix_query(punctuation=punctuation, sequence=sequence, endpoint_mode="to"))
    assert scope.episode is None
    assert scope.has_episode_constraint is True


@pytest.mark.parametrize("endpoint_mode", MATRIX["endpoint_mode"].keys())
def test_generated_endpoint_modes_preserve_constraint(endpoint_mode: str):
    scope = _scope(_matrix_query(punctuation="none", sequence="and", endpoint_mode=endpoint_mode))
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.has_endpoint_constraint is True
    assert scope.endpoint_strict is (endpoint_mode in {"to", "into"})
