"""R2-SD1: canonical soft-direction route-phase authority."""
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

SOFT_QUERY = "Trace Commander Delta from Harbor North toward Fort VII during Campaign Gold."
SOFT_QUERY_NO_EPISODE = "Trace Commander Delta from Harbor North toward Fort VII."
TERRA_COMPETING = (
    "Trace Commander Delta from Port A to City B followed by Commander Delta moving from "
    "City C to Fort D."
)


def _scope(query: str):
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


def test_grok_unrelated_route_with_episode_scope_rejects():
    admission, component_admitted = _admission(
        SOFT_QUERY,
        "Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_gold_episode_does_not_rescue_unrelated_route_phase():
    admission, component_admitted = _admission(
        SOFT_QUERY,
        "During Campaign Gold, Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_unrelated_route_without_gold_evidence_still_rejects():
    admission, component_admitted = _admission(
        SOFT_QUERY,
        "Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


@pytest.mark.parametrize("direction", ["toward", "towards"])
def test_towards_variant_rejects_unrelated_route(direction: str):
    query = f"Trace Commander Delta from Harbor North {direction} Fort VII during Campaign Gold."
    admission, component_admitted = _admission(
        query,
        "During Campaign Gold, Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_soft_origin_touch_matches():
    admission, component_admitted = _admission(
        SOFT_QUERY_NO_EPISODE,
        "Commander Delta marched from Harbor North to Ridge Pass.",
        origin="Harbor North",
        destination="Ridge Pass",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_soft_destination_touch_matches():
    admission, component_admitted = _admission(
        SOFT_QUERY_NO_EPISODE,
        "Commander Delta marched from Ridge Pass toward Fort VII.",
        origin="Ridge Pass",
        destination="Fort VII",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_soft_direct_pair_matches():
    admission, component_admitted = _admission(
        SOFT_QUERY_NO_EPISODE,
        "Commander Delta marched from Harbor North toward Fort VII.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_soft_reverse_is_wrong():
    admission, _ = _admission(
        SOFT_QUERY_NO_EPISODE,
        "Commander Delta marched from Fort VII to Harbor North.",
        origin="Fort VII",
        destination="Harbor North",
    )
    assert admission.route_phase_match is AuthorityState.WRONG
    assert not admission.admitted


def test_soft_neither_endpoint_touch_is_not_match():
    admission, _ = _admission(
        SOFT_QUERY_NO_EPISODE,
        "Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted


@pytest.mark.parametrize(
    ("text", "origin", "destination", "expects_match"),
    [
        ("Commander Delta marched from Harbor North to Ridge Pass.", "Harbor North", "Ridge Pass", True),
        ("Commander Delta marched from Ridge Pass toward Fort VII.", "Ridge Pass", "Fort VII", True),
        ("Commander Delta marched from Harbor North toward Fort VII.", "Harbor North", "Fort VII", True),
        ("Commander Delta marched from Fort VII to Harbor North.", "Fort VII", "Harbor North", False),
        ("Commander Delta marched from Station 12 to Gate Q.", "Station 12", "Gate Q", False),
    ],
)
def test_soft_direction_endpoint_touch_matrix(text, origin, destination, expects_match):
    admission, component_admitted = _admission(
        SOFT_QUERY_NO_EPISODE,
        text,
        origin=origin,
        destination=destination,
    )
    if expects_match:
        assert admission.route_phase_match is AuthorityState.MATCH
        assert admission.admitted
    else:
        assert admission.route_phase_match is not AuthorityState.MATCH
        assert not admission.admitted
    assert admission.admitted == component_admitted


@pytest.mark.parametrize(
    "query",
    [
        SOFT_QUERY,
        SOFT_QUERY_NO_EPISODE,
    ],
)
@pytest.mark.parametrize(
    ("text", "origin", "destination"),
    [
        ("Commander Delta marched from Station 12 to Gate Q.", "Station 12", "Gate Q"),
        ("During Campaign Gold, Commander Delta marched from Station 12 to Gate Q.", "Station 12", "Gate Q"),
    ],
)
def test_episode_never_upgrades_unrelated_route_phase(query, text, origin, destination):
    admission, _ = _admission(query, text, origin=origin, destination=destination)
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted


def test_strict_endpoint_controls_unchanged():
    query = "Trace Commander Delta from Harbor North to Fort VII."
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Harbor North to Fort VII.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted

    reverse, _ = _admission(
        query,
        "Commander Delta marched from Fort VII to Harbor North.",
        origin="Fort VII",
        destination="Harbor North",
    )
    assert reverse.route_phase_match is AuthorityState.WRONG
    assert not reverse.admitted

    unrelated, _ = _admission(
        query,
        "Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert unrelated.route_phase_match is not AuthorityState.MATCH
    assert not unrelated.admitted


def test_broad_query_unchanged():
    query = "Show Commander Delta's route."
    scope = _scope(query)
    assert scope.has_endpoint_constraint is False
    admission, component_admitted = _admission(
        query,
        "Commander Delta marched from Station 12 to Gate Q.",
        origin="Station 12",
        destination="Gate Q",
    )
    assert admission.admitted
    assert admission.admitted == component_admitted


def test_ambiguous_endpoint_regression():
    scope = _scope(TERRA_COMPETING)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True
    admission, component_admitted = _admission(
        TERRA_COMPETING,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert not admission.admitted
    assert admission.admitted == component_admitted
