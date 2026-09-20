"""V1.1G9G-R2R4.3: query episode/destination tail boundaries."""
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

DUAL_QUERY = "Trace Commander Alpha from Port A to City B during Campaign Red and Campaign Blue."
DUAL_QUERY_REVERSED = (
    "Trace Commander Alpha from Port A to City B during Campaign Blue and Campaign Red."
)


def _assert_roles(query: str, *, subject=None, origin=None, destination=None, episode=None):
    scope = parse_query_route_scope((query,))
    if subject is not None:
        assert scope.subject == subject
    if origin is not None:
        assert scope.origin == origin
    if destination is not None:
        assert scope.destination == destination
    if episode is not None:
        assert scope.episode == episode
    return scope


def _admission(query: str, evidence_text: str):
    ev = _evidence(evidence_text, eid="ev-a")
    event = _event(
        "event-a",
        evidence_text,
        origin="Port A",
        destination="City B",
        evidence_id="ev-a",
    )
    observations = [
        _observation("a1", label="Port A", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label="City B", event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
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
        (query,),
    )
    assembly = assemble_observation_components(
        observations, [relation], [event], [ev], query_contexts=(query,),
    )
    edge = (relation.earlier_observation_id, relation.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    return admission, component_admitted


def test_episode_stops_before_while_clause():
    scope = _assert_roles(
        "Show Commander Alpha's route during Campaign Red while General Gamma held City Z.",
        subject="Commander Alpha",
        episode="Campaign Red",
    )
    assert "while" not in (scope.episode or "").casefold()
    assert scope.has_episode_constraint is True


@pytest.mark.parametrize(
    "query",
    [DUAL_QUERY, DUAL_QUERY_REVERSED],
)
def test_coordinated_campaigns_are_ambiguous(query: str):
    scope = parse_query_route_scope((query,))
    assert scope.subject == "Commander Alpha"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.episode is None
    assert scope.has_episode_constraint is True


@pytest.mark.parametrize(
    "query",
    [
        "During Campaign Red, trace Commander Alpha from Port A to City B.",
        "Trace Commander Alpha during Campaign Red while Commander Beta operated nearby.",
        "During Campaign Red; trace Commander Alpha from Port A to City B.",
        "During Campaign Red — trace Commander Alpha from Port A to City B.",
    ],
)
def test_episode_excludes_tail_clauses(query: str):
    scope = parse_query_route_scope((query,))
    assert scope.episode == "Campaign Red"
    assert "while" not in (scope.episode or "").casefold()
    assert "trace" not in (scope.episode or "").casefold()


def test_destination_stops_before_after_clause():
    scope = _assert_roles(
        "During Campaign Red, Commander Alpha moved from Port A to City B after meeting Commander Beta.",
        origin="Port A",
        destination="City B",
        episode="Campaign Red",
    )
    assert "after" not in (scope.destination or "").casefold()
    assert "beta" not in (scope.destination or "").casefold()


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B while Beta operated nearby.",
        "Trace Commander Alpha from Port A to City B after the battle.",
        "Trace Commander Alpha from Port A to City B during Campaign Red.",
        "From Port A to City B, trace Commander Alpha during Campaign Red.",
        "From Port A to City B; trace Commander Alpha during Campaign Red.",
        "From Port A to City B — trace Commander Alpha during Campaign Red.",
    ],
)
def test_destination_stops_before_structural_tails(query: str):
    assert parse_query_route_scope((query,)).destination == "City B"


@pytest.mark.parametrize(
    ("origin", "destination"),
    [
        ("New Market", "Warwick"),
        ("Battle Creek", "Campaign Hill"),
    ],
)
def test_legitimate_place_names_are_not_truncated(origin: str, destination: str):
    scope = parse_query_route_scope((f"Trace movement from {origin} to {destination}.",))
    assert scope.origin == origin
    assert scope.destination == destination


def test_dual_episode_query_does_not_match_red_only_evidence():
    admission, component_admitted = _admission(
        DUAL_QUERY,
        "During Campaign Red, Commander Alpha marched from Port A to City B.",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_dual_episode_query_does_not_match_blue_only_evidence():
    admission, component_admitted = _admission(
        DUAL_QUERY,
        "During Campaign Blue, Commander Alpha marched from Port A to City B.",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_single_campaign_red_query_still_matches_red_evidence():
    admission, component_admitted = _admission(
        "Trace Commander Alpha from Port A to City B during Campaign Red.",
        "During Campaign Red, Commander Alpha marched from Port A to City B.",
    )
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.admitted
    assert admission.admitted == component_admitted


@pytest.mark.parametrize(
    "query",
    [
        "Show routes during Campaign Red.",
        "Show routes during the Northern War.",
        "Show routes in Campaign Red.",
        "Show routes in the Northern War.",
    ],
)
def test_single_episode_controls(query: str):
    scope = parse_query_route_scope((query,))
    assert scope.subject is None
    assert scope.has_episode_constraint is True
    assert scope.episode in {"Campaign Red", "Northern War"}
