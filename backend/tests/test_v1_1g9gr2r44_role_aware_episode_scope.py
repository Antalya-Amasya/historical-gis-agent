"""V1.1G9G-R2R4.4: role-aware episode scope parsing for campaign-named endpoints."""
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


def _assert_scope(
    query: str,
    *,
    subject=None,
    origin=None,
    destination=None,
    episode=None,
    has_episode_constraint=None,
):
    scope = parse_query_route_scope((query,))
    if subject is not None:
        assert scope.subject == subject
    if origin is not None:
        assert scope.origin == origin
    if destination is not None:
        assert scope.destination == destination
    if episode is not None:
        assert scope.episode == episode
    if has_episode_constraint is not None:
        assert scope.has_episode_constraint is has_episode_constraint
    return scope


def _admission(query: str, evidence_text: str, *, origin: str, destination: str):
    ev = _evidence(evidence_text, eid="ev-a")
    event = _event(
        "event-a",
        evidence_text,
        origin=origin,
        destination=destination,
        evidence_id="ev-a",
    )
    observations = [
        _observation(
            "a1",
            label=origin,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
        ),
        _observation(
            "a2",
            label=destination,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.DESTINATION,
        ),
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


def test_campaign_named_origin_does_not_become_episode():
    _assert_scope(
        "Trace Commander Alpha from Campaign Hill to Fort II during Campaign Red.",
        subject="Commander Alpha",
        origin="Campaign Hill",
        destination="Fort II",
        episode="Campaign Red",
        has_episode_constraint=True,
    )


def test_campaign_named_destination_does_not_become_episode():
    _assert_scope(
        "Trace Commander Alpha from Port A to Campaign Hill during Campaign Red.",
        origin="Port A",
        destination="Campaign Hill",
        episode="Campaign Red",
    )


def test_both_campaign_named_endpoints_keep_single_episode():
    _assert_scope(
        "Trace Commander Alpha from Campaign Hill to Campaign Gate during Campaign Red.",
        origin="Campaign Hill",
        destination="Campaign Gate",
        episode="Campaign Red",
    )


@pytest.mark.parametrize("query", [DUAL_QUERY, DUAL_QUERY_REVERSED])
def test_true_dual_episode_query_is_ambiguous(query: str):
    scope = _assert_scope(
        query,
        subject="Commander Alpha",
        origin="Port A",
        destination="City B",
        episode=None,
        has_episode_constraint=True,
    )
    assert scope.episode is None


def test_episode_first_query_does_not_rescan_endpoint_span():
    _assert_scope(
        "During Campaign Red, trace Commander Alpha from Campaign Hill to Fort II.",
        subject="Commander Alpha",
        origin="Campaign Hill",
        destination="Fort II",
        episode="Campaign Red",
    )


def test_possessive_query_with_campaign_named_origin():
    _assert_scope(
        "Commander Alpha's route from Campaign Hill to Fort II in Campaign Red.",
        subject="Commander Alpha",
        origin="Campaign Hill",
        destination="Fort II",
        episode="Campaign Red",
    )


@pytest.mark.parametrize(
    "query",
    [
        "Trace movement from Battle Creek to Fort II during Campaign Red.",
        "Trace movement from Warwick to Campaign Hill during Campaign Red.",
    ],
)
def test_war_battle_like_places_remain_endpoints(query: str):
    scope = parse_query_route_scope((query,))
    assert scope.subject is None
    assert scope.episode == "Campaign Red"
    assert "battle" not in (scope.episode or "").casefold()
    assert "warwick" not in (scope.episode or "").casefold()


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B during Campaign Red, later during Campaign Blue.",
        "Trace Commander Alpha from Port A to City B during Campaign Red / Campaign Blue.",
    ],
)
def test_multiple_episode_markers_fail_closed(query: str):
    scope = parse_query_route_scope((query,))
    assert scope.has_episode_constraint is True
    assert scope.episode is None


def test_subjectless_campaign_named_endpoint_query():
    scope = parse_query_route_scope(
        ("Trace movement from Campaign Hill to Fort II during Campaign Red.",),
    )
    assert scope.subject is None
    assert scope.origin == "Campaign Hill"
    assert scope.destination == "Fort II"
    assert scope.episode == "Campaign Red"


def test_campaign_hill_forward_route_phase():
    query = "Trace Commander Alpha from Campaign Hill to Fort II during Campaign Red."
    admission, component_admitted = _admission(
        query,
        "Commander Alpha marched from Campaign Hill to Fort II.",
        origin="Campaign Hill",
        destination="Fort II",
    )
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.admitted == component_admitted


def test_campaign_hill_reverse_route_phase_rejected():
    query = "Trace Commander Alpha from Campaign Hill to Fort II during Campaign Red."
    admission, component_admitted = _admission(
        query,
        "Commander Alpha marched from Fort II to Campaign Hill.",
        origin="Fort II",
        destination="Campaign Hill",
    )
    assert admission.route_phase_match is AuthorityState.WRONG
    assert not admission.admitted
    assert admission.admitted == component_admitted


def test_dual_episode_query_does_not_match_red_only_evidence():
    admission, component_admitted = _admission(
        DUAL_QUERY,
        "During Campaign Red, Commander Alpha marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted
