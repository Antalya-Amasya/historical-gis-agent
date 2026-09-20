"""G9Q1-R4.1: destination spans must end before later route-frame starts."""
from __future__ import annotations

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.query_scope_parser import (
    QuerySpanRole,
    _SpanRegistry,
    _claim_context,
    _claim_subject,
    _classify_route_frame_group,
    _discover_route_frame_candidates,
    _FrameGroup,
    parse_query_scope,
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


def _frames(query: str):
    registry = _SpanRegistry(query)
    _claim_subject(registry)
    _claim_context(registry)
    candidates = _discover_route_frame_candidates(registry)
    group = _classify_route_frame_group(candidates, query)
    return candidates, group


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
    return classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event},
        {ev.id: ev},
        (query,),
    )


def test_terra_followed_by_same_actor_moving_from_is_competing():
    candidates, group = _frames(TERRA_QUERY)
    assert [(item.origin, item.destination) for item in candidates] == [
        ("Port A", "City B"),
        ("City C", "Fort D"),
    ]
    assert "followed" not in candidates[0].destination.casefold()
    assert group is _FrameGroup.COMPETING
    scope = _scope(TERRA_QUERY)
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B, followed by movement from City C to Fort D.",
        "Trace Commander Delta from Port A to City B, after which Commander Delta moved from City C to Fort D.",
        "Trace Commander Delta from Port A to City B; Commander Delta then moved from City C to Fort D.",
        "Trace Commander Delta from Port A to City B and later moved from City C to Fort D.",
    ],
)
def test_same_actor_complete_frames_are_competing(query: str):
    candidates, group = _frames(query)
    pairs = [(item.origin, item.destination) for item in candidates if item.competing]
    assert ("Port A", "City B") in pairs
    assert ("City C", "Fort D") in pairs
    assert group is _FrameGroup.COMPETING
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert "followed" not in (scope.destination or "").casefold()


def test_competing_frames_are_order_independent():
    forward = "Trace Commander Delta from Port A to City B followed by Commander Delta moving from City C to Fort D."
    reverse = "Trace Commander Delta from City C to Fort D followed by Commander Delta moving from Port A to City B."
    _, group_forward = _frames(forward)
    _, group_reverse = _frames(reverse)
    assert group_forward is _FrameGroup.COMPETING
    assert group_reverse is _FrameGroup.COMPETING
    assert _scope(forward).origin is None
    assert _scope(reverse).origin is None


def test_while_other_actor_frame_remains_context():
    query = "Trace Commander Delta from Port A to City B while Commander Sigma moved from City C to Fort D."
    candidates, group = _frames(query)
    primary = [item for item in candidates if not item.in_context]
    context = [item for item in candidates if item.in_context]
    assert [(item.origin, item.destination) for item in primary] == [("Port A", "City B")]
    assert [(item.origin, item.destination) for item in context] == [("City C", "Fort D")]
    assert all(not item.competing for item in context)
    assert group is _FrameGroup.SINGLE_PRIMARY
    scope = _scope(query)
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"


def test_different_actor_non_context_clears_subject_and_endpoints():
    query = "Trace Commander Delta from Port A to City B and Commander Sigma from City C to Fort D."
    scope = _scope(query)
    assert scope.subject is None
    assert scope.origin is None
    assert scope.destination is None
    _, group = _frames(query)
    assert group is _FrameGroup.COMPETING


def test_true_context_without_second_frame_keeps_primary():
    query = "Trace Commander Delta from Port A to City B after meeting Commander Sigma."
    candidates, group = _frames(query)
    assert [(item.origin, item.destination) for item in candidates] == [("Port A", "City B")]
    assert group is _FrameGroup.SINGLE_PRIMARY
    scope = _scope(query)
    assert scope.origin == "Port A"
    assert scope.destination == "City B"


def test_true_context_place_is_not_a_second_frame():
    query = "Trace Commander Delta from Port A to City B after meeting Commander Sigma at Fort C."
    candidates, group = _frames(query)
    assert [(item.origin, item.destination) for item in candidates] == [("Port A", "City B")]
    assert group is _FrameGroup.SINGLE_PRIMARY
    scope = _scope(query)
    assert scope.destination == "City B"
    assert "fort c" not in (scope.destination or "").casefold()


def test_partial_followed_by_movement_toward_fails_closed():
    query = "Trace Commander Delta from Port A to City B followed by movement toward Fort C."
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    parsed = parse_query_scope(query)
    assert "toward" not in (parsed.destination or "").casefold()
    assert "followed" not in (parsed.destination or "").casefold()


def test_postposed_via_is_not_a_second_frame():
    parsed = parse_query_scope("Trace movement from George to Helena via Pass Omega.")
    assert parsed.origin == "George"
    assert parsed.destination == "Helena"
    assert any(span.role is QuerySpanRole.VIA and "Pass Omega" in span.text for span in parsed.spans)
    _, group = _frames("Trace movement from George to Helena via Pass Omega.")
    assert group is _FrameGroup.SINGLE_PRIMARY


def test_competing_frames_do_not_admit_first_pair():
    admission = _admission(
        TERRA_QUERY,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert _scope(TERRA_QUERY).has_endpoint_constraint is True
