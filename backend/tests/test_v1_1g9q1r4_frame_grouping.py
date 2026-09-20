"""G9Q1-R4: route/episode frame grouping and sequence ambiguity."""
from __future__ import annotations

import itertools

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.query_route_admission import (
    AuthorityState,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.query_scope_parser import QuerySpanRole, parse_query_scope, parse_query_scope_spans
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
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
    return admission


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B and from City C to Fort D.",
        "Trace Commander Delta from Port A to City B, then from City C to Fort D.",
        "Trace Commander Delta from Port A to City B after which Commander Delta moved from City C to Fort D.",
        "Trace Commander Delta from Port A to City B followed by movement from City C to Fort D.",
    ],
)
def test_multiple_route_frames_fail_closed(query: str):
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


def test_two_subject_owned_frames_clear_subject_and_endpoints():
    query = "Trace Commander Delta from Port A to City B and Commander Sigma from City C to Fort D."
    scope = _scope(query)
    assert scope.subject is None
    assert scope.origin is None
    assert scope.destination is None


def test_two_subject_frames_do_not_admit_delta_only():
    query = "Trace Commander Delta from Port A to City B and Commander Sigma from City C to Fort D."
    scope = _scope(query)
    admission = _admission(
        query,
        "Commander Delta marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert scope.subject is None
    assert scope.has_endpoint_constraint is True
    assert admission.route_phase_match is not AuthorityState.MATCH
    assert not admission.admitted


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B, then to Fort C.",
        "Trace Commander Delta from Port A to City B then to Fort C.",
        "Trace Commander Delta from Port A to City B then toward Fort C.",
        "Trace Commander Delta from Port A to City B and onward into Fort C.",
    ],
)
def test_same_actor_continuation_fails_closed(query: str):
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


def test_true_context_preserves_primary_route():
    scope = _scope(
        "Trace Commander Delta from Port A to City B while Commander Sigma remained at Gate Q."
    )
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"


def test_context_second_route_does_not_replace_primary():
    scope = _scope(
        "Trace Commander Delta from Port A to City B while Commander Sigma moved from City C to Fort D."
    )
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"


def test_postposed_via_is_single_frame():
    parsed = parse_query_scope("Trace movement from George to Helena via Pass Omega.")
    assert parsed.origin == "George"
    assert parsed.destination == "Helena"
    assert any(span.role is QuerySpanRole.VIA for span in parsed.spans)
    assert "via" not in (parsed.destination or "").casefold()


@pytest.mark.parametrize(
    "connector",
    [
        "and",
        "or",
        "/",
        "before",
        "after",
        "then",
        "followed by",
        "subsequently",
    ],
)
def test_multiple_episode_candidates_fail_closed(connector: str):
    if connector == "/":
        episode = " during Campaign Gold / Campaign Silver"
    elif connector in {"before", "after", "then", "followed by", "subsequently"}:
        episode = f" during Campaign Gold {connector} Campaign Silver"
    else:
        episode = f" during Campaign Gold {connector} Campaign Silver"
    query = f"Trace Commander Delta from Port A to City B{episode}."
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True
    assert parse_query_scope(query).episode_ambiguous is True


def test_context_episode_does_not_compete_with_primary_silver():
    query = (
        "During Campaign Silver, trace Commander Delta from Port A to City B while "
        "Commander Sigma operated during Campaign Gold."
    )
    scope = _scope(query)
    assert scope.episode == "Campaign Silver"
    assert parse_query_scope(query).episode_ambiguous is False


MATRIX = {
    "route_frames": {
        "single": "Trace Commander Delta from {origin} to {destination}",
        "two_same": "Trace Commander Delta from {origin} to {destination} and from City C to Fort D",
        "two_subjects": "Trace Commander Delta from {origin} to {destination} and Commander Sigma from City C to Fort D",
        "context_route": "Trace Commander Delta from {origin} to {destination} while Commander Sigma moved from City C to Fort D",
        "continuation": "Trace Commander Delta from {origin} to {destination}, then to Fort C",
    },
    "origin": {"ordinary": "Port A", "campaign": "Campaign Hill"},
    "destination": {"ordinary": "City B", "campaign": "Campaign Gate"},
    "via": {"absent": "", "postposed": " via Pass Omega", "interposed": " through Campaign Ridge"},
    "episode": {
        "absent": "",
        "single": " during Campaign Gold",
        "two": " during Campaign Gold then Campaign Silver",
        "context_second": " during Campaign Gold while Commander Sigma operated during Campaign Silver",
    },
}


def _matrix_query(**selection):
    route = MATRIX["route_frames"][selection["route_frames"]].format(
        origin=MATRIX["origin"][selection["origin"]],
        destination=MATRIX["destination"][selection["destination"]],
    )
    via = MATRIX["via"][selection["via"]]
    if via.startswith(" via") or via.startswith(" through"):
        if " to " in route and via.startswith(" via"):
            head, tail = route.rsplit(" to ", 1)
            route = f"{head} to {tail.split('.')[0]}{via}"
        elif via.startswith(" through"):
            head, tail = route.rsplit(" from ", 1)
            origin, rest = tail.split(" to ", 1)
            route = f"{head} from {origin}{via} to {rest}"
    episode = MATRIX["episode"][selection["episode"]]
    return f"{route}{episode}."


@pytest.mark.parametrize(
    "selection",
    [
        {"route_frames": "single", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "single"},
        {"route_frames": "single", "origin": "ordinary", "destination": "ordinary", "via": "postposed", "episode": "absent"},
        {"route_frames": "single", "origin": "campaign", "destination": "ordinary", "via": "interposed", "episode": "single"},
        {"route_frames": "two_same", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "absent"},
        {"route_frames": "two_subjects", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "absent"},
        {"route_frames": "context_route", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "absent"},
        {"route_frames": "continuation", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "absent"},
        {"route_frames": "single", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "two"},
        {"route_frames": "single", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "context_second"},
    ],
)
def test_frame_grouping_matrix(selection: dict):
    query = _matrix_query(**selection)
    scope = _scope(query)
    parsed = parse_query_scope(query)

    if selection["route_frames"] in {"two_same", "two_subjects", "continuation"}:
        assert scope.origin is None
        assert scope.destination is None
    elif selection["route_frames"] == "context_route":
        assert scope.origin == "Port A"
        assert scope.destination == "City B"
    elif selection["route_frames"] == "single":
        assert scope.origin is not None
        assert scope.destination is not None

    if selection["route_frames"] == "two_subjects":
        assert scope.subject is None
    elif selection["route_frames"] in {"single", "context_route", "continuation", "two_same"}:
        assert scope.subject == "Commander Delta"

    if selection["episode"] == "two":
        assert scope.episode is None
        assert scope.has_episode_constraint is True
    elif selection["episode"] == "single":
        assert scope.episode == "Campaign Gold"
    elif selection["episode"] == "context_second":
        assert scope.episode == "Campaign Gold"
        assert parsed.episode_ambiguous is False

    if selection["via"] == "postposed":
        assert any(span.role is QuerySpanRole.VIA for span in parsed.spans)


def test_generated_frame_grouping_samples():
    samples = list(itertools.islice(
        itertools.product(
            ("single", "two_same", "context_route"),
            ("ordinary",),
            ("ordinary",),
            ("absent", "interposed"),
            ("absent", "two"),
        ),
        12,
    ))
    for route_frames, origin, destination, via, episode in samples:
        if via != "absent" and route_frames != "single":
            continue
        query = _matrix_query(
            route_frames=route_frames,
            origin=origin,
            destination=destination,
            via=via,
            episode=episode,
        )
        scope = _scope(query)
        if route_frames in {"two_same", "two_subjects"}:
            assert scope.origin is None
            assert scope.destination is None
        elif route_frames == "context_route":
            assert scope.origin == "Port A"
            assert scope.destination == "City B"
        if episode == "two":
            assert scope.episode is None
