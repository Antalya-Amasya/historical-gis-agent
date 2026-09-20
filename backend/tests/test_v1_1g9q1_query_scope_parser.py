"""G9Q1: span-ownership query scope parser."""
from __future__ import annotations

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
from backend.app.routes.query_scope_parser import (
    QuerySpanRole,
    SpanStatus,
    parse_query_scope,
    parse_query_scope_spans,
)
from backend.app.routes.route_observations import ObservationOrderingAuthority
from backend.tests.test_v1_1g9gr2r2_canonical_authority_closure import (
    _event,
    _evidence,
    _observation,
    _relation,
)

EQUIVALENT = QueryRouteScope(
    subject="Commander Alpha",
    origin="Port A",
    destination="City B",
    episode="Campaign Red",
    has_episode_constraint=True,
    has_endpoint_constraint=True,
)
_INCOMPATIBLE = {
    QuerySpanRole.SUBJECT: {QuerySpanRole.ORIGIN, QuerySpanRole.VIA, QuerySpanRole.DESTINATION, QuerySpanRole.EPISODE},
    QuerySpanRole.ORIGIN: {QuerySpanRole.EPISODE, QuerySpanRole.DESTINATION, QuerySpanRole.SUBJECT},
    QuerySpanRole.VIA: {QuerySpanRole.EPISODE, QuerySpanRole.ORIGIN, QuerySpanRole.DESTINATION},
    QuerySpanRole.DESTINATION: {QuerySpanRole.EPISODE, QuerySpanRole.SUBJECT, QuerySpanRole.ORIGIN},
    QuerySpanRole.EPISODE: {QuerySpanRole.ORIGIN, QuerySpanRole.VIA, QuerySpanRole.DESTINATION, QuerySpanRole.SUBJECT},
}


def _assert_no_incompatible_overlap(spans):
    for left in spans:
        for right in spans:
            if left is right:
                continue
            if left.role not in _INCOMPATIBLE.get(right.role, set()):
                continue
            if left.start < right.end and left.end > right.start:
                pytest.fail(f"overlap {left.role}/{right.role}: {left.text!r} vs {right.text!r}")


def _scope(query: str) -> QueryRouteScope:
    return parse_query_route_scope((query,))


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B during Campaign Red.",
        "During Campaign Red, trace Commander Alpha from Port A to City B.",
        "From Port A to City B, trace Commander Alpha during Campaign Red.",
        "Commander Alpha's route from Port A to City B in Campaign Red.",
    ],
)
def test_equivalent_word_orders(query: str):
    scope = _scope(query)
    assert scope == EQUIVALENT


def test_through_campaign_ridge_is_via_not_episode():
    query = (
        "Trace Commander Alpha from Campaign Hill through Campaign Ridge to Fort II "
        "during Campaign Gold."
    )
    parsed = parse_query_scope(query)
    assert parsed.subject == "Commander Alpha"
    assert parsed.origin == "Campaign Hill"
    assert parsed.destination == "Fort II"
    assert parsed.episode == "Campaign Gold"
    via = [span for span in parsed.spans if span.role is QuerySpanRole.VIA]
    assert len(via) == 1
    assert via[0].text == "Campaign Ridge"
    _assert_no_incompatible_overlap(parsed.spans)


@pytest.mark.parametrize(
    ("query", "destination"),
    [
        ("Trace Commander Alpha from Harbor North to Fort VII before meeting General Sigma.", "Fort VII"),
        ("Trace Commander Alpha from Harbor North to Fort VII when General Sigma arrives.", "Fort VII"),
    ],
)
def test_context_tail_does_not_enter_destination(query: str, destination: str):
    scope = _scope(query)
    assert scope.destination == destination
    context = [span for span in parse_query_scope_spans(query) if span.role is QuerySpanRole.CONTEXT]
    assert context


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B during Campaign Gold and Campaign Silver.",
        "Trace Commander Alpha from Port A to City B during Campaign Gold or Campaign Silver.",
        "Trace Commander Alpha from Port A to City B during Campaign Gold / Campaign Silver.",
    ],
)
def test_coordinated_episodes_are_ambiguous(query: str):
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True
    parsed = parse_query_scope(query)
    assert parsed.episode_ambiguous is True


def test_campaign_hill_origin_keeps_episode_red():
    scope = _scope("Trace Commander Alpha from Campaign Hill to Fort II during Campaign Red.")
    assert scope.origin == "Campaign Hill"
    assert scope.destination == "Fort II"
    assert scope.episode == "Campaign Red"


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B and from City C to Fort D.",
        "Trace Commander Alpha from Port A to City B or from City C to Fort D.",
        "Trace Commander Alpha from Port A to City B / from City C to Fort D.",
        "Trace Commander Alpha from Port A to City B, then from City C to Fort D.",
        "Trace Commander Alpha from City C to Fort D and from Port A to City B.",
        "Trace Commander Alpha from Port A to City B and then toward City C.",
    ],
)
def test_competing_or_continuing_route_frames_emit_no_endpoint_authority(query: str):
    scope = _scope(query)
    assert scope.subject == "Commander Alpha"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


def test_multiple_subject_owned_route_frames_emit_no_combined_route_authority():
    scope = _scope(
        "Trace Commander Alpha from Port A to City B and Commander Beta from City C to Fort D."
    )
    assert scope.subject is None
    assert scope.origin is None
    assert scope.destination is None


def test_ambiguous_route_frames_do_not_destroy_confirmed_episode_scope():
    scope = _scope(
        "During Campaign Red, trace Commander Alpha from Port A to City B and from City C to Fort D."
    )
    assert scope.episode == "Campaign Red"
    assert scope.has_episode_constraint is True
    assert scope.origin is None
    assert scope.destination is None


@pytest.mark.parametrize(
    ("origin", "destination"),
    [
        ("Battle Creek", "Fort II"),
        ("Warwick", "Campaign Hill"),
        ("Expedition Bay", "General Pass"),
    ],
)
def test_role_looking_places_stay_endpoints(origin: str, destination: str):
    query = f"Trace movement from {origin} to {destination} during Campaign Red."
    scope = _scope(query)
    assert scope.subject is None
    assert scope.origin == origin
    assert scope.destination == destination
    assert scope.episode == "Campaign Red"


def _admission(query: str, text: str, *, origin: str, destination: str):
    ev = _evidence(text, eid="ev-a")
    event = _event("event-a", text, origin=origin, destination=destination, evidence_id="ev-a")
    observations = [
        _observation("a1", label=origin, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.ORIGIN),
        _observation("a2", label=destination, event_id="event-a", evidence_id="ev-a", place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation("a1", "a2", event_ids=("event-a",), evidence_refs=("ev-a",), authority=ObservationOrderingAuthority.AFTER_SUBORDINATE)
    admission = classify_observation_relation_admission(
        relation, {o.observation_id: o for o in observations}, {event.id: event}, {ev.id: ev}, (query,),
    )
    assembly = assemble_observation_components(observations, [relation], [event], [ev], query_contexts=(query,))
    edge = (relation.earlier_observation_id, relation.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    return admission, component_admitted


def test_dual_episode_query_does_not_admit_red_only_evidence():
    query = "Trace Commander Alpha from Port A to City B during Campaign Gold and Campaign Silver."
    admission, component_admitted = _admission(
        query,
        "During Campaign Gold, Commander Alpha marched from Port A to City B.",
        origin="Port A",
        destination="City B",
    )
    assert admission.episode_match is not AuthorityState.MATCH
    assert not admission.admitted
    assert admission.admitted == component_admitted


MATRIX = {
    "subject": {
        "explicit": "Trace Commander Alpha from {origin} to {destination}{tail}.",
        "possessive": "Commander Alpha's route from {origin} to {destination}{tail}.",
        "absent": "Trace movement from {origin} to {destination}{tail}.",
    },
    "origin": {"ordinary": "Port A", "campaign": "Campaign Hill", "person": "Alexander"},
    "destination": {"ordinary": "City B", "campaign": "Campaign Gate", "person": "Victoria"},
    "via": {"absent": "", "ordinary": " across River X", "campaign": " through Campaign Ridge"},
    "episode": {
        "absent": "",
        "single": " during Campaign Red",
        "and": " during Campaign Red and Campaign Blue",
        "or": " during Campaign Red or Campaign Blue",
        "slash": " during Campaign Red / Campaign Blue",
    },
    "order": {
        "episode-after": "{body}{episode}.",
        "episode-before": "{episode}, {body}.",
    },
    "tail": {
        "none": "",
        "before": " before meeting General Sigma",
        "after": " after the battle",
        "while": " while General Sigma operated nearby",
        "when": " when General Sigma arrives",
    },
}


def _matrix_query(**selection):
    origin = MATRIX["origin"][selection["origin"]]
    destination = MATRIX["destination"][selection["destination"]]
    via = MATRIX["via"][selection["via"]]
    episode = MATRIX["episode"][selection["episode"]]
    tail = MATRIX["tail"][selection["tail"]]
    body = MATRIX["subject"][selection["subject"]].format(origin=origin, destination=destination, tail=via)
    if selection["order"] == "episode-after":
        return f"{body.rstrip('.')}{episode}{tail}."
    if selection["episode"] == "single":
        return f"During Campaign Red, {body.lstrip().rstrip('.')}{tail}."
    return f"{body.rstrip('.')}{episode}{tail}."


@pytest.mark.parametrize(
    "selection",
    [
        {"subject": "explicit", "origin": "campaign", "destination": "ordinary", "via": "campaign", "episode": "single", "order": "episode-after", "tail": "none"},
        {"subject": "absent", "origin": "person", "destination": "person", "via": "absent", "episode": "single", "order": "episode-after", "tail": "none"},
        {"subject": "explicit", "origin": "ordinary", "destination": "campaign", "via": "absent", "episode": "single", "order": "episode-before", "tail": "before"},
        {"subject": "possessive", "origin": "campaign", "destination": "ordinary", "via": "absent", "episode": "and", "order": "episode-after", "tail": "none"},
        {"subject": "explicit", "origin": "ordinary", "destination": "ordinary", "via": "ordinary", "episode": "or", "order": "episode-after", "tail": "when"},
        {"subject": "explicit", "origin": "campaign", "destination": "campaign", "via": "absent", "episode": "slash", "order": "episode-after", "tail": "while"},
    ],
)
def test_compact_combinatorial_matrix(selection: dict):
    query = _matrix_query(**selection)
    parsed = parse_query_scope(query)
    scope = _scope(query)
    _assert_no_incompatible_overlap(parsed.spans)
    if selection["subject"] == "absent":
        assert scope.subject is None
    elif selection["subject"] == "possessive":
        assert scope.subject == "Commander Alpha"
    else:
        assert scope.subject == "Commander Alpha"
    if selection["origin"] == "person":
        assert scope.origin == "Alexander"
    elif selection["origin"] == "campaign":
        assert scope.origin == "Campaign Hill"
    else:
        assert scope.origin == "Port A"
    if selection["episode"] in {"and", "or", "slash"}:
        assert scope.episode is None
        assert scope.has_episode_constraint is True
    elif selection["episode"] == "single":
        assert scope.episode == "Campaign Red"
    if selection["tail"] not in {"none", "when"}:
        assert "meeting" not in (scope.destination or "").casefold()
        assert "battle" not in (scope.destination or "").casefold()
        assert "sigma" not in (scope.destination or "").casefold()


def test_generated_sample_matrix_has_no_unsafe_overlap():
    samples = list(itertools.islice(
        itertools.product(
            ("explicit", "absent"),
            ("ordinary", "campaign"),
            ("ordinary", "campaign"),
            ("absent", "campaign"),
            ("single", "and"),
            ("episode-after",),
            ("none", "before"),
        ),
        12,
    ))
    for subject, origin, destination, via, episode, order, tail in samples:
        query = _matrix_query(
            subject=subject,
            origin=origin,
            destination=destination,
            via=via,
            episode=episode,
            order=order,
            tail=tail,
        )
        parsed = parse_query_scope(query)
        _assert_no_incompatible_overlap(parsed.spans)
