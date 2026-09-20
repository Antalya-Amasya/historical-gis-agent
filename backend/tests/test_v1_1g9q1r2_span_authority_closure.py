"""G9Q1-R2: span authority closure for local reconciliation gaps."""
from __future__ import annotations

import itertools

import pytest

from backend.app.models import EventPlaceRole
from backend.app.routes.observation_components import assemble_observation_components
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

RIDGE_GOLD_QUERY = (
    "Trace Commander Delta from Harbor North through Campaign Ridge to Fort VII "
    "during Campaign Gold."
)
DELTA_WHILE_SIGMA_QUERY = (
    "Show Commander Delta's route while Commander Sigma moved from Station 12 to Gate Q."
)
DELTA_ROUTE_WHILE_SIGMA_AT_GATE = (
    "Trace Commander Delta from Harbor North to Fort VII while Commander Sigma remained at Gate Q."
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
    obs_by = {item.observation_id: item for item in observations}
    relation = _relation(
        "a1",
        "a2",
        event_ids=("event-a",),
        evidence_refs=("ev-a",),
        authority=ObservationOrderingAuthority.AFTER_SUBORDINATE,
    )
    admission = classify_observation_relation_admission(
        relation, obs_by, {event.id: event}, {ev.id: ev}, (query,),
    )
    assembly = assemble_observation_components(
        observations, [relation], [event], [ev], query_contexts=(query,),
    )
    edge = (relation.earlier_observation_id, relation.later_observation_id)
    component_admitted = any(edge in component.relation_ids for component in assembly.components)
    return admission, component_admitted


def test_via_campaign_ridge_does_not_become_query_episode():
    parsed = parse_query_scope(RIDGE_GOLD_QUERY)
    assert parsed.episode == "Campaign Gold"
    via = [span for span in parsed.spans if span.role is QuerySpanRole.VIA]
    assert len(via) == 1
    assert via[0].text == "Campaign Ridge"


def test_gold_evidence_may_match_gold_episode_query():
    admission, component_admitted = _admission(
        RIDGE_GOLD_QUERY,
        "During Campaign Gold, Commander Delta marched from Harbor North to Fort VII.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert admission.episode_match is AuthorityState.MATCH
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


def test_context_route_frame_does_not_contaminate_primary_subject():
    scope = _scope(DELTA_WHILE_SIGMA_QUERY)
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is False


def test_primary_route_preserved_when_context_has_no_competing_frame():
    scope = _scope(DELTA_ROUTE_WHILE_SIGMA_AT_GATE)
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Harbor North"
    assert scope.destination == "Fort VII"


@pytest.mark.parametrize(
    "connector",
    [
        "and",
        "or",
        "/",
        "versus",
        "before",
    ],
)
def test_competing_episode_connectors_are_ambiguous(connector: str):
    if connector == "/":
        episode = " during Campaign Gold / Campaign Silver"
    elif connector == "before":
        episode = " during Campaign Gold before Campaign Silver"
    else:
        episode = f" during Campaign Gold {connector} Campaign Silver"
    query = f"Trace Commander Delta from Port A to City B{episode}."
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True
    assert parse_query_scope(query).episode_ambiguous is True


def test_repeated_during_episodes_are_ambiguous():
    query = "Trace Commander Delta from Port A to City B during Campaign Gold, later during Campaign Silver."
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is True


@pytest.mark.parametrize(
    ("tail", "destination"),
    [
        ("until Commander Sigma arrived", "Fort VII"),
        ("upon reaching the coast", "Fort VII"),
        ("subsequently joining Sigma", "Fort VII"),
        ("after the battle ended", "Fort VII"),
        ("because Sigma diverted north", "Fort VII"),
        ("although Sigma remained behind", "Fort VII"),
        ("following the ceasefire", "Fort VII"),
        ("prior to Sigma's arrival", "Fort VII"),
        ("once Sigma departed", "Fort VII"),
        ("afterward linking with Sigma", "Fort VII"),
    ],
)
def test_context_tails_do_not_enter_destination(tail: str, destination: str):
    query = f"Trace Commander Delta from Harbor North to Fort VII {tail}."
    scope = _scope(query)
    assert scope.destination == destination
    context = [span for span in parse_query_scope_spans(query) if span.role is QuerySpanRole.CONTEXT]
    assert context


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B and onward into Fort C.",
        "Trace Commander Delta from Port A to City B before proceeding to Fort C.",
        "Trace Commander Delta from Port A to City B later moving from Fort C to Gate D.",
    ],
)
def test_partial_continuations_clear_endpoint_authority(query: str):
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None
    assert scope.has_endpoint_constraint is True


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Delta from Port A to City B and from City C to Fort D.",
        "Trace Commander Delta from Port A to City B or from City C to Fort D.",
        "Trace Commander Delta from Port A to City B / from City C to Fort D.",
        "Trace Commander Delta from Port A to City B, then from City C to Fort D.",
    ],
)
def test_multiple_route_frames_remain_unsupported(query: str):
    scope = _scope(query)
    assert scope.origin is None
    assert scope.destination is None


MATRIX = {
    "subject": {
        "explicit": "Trace Commander Delta from {origin}{via} to {destination}{tail}.",
        "context_actor": (
            "Show Commander Delta's route while Commander Sigma moved from Station 12 to Gate Q{tail}."
        ),
    },
    "origin": {"ordinary": "Port A", "campaign": "Campaign Hill"},
    "destination": {"ordinary": "City B", "campaign": "Campaign Gate"},
    "via": {"absent": "", "campaign": " through Campaign Ridge"},
    "episode": {
        "absent": "",
        "single": " during Campaign Gold",
        "and": " during Campaign Gold and Campaign Silver",
        "or": " during Campaign Gold or Campaign Silver",
        "slash": " during Campaign Gold / Campaign Silver",
        "versus": " during Campaign Gold versus Campaign Silver",
        "before": " during Campaign Gold before Campaign Silver",
    },
    "tail": {
        "none": "",
        "until": " until Commander Sigma arrived",
        "upon": " upon reaching the coast",
        "subsequently": " subsequently joining Sigma",
        "while": " while Commander Sigma operated nearby",
        "because": " because Commander Sigma diverted",
        "although": " although Commander Sigma remained behind",
        "after": " after the battle ended",
        "before_tail": " before meeting Commander Sigma",
    },
}


def _matrix_query(**selection):
    if selection["subject"] == "context_actor":
        return MATRIX["subject"]["context_actor"].format(tail=MATRIX["tail"][selection["tail"]])
    origin = MATRIX["origin"][selection["origin"]]
    destination = MATRIX["destination"][selection["destination"]]
    via = MATRIX["via"][selection["via"]]
    episode = MATRIX["episode"][selection["episode"]]
    tail = MATRIX["tail"][selection["tail"]]
    body = MATRIX["subject"]["explicit"].format(
        origin=origin,
        destination=destination,
        via=via,
        tail="",
    )
    return f"{body.rstrip('.')}{episode}{tail}."


@pytest.mark.parametrize(
    "selection",
    [
        {"subject": "explicit", "origin": "campaign", "destination": "ordinary", "via": "campaign", "episode": "single", "tail": "none"},
        {"subject": "explicit", "origin": "ordinary", "destination": "ordinary", "via": "campaign", "episode": "single", "tail": "until"},
        {"subject": "context_actor", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "absent", "tail": "none"},
        {"subject": "explicit", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "and", "tail": "subsequently"},
        {"subject": "explicit", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "versus", "tail": "upon"},
        {"subject": "explicit", "origin": "ordinary", "destination": "ordinary", "via": "absent", "episode": "before", "tail": "because"},
        {"subject": "explicit", "origin": "ordinary", "destination": "campaign", "via": "absent", "episode": "or", "tail": "although"},
        {"subject": "explicit", "origin": "campaign", "destination": "ordinary", "via": "absent", "episode": "slash", "tail": "after"},
    ],
)
def test_expanded_combinatorial_matrix_has_no_unsafe_false_authority(selection: dict):
    query = _matrix_query(**selection)
    scope = _scope(query)
    parsed = parse_query_scope(query)

    if selection["subject"] == "context_actor":
        assert scope.subject == "Commander Delta"
        assert scope.origin is None
        assert scope.destination is None
    elif selection["via"] == "campaign":
        assert parsed.spans and any(span.role is QuerySpanRole.VIA for span in parsed.spans)

    if selection["episode"] in {"and", "or", "slash", "versus", "before"}:
        assert scope.episode is None
        assert scope.has_episode_constraint is True
    elif selection["episode"] == "single":
        assert scope.episode == "Campaign Gold"

    if selection["tail"] not in {"none"} and selection["subject"] != "context_actor":
        assert "sigma" not in (scope.destination or "").casefold()
        assert "battle" not in (scope.destination or "").casefold()
        assert "coast" not in (scope.destination or "").casefold()


def test_generated_matrix_sample_has_no_unsafe_overlap():
    samples = list(itertools.islice(
        itertools.product(
            ("ordinary", "campaign"),
            ("ordinary",),
            ("absent", "campaign"),
            ("single", "before", "versus"),
            ("none", "until", "subsequently"),
        ),
        18,
    ))
    for origin, destination, via, episode, tail in samples:
        query = _matrix_query(
            subject="explicit",
            origin=origin,
            destination=destination,
            via=via,
            episode=episode,
            tail=tail,
        )
        scope = _scope(query)
        if episode in {"before", "versus"}:
            assert scope.episode is None
            assert scope.has_episode_constraint is True
        if tail != "none":
            assert "sigma" not in (scope.destination or "").casefold()
