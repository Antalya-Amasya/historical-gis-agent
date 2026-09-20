"""G9Q1-R3: context authority isolation."""
from __future__ import annotations

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

TERRA_QUERY = (
    "Show Commander Delta's route while Commander Sigma moved from Station 12 to Gate Q "
    "during Campaign Gold."
)
PRIMARY_BEFORE_CONTEXT = (
    "During Campaign Silver, show Commander Delta's route while Commander Sigma moved "
    "during Campaign Gold."
)
LEADING_CONTEXT = (
    "While Commander Sigma moved during Campaign Gold, trace Commander Delta from Harbor "
    "North to Fort VII during Campaign Silver."
)
PRIMARY_ROUTE_CONTEXT_EPISODE = (
    "Trace Commander Delta from Harbor North to Fort VII while Commander Sigma operated "
    "during Campaign Gold."
)
CONTEXT_ROUTE_EPISODE = (
    "Show Commander Delta's route while Commander Sigma moved through Campaign Ridge "
    "during Campaign Gold."
)


def _scope(query: str):
    return parse_query_route_scope((query,))


def _admission(
    query: str,
    text: str,
    *,
    origin: str | None = None,
    destination: str | None = None,
    actor: str = "Commander Delta",
):
    ev = _evidence(text, eid="ev-a")
    event = _event(
        "event-a",
        text,
        origin=origin or "Harbor North",
        destination=destination or "Fort VII",
        evidence_id="ev-a",
    )
    observations = [
        _observation(
            "a1",
            label=event.place_bindings[0].mention.raw_text,
            event_id="event-a",
            evidence_id="ev-a",
            place_role=EventPlaceRole.ORIGIN,
            actor_text=actor,
        ),
        _observation(
            "a2",
            label=event.place_bindings[1].mention.raw_text,
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


def test_terra_reproduction_has_no_primary_episode():
    scope = _scope(TERRA_QUERY)
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.episode is None
    assert scope.has_episode_constraint is False


def test_terra_reproduction_has_no_false_gold_admission():
    scope = _scope(TERRA_QUERY)
    admission, component_admitted = _admission(
        TERRA_QUERY,
        "During Campaign Gold, Commander Delta marched from Harbor North to Fort VII.",
        origin="Harbor North",
        destination="Fort VII",
    )
    assert not scope.has_episode_constraint
    assert admission.episode_match is not AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.UNKNOWN
    assert admission.admitted == component_admitted


def test_primary_episode_before_context():
    scope = _scope(PRIMARY_BEFORE_CONTEXT)
    assert scope.subject == "Commander Delta"
    assert scope.episode == "Campaign Silver"
    assert scope.has_episode_constraint is True
    assert parse_query_scope(PRIMARY_BEFORE_CONTEXT).episode_ambiguous is False


def test_primary_episode_after_leading_context():
    scope = _scope(LEADING_CONTEXT)
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Harbor North"
    assert scope.destination == "Fort VII"
    assert scope.episode == "Campaign Silver"
    assert scope.has_episode_constraint is True


def test_primary_route_with_context_episode():
    scope = _scope(PRIMARY_ROUTE_CONTEXT_EPISODE)
    assert scope.subject == "Commander Delta"
    assert scope.origin == "Harbor North"
    assert scope.destination == "Fort VII"
    assert scope.episode is None
    assert scope.has_episode_constraint is False


def test_context_route_and_episode_do_not_donate_primary_authority():
    scope = _scope(CONTEXT_ROUTE_EPISODE)
    assert scope.subject == "Commander Delta"
    assert scope.origin is None
    assert scope.destination is None
    assert scope.episode is None
    assert scope.has_episode_constraint is False


@pytest.mark.parametrize(
    "marker",
    ["while", "when", "because", "although"],
)
def test_context_markers_isolate_nested_episodes(marker: str):
    query = (
        f"Trace Commander Delta from Harbor North to Fort VII {marker} Commander Sigma "
        "operated during Campaign Gold."
    )
    scope = _scope(query)
    assert scope.episode is None
    assert scope.has_episode_constraint is False
    context = [span for span in parse_query_scope_spans(query) if span.role is QuerySpanRole.CONTEXT]
    assert context


def test_primary_episode_before_context_clause():
    scope = _scope("During Campaign Gold, trace Commander Delta from Port A to City B while Sigma waited.")
    assert scope.episode == "Campaign Gold"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"


def test_primary_multi_episode_still_ambiguous():
    scope = _scope("Trace Commander Delta from Port A to City B during Campaign Gold and Campaign Silver.")
    assert scope.episode is None
    assert scope.has_episode_constraint is True


def test_primary_silver_with_context_gold_is_not_ambiguous():
    scope = _scope(PRIMARY_BEFORE_CONTEXT)
    assert scope.episode == "Campaign Silver"
    assert parse_query_scope(PRIMARY_BEFORE_CONTEXT).episode_ambiguous is False
