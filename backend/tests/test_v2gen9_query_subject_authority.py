"""Query identity constrains route admission without supplying evidence actors."""
import pytest

from backend.app.agent.evidence_support import assess_evidence_support, extract_subject_terms
from backend.app.agent.loop import infer_requested_output
from backend.app.models import EventActorStatus
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.query_route_admission import (
    AuthorityState, classify_observation_relation_admission, parse_query_route_scope,
)
from backend.tests.test_agent_loop import ev
from backend.tests.test_v2gen8_non_completion_authority import SyntheticPlaces

CAESAR_QUERIES = [
    "Trace Caesar's route from Rome to Capua.",
    "Show Caesar route from Rome to Capua.",
    "展示凯撒相关路线", "凯撒的行军路线",
    "Show the route associated with Caesar.",
    "Reconstruct the movement involving Caesar.",
]


def control(query, actor, context=""):
    evidence = ev("actor-proof", context + f"{actor} marched from Rome to Capua.")
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([evidence])
    before = [event.actor.model_dump() for event in events]
    events, _ = HistoricalEventPlaceResolver(SyntheticPlaces()).resolve(events)
    result = EventAnchorRouteBuilder().build_with_diagnostics(
        events, [evidence], event_id="control", name="control", period="unspecified",
        query_contexts=(query,),
    )
    admissions = [classify_observation_relation_admission(
        relation, {o.observation_id: o for o in result.observations},
        {event.id: event for event in events}, {evidence.id: evidence}, (query,),
    ) for relation in result.observation_relations]
    assert before == [event.actor.model_dump() for event in events]
    return evidence, events, result, admissions


@pytest.mark.parametrize("query", CAESAR_QUERIES)
@pytest.mark.parametrize("actor,allowed", [("Caesar", True), ("Pompey", False)])
def test_requested_actor_authority_survives_query_form(query, actor, allowed):
    evidence, events, result, admissions = control(query, actor, "Caesar remained in the camp. ")
    assert infer_requested_output(query) == "historical_route"
    assert parse_query_route_scope((query,)).subject.casefold() == "caesar"
    assert assess_evidence_support(query, "historical_route", [evidence]).status == "sufficient"
    assert len(events) == 1 and events[0].actor.actor_text == actor
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert admissions and all(a.admitted is allowed for a in admissions)
    assert all(a.subject_match is (AuthorityState.MATCH if allowed else AuthorityState.WRONG) for a in admissions)
    assert (result.route is not None) is allowed


@pytest.mark.parametrize("query,name", [
    ("追踪汉尼拔翻越阿尔卑斯的路线", "Hannibal"),
    ("展示庞培从罗马到布伦迪西姆的路线", "Pompey"),
    ("Show Hannibal route across the Alps.", "Hannibal"),
    ("Display Pompey route to Brundisium.", "Pompey"),
    ("展示波利比乌斯相关路线", "Polybius"),
])
@pytest.mark.parametrize("wrong", [False, True])
def test_other_registry_identities_activate_subject_veto(query, name, wrong):
    actor = "Ariston" if wrong else name
    _, events, result, admissions = control(query, actor)
    assert parse_query_route_scope((query,)).subject.casefold() == name.casefold()
    assert events[0].actor.actor_text == actor
    assert admissions
    assert all(a.subject_match is (AuthorityState.WRONG if wrong else AuthorityState.MATCH) for a in admissions)
    # English endpoint constraints may independently reject these Rome->Capua
    # fixtures; this matrix isolates the requested subject check.
    if wrong:
        assert result.route is None


@pytest.mark.parametrize("query", ["Compare Caesar and Pompey routes.",
                                  "Show the routes of Caesar and Pompey."])
@pytest.mark.parametrize("actor", ["Caesar", "Pompey"])
def test_ambiguous_people_do_not_select_arbitrary_subject(query, actor):
    scope = parse_query_route_scope((query,))
    assert scope.subject is None and scope.subject_ambiguous
    _, events, result, admissions = control(query, actor)
    assert events[0].actor.actor_text == actor
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)
    assert result.route is None


def test_explicit_possessive_subject_beats_contextual_comparison_person():
    query = "Show Caesar's route while comparing it with Pompey."
    assert parse_query_route_scope((query,)).subject == "Caesar"
    assert control(query, "Caesar")[2].route is not None
    assert control(query, "Pompey")[2].route is None


@pytest.mark.parametrize("query", ["Show Rome route to Capua.", "展示罗马到卡普阿的路线",
    "Show the Battle of Cannae route.", "展示坎尼会战相关路线", "Show Gaul route.", "展示阿尔卑斯相关路线"])
def test_places_and_events_do_not_become_people(query):
    scope = parse_query_route_scope((query,))
    assert scope.subject is None and not scope.subject_ambiguous


@pytest.mark.parametrize("query", ["Show the route to Capua described by Caesar.",
    "Show the route recorded by Livy.", "Show the route according to Polybius."])
def test_source_identity_is_not_requested_mover(query):
    scope = parse_query_route_scope((query,))
    assert scope.subject is None and not scope.subject_ambiguous
    assert control(query, "Pompey")[2].route is not None


@pytest.mark.parametrize("actor,allowed", [("Neralis Vexon", True), ("Terenos Qavik", False)])
def test_unknown_literal_subject_stays_investigable_and_checked(actor, allowed):
    query = "Show Neralis Vexon's route from Rome to Capua."
    assert extract_subject_terms(query) == ()
    assert parse_query_route_scope((query,)).subject == "Neralis Vexon"
    evidence, events, result, admissions = control(query, actor)
    assert assess_evidence_support(query, "historical_route", [evidence]).status == "unassessed"
    assert events[0].actor.actor_text == actor
    assert all(a.admitted is allowed for a in admissions)
    assert (result.route is not None) is allowed


@pytest.mark.parametrize("query", ["Who was Caesar?", "Explain Caesar's political role.", "Where was Pompey?"])
def test_non_route_queries_do_not_activate_fallback(query):
    assert infer_requested_output(query) != "historical_route"
    scope = parse_query_route_scope((query,))
    assert scope.subject is None and not scope.subject_ambiguous


def test_identity_alias_is_compatibility_only():
    query = "展示凯撒相关路线"
    _, events, result, admissions = control(query, "Julius Caesar")
    assert events[0].actor.actor_text == "Julius Caesar"
    assert result.route is not None and all(a.subject_match is AuthorityState.MATCH for a in admissions)


def test_alias_substring_does_not_invent_a_person():
    assert parse_query_route_scope(("Show Caesarea route.",)).subject is None


def test_canonical_query_alone_defines_fallback_subject():
    scope = parse_query_route_scope(("Show Caesar route.", "Pompey movement"))
    assert scope.subject == "caesar" and not scope.subject_ambiguous
