"""Requested mover constraints survive interrogative syntax and parse misses."""
import socket
import pytest
from backend.app.models import AgentState, EventActorStatus
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.routes.query_route_admission import parse_query_route_scope, AuthorityState
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from test_agent_loop import ev, Retriever, call, terminal
from test_v2auth1_strict_endpoint_admission import build, labels, CanonicalPlaces


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Subject tests must not contact providers")
    monkeypatch.setattr(socket.socket, "connect", blocked)


FORMS = (
    "How did {name} move from Rome to Capua",
    "How did {name} travel from Rome to Capua",
    "How did {name} get from Rome to Capua",
    "How did {name} proceed from Rome to Capua",
    "Which route did {name} take from Rome to Capua",
    "What route did {name} use from Rome to Capua",
    "How was {name}'s movement from Rome to Capua conducted",
)
NAMES = ("Marcus", "Aurelius Nestor", "Cassius Varro Minor", "Aelius Dorion",
         "Nikanor Philotas", "Tiberon Melanthes", "Dorieus Cassianus",
         "Dorion", "Philotas", "Neratius")


@pytest.mark.parametrize("template", FORMS)
@pytest.mark.parametrize("name", NAMES)
def test_clear_question_subjects_do_not_require_registry(template, name):
    for punctuation in ("", ".", "?"):
        s = parse_query_route_scope((template.format(name=name) + punctuation,))
        assert s.subject == name and not s.subject_unresolved and not s.subject_ambiguous


@pytest.mark.parametrize("template", FORMS)
@pytest.mark.parametrize("actor,allowed", [("Marcus", True), ("Lucius", False)])
def test_question_actor_admission(template, actor, allowed):
    q = template.format(name="Marcus")
    result, events, admissions = build([f"{actor} marched from Rome to Capua."], q)
    assert events[0].actor.actor_text == actor
    assert events[0].actor.actor_status is EventActorStatus.EXPLICIT
    assert admissions and all(a.subject_match is (AuthorityState.MATCH if allowed else AuthorityState.WRONG)
                              and a.admitted is allowed for a in admissions)
    assert labels(result) == (["Roma", "Capua"] if allowed else [])


def test_where_question_preserves_single_endpoint_subject():
    s = parse_query_route_scope(("Where did Marcus march from Rome?",))
    assert s.subject == "Marcus" and not s.subject_unresolved


@pytest.mark.parametrize("name", NAMES[1:])
def test_unseen_actor_matching_and_wrong_actor(name):
    query = f"How did {name} move from Rome to Capua"
    for actor in (name, "Lucius"):
        result, events, admissions = build([f"{actor} marched from Rome to Capua."], query)
        assert events[0].actor.actor_text == actor
        assert bool(labels(result)) is (actor == name)
        assert all(a.admitted is (actor == name) for a in admissions)


@pytest.mark.parametrize("phrase", ["marcus", "Dr. Marcus", "Marcus 2", "the traveler",
                                   "Marcus and the commander"])
def test_expected_unresolved_subject_is_not_unconstrained(phrase):
    query = f"How did {phrase} move from Rome to Capua"
    s = parse_query_route_scope((query,))
    assert s.subject is None and s.subject_unresolved
    result, _, admissions = build(["Lucius marched from Rome to Capua."], query)
    assert not labels(result)
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted
                              and "QUERY_SUBJECT_UNKNOWN" in a.reason_codes for a in admissions)


@pytest.mark.parametrize("query", ["Show the route from Rome to Capua.",
    "What route connected Rome and Capua?", "Reconstruct movement from Rome to Capua."])
def test_genuine_subjectless_queries_remain_usable(query):
    s = parse_query_route_scope((query,))
    assert s.subject is None and not s.subject_unresolved and not s.subject_ambiguous
    result, _, admissions = build(["Lucius marched from Rome to Capua."], query)
    assert labels(result) == ["Roma", "Capua"]
    assert admissions and all(a.admitted for a in admissions)


@pytest.mark.parametrize("query", ["How did Marcus and Lucius move from Rome to Capua",
    "Which route did Marcus or Pompey take?"])
def test_multi_person_questions_fail_closed(query):
    s = parse_query_route_scope((query,))
    assert s.subject is None and s.subject_unresolved and s.subject_ambiguous
    result, _, admissions = build(["Marcus marched from Rome to Capua."], query)
    assert not labels(result)
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)


@pytest.mark.parametrize("query", ["How did Rome connect to Capua?",
    "How did Rhodes link to the mainland?", "How did Cannae relate to Rome?",
    "How did Rome move from Rome to Capua"])
def test_places_are_not_persons(query):
    s = parse_query_route_scope((query,))
    assert s.subject is None and not s.subject_unresolved


@pytest.mark.parametrize("role", ["army", "consul", "commander"])
def test_roles_do_not_become_named_people(role):
    q = f"How did the {role} move from Rome to Capua"
    s = parse_query_route_scope((q,))
    assert s.subject is None and s.subject_unresolved
    assert not labels(build(["Lucius marched from Rome to Capua."], q)[0])


@pytest.mark.parametrize("query", ["According to Livy, how did Marcus move from Rome to Capua",
    "Polybius asks how Marcus moved from Rome to Capua.",
    "How did Marcus move from Rome to Capua, then summarize the evidence.",
    "How did Marcus move from Rome to Capua, then explain the evidence using Lucius's movement from Capua to Rome."])
def test_source_and_presentation_do_not_replace_mover(query):
    s = parse_query_route_scope((query,))
    assert (s.subject, s.origin, s.destination) == ("Marcus", "Rome", "Capua")
    assert not labels(build(["Lucius marched from Rome to Capua."], query)[0])


def test_unknown_evidence_actor_not_filled_by_query():
    q = "How did Marcus move from Rome to Capua"
    result, events, admissions = build(["He marched from Rome to Capua."], q)
    assert not labels(result)
    assert all(e.actor.actor_status is EventActorStatus.UNKNOWN and e.actor.actor_text != "Marcus" for e in events)
    assert all(a.subject_match is not AuthorityState.MATCH for a in admissions)


def test_query_does_not_change_occurrence_completion_or_mode():
    text = "Lucius marched from Rome to Capua."
    m = HistoricalPlaceMentionExtractor()
    claims = m.movement_claims([ev("p", text)], event_id="p")
    assert len(claims) == 1 and claims[0].source_place == "Roma" and claims[0].destination_place == "Capua"
    assert not labels(build([text], "How did Marcus move from Rome to Capua")[0])
    incomplete = "Marcus attempted to reach Capua."
    assert not labels(build([incomplete], "How did Marcus move from Rome to Capua")[0])
    extractor = EvidenceGroundedHistoricalEventExtractor()
    evidence = [ev("p", text)]
    before, _ = extractor.extract(evidence)
    rejected, _ = extractor.extract(evidence, query="How did Marcus move from Rome to Capua")
    assert rejected == []
    after, _ = extractor.extract(evidence, query="How did Lucius move from Rome to Capua")
    assert [e.actor.model_dump() for e in before] == [e.actor.model_dump() for e in after]
    assert [e.source_statements for e in before] == [e.source_statements for e in after]


def test_question_cannot_resolve_source_endpoint_alternatives():
    q = "How did Marcus move from Rome to Capua?"
    result, _, _ = build(["Marcus marched from Rome to Capua or Neapolis."], q)
    assert not labels(result) and not result.observation_relations


@pytest.mark.parametrize("actor,expected", [("Lucius", []), ("Marcus", ["Roma", "Capua"]), ("He", [])])
def test_exact_ordinary_agent_path(actor, expected):
    q = "How did Marcus move from Rome to Capua"
    a = HistoricalGisAgent(ScriptedLLMProvider([
        call("search_historical_evidence", {"query": q}),
        call("build_historical_route", {"event_id": "subject", "name": "subject", "period": "unspecified"}),
        terminal("The source describes movement.", ["p"]),
    ]), Retriever([ev("p", f"{actor} marched from Rome to Capua.")]), CanonicalPlaces(), max_grounding_corrections=0)
    _, state = a.respond(q, AgentState(session_id="subject2"))
    actual = [p.historical_place.canonical_name for p in state.historical_route.ordered_points] if state.historical_route else []
    assert actual == expected
    assert any(t.tool_name == "build_historical_route" for t in state.tool_history)


def test_unparsed_question_modifier_does_not_remove_person_constraint():
    query = "How did Marcus cautiously navigate from Rome to Capua"
    s = parse_query_route_scope((query,))
    assert s.subject is None and s.subject_unresolved
    result, _, admissions = build(["Lucius marched from Rome to Capua."], query)
    assert not labels(result)
    assert admissions and all(a.subject_match is AuthorityState.UNKNOWN and not a.admitted for a in admissions)
