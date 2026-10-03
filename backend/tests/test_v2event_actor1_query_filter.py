"""Explicit actor authority precedes lexical, geographic and retrieval context."""
import pytest
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentState, EventActorStatus
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor as Extractor
from backend.app.routes.query_route_admission import AuthorityState, parse_query_route_scope
from backend.tests.test_agent_loop import ev, Retriever, call, terminal
from backend.tests.test_v2auth1_strict_endpoint_admission import CanonicalPlaces

QUERY = "Marcus military actions in Gaul"


def extract(text, query=QUERY):
    return Extractor().extract([ev("actor-filter", text)], query=query)[0]


@pytest.mark.parametrize("text", [
    "Lucius captured Athens in Gaul.",
    "Lucius captured Alesia in Gaul.",
    "Lucius marched through Gaul.",
    "Lucius besieged a Gallic city.",
    "Lucius fought at Alesia in Gaul.",
    "Lucius sailed from Rome to Capua in Gaul.",
    "Lucius departed from Rome in Gaul.",
    "Lucius arrived at Capua in Gaul.",
    "Lucius was elected consul in Gaul.",
    "Lucius issued a decree in Gaul.",
])
def test_explicit_wrong_actor_veto_precedes_region_and_action(text):
    before = extract(text, None)
    assert before and before[0].actor.actor_text == "Lucius"
    assert extract(text) == []


def test_same_action_cannot_rescue_wrong_actor():
    assert extract("Lucius captured Alesia in Gaul.", "Marcus captured cities in Gaul") == []


@pytest.mark.parametrize("actor", ["Marcus", "Aelius Dorion", "Tiberon Melanthes", "Marcus Tullianus Severus", "Dorion", "Philotas"])
def test_fresh_matching_actor_is_preserved_without_query_mutation(actor):
    text = f"{actor} captured Olynthara in Meridia."
    query = f"{actor} military actions in Meridia"
    assert parse_query_route_scope((query,)).subject == actor
    before = extract(text, None)
    after = extract(text, query)
    assert before and after
    assert [e.model_dump() for e in after] == [e.model_dump() for e in before]


@pytest.mark.parametrize("query_actor,event_actor", [
    ("Aelius Dorion", "Nikanor Philotas"),
    ("Tiberon Melanthes", "Dorieus Cassianus"),
    ("Dorion", "Philotas"),
    ("Marcus Tullianus Severus", "Nikanor Philotas"),
])
def test_unseen_distinct_actors(query_actor, event_actor):
    text = f"{event_actor} captured Olynthara in Meridia."
    assert extract(text, None)[0].actor.actor_text == event_actor
    assert extract(text, f"{query_actor} military actions in Meridia") == []


@pytest.mark.parametrize("actor", ["He", "The army", "The commander", "The consul", "Marcus and Lucius", "Marcus or Lucius"])
def test_unknown_role_and_compound_actors_are_not_query_filled(actor):
    text = f"{actor} captured Alesia in Gaul."
    before, after = extract(text, None), extract(text)
    assert before and after
    assert all(e.actor.actor_status is EventActorStatus.UNKNOWN and e.actor.actor_text is None for e in after)
    assert [e.model_dump() for e in after] == [e.model_dump() for e in before]
    assert Extractor._query_actor_compatibility(after[0].actor, (QUERY,)) is AuthorityState.UNKNOWN


@pytest.mark.parametrize("place", ["Rhodes", "Cannae", "Gaul", "Rome"])
def test_known_place_cannot_be_actor(place):
    query = f"{place} military actions in Gaul"
    assert parse_query_route_scope((query,)).subject is None
    events = extract(f"{place} captured Alesia in Gaul.", None)
    assert events and all(e.actor.actor_status is EventActorStatus.UNKNOWN for e in events)


def test_generic_query_retains_existing_actor_unconstrained_behavior():
    text = "Marcus captured Alesia. Lucius marched through Gaul."
    events = extract(text, "Military actions in Gaul")
    assert {e.actor.actor_text for e in events} == {"Marcus", "Lucius"}


def test_same_parent_and_retrieval_hint_cannot_override_canonical_actor():
    text = "Marcus captured Alesia in Gaul. Lucius captured Athens in Gaul."
    events, _ = Extractor().extract([ev("parent", text)], query_contexts=(QUERY, "Lucius captured Athens in Gaul"))
    assert events and all(e.actor.actor_text == "Marcus" for e in events)
    assert all(e.evidence_refs == ["parent"] for e in events)


@pytest.mark.parametrize("text,actor", [
    ("According to Livy, Marcus captured Alesia in Gaul.", "Marcus"),
    ("Polybius says Lucius captured Alesia in Gaul.", "Lucius"),
    ("Marcus said Lucius captured Alesia in Gaul.", "Lucius"),
])
def test_source_and_matrix_subjects_do_not_replace_local_actor(text, actor):
    events = extract(text, None)
    assert events and events[0].actor.actor_text == actor
    assert bool(extract(text)) == (actor == "Marcus")


@pytest.mark.parametrize("text", ["Lucius fought near Alesia.", "Lucius planned to attack Alesia in Gaul.", "Lucius was prevented from marching to Capua in Gaul."])
def test_unsupported_or_nonasserted_forms_do_not_bypass_veto(text):
    assert extract(text) == []


def test_possible_abbreviation_is_unknown_not_match():
    event = extract("Quintus Fabius Pictor captured Alesia in Gaul.", None)[0]
    assert Extractor._query_actor_compatibility(event.actor, ("Pictor military actions in Gaul",)) is AuthorityState.UNKNOWN
    assert Extractor._query_actor_compatibility(event.actor, ("Lucius military actions in Gaul",)) is AuthorityState.WRONG


def test_question_form_wrong_actor_is_filtered():
    assert extract("Lucius marched from Rome to Capua.", "How did Marcus move from Rome to Capua?") == []


@pytest.mark.parametrize("actor,expected", [("Lucius", []), ("Marcus", ["Marcus"])])
def test_ordinary_agent_search_filters_structured_events_before_terminal(actor, expected):
    agent = HistoricalGisAgent(ScriptedLLMProvider([
        call("search_historical_evidence", {"query": QUERY}),
        terminal("The source describes an event.", ["p"]),
    ]), Retriever([ev("p", f"{actor} captured Athens in Gaul.")]), CanonicalPlaces(), max_grounding_corrections=0)
    _, state = agent.respond(QUERY, AgentState(session_id="event-actor-filter"))
    assert [e.actor.actor_text for e in state.historical_events] == expected
    assert any(t.tool_name == "search_historical_evidence" for t in state.tool_history)


@pytest.mark.parametrize("query", ["Marcus's military actions in Gaul", "Marcus’s military actions in Gaul"])
def test_possessive_event_subject_is_clean(query):
    assert parse_query_route_scope((query,)).subject == "Marcus"
    assert extract("Marcus captured Alesia in Gaul.", query)
    assert extract("Lucius captured Alesia in Gaul.", query) == []
