"""Structured non-completion is visible to answers without entering route authority."""
import copy
import json
import pytest
from backend.app.agent.tools import AgentToolRegistry, incomplete_movement_answer_context
from backend.app.agent.loop import _model_result
from backend.app.models import AgentState, AgentModelResponse
from backend.tests.test_agent_loop import Retriever, ev, agent, call, terminal
from backend.tests.test_v2gen8_non_completion_authority import SyntheticPlaces

CONTROLS = [
 ("Marcus attempted to enter City Alpha but was prevented.", "PREVENTED"),
 ("Marcus advanced toward City Alpha but stopped before reaching it.", "ABORTED"),
 ("Marcus planned to march to City Alpha.", "PLANNED"),
 ("Marcus did not enter City Alpha.", "NEGATED")]
QUERY = "Trace Marcus's route to City Alpha."

def search(text, items=None):
    source = ev("negative", text)
    state = AgentState(session_id="context", user_query=QUERY, requested_output="historical_route")
    tools = AgentToolRegistry(Retriever(items or [source]), SyntheticPlaces())
    payload, _ = tools.execute("search_historical_evidence", {"query": QUERY}, state)
    return source, state, tools, payload

@pytest.mark.parametrize("text,outcome", CONTROLS)
def test_structured_visibility_has_no_completed_authority(text, outcome):
    source, state, tools, payload = search(text)
    context = payload["result"]["incomplete_movement_facts"]
    assert context == _model_result("search_historical_evidence", payload, state)["incomplete_movement_facts"]
    assert len(context) == 1 and context[0]["outcome"] == outcome
    assert context[0]["actor"] == {"actor_text": "Marcus", "actor_status": "EXPLICIT"}
    assert context[0]["evidence_refs"] == [source.id]
    assert context[0]["context_role"] == "NON_COMPLETION_EXPLANATION_ONLY"
    assert all(m["role"] == "UNKNOWN" for field in ["destination_mentions", "context_place_mentions"] for m in context[0][field])
    assert not state.historical_events and not state.supporting_evidence
    before = copy.deepcopy(state.historical_event_diagnostics)
    tools.execute("build_historical_route", {"event_id":"a", "name":"a", "period":"unspecified"}, state)
    assert state.historical_event_diagnostics == before
    assert state.historical_route is None and state.historical_route_presentation is None
    assert state.historical_route_diagnostics["observation_count"] == 0
    assert state.historical_route_diagnostics["transition_constraint_count"] == 0

@pytest.mark.parametrize("text,outcome", CONTROLS)
def test_bootstrap_context_reaches_exact_provider_messages(text, outcome):
    subject = agent([AgentModelResponse(content="No completed route can be reconstructed.")], [ev("negative", text)], max_steps=1)
    _, state = subject.respond(QUERY, AgentState(session_id="bootstrap"))
    messages = subject.provider.requests[0]["messages"]
    message = next(m for m in messages if '"incomplete_movement_facts"' in m["content"])
    context = json.loads(message["content"][message["content"].index('{'):])["incomplete_movement_facts"]
    assert context[0]["outcome"] == outcome
    assert "explanation only" in message["content"]
    assert not any('"incomplete_movement_facts"' in m["content"] for m in state.messages)


def test_explicit_search_context_reaches_next_provider_call():
    text = CONTROLS[0][0]
    subject = agent([call("search_historical_evidence", {"query":"Marcus"}), AgentModelResponse(content="The source records a prevented attempt.")], [ev("negative", text)], max_steps=2)
    subject.respond("Explain Marcus.", AgentState(session_id="explicit"))
    messages = subject.provider.requests[1]["messages"]
    tool = next(json.loads(m["content"]) for m in messages if m["role"] == "tool")
    assert tool["result"]["incomplete_movement_facts"][0]["outcome"] == "PREVENTED"


def test_mixed_prevented_and_completed_clauses_remain_separate():
    _, state, _, payload = search("Marcus was prevented from entering City Alpha, but later entered City Alpha.")
    fact = payload["result"]["incomplete_movement_facts"][0]
    assert fact["outcome"] == "PREVENTED" and "later entered" not in fact["source_statement"]
    assert len(state.historical_events) == 1
    assert "later entered" in state.historical_events[0].summary
    assert state.historical_events[0].actor.actor_text is None

@pytest.mark.parametrize("completion,expected", [("Bion later entered City Alpha.", None), ("Bion entered City Alpha.", "Bion")])
def test_mixed_actors_are_not_propagated(completion, expected):
    _, state, _, payload = search(CONTROLS[1][0] + " " + completion)
    assert payload["result"]["incomplete_movement_facts"][0]["actor"]["actor_text"] == "Marcus"
    assert len(state.historical_events) == 1 and state.historical_events[0].actor.actor_text == expected


def test_query_does_not_manufacture_negative_facts():
    state = AgentState(session_id="query", user_query="Was Marcus prevented from entering City Alpha?")
    tools = AgentToolRegistry(Retriever([ev("ordinary", "Marcus remained in camp.")]), SyntheticPlaces())
    payload, _ = tools.execute("search_historical_evidence", {"query":state.user_query}, state)
    assert payload["result"]["incomplete_movement_facts"] == []

@pytest.mark.parametrize("cited", [False, True])
def test_negative_answer_and_citation_support_contract(cited):
    source = ev("negative", CONTROLS[0][0])
    answer = "The source records an attempted entry that was prevented; no completed arrival is supported."
    response = terminal(answer, [source.id]) if cited else AgentModelResponse(content=answer)
    subject = agent([response], [source], max_steps=1)
    reply, state = subject.respond(QUERY, AgentState(session_id="citation"))
    assert answer in reply and state.historical_route is None
    assert state.supporting_evidence == ([source] if cited else [])
    assert not any("unsupported_route" in w for w in state.warnings)


def test_context_is_not_limited_to_first_eight_excerpt_windows():
    source = ev("negative", "Background " * 70 + ". " + CONTROLS[0][0])
    items = [ev(str(i), "The senate met.") for i in range(8)] + [source]
    _, state, _, payload = search(source.text, items)
    model = _model_result("search_historical_evidence", payload, state)
    assert source.id not in [e["id"] for e in model["evidence"]]
    assert model["incomplete_movement_facts"][0]["outcome"] == "PREVENTED"


def test_empty_and_stale_context_are_not_exposed():
    _, state, _, _ = search(CONTROLS[0][0])
    state.historical_evidence = []
    assert incomplete_movement_answer_context(state) == []
    subject = agent([AgentModelResponse(content="The evidence is insufficient.")], [])
    subject.respond("Explain an unrelated matter.", state)
    assert state.historical_event_diagnostics is None
    assert not any('"incomplete_movement_facts"' in m["content"] for m in subject.provider.requests[0]["messages"])
    assert incomplete_movement_answer_context(AgentState(session_id="empty")) == []


def test_context_is_bounded_deduplicated_and_does_not_mutate_state():
    _, state, _, _ = search(CONTROLS[0][0])
    original = state.historical_event_diagnostics["extraction"]["incomplete_movement_facts"][0]
    facts = [copy.deepcopy(original), copy.deepcopy(original)]
    for i in range(12):
        item = copy.deepcopy(original)
        item["source_statement"] = str(i) + "x" * 1000
        item["destination_mentions"] *= 20
        facts.append(item)
    state.historical_event_diagnostics["extraction"]["incomplete_movement_facts"] = facts
    before = copy.deepcopy(facts)
    visible = incomplete_movement_answer_context(state)
    assert len(visible) == 8 and sum(v["source_statement"] == original["source_statement"] for v in visible) == 1
    assert all(len(v["source_statement"]) <= 500 and len(v["destination_mentions"]) <= 8 for v in visible)
    assert facts == before and state.supporting_evidence == []
