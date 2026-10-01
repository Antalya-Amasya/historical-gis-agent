"""Negative terminal information survives without licensing positive routes."""
from time import perf_counter
import pytest

from backend.app.agent.evidence_support import render_evidence_citations, validate_evidence_citations
from backend.app.agent.loop import BoundedAgentLoop, NO_ROUTE_TERMINAL_GUARDRAIL, _explicitly_insufficient
from backend.app.models import AgentModelResponse, AgentState
from backend.tests.test_agent_loop import agent, call, ev, terminal
from backend.tests.test_v1_1h2_no_route_terminal_arbitration import _loop, _route_state

NEGATIVES = [
    "The evidence does not allow me to reconstruct the route.",
    "The evidence does not permit reconstruction of the route.",
    "The evidence is too thin to draw a route.",
    "There is not enough evidence to draw a route.",
    "I will not draw a route from this evidence.",
    "The evidence fails to establish the route.",
    "The sources do not mention any route.",
    "The evidence is silent on the route.",
    "No reliable route can be reconstructed.",
    "The route cannot be established from these sources.",
    "The sources do not establish a completed journey.",
    "The movement was prevented before arrival.",
    "The movement was abandoned, so no route can be drawn.",
    "A Roman road existed between the cities, but the sources do not establish that this actor followed it.",
    "The sources mention a possible road, but they do not establish that the actor used it.",
]
POSITIVES = [
    "The route went from A to B.", "The route likely went from A to B.",
    "The actor probably travelled from A to B.", "The route may have passed through C.",
    "The most plausible route is A through C to B.", "The movement was probably by sea.",
    "A route from A to B is likely.", "No route is proven, but the most likely route was A to B.",
    "It cannot be ruled out that the route followed A to B.",
    "It is not impossible that the route went from A to B.",
    "The evidence does not disprove that the route went from A to B.",
    "We cannot exclude a route through C.",
    "I cannot say that the route definitely did not pass through C.",
    "No exact route is known, but it probably followed the coast.",
    "No route can be established from the evidence. However, it probably went from A to B.",
    "The evidence is insufficient, but the route probably went from A to B.",
    "No reliable route can be proven, though the actor most likely travelled by sea.",
    "The geometry follows A → B.",
]
CHINESE_NEGATIVES = ["证据不足以支持完整路线。", "现有史料无法证明这段移动已经完成。",
    "史料表明这次进入行动被阻止，因此不应重建一条已完成路线。",
    "现有材料没有建立从甲地到乙地的完整历史路线。", "无法生成路线。"]


@pytest.mark.parametrize("text", NEGATIVES + CHINESE_NEGATIVES)
def test_safe_negative_polarity_and_exact_terminal_prose(text):
    assert not BoundedAgentLoop._final_answer_asserts_unsupported_route(text)
    reply, state = _loop()._finish(text, _route_state(), perf_counter())
    assert reply == text
    assert state.historical_route is None and state.historical_route_presentation is None


@pytest.mark.parametrize("text", POSITIVES + ["没有确凿路线，但最可能的路线是从甲地经过丙地到乙地。"])
def test_positive_or_speculative_claim_cannot_hide_in_refusal(text):
    source = ev("cited", "A source passage.")
    text += " " + render_evidence_citations((source.id,), [source])
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route(text)
    reply, state = _loop()._finish(text, _route_state(historical_evidence=[source]), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL
    assert state.supporting_evidence == []


INSUFFICIENCY_ANSWERS = [
    "The movement was prevented, so no completed route should be reconstructed.",
    "The sources disagree on whether the army reached Rome, so no route can be drawn.",
    "No reliable route can be produced from these sources.",
    "Evidence is insufficient to build a route.", "The evidence is insufficient.", "无法生成路线。",
]


def insufficient_path(answer, evidence):
    return agent([call("search_historical_evidence", {"query": "Show Caesar route"}),
                  AgentModelResponse(content=answer)], evidence, max_steps=2).respond(
        "Show Caesar route.", AgentState(session_id="negative"))


def negative_source():
    return ev("negative", "The movement was prevented. The sources disagree on whether the army reached Rome.")


@pytest.mark.parametrize("answer", INSUFFICIENCY_ANSWERS)
def test_insufficiency_branch_preserves_safe_grounded_prose(answer):
    source = negative_source()
    reply, state = insufficient_path(answer, [source])
    assert reply == answer
    assert state.final_grounding_status == "grounded"
    assert state.supporting_evidence == []  # No citation was printed.
    assert state.historical_evidence == [source]
    assert state.historical_route is None and state.historical_route_presentation is None


def test_insufficiency_keyword_helper_contract_is_not_broadened():
    assert [_explicitly_insufficient(t) for t in INSUFFICIENCY_ANSWERS] == [False, False, False, True, True, True]


def test_insufficiency_path_preserves_valid_citation_only():
    source = negative_source()
    noise = ev("noise", "Unrelated evidence.")
    answer = INSUFFICIENCY_ANSWERS[1] + " " + render_evidence_citations((source.id,), [source])
    reply, state = insufficient_path(answer, [source, noise])
    assert reply == answer
    assert not validate_evidence_citations(reply, state.historical_evidence, require_citation=True)
    assert state.supporting_evidence == [source]
    assert state.historical_evidence == [source, noise]


@pytest.mark.parametrize("corruption", ["id", "metadata"])
def test_invalid_negative_citation_fails_grounding(corruption):
    source = negative_source()
    citation = render_evidence_citations((source.id,), [source])
    citation = citation.replace("negative", "fabricated-id") if corruption == "id" else citation.replace("Polybius", "Fabricated")
    reply, state = insufficient_path(INSUFFICIENCY_ANSWERS[0] + " " + citation, [source])
    assert citation not in reply
    assert state.supporting_evidence == []
    assert state.final_grounding_status == "guardrail_fallback"


@pytest.mark.parametrize("answer", [POSITIVES[15], POSITIVES[16]])
def test_insufficiency_language_does_not_bypass_positive_guard(answer):
    reply, state = insufficient_path(answer, [negative_source()])
    assert answer not in reply and state.supporting_evidence == []
    assert state.historical_route is None


def test_zero_evidence_does_not_gain_historical_negative_detail():
    reply, state = insufficient_path(INSUFFICIENCY_ANSWERS[0], [])
    assert "movement was prevented" not in reply
    assert "insufficient" in reply
    assert state.supporting_evidence == []


def test_prevented_entry_cited_submission_keeps_negative_history_and_no_geometry():
    source = ev("prevented", "Ariston attempted to enter Port Delta but was prevented.")
    answer = "The movement was prevented, so no completed route should be reconstructed."
    reply, state = agent([terminal(answer, [source.id])], [source]).respond(
        "Reconstruct Ariston's route to Port Delta only if entry completed.", AgentState(session_id="prevented"))
    assert answer in reply
    assert not validate_evidence_citations(reply, [source], require_citation=True)
    assert state.supporting_evidence == [source]
    assert state.historical_route is None and state.historical_route_presentation is None


def test_disagreement_keeps_only_cited_support():
    sources = [ev("a", "Source A indicates movement toward Rome."),
               ev("b", "Source B does not establish arrival.")]
    answer = "The sources disagree on whether the army reached Rome, so no completed route can be reconstructed."
    reply, state = agent([terminal(answer, [s.id for s in sources])], sources).respond(
        "Reconstruct a route only if arrival is established.", AgentState(session_id="disagreement"))
    assert answer in reply and state.supporting_evidence == sources
    assert state.historical_route is None and state.historical_route_presentation is None


def test_provider_error_is_operational_not_historical_insufficiency():
    class BrokenProvider:
        def complete(self, *_):
            raise RuntimeError("synthetic provider failure")
    subject = agent([])
    subject.provider = BrokenProvider()
    reply, state = subject.respond("Show a route.", AgentState(session_id="error"))
    assert state.status == "provider_error"
    assert "provider is unavailable" in reply
    assert state.supporting_evidence == []


def test_tool_failure_is_not_reinterpreted_as_negative_history():
    bad = call("not_allowed", {}, "same")
    reply, state = agent([bad, bad, bad]).respond("x", AgentState(session_id="tool-failure"))
    assert state.status == "tool_failure"
    assert state.historical_route is None and state.supporting_evidence == []


@pytest.mark.parametrize("text", [
    "The evidence is insufficient nevertheless the actor travelled from A to B.",
    "The actor does not hesitate before he travelled from A to B.",
    "The route went from A to B despite the movement being aborted.",
    "Route from A to B.", "Waypoints: A and B.",
])
def test_unrelated_negative_words_do_not_cancel_route_assertions(text):
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route(text)
