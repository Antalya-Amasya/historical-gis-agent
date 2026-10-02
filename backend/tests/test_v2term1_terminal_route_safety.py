"""State-grounded terminal propositions; fixtures make no historical truth claims."""
from copy import deepcopy
from time import perf_counter

import pytest

from backend.app.agent.loop import BoundedAgentLoop, NO_ROUTE_TERMINAL_GUARDRAIL
from backend.app.agent.evidence_support import validate_evidence_citations
from backend.app.models import AgentState
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_agent_loop import agent, ev, terminal
from backend.tests.test_v1_1h2_no_route_terminal_arbitration import _loop, _route_state
from backend.tests.test_g4d_route_preservation import structured_route

SOURCES = {
    "ABORTED": "Marcus advanced toward Capua but stopped before reaching it.",
    "PREVENTED": "Marcus attempted to enter Capua but was prevented.",
    "NEGATED": "Marcus did not enter Capua.",
    "PLANNED": "Marcus planned to march to Capua.",
}
NEGATIVES = (
    ("ABORTED", "The journey appears to have been abandoned before the destination was reached."),
    ("ABORTED", "The movement ended before arrival."),
    ("PREVENTED", "The force was prevented from entering the city."),
    ("PLANNED", "The sources describe an intended march, not a completed one."),
    ("NEGATED", "The text explicitly states that he did not reach the destination."),
    ("ABORTED", "The expedition seems to have failed before arrival."),
    ("ABORTED", "The force appears not to have completed the journey."),
    ("ABORTED", "The movement seems to have ended before arrival."),
    ("ABORTED", "史料表明这次行程在抵达目的地前已经中止。"),
    ("PLANNED", "资料只支持“计划前往”，不能证明实际抵达。"),
)
SMUGGLING = (
    "The sources do not prove a route; nevertheless, the most plausible path was through C.",
    "The evidence is insufficient, but the likely route was A → C → B.",
    "The route is uncertain; however, he probably travelled via C.",
    "The sources cannot establish the route, though the most plausible path is through C.",
    "The sources do not say where he went, yet he may have travelled through C.",
    "The sources do not prove a route. The most plausible path was through C.",
    "There is insufficient evidence for the exact route. He probably travelled through C.",
    "Although the surviving sources do not preserve an exact itinerary, the most likely route would have passed through C.",
    "Despite the uncertainty, he probably moved from A to B via C.",
    "The route is unknown; however, probably via C.",
    "The sources do not establish the route; nevertheless the path would be through C.",
    "史料无法证明具体路线，但最可能是从甲地经丙地到乙地。",
    "路线不确定，不过大概经丙地到乙地。",
    "史料无法确定路线。然而他可能从甲地到乙地。",
)


def negative_state(kind):
    source = ev("negative", SOURCES[kind])
    events, diagnostics = EvidenceGroundedHistoricalEventExtractor().extract([source])
    assert events == [] and diagnostics["incomplete_movement_facts"][0]["outcome"] == kind
    return _route_state(historical_evidence=[source], historical_events=events,
                        historical_event_diagnostics={"extraction": diagnostics})


def geometry():
    return {"geojson": {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"segment_role": "terrain"},
         "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 2]]}},
    ]}}


@pytest.mark.parametrize("kind,answer", NEGATIVES)
def test_supported_noncompletion_survives_and_does_not_mutate_history(kind, answer):
    state = negative_state(kind)
    before = deepcopy(state.historical_event_diagnostics)
    assert not BoundedAgentLoop._final_answer_asserts_unsupported_route(answer)
    reply, finished = _loop()._finish(answer, state, perf_counter())
    assert reply != NO_ROUTE_TERMINAL_GUARDRAIL and "insufficient" not in reply.lower()
    assert finished.historical_events == [] and finished.historical_route is None
    assert finished.historical_route_presentation is None
    assert finished.historical_event_diagnostics == before
    assert reply == answer or SOURCES[kind] in reply


@pytest.mark.parametrize("kind,answer", NEGATIVES)
def test_supported_negative_survives_public_grounded_submission(kind, answer):
    source = ev("negative", SOURCES[kind])
    reply, state = agent([terminal(answer, [source.id])], [source]).respond(
        "Trace Marcus's route from Rome to Capua.", AgentState(session_id="term-public"))
    assert reply != NO_ROUTE_TERMINAL_GUARDRAIL and "insufficient" not in reply.lower()
    assert reply.startswith(answer) or SOURCES[kind] in reply
    assert state.historical_route is None and state.historical_route_presentation is None
    assert not validate_evidence_citations(reply, [source], require_citation=True)


@pytest.mark.parametrize("kind,answer", NEGATIVES)
@pytest.mark.parametrize("unrelated", (False, True))
def test_candidate_negative_wording_is_not_evidence(kind, answer, unrelated):
    state = _route_state(historical_evidence=[ev("unrelated", "Marcus remained in camp.")] if unrelated else [])
    reply, finished = _loop()._finish(answer, state, perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL and finished.supporting_evidence == []
    assert finished.historical_route is None


@pytest.mark.parametrize("answer", SMUGGLING)
def test_disclaimer_does_not_authorize_a_positive_route(answer):
    assert BoundedAgentLoop._final_answer_asserts_unsupported_route(answer)
    reply, state = _loop()._finish(answer, _route_state(), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL and state.supporting_evidence == []


@pytest.mark.parametrize("answer", SMUGGLING)
def test_positive_rewrite_keeps_the_supported_negative_fact(answer):
    state = negative_state("PREVENTED")
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert SOURCES["PREVENTED"] in reply and answer not in reply
    assert "through C" not in reply and "→" not in reply
    assert state.supporting_evidence == state.historical_evidence
    assert not validate_evidence_citations(reply, state.historical_evidence, require_citation=True)


@pytest.mark.parametrize("answer", (
    "The journey was abandoned.",
    "Pompey was prevented from entering Capua.",
))
def test_wrong_negative_kind_or_actor_is_replaced_by_documentary_state(answer):
    state = negative_state("PLANNED" if "abandoned" in answer else "PREVENTED")
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert answer not in reply and "Pompey" not in reply and "abandoned" not in reply
    assert state.historical_evidence[0].text in reply


@pytest.mark.parametrize("answer", (
    "The available evidence does not establish the route.",
    "The sources cannot establish an exact itinerary.",
    "现有史料无法证明具体路线。",
))
def test_unknown_is_an_epistemic_refusal_not_a_negative_historical_fact(answer):
    reply, state = _loop()._finish(answer, _route_state(), perf_counter())
    assert reply == answer and state.historical_route is None


def test_failed_to_reach_state_uses_existing_aborted_authority():
    state = negative_state("ABORTED")
    reply, _ = _loop()._finish("The journey failed to reach its destination.", state, perf_counter())
    assert reply != NO_ROUTE_TERMINAL_GUARDRAIL


def test_supported_positive_route_is_allowed():
    state = _route_state(historical_route=structured_route(), historical_route_diagnostics={"canonical_completeness": "COMPLETE"})
    answer = "The evidence supports movement from Genava to Alpes."
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert reply == answer


def test_partial_route_cannot_gain_an_unsupported_middle_or_endpoint():
    state = _route_state(historical_route=structured_route(), historical_route_diagnostics={"canonical_completeness": "PARTIAL"})
    route_before = state.historical_route.model_dump()
    reply, state = _loop()._finish("The route went from Genava through C to Alpes and then to Rome.", state, perf_counter())
    assert "partial" in reply.lower() and "Genava" in reply and "Alpes" in reply
    assert "through C" not in reply and "Rome" not in reply
    assert state.historical_route.model_dump() == route_before


def test_actual_simulation_can_be_described_without_becoming_history():
    answer = "The historical route is not established. The GIS simulation uses A→C→B as a plausible model."
    state = _route_state(historical_route_presentation=geometry())
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply == answer and state.historical_route is None
    assert state.historical_route_presentation == geometry()
    reply, _ = _loop()._finish(answer, _route_state(), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


@pytest.mark.parametrize("answer", (
    "He sailed from A to B using the GIS simulation.",
    "The GIS simulation confirms the historical route A→C→B.",
    "The GIS simulation uses A→C→B. He probably followed it.",
    "The route was probably A→C→B, according to GIS simulation.",
))
def test_simulation_mention_cannot_license_a_historical_assertion(answer):
    state = _route_state(historical_route_presentation=geometry())
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


def test_negative_fact_and_simulation_remain_separate():
    state = negative_state("PREVENTED")
    state.historical_route_presentation = geometry()
    answer = "The movement was prevented. The GIS simulation uses A→C→B as a plausible model."
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert "prevented" in reply and state.historical_route is None
    assert "most plausible historical" not in reply


def test_empty_failed_or_point_presentation_cannot_authorize_a_simulation_path():
    failed = geometry()
    failed["geojson"]["features"][0]["properties"]["segment_role"] = "failed_gap"
    point = geometry()
    point["geojson"]["features"][0]["geometry"] = {"type": "Point", "coordinates": [0, 0]}
    for value in ({}, {"geojson": {"features": []}}, failed, point):
        reply, _ = _loop()._finish("The GIS simulation uses A→B.", _route_state(historical_route_presentation=value), perf_counter())
        assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


def test_public_submission_cannot_create_a_negative_fact_from_an_unrelated_source():
    source = ev("unrelated", "Marcus remained in camp.")
    answer = "The journey appears to have been abandoned before the destination was reached."
    reply, state = agent([terminal(answer, [source.id])], [source]).respond(
        "Trace Marcus's route from Rome to Capua.", AgentState(session_id="term-no-proof"))
    assert answer not in reply and "abandoned" not in reply
    assert state.historical_route is None


def test_explicit_unsupported_nonexecution_does_not_survive():
    reply, _ = _loop()._finish("The movement was not executed.", _route_state(), perf_counter())
    assert reply == NO_ROUTE_TERMINAL_GUARDRAIL


@pytest.mark.parametrize("answer", (
    "The force was not prevented from entering the city.",
    "The journey was never abandoned.",
))
def test_opposite_polarity_cannot_borrow_a_noncompletion_keyword(answer):
    state = negative_state("PREVENTED" if "prevented" in answer else "ABORTED")
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert answer not in reply and SOURCES["PREVENTED" if "prevented" in answer else "ABORTED"] in reply


def test_negative_claim_does_not_override_an_admitted_positive_route():
    state = _route_state(historical_route=structured_route(), historical_route_diagnostics={"canonical_completeness": "COMPLETE"})
    reply, _ = _loop()._finish("The journey was abandoned.", state, perf_counter())
    assert "abandoned" not in reply and "Genava" in reply and "Alpes" in reply


def test_negative_clause_does_not_hide_partial_route_completion():
    state = negative_state("PREVENTED")
    state.historical_route = structured_route()
    state.historical_route_diagnostics = {"canonical_completeness": "PARTIAL"}
    reply, _ = _loop()._finish("The movement was prevented. The route went from Genava through C to Alpes.", state, perf_counter())
    assert "through C" not in reply and "partial" in reply.lower()


def test_public_terminal_submission_retains_only_explicit_existing_simulation():
    source = ev("negative", SOURCES["PLANNED"])
    state = AgentState(session_id="term-simulation-public")
    answer = "The historical route is not established. The GIS simulation uses A→C→B as a plausible model."
    subject = agent([terminal(answer, [source.id])], [source])
    original = subject.provider.complete

    def complete(*args, **kwargs):
        # Deterministic terminal-state seam: GIS is not run or modified.
        state.historical_route_presentation = geometry()
        return original(*args, **kwargs)

    subject.provider.complete = complete
    reply, state = subject.respond("Trace Marcus's route from Rome to Capua.", state)
    assert answer in reply and state.historical_route is None
    assert not validate_evidence_citations(reply, [source], require_citation=True)


def test_wrong_citation_cannot_supply_negative_authority():
    state = negative_state("PREVENTED")
    unrelated = ev("unrelated", "Marcus remained in camp.")
    state.historical_evidence.append(unrelated)
    answer = "The movement was prevented. [Evidence: unrelated — Polybius, Histories, Book III]"
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert answer not in reply and SOURCES["PREVENTED"] in reply
    assert state.supporting_evidence == [state.historical_evidence[0]]


@pytest.mark.parametrize("source_text", (
    "The sources do not establish that the movement was prevented.",
    "The sources cannot confirm that the journey was abandoned.",
))
def test_unknown_documentary_qualification_cannot_become_a_negative_fact(source_text):
    state = _route_state(historical_evidence=[ev("uncertain", source_text)])
    answer = "The movement was prevented." if "prevented" in source_text else "The journey was abandoned."
    reply, _ = _loop()._finish(answer, state, perf_counter())
    assert not reply.startswith(answer)
    assert source_text in reply or reply == NO_ROUTE_TERMINAL_GUARDRAIL
