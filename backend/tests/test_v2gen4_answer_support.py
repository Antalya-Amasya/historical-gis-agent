"""Final answer support is validated citation provenance, not retrieval rank."""
from time import perf_counter

import pytest

from backend.app.agent.evidence_support import final_answer_support, render_evidence_citations
from backend.app.agent.route_summary import admitted_route_summary
from backend.app.models import AgentState, ChatResponse
from backend.app.route_result_status import derive_route_result_status
from backend.tests.test_agent_loop import ev, agent, terminal
from backend.tests.test_v1_1h2_no_route_terminal_arbitration import _loop, _route_state
from backend.tests.test_g4d_route_preservation import structured_route


def test_valid_citations_match_objects_in_citation_order_without_duplicates():
    evidence = [ev("one", "First source."), ev("two", "Second source.")]
    answer = render_evidence_citations(("two", "one", "two"), evidence)
    assert final_answer_support(answer, evidence) == [evidence[1], evidence[0]]
    assert [e.id for e in evidence] == ["one", "two"]


def test_invalid_id_and_metadata_are_not_support():
    source = ev("one", "Source.")
    valid = render_evidence_citations(("one",), [source])
    assert final_answer_support(valid.replace("Polybius", "Fabricated"), [source]) == []
    assert final_answer_support(valid.replace("one", "missing"), [source]) == []
    assert final_answer_support("Evidence references: one.", [source]) == []


def test_zero_citations_clear_previous_support_without_mutating_workspace():
    source = ev("one", "Source.")
    state = _route_state(historical_evidence=[source], supporting_evidence=[source])
    reply, state = _loop()._finish("Evidence is insufficient.", state, perf_counter())
    assert state.supporting_evidence == []
    assert state.historical_evidence == [source]
    assert reply == "Evidence is insufficient."


def test_rejected_positive_answer_drops_its_previously_valid_citation():
    source = ev("one", "Source.")
    state = _route_state(historical_evidence=[source])
    answer = "The route went from A to B. " + render_evidence_citations(("one",), [source])
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert "[Evidence:" not in reply
    assert state.supporting_evidence == []


def test_setovia_shaped_twenty_retrieved_one_final_support_serializes_separately():
    source = ev("prevented", "The attempted entry into Setovia was prevented.")
    workspace = [source, *[ev(f"noise-{i}", f"Unrelated source passage {i}.") for i in range(19)]]
    answer = source.text + " A completed route cannot be reconstructed."
    reply, state = agent([terminal(answer, [source.id])], workspace).respond(
        "Reconstruct a route into Setovia only if entry was completed.", AgentState(session_id="support"))
    assert answer in reply and source.id in reply
    assert state.supporting_evidence == [source]
    assert len(state.historical_evidence) == 20
    assert state.historical_route is None
    response = ChatResponse(session_id=state.session_id, reply=reply, state=state, route_result_status=derive_route_result_status(state).value)
    payload = response.model_dump(mode="json")
    assert len(payload["state"]["historical_evidence"]) == 20
    assert [e["id"] for e in payload["state"]["supporting_evidence"]] == ["prevented"]


@pytest.mark.parametrize("partial", [False, True])
def test_deterministic_route_summary_uses_only_references_it_actually_prints(partial):
    workspace = [ev(str(i), "Source.") for i in range(6)]
    route = structured_route()
    route.evidence_refs = ["1", "2", "missing", "3", "4"]
    state = _route_state(historical_evidence=workspace, historical_route=route,
        historical_route_diagnostics={"canonical_completeness": "PARTIAL" if partial else "FULL"})
    answer = admitted_route_summary(state)
    reply, state = _loop()._finish(answer, state, perf_counter())
    assert reply == answer
    assert [e.id for e in state.supporting_evidence] == ["1", "2", "3"]
    assert "4" not in reply.split("Evidence references: ")[-1]
