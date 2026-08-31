from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentModelResponse, AgentState, Evidence
from backend.tests.test_agent_loop import Geo, Retriever, call, terminal


def sallust(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Sallust",
        work="Catiline + Jugurthine War",
        locator="section unavailable",
        excerpt=text,
        text=text,
        book="",
        page_start=1,
        page_end=1,
        source_file="sallust.epub",
        source_type="primary_source",
    )


GRACCHUS_EVIDENCE = [
    sallust(
        "g1",
        "Thus when Tiberius and Caius Gracchus began to vindicate the liberty of the people, "
        "and to expose the misconduct of the few, the nobility endeavored to stop them.",
    )
]

GRACCHUS_QUERY = (
    "What were the main reforms proposed by Tiberius Gracchus, "
    "and why did they cause political conflict?"
)

GROUNDED_GRACCHUS_ANSWER = (
    "According to Sallust, Tiberius Gracchus began to vindicate the liberty of the people "
    "and expose noble misconduct, which caused political conflict."
)


def test_grounded_ordinary_qa_prose_does_not_trigger_insufficient_evidence_fallback():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Tiberius Gracchus reforms"}),
        AgentModelResponse(content=GROUNDED_GRACCHUS_ANSWER),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever(GRACCHUS_EVIDENCE), Geo()).respond(
        GRACCHUS_QUERY,
        AgentState(session_id="gracchus-prose"),
    )
    assert reply == GROUNDED_GRACCHUS_ANSWER
    assert state.status == "completed"
    assert state.final_grounding_status == "grounded"
    assert "insufficient to support a reliable answer" not in reply


def test_unsupported_ordinary_qa_prose_still_fails_closed_after_correction():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Tiberius Gracchus reforms"}),
        AgentModelResponse(content="Atlantis was the decisive reform site."),
        AgentModelResponse(content="Atlantis was the decisive reform site."),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever(GRACCHUS_EVIDENCE), Geo()).respond(
        GRACCHUS_QUERY,
        AgentState(session_id="gracchus-unsupported"),
    )
    assert reply == "The current retrieved historical evidence is insufficient to support a reliable answer."
    assert state.status == "completed_with_guardrail"
    assert state.final_grounding_status == "guardrail_fallback"
    assert "Atlantis" not in reply


def test_catiline_style_submit_grounded_answer_still_passes():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Catiline conspiracy"}),
        terminal("Sallust records that Catiline planned violence against the consuls.", ["c1"]),
    ])
    evidence = [sallust("c1", "Catiline and Autronius prepared to assassinate the consuls in the Capitol.")]
    reply, state = HistoricalGisAgent(provider, Retriever(evidence), Geo()).respond(
        "What happened during the Catiline conspiracy, and why was it politically significant?",
        AgentState(session_id="catiline-terminal"),
    )
    assert "Catiline" in reply
    assert state.final_grounding_status == "grounded"
    assert "[Evidence: c1" in reply


def test_submit_grounded_answer_ignores_erroneous_insufficient_flag_when_provenance_is_grounded():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Tiberius Gracchus reforms"}),
        terminal(GROUNDED_GRACCHUS_ANSWER, ["g1"], insufficient_evidence=True),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever(GRACCHUS_EVIDENCE), Geo()).respond(
        GRACCHUS_QUERY,
        AgentState(session_id="gracchus-submit-flag"),
    )
    assert "insufficient to support a reliable answer" not in reply
    assert state.final_grounding_status == "grounded"
    assert "[Evidence: g1" in reply
