from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever


class Retriever(HistoricalRetriever):
    def retrieve(self, _query, top_k=5, filters=None):
        return [
            Evidence(
                id="evidence-1", author="Polybius", work="Histories", locator="Book III",
                excerpt="Hannibal crossed the Alps.", text="Hannibal crossed the Alps.",
            )
        ]


class Geography:
    def call(self, _tool, _arguments):
        return {"found": False}


class RaisingProvider:
    def __init__(self, message):
        self.message = message

    def complete(self, _messages, _tools):
        raise RuntimeError(self.message)


def call(name, arguments, identifier):
    return AgentModelResponse(
        tool_calls=[AgentToolCall(id=identifier, name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def test_successful_provider_timing_is_content_free_and_preserves_answer_flow():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal"}, "search"),
        AgentModelResponse(content="The current evidence is insufficient to support a reliable answer."),
    ])
    reply, state = HistoricalGisAgent(provider, Retriever(), Geography(), max_steps=3).respond(
        "What happened to Hannibal?", AgentState(session_id="provider-timing-success")
    )

    assert reply
    assert [item.call_number for item in state.provider_call_timing] == [1, 2]
    assert [item.agent_step for item in state.provider_call_timing] == [1, 2]
    assert all(item.status == "SUCCESS" and item.elapsed_ms >= 0 for item in state.provider_call_timing)
    assert state.tool_results["provider_timing_summary"]["successful_calls"] == 2
    assert "Hannibal crossed the Alps" not in str(state.provider_call_timing)
    assert set(state.provider_call_timing[0].model_dump()) == {
        "call_number", "agent_step", "started_ms", "finished_ms", "elapsed_ms", "status",
    }


def test_timeout_and_error_are_recorded_without_replaying_provider_calls():
    timeout_reply, timeout_state = HistoricalGisAgent(
        RaisingProvider("provider timed out"), Retriever(), Geography(), max_steps=3
    ).respond("ordinary question", AgentState(session_id="provider-timing-timeout"))
    error_reply, error_state = HistoricalGisAgent(
        RaisingProvider("provider unavailable"), Retriever(), Geography(), max_steps=3
    ).respond("ordinary question", AgentState(session_id="provider-timing-error"))

    assert timeout_reply == error_reply == "The configured language-model provider is unavailable."
    assert [item.status for item in timeout_state.provider_call_timing] == ["TIMEOUT"]
    assert [item.status for item in error_state.provider_call_timing] == ["ERROR"]
    assert timeout_state.tool_results["provider_timing_summary"]["total_calls"] == 1
    assert error_state.tool_results["provider_timing_summary"]["failed_calls"] == 1
