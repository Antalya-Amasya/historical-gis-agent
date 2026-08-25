import pytest
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.deepseek import DeepSeekLLMProvider
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.model_router import ModelRouter, ModelSelectionError
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever


class Retriever(HistoricalRetriever):
    def retrieve(self, query, top_k=5, filters=None):
        return [Evidence(id="one", author="Polybius", work="Histories", locator="Book III", excerpt="Alps", text="Alps")]


class Geo:
    def call(self, tool, arguments):
        return {"found": False}


def tool_call(query="Alps"):
    return AgentModelResponse(tool_calls=[AgentToolCall(id="search", name="search_historical_evidence", arguments={"query": query})], finish_reason="tool_calls")


def router(flash="flash-id", pro="pro-id", legacy=None):
    return ModelRouter(flash, pro, legacy, "flash_first")


@pytest.mark.parametrize("intent,requested_output", [("historical_question", "answer"), ("historical_route", "historical_route"), ("geography_question", "geography_fact")])
def test_default_policy_selects_flash_for_normal_requests(intent, requested_output):
    selected = router().select_model(intent, requested_output)
    assert selected.tier == "flash" and selected.model_id == "flash-id" and selected.reason == "flash_first"


@pytest.mark.parametrize("quality_mode", ["economy", "balanced"])
def test_economy_and_balanced_select_flash(quality_mode):
    assert router().select_model("answer", "answer", quality_mode=quality_mode).tier == "flash"


def test_high_quality_selects_pro():
    selected = router().select_model("answer", "answer", quality_mode="high")
    assert selected.tier == "pro" and selected.model_id == "pro-id" and selected.reason == "quality_mode_high"


def test_flash_uses_legacy_fallback_and_missing_config_is_explicit():
    selected = router(flash=None, pro="pro-id", legacy="legacy-id").select_model("answer", "answer")
    assert selected.tier == "flash" and selected.model_id == "legacy-id" and selected.reason == "legacy_model_fallback"
    with pytest.raises(ModelSelectionError, match="Flash model"):
        router(flash=None, pro="pro-id", legacy=None).select_model("answer", "answer")


def test_high_quality_without_pro_is_explicit_error():
    with pytest.raises(ModelSelectionError, match="Pro model"):
        router(pro=None).select_model("answer", "answer", quality_mode="high")


def test_router_does_not_depend_on_historical_person_names():
    hannibal = router().select_model("answer", "answer")
    caesar = router().select_model("answer", "answer")
    assert hannibal == caesar


def test_agent_uses_one_selected_model_for_all_continuations():
    scripted = ScriptedLLMProvider([tool_call(), AgentModelResponse(content="Grounded answer.")])
    selected_ids = []
    def provider_factory(model_id):
        selected_ids.append(model_id)
        return scripted
    subject = HistoricalGisAgent(None, Retriever(), Geo(), model_router=router(), provider_factory=provider_factory)
    _, state = subject.respond("What does Polybius say?", AgentState(session_id="router-answer"))
    assert selected_ids == ["flash-id"] and len(scripted.requests) == 2
    assert state.selected_model_tier == "flash" and state.selected_model_id == "flash-id"
    assert state.model_policy == "flash_first" and state.quality_mode is None


def test_high_quality_agent_selects_pro_once():
    scripted = ScriptedLLMProvider([AgentModelResponse(content="Done.")])
    selected_ids = []
    subject = HistoricalGisAgent(None, Retriever(), Geo(), model_router=router(), provider_factory=lambda model_id: selected_ids.append(model_id) or scripted)
    state = AgentState(session_id="router-high", assumptions={"quality_mode": "high"})
    subject.respond("What does Polybius say?", state)
    assert selected_ids == ["pro-id"] and state.selected_model_tier == "pro" and state.quality_mode == "high"


def test_deepseek_provider_receives_selected_model_id():
    selected = router().select_model("answer", "answer")
    provider = DeepSeekLLMProvider("test-key", "https://example.invalid", selected.model_id)
    assert provider.model == "flash-id"
