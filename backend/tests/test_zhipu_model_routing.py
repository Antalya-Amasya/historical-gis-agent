"""Audit tests for Zhipu/DeepSeek model routing defaults."""
from __future__ import annotations

from pathlib import Path

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.agent.model_router import ModelRouter
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever

AIR = "glm-4.5-air"
PRO = "glm-4.7"


class Retriever(HistoricalRetriever):
    def retrieve(self, query, top_k=5, filters=None):
        return [
            Evidence(
                id="one",
                author="Polybius",
                work="Histories",
                locator="Book III",
                excerpt="Alps",
                text="Alps",
            )
        ]


class Geo:
    def call(self, tool, arguments):
        return {"found": False}


def zhipu_router(flash=AIR, pro=PRO, legacy=AIR):
    return ModelRouter(flash, pro, legacy, "flash_first")


def call(name, arguments, ident="x"):
    return AgentModelResponse(
        tool_calls=[AgentToolCall(id=ident, name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def test_zhipu_flash_first_selects_air():
    selected = zhipu_router().select_model("answer", "answer")
    assert selected.tier == "flash"
    assert selected.model_id == AIR


def test_zhipu_explicit_high_selects_pro():
    selected = zhipu_router().select_model("answer", "answer", quality_mode="high")
    assert selected.tier == "pro"
    assert selected.model_id == PRO
    assert selected.reason == "quality_mode_high"


def test_zhipu_legacy_fallback_uses_air_not_pro():
    selected = zhipu_router(flash=None, pro=PRO, legacy=AIR).select_model("answer", "answer")
    assert selected.tier == "flash"
    assert selected.model_id == AIR
    assert selected.reason == "legacy_model_fallback"


def test_normal_agent_request_uses_air_for_all_provider_instances():
    scripted = ScriptedLLMProvider(
        [
            call("search_historical_evidence", {"query": "Alps"}),
            call(
                "submit_grounded_answer",
                {
                    "answer": "Grounded answer.",
                    "evidence_ids": ["one"],
                    "insufficient_evidence": False,
                },
                "answer",
            ),
        ]
    )
    selected_ids: list[str] = []

    def provider_factory(model_id: str):
        selected_ids.append(model_id)
        return scripted

    subject = HistoricalGisAgent(
        None,
        Retriever(),
        Geo(),
        model_router=zhipu_router(),
        provider_factory=provider_factory,
    )
    _, state = subject.respond("What does Polybius say?", AgentState(session_id="air-default"))
    assert selected_ids == [AIR]
    assert state.selected_model_id == AIR
    assert state.selected_model_tier == "flash"


def test_grounding_correction_does_not_upgrade_to_pro():
    scripted = ScriptedLLMProvider(
        [
            call(
                "submit_grounded_answer",
                {
                    "answer": "Hannibal crossed the Rhine in 218 BCE.",
                    "evidence_ids": ["one"],
                    "insufficient_evidence": False,
                },
                "answer",
            ),
            call(
                "submit_grounded_answer",
                {
                    "answer": "Polybius mentions the Alps.",
                    "evidence_ids": ["one"],
                    "insufficient_evidence": False,
                },
                "answer2",
            ),
        ]
    )
    selected_ids: list[str] = []

    subject = HistoricalGisAgent(
        None,
        Retriever(),
        Geo(),
        model_router=zhipu_router(),
        provider_factory=lambda model_id: selected_ids.append(model_id) or scripted,
        max_grounding_corrections=1,
    )
    _, state = subject.respond("What does Polybius say about the Alps?", AgentState(session_id="grounding-air"))
    assert selected_ids == [AIR]
    assert state.grounding_corrections == 1
    assert state.selected_model_id == AIR


def test_completion_correction_does_not_upgrade_to_pro():
    scripted = ScriptedLLMProvider(
        [
            call("search_historical_evidence", {"query": "Hannibal route"}),
            call(
                "build_historical_route",
                {"event_id": "test", "name": "Test route", "period": "218 BCE"},
            ),
            AgentModelResponse(content="The retrieved evidence was insufficient to construct a historical route."),
            AgentModelResponse(content="Route evidence remains insufficient."),
        ]
    )
    selected_ids: list[str] = []

    subject = HistoricalGisAgent(
        None,
        Retriever(),
        Geo(),
        model_router=zhipu_router(),
        provider_factory=lambda model_id: selected_ids.append(model_id) or scripted,
        max_steps=4,
        max_completion_corrections=1,
    )
    _, state = subject.respond(
        "Build Hannibal's route across the Alps.",
        AgentState(session_id="completion-air"),
    )
    assert selected_ids == [AIR]
    assert state.selected_model_id == AIR


def test_tool_continuation_keeps_default_air_tier():
    scripted = ScriptedLLMProvider(
        [
            call("search_historical_evidence", {"query": "Alps"}),
            call(
                "submit_grounded_answer",
                {
                    "answer": "Grounded answer.",
                    "evidence_ids": ["one"],
                    "insufficient_evidence": False,
                },
                "answer",
            ),
        ]
    )
    selected_ids: list[str] = []

    subject = HistoricalGisAgent(
        None,
        Retriever(),
        Geo(),
        model_router=zhipu_router(),
        provider_factory=lambda model_id: selected_ids.append(model_id) or scripted,
    )
    subject.respond("What does Polybius say?", AgentState(session_id="continuation-air"))
    assert selected_ids == [AIR]
    assert len(scripted.requests) == 2


def test_unknown_legacy_safe_path_does_not_select_pro_by_default():
    selected = zhipu_router(flash=None, pro=PRO, legacy=AIR).select_model(
        "historical_route",
        "historical_route",
    )
    assert selected.model_id == AIR
    assert selected.tier == "flash"


def test_deepseek_flash_pro_mapping_regression():
    router = ModelRouter("deepseek-flash", "deepseek-pro", "deepseek-legacy", "flash_first")
    assert router.select_model("answer", "answer").model_id == "deepseek-flash"
    assert router.select_model("answer", "answer", quality_mode="high").model_id == "deepseek-pro"
    assert router.select_model("answer", "answer", quality_mode="balanced").model_id == "deepseek-flash"
    legacy_router = ModelRouter(None, "deepseek-pro", "deepseek-legacy", "flash_first")
    assert legacy_router.select_model("answer", "answer").model_id == "deepseek-legacy"


def test_production_code_has_no_hardcoded_glm_runtime_models():
    app_root = Path(__file__).resolve().parents[1] / "app"
    forbidden = ("glm-4.5-flash", "glm-4.7", "glm-4.5-air")
    hits: list[str] = []
    for path in app_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(app_root.parent)}:{token}")
    assert hits == [], f"Hardcoded glm model IDs found in production code: {hits}"
