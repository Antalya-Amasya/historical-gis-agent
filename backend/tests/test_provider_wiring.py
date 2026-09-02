import json

import httpx
import pytest

from backend.app.agent.llm.deepseek import DeepSeekLLMProvider
from backend.app.agent.llm.zhipu import ZhipuLLMProvider
from backend.app.agent.model_router import ModelRouter
from backend.app.core.config import Settings
from backend.app.main import build_agent


def test_deepseek_still_contains_thinking_field():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    DeepSeekLLMProvider(
        "test-key",
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ).complete([], [])
    assert captured["thinking"] == {"type": "disabled"}


def test_build_agent_deepseek(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "llm")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL_FLASH", "flash-model")
    monkeypatch.setenv("DEEPSEEK_MODEL_PRO", "pro-model")
    monkeypatch.setenv("AGENT_MODEL_POLICY", "flash_first")
    settings = Settings()
    monkeypatch.setattr("backend.app.main.settings", settings)
    agent = build_agent()
    assert agent.model_router is not None
    selected = agent.model_router.select_model("answer", "answer")
    provider = agent.provider_factory(selected.model_id)
    assert isinstance(provider, DeepSeekLLMProvider)
    assert provider.model == "flash-model"


def test_build_agent_zhipu(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "llm")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "zhipu")
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    monkeypatch.setenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    monkeypatch.setenv("ZHIPU_MODEL_FLASH", "glm-4.5-air")
    monkeypatch.setenv("ZHIPU_MODEL_PRO", "glm-4.7")
    monkeypatch.setenv("ZHIPU_MODEL", "glm-4.5-air")
    monkeypatch.setenv("AGENT_MODEL_POLICY", "flash_first")
    settings = Settings()
    monkeypatch.setattr("backend.app.main.settings", settings)
    agent = build_agent()
    assert agent.model_router is not None
    selected = agent.model_router.select_model("answer", "answer")
    provider = agent.provider_factory(selected.model_id)
    assert isinstance(provider, ZhipuLLMProvider)
    assert provider.model == "glm-4.5-air"


def test_build_agent_unknown_provider_uses_fake(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "llm")
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "unknown")
    settings = Settings()
    monkeypatch.setattr("backend.app.main.settings", settings)
    agent = build_agent()
    assert agent.model_router is None
    assert agent.provider is not None


def test_zhipu_model_policy_flash_first():
    router = ModelRouter("glm-4.5-air", "glm-4.7", "glm-4.5-air", "flash_first")
    selected = router.select_model("answer", "answer")
    assert selected.tier == "flash"
    assert selected.model_id == "glm-4.5-air"


def test_zhipu_model_policy_high_selects_pro():
    router = ModelRouter("glm-4.5-air", "glm-4.7", "glm-4.5-air", "flash_first")
    selected = router.select_model("answer", "answer", quality_mode="high")
    assert selected.tier == "pro"
    assert selected.model_id == "glm-4.7"
