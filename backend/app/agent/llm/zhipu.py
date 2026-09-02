"""Zhipu OpenAI-compatible chat-completions adapter."""
from __future__ import annotations

import httpx

from backend.app.agent.llm.base import LLMProvider
from backend.app.agent.llm.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleProviderError,
)
from backend.app.models import AgentModelResponse


class ZhipuProviderError(OpenAICompatibleProviderError):
    pass


class ZhipuLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str | None,
        timeout_s: float = 30,
        connect_timeout_s: float = 10,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._provider = OpenAICompatibleChatProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider_name="Zhipu",
            timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s,
            http_client=http_client,
            error_class=ZhipuProviderError,
        )

    @staticmethod
    def to_api_tools(tools: list[dict]) -> list[dict]:
        return OpenAICompatibleChatProvider.to_api_tools(tools)

    def complete(self, messages: list[dict], tools: list[dict]) -> AgentModelResponse:
        return self._provider.complete(messages, tools)
