"""Shared OpenAI-compatible chat/completions adapter for LLM providers."""
from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Callable, Type

import httpx

from backend.app.agent.llm.base import LLMProvider
from backend.app.models import AgentModelResponse, AgentToolCall

logger = logging.getLogger(__name__)


class OpenAICompatibleProviderError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


class OpenAICompatibleChatProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str | None,
        provider_name: str = "LLM",
        timeout_s: float = 30,
        connect_timeout_s: float = 10,
        http_client: httpx.Client | None = None,
        extra_payload: dict | None = None,
        error_class: Type[OpenAICompatibleProviderError] = OpenAICompatibleProviderError,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name
        self.timeout_s = timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.http_client = http_client
        self.extra_payload = extra_payload or {}
        self.error_class = error_class
        self.completion_url = build_chat_completions_url(self.base_url)

    @staticmethod
    def to_api_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def _error(self, message: str, *, http_status: int | None = None) -> OpenAICompatibleProviderError:
        return self.error_class(message, http_status=http_status)

    def complete(self, messages: list[dict], tools: list[dict]) -> AgentModelResponse:
        if not self.api_key:
            raise self._error(f"{self.provider_name} API key is missing")
        if not self.model:
            raise self._error(f"{self.provider_name} model is not configured")
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "tools": self.to_api_tools(tools),
            "tool_choice": "auto",
        }
        payload.update(self.extra_payload)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self.http_client or httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s))
        close_client = self.http_client is None
        started = perf_counter()
        logger.info("%s_llm_request_started model=%s", self.provider_name.lower(), self.model)
        try:
            response = client.post(
                self.completion_url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s),
            )
            logger.info(
                "%s_llm_http_response status=%s latency_ms=%s model=%s",
                self.provider_name.lower(),
                response.status_code,
                int((perf_counter() - started) * 1000),
                self.model,
            )
        except httpx.TimeoutException as exc:
            raise self._error(f"{self.provider_name} request timed out") from exc
        except httpx.HTTPError as exc:
            raise self._error(f"{self.provider_name} network request failed") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code == 401:
            raise self._error(f"{self.provider_name} authentication failed (401)", http_status=401)
        if response.status_code == 429:
            raise self._error(f"{self.provider_name} rate limit reached (429)", http_status=429)
        if response.is_error:
            raise self._error(
                f"{self.provider_name} API request failed ({response.status_code})",
                http_status=response.status_code,
            )
        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise self._error(f"{self.provider_name} response was malformed") from exc
        calls: list[AgentToolCall] = []
        for raw in message.get("tool_calls") or []:
            try:
                arguments = json.loads(raw["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                calls.append(
                    AgentToolCall(
                        id=raw["id"],
                        name=raw["function"]["name"],
                        arguments=arguments,
                    )
                )
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise self._error(f"{self.provider_name} returned malformed tool arguments") from exc
        return AgentModelResponse(
            content=message.get("content"),
            tool_calls=calls,
            finish_reason=body["choices"][0].get("finish_reason") or "stop",
            usage={
                key: int(value)
                for key, value in (body.get("usage") or {}).items()
                if isinstance(value, (int, float))
            },
            http_status=response.status_code,
        )
