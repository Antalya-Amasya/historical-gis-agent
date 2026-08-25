"""Minimal DeepSeek OpenAI-compatible chat-completions adapter."""
from __future__ import annotations
import json
import logging
from time import perf_counter
import httpx
from backend.app.agent.llm.base import LLMProvider
from backend.app.models import AgentModelResponse, AgentToolCall

logger = logging.getLogger(__name__)
class DeepSeekProviderError(RuntimeError): pass
class DeepSeekLLMProvider(LLMProvider):
    def __init__(self, api_key: str | None, base_url: str, model: str | None, timeout_s: float = 30, connect_timeout_s: float = 10, http_client: httpx.Client | None = None):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.timeout_s, self.connect_timeout_s, self.http_client = timeout_s, connect_timeout_s, http_client
    @staticmethod
    def to_api_tools(tools: list[dict]) -> list[dict]:
        return [{"type":"function","function":{"name":tool["name"],"description":tool["description"],"parameters":tool["input_schema"]}} for tool in tools]
    def complete(self, messages: list[dict], tools: list[dict]) -> AgentModelResponse:
        if not self.api_key: raise DeepSeekProviderError("DeepSeek API key is missing")
        if not self.model: raise DeepSeekProviderError("DeepSeek model is not configured")
        payload={"model":self.model,"messages":messages,"tools":self.to_api_tools(tools),"tool_choice":"auto","thinking":{"type":"disabled"}}
        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json"}
        client=self.http_client or httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s))
        close_client=self.http_client is None
        started=perf_counter()
        logger.info("deepseek_llm_request_started model=%s", self.model)
        try:
            response=client.post(f"{self.base_url}/chat/completions",headers=headers,json=payload,timeout=httpx.Timeout(self.timeout_s, connect=self.connect_timeout_s))
            logger.info("deepseek_llm_http_response status=%s latency_ms=%s model=%s", response.status_code, int((perf_counter()-started)*1000), self.model)
        except httpx.TimeoutException as exc: raise DeepSeekProviderError("DeepSeek request timed out") from exc
        except httpx.HTTPError as exc: raise DeepSeekProviderError("DeepSeek network request failed") from exc
        finally:
            if close_client: client.close()
        if response.status_code == 401: raise DeepSeekProviderError("DeepSeek authentication failed (401)")
        if response.status_code == 429: raise DeepSeekProviderError("DeepSeek rate limit reached (429)")
        if response.is_error: raise DeepSeekProviderError(f"DeepSeek API request failed ({response.status_code})")
        try: body=response.json(); message=body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc: raise DeepSeekProviderError("DeepSeek response was malformed") from exc
        calls=[]
        for raw in message.get("tool_calls") or []:
            try:
                arguments=json.loads(raw["function"]["arguments"])
                if not isinstance(arguments,dict): raise ValueError("arguments must be an object")
                calls.append(AgentToolCall(id=raw["id"],name=raw["function"]["name"],arguments=arguments))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc: raise DeepSeekProviderError("DeepSeek returned malformed tool arguments") from exc
        return AgentModelResponse(content=message.get("content"),tool_calls=calls,finish_reason=body["choices"][0].get("finish_reason") or "stop", usage={key:int(value) for key,value in (body.get("usage") or {}).items() if isinstance(value,(int,float))}, http_status=response.status_code)
