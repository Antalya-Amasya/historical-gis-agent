import json

import httpx
import pytest

from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.zhipu import ZhipuLLMProvider, ZhipuProviderError
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever

TOOLS = [
    {
        "name": "test_tool",
        "description": "test",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
]


def provider(handler, *, base_url="https://open.bigmodel.cn/api/paas/v4"):
    return ZhipuLLMProvider(
        "test-key",
        base_url,
        "glm-4.5-air",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def response(status, body):
    return httpx.Response(status, json=body)


def body(message, finish="stop", usage=None):
    payload = {"choices": [{"message": message, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_zhipu_normal_text():
    result = provider(lambda request: response(200, body({"content": "OK"}))).complete(
        [{"role": "user", "content": "只回复 OK"}],
        [],
    )
    assert result.content == "OK"
    assert result.http_status == 200


def test_zhipu_single_tool_call():
    calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"value": "hello"}'},
        }
    ]
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": calls}, "tool_calls"))
    ).complete([{"role": "user", "content": "call test_tool"}], TOOLS)
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "test_tool"
    assert result.tool_calls[0].arguments == {"value": "hello"}


def test_zhipu_multiple_tool_calls():
    calls = [
        {
            "id": "a",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"value": "one"}'},
        },
        {
            "id": "b",
            "type": "function",
            "function": {"name": "test_tool", "arguments": '{"value": "two"}'},
        },
    ]
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": calls}, "tool_calls"))
    ).complete([], TOOLS)
    assert len(result.tool_calls) == 2


def test_zhipu_malformed_tool_arguments():
    malformed = {
        "tool_calls": [
            {"id": "a", "function": {"name": "test_tool", "arguments": "not-json"}}
        ]
    }
    with pytest.raises(ZhipuProviderError, match="malformed tool arguments"):
        provider(lambda request: response(200, body(malformed))).complete([], TOOLS)


def test_zhipu_does_not_contain_deepseek_thinking_field():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return response(200, body({"content": "ok"}))

    provider(handler).complete([], [])
    assert "thinking" not in captured
    assert captured["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "authentication"),
        (429, "rate limit"),
        (500, "API request failed"),
    ],
)
def test_zhipu_http_failures(status, message):
    with pytest.raises(ZhipuProviderError, match=message):
        provider(lambda request: response(status, {"error": "x"})).complete([], [])


def test_zhipu_timeout():
    def timeout(request):
        raise httpx.TimeoutException("timeout", request=request)

    with pytest.raises(ZhipuProviderError, match="timed out"):
        provider(timeout).complete([], [])


def test_zhipu_base_url_with_trailing_slash():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return response(200, body({"content": "ok"}))

    provider(handler, base_url="https://open.bigmodel.cn/api/paas/v4/").complete([], [])
    assert captured["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"


def test_zhipu_tool_result_continuation_roundtrip():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if not any(message["role"] == "tool" for message in payload["messages"]):
            call = {
                "id": "search",
                "type": "function",
                "function": {
                    "name": "search_historical_evidence",
                    "arguments": json.dumps({"query": "Cannae", "top_k": 8}),
                },
            }
            return response(200, body({"content": None, "tool_calls": [call]}, "tool_calls"))
        tool_message = next(
            message for message in reversed(payload["messages"]) if message["role"] == "tool"
        )
        assert tool_message["tool_call_id"] == "search"
        terminal = {
            "id": "answer",
            "type": "function",
            "function": {
                "name": "submit_grounded_answer",
                "arguments": json.dumps(
                    {
                        "answer": "Polybius records Cannae.",
                        "evidence_ids": ["cannae-1"],
                        "insufficient_evidence": False,
                    }
                ),
            },
        }
        return response(200, body({"content": None, "tool_calls": [terminal]}, "tool_calls"))

    class Retriever(HistoricalRetriever):
        def retrieve(self, query, top_k=5, filters=None):
            return [
                Evidence(
                    id="cannae-1",
                    author="Polybius",
                    work="Histories",
                    locator="Book III",
                    excerpt="Cannae battle.",
                    text="Cannae battle.",
                )
            ]

    zhipu_provider = ZhipuLLMProvider(
        "test-key",
        "https://open.bigmodel.cn/api/paas/v4",
        "glm-4.5-air",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    reply, state = HistoricalGisAgent(zhipu_provider, Retriever()).respond(
        "What happened at Cannae?", AgentState(session_id="zhipu-roundtrip")
    )
    assert len(requests) == 2
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert "Polybius records Cannae" in reply
    assert state.final_grounding_status == "grounded"
