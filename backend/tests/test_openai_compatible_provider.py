import json

import httpx
import pytest

from backend.app.agent.llm.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleProviderError,
    build_chat_completions_url,
)

TOOLS = [
    {
        "name": "search_historical_evidence",
        "description": "search",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


def provider(handler, *, base_url="https://api.example.com", extra_payload=None):
    return OpenAICompatibleChatProvider(
        api_key="test-key",
        base_url=base_url,
        model="test-model",
        provider_name="Test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        extra_payload=extra_payload,
    )


def response(status, body):
    return httpx.Response(status, json=body)


def body(message, finish="stop", usage=None):
    payload = {"choices": [{"message": message, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_normal_assistant_text():
    result = provider(lambda request: response(200, body({"content": "Hello"}))).complete(
        [{"role": "user", "content": "hi"}],
        [],
    )
    assert result.content == "Hello"
    assert not result.tool_calls
    assert result.finish_reason == "stop"


def test_single_tool_call():
    calls = [
        {
            "id": "a",
            "type": "function",
            "function": {
                "name": "search_historical_evidence",
                "arguments": '{"query": "x"}',
            },
        }
    ]
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": calls}, "tool_calls"))
    ).complete([], TOOLS)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "a"
    assert result.tool_calls[0].name == "search_historical_evidence"
    assert result.tool_calls[0].arguments == {"query": "x"}


def test_multiple_tool_calls():
    calls = [
        {
            "id": "a",
            "type": "function",
            "function": {
                "name": "search_historical_evidence",
                "arguments": '{"query": "x"}',
            },
        },
        {
            "id": "b",
            "type": "function",
            "function": {
                "name": "search_historical_evidence",
                "arguments": '{"query": "y"}',
            },
        },
    ]
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": calls}, "tool_calls"))
    ).complete([], TOOLS)
    assert [(c.id, c.arguments) for c in result.tool_calls] == [
        ("a", {"query": "x"}),
        ("b", {"query": "y"}),
    ]


def test_tool_arguments_json_decoding():
    calls = [
        {
            "id": "a",
            "type": "function",
            "function": {
                "name": "search_historical_evidence",
                "arguments": '{"query": "decoded"}',
            },
        }
    ]
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": calls}, "tool_calls"))
    ).complete([], TOOLS)
    assert result.tool_calls[0].arguments["query"] == "decoded"


def test_malformed_tool_arguments_error():
    malformed = {
        "tool_calls": [
            {
                "id": "a",
                "function": {
                    "name": "search_historical_evidence",
                    "arguments": "not-json",
                },
            }
        ]
    }
    with pytest.raises(OpenAICompatibleProviderError, match="malformed tool arguments"):
        provider(lambda request: response(200, body(malformed))).complete([], TOOLS)


def test_usage_parsing():
    result = provider(
        lambda request: response(
            200,
            body({"content": "ok"}, usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}),
        )
    ).complete([], [])
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_finish_reason():
    result = provider(
        lambda request: response(200, body({"content": None, "tool_calls": []}, "tool_calls"))
    ).complete([], [])
    assert result.finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "authentication failed"),
        (429, "rate limit"),
        (500, "API request failed"),
    ],
)
def test_http_failures(status, message):
    with pytest.raises(OpenAICompatibleProviderError, match=message):
        provider(lambda request: response(status, {"error": "x"})).complete([], [])


def test_timeout():
    def timeout(request):
        raise httpx.TimeoutException("timeout", request=request)

    with pytest.raises(OpenAICompatibleProviderError, match="timed out"):
        provider(timeout).complete([], [])


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.example.com", "https://api.example.com/chat/completions"),
        ("https://api.example.com/", "https://api.example.com/chat/completions"),
        (
            "https://api.example.com/chat/completions",
            "https://api.example.com/chat/completions",
        ),
        (
            "https://api.example.com/chat/completions/",
            "https://api.example.com/chat/completions",
        ),
    ],
)
def test_base_url_slash_handling(base_url, expected):
    assert build_chat_completions_url(base_url) == expected
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return response(200, body({"content": "ok"}))

    provider(handler, base_url=base_url).complete([], [])
    assert captured["url"] == expected


def test_extra_payload_merged():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return response(200, body({"content": "ok"}))

    provider(handler, extra_payload={"thinking": {"type": "disabled"}}).complete([], [])
    assert captured["thinking"] == {"type": "disabled"}
