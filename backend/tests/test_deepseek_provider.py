import json
import httpx, pytest
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.deepseek import DeepSeekLLMProvider, DeepSeekProviderError
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever

TOOLS=[{"name":"search_historical_evidence","description":"search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]
def provider(handler): return DeepSeekLLMProvider("test-key","https://api.deepseek.com","deepseek-v4-flash",http_client=httpx.Client(transport=httpx.MockTransport(handler)))
def response(status, body): return httpx.Response(status,json=body)
def body(message, finish="stop"): return {"choices":[{"message":message,"finish_reason":finish}]}

def test_plain_response_and_schema_conversion():
    captured={}
    def handler(request): captured.update(json.loads(request.content)); return response(200,body({"content":"Grounded answer"}))
    result=provider(handler).complete([{"role":"user","content":"x"}],TOOLS)
    assert result.content=="Grounded answer" and not result.tool_calls
    assert captured["tools"][0]["type"]=="function" and captured["tools"][0]["function"]["parameters"]==TOOLS[0]["input_schema"]
    assert captured["tool_choice"]=="auto" and captured["thinking"]["type"]=="disabled"

def test_single_and_multiple_tool_calls():
    calls=[{"id":"a","type":"function","function":{"name":"search_historical_evidence","arguments":"{\"query\": \"x\"}"}},{"id":"b","type":"function","function":{"name":"search_historical_evidence","arguments":"{\"query\": \"y\"}"}}]
    result=provider(lambda request: response(200,body({"content":None,"tool_calls":calls},"tool_calls"))).complete([],TOOLS)
    assert [(c.id,c.name,c.arguments) for c in result.tool_calls]==[("a","search_historical_evidence",{"query":"x"}),("b","search_historical_evidence",{"query":"y"})]

def test_malformed_arguments_and_response_are_safe_failures():
    malformed={"tool_calls":[{"id":"a","function":{"name":"search_historical_evidence","arguments":"not-json"}}]}
    with pytest.raises(DeepSeekProviderError,match="malformed tool arguments"): provider(lambda request: response(200,body(malformed))).complete([],TOOLS)
    with pytest.raises(DeepSeekProviderError,match="malformed"): provider(lambda request: response(200,{})).complete([],TOOLS)

@pytest.mark.parametrize(("status","message"),[(401,"authentication"),(429,"rate limit"),(500,"API request failed")])
def test_http_failures_are_diagnostic_without_headers(status,message):
    with pytest.raises(DeepSeekProviderError,match=message): provider(lambda request: response(status,{"error":"x"})).complete([],TOOLS)

def test_timeout_missing_key_and_network_failure():
    with pytest.raises(DeepSeekProviderError,match="missing"): DeepSeekLLMProvider(None,"https://api.deepseek.com","x").complete([],TOOLS)
    def timeout(request): raise httpx.TimeoutException("timeout",request=request)
    with pytest.raises(DeepSeekProviderError,match="timed out"): provider(timeout).complete([],TOOLS)
    def network(request): raise httpx.NetworkError("network",request=request)
    with pytest.raises(DeepSeekProviderError,match="network"): provider(network).complete([],TOOLS)


def test_mock_http_real_provider_path_generates_from_tool_evidence_not_fake_canned_text():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if not any(message["role"] == "tool" for message in payload["messages"]):
            call = {"id":"search","type":"function","function":{
                "name":"search_historical_evidence",
                "arguments":json.dumps({"query":"Battle of Cannae","top_k":8}),
            }}
            return response(200,body({"content":None,"tool_calls":[call]},"tool_calls"))
        tool_payload = json.loads(next(
            message["content"] for message in reversed(payload["messages"]) if message["role"] == "tool"
        ))
        assert tool_payload["result"]["evidence"][0]["id"] == "cannae-1"
        assert tool_payload["result"]["historical_events"][0]["event_type"] == "BATTLE"
        return response(200,body({"content":"Polybius records a battle at Cannae. [Evidence: cannae-1 " + chr(0x2014) + " Polybius, Histories, Book III]"}))

    class Retriever(HistoricalRetriever):
        def retrieve(self, query, top_k=5, filters=None):
            return [Evidence(
                id="cannae-1", author="Polybius", work="Histories", locator="Book III",
                excerpt="The Romans fought a battle at Cannae.", text="The Romans fought a battle at Cannae.",
            )]

    real_provider = DeepSeekLLMProvider(
        "test-key", "https://api.deepseek.com", "deepseek-test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    reply, state = HistoricalGisAgent(real_provider, Retriever()).respond(
        "What happened at the Battle of Cannae?", AgentState(session_id="deepseek-contract")
    )
    assert len(requests) == 2
    assert "Polybius records a battle at Cannae" in reply
    assert "available primary-source evidence supports a cautious historical answer" not in reply
    assert state.final_grounding_status == "grounded"
