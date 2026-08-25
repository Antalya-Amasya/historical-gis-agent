from backend.app.agent.llm.base import LLMProvider
from backend.app.models import AgentModelResponse, AgentToolCall

class ScriptedLLMProvider(LLMProvider):
    """Deterministic test provider. Scripts are explicit; it is not a real LLM."""
    def __init__(self, responses: list[AgentModelResponse]): self.responses, self.requests = list(responses), []
    def complete(self, messages: list[dict[str, str]], tools: list[dict]) -> AgentModelResponse:
        self.requests.append({"messages": list(messages), "tools": list(tools)})
        if not self.responses: return AgentModelResponse(content="Scripted provider has no further response.")
        return self.responses.pop(0)

class RuleBasedFakeLLMProvider(LLMProvider):
    """Development-only fake tool caller; decisions are generic intent cues, never historic entities."""
    def complete(self, messages: list[dict[str, str]], tools: list[dict]) -> AgentModelResponse:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tool_messages = [m for m in messages if m["role"] == "tool"]
        is_route = any(token in user.lower() for token in ("route", "路线", "行军"))
        if not tool_messages:
            return AgentModelResponse(tool_calls=[AgentToolCall(id="search-1", name="search_historical_evidence", arguments={"query": user, "top_k": 8})], finish_reason="tool_calls")
        last = tool_messages[-1]["content"]
        if "search_historical_evidence" in last:
            if '"evidence_count": 0' in last:
                return AgentModelResponse(content="Current corpus does not contain sufficient retrieved evidence for this request.")
            if is_route:
                return AgentModelResponse(tool_calls=[AgentToolCall(id="route-1", name="build_historical_route", arguments={"event_id": "evidence-driven-route", "name": "Evidence-supported historical route", "period": "unspecified"})], finish_reason="tool_calls")
            return AgentModelResponse(content="I retrieved historical evidence for this question. The response is limited to the returned primary-source evidence and does not assert a route.")
        if "build_historical_route" in last:
            if '"route_points": 0' in last:
                return AgentModelResponse(content="The retrieved evidence was insufficient to construct a historical route, so no route was returned.")
            return AgentModelResponse(content="I built an evidence-supported schematic historical route from the retrieved sources. It is not an exact march track.")
        return AgentModelResponse(content="The requested tool result has been recorded.")
