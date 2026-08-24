from backend.app.geography.mcp_client import GeographyMcpClient
from backend.app.models import AgentState, HistoricalEvent, HistoricalPlace
from backend.app.rag.retriever import EmptyHistoricalRetriever, HistoricalRetriever
from backend.app.routes.extractor import HistoricalRouteExtractor


class MockAgent:
    def __init__(self, geography_client: GeographyMcpClient | None = None, evidence_retriever: HistoricalRetriever | None = None):
        self.geography_client = geography_client or GeographyMcpClient()
        self.evidence_retriever = evidence_retriever or EmptyHistoricalRetriever()
        self.route_extractor = HistoricalRouteExtractor(self.geography_client)

    def respond(self, message: str, state: AgentState) -> tuple[str, AgentState]:
        state.messages.append({"role": "user", "content": message})
        if any(token in message.lower() for token in ("hannibal", "汉尼拔", "alps", "阿尔卑斯")):
            evidence = self.evidence_retriever.retrieve(message, top_k=10)
            state.historical_evidence = evidence
            route = self.route_extractor.extract_hannibal_218(evidence)
            if route:
                places = [point.historical_place for point in route.ordered_points]
                state.historical_route = route
                summary = "Evidence-supported stages resolved through Geography MCP; schematic links only."
                reply = "已根据检索到的史料生成示意性历史行动路线；各节点和坐标均由 Geography MCP 返回。"
            else:
                places = []
                for name in ("Carthago", "Carthago Nova"):
                    result = self.geography_client.call("resolve_ancient_place", {"name": name})
                    if result.get("found"):
                        places.append(HistoricalPlace.model_validate({key: value for key, value in result.items() if key != "found"}))
                summary = "MCP-resolved campaign-context places; evidence is insufficient to assert a route."
                reply = "已识别演示事件；证据不足时不生成路线；不主张具体路线，地点由 Geography MCP 返回。"
            state.current_event = HistoricalEvent(id="hannibal-alps-218-bc", name="Hannibal's Italian campaign approach", period="218 BCE", summary=summary, places=places, evidence=evidence, uncertainty_note="Historical Route is a source-driven reconstruction; it is not an exact march track.")
        else:
            reply = "Mock Agent 已收到问题。"
        state.messages.append({"role": "assistant", "content": reply})
        return reply, state
