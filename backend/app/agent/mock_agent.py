from backend.app.models import AgentState, HistoricalEvent, HistoricalPlace
from backend.app.geography.mcp_client import GeographyMcpClient


class MockAgent:
    def __init__(self, geography_client: GeographyMcpClient | None = None):
        self.geography_client = geography_client or GeographyMcpClient()

    def respond(self, message: str, state: AgentState) -> tuple[str, AgentState]:
        state.messages.append({"role": "user", "content": message})
        if any(token in message.lower() for token in ("hannibal", "汉尼拔", "alps", "阿尔卑斯")):
            places = []
            for name in ("Carthago", "Carthago Nova"):
                result = self.geography_client.call("resolve_ancient_place", {"name": name})
                if result.get("found"):
                    places.append(HistoricalPlace.model_validate({key:value for key,value in result.items() if key != "found"}))
            event = HistoricalEvent(id="hannibal-alps-218-bc", name="Hannibal s Alpine crossing", period="218 BCE", summary="MCP-resolved campaign-context places; no route is asserted.", places=places, uncertainty_note="The exact Alpine route remains disputed.")
            state.current_event = event
            reply = "已识别演示事件；地点由 Geography MCP 返回，不主张具体路线。"
        else:
            reply = "Mock Agent 已收到问题。"
        state.messages.append({"role": "assistant", "content": reply})
        return reply, state
