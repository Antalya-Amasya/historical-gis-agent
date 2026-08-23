from backend.app.models import AgentState, HistoricalEvent
from geography_mcp.tools import hannibal_demo_places


class MockAgent:
    def respond(self, message: str, state: AgentState) -> tuple[str, AgentState]:
        state.messages.extend([{"role": "user", "content": message}])
        if any(token in message.lower() for token in ("hannibal", "汉尼拔", "alps", "阿尔卑斯")):
            event = HistoricalEvent(
                id="hannibal-alps-218-bc",
                name="Hannibal's Alpine crossing",
                period="218 BCE",
                summary=(
                    "Two fixed, source-attributed places provide campaign context for the 218 BCE demo. "
                    "They are markers only and do not draw or imply a historical route."
                ),
                places=hannibal_demo_places(),
                uncertainty_note=(
                    "The exact Alpine route remains disputed. Phase 1 deliberately shows no route, "
                    "pass, or inferred intermediate location."
                ),
            )
            state.current_event = event
            state.historical_period = "218 BCE"
            reply = (
                "已识别演示事件：公元前218年汉尼拔翻越阿尔卑斯。"
                "地图将展示经过人工审计、附 Pleiades 来源的地点标记；"
                "不主张具体路线或通道。"
            )
        else:
            reply = "Mock Agent 已收到问题。Phase 1 目前支持汉尼拔翻越阿尔卑斯的地点标记演示。"
        state.messages.append({"role": "assistant", "content": reply})
        return reply, state
