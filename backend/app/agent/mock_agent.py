from backend.app.models import AgentState, HistoricalEvent, HistoricalPlace


ALPS_PLACE = HistoricalPlace(
    id="demo-alps",
    canonical_name="Alpes",
    modern_name="Alps",
    latitude=45.85,
    longitude=7.35,
    period="-218",
    source="Phase 0 demo place repository",
    confidence=0.55,
    uncertain=True,
)


class MockAgent:
    def respond(self, message: str, state: AgentState) -> tuple[str, AgentState]:
        state.messages.extend([{"role": "user", "content": message}])
        if any(token in message.lower() for token in ("hannibal", "汉尼拔", "alps", "阿尔卑斯")):
            event = HistoricalEvent(
                id="hannibal-alps-218-bc",
                name="Hannibal's Alpine crossing",
                period="-218",
                summary="A Phase 0 demo event representing Hannibal's 218 BC crossing of the Alps.",
                places=[ALPS_PLACE],
                uncertainty_note="The exact Alpine route remains disputed; no route is asserted in Phase 0.",
            )
            state.current_event = event
            state.historical_period = "-218"
            reply = (
                "已识别演示事件：公元前218年汉尼拔翻越阿尔卑斯。"
                "地图阶段将使用 Alpes 标记；具体通道与路线仍存在重大史料争议，"
                "Phase 0 不对实际路线作判断。"
            )
        else:
            reply = "Mock Agent 已收到问题。Phase 0 目前支持汉尼拔翻越阿尔卑斯的最小演示。"
        state.messages.append({"role": "assistant", "content": reply})
        return reply, state

