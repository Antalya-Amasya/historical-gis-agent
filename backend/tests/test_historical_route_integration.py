from backend.app.agent.mock_agent import MockAgent
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever


class FakeRetriever(HistoricalRetriever):
    def retrieve(self, query: str, top_k: int = 5, filters=None) -> list[Evidence]:
        return [Evidence(id="nova", author="Polybius", work="Histories", locator="III", excerpt="New Carthage"), Evidence(id="rhone", author="Polybius", work="Histories", locator="III", excerpt="Rhone"), Evidence(id="po", author="Polybius", work="Histories", locator="III", excerpt="Po Valley")]


class FakeMcpClient:
    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        values = {"Carthago Nova": (37.6, -0.98, "265849"), "Rhodanus": (43.33, 4.85, "148168"), "Padus": (44.95, 12.43, "393469")}
        latitude, longitude, source_id = values[arguments["name"]]
        return {"found": True, "id": f"pleiades-{source_id}", "canonical_name": arguments["name"], "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places", "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}", "confidence": 0.9, "uncertain": False}


def test_mock_agent_uses_retrieved_evidence_then_mcp_to_build_route() -> None:
    _, state = MockAgent(geography_client=FakeMcpClient(), evidence_retriever=FakeRetriever()).respond("展示汉尼拔218 BCE路线", AgentState(session_id="test"))
    assert state.historical_route is not None
    assert state.historical_route.geometry.coordinates[0] == (-0.98, 37.6)
    assert state.historical_evidence[0].id == "nova"
