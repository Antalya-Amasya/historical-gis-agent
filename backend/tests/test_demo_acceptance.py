"""Offline Phase 18 acceptance checks; no live LLM or external map services."""
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import RuleBasedFakeLLMProvider, ScriptedLLMProvider
from backend.app.historical_campaign_presentation_service import CaesarCampaignPresentationFactory
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.rag.retriever import HistoricalRetriever


class Retriever(HistoricalRetriever):
    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, query, top_k=5, filters=None):
        return list(self.evidence)


class Geography:
    values = {
        "Carthago Nova": (37.599896, -0.98452, "265849", "exact_site"),
        "Rhodanus": (43.33167, 4.84861, "148168", "representative_point"),
        "Alpes": (43.74465275, 7.40183905, "783", "regional_centroid"),
        "Padus": (44.952389, 12.432028, "393469", "representative_point"),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        latitude, longitude, source_id, role = self.values[arguments["name"]]
        return {"found": True, "id": f"pleiades-{source_id}", "canonical_name": arguments["name"], "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places", "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}", "confidence": 0.9, "uncertain": role != "exact_site", "coordinate_role": role}


def evidence(identifier, author, book, chapter, text):
    return Evidence(id=identifier, author=author, work="Histories" if author == "Polybius" else "Ab Urbe Condita", locator=f"Book {book}", book=book, chapter=chapter, excerpt=text, text=text)


def tool(name, arguments, identifier):
    return AgentModelResponse(tool_calls=[AgentToolCall(id=identifier, name=name, arguments=arguments)], finish_reason="tool_calls")


def test_hannibal_demo_request_returns_intent_evidence_backed_line_and_markers():
    retrieved = [
        evidence("polybius-nova", "Polybius", "III", "33", "Hannibal departed from New Carthage."),
        evidence("polybius-rhone", "Polybius", "III", "42", "He crossed the Rhone and entered the Alps."),
        evidence("polybius-padus", "Polybius", "III", "50", "The Alps would bring him into the plains of the Padus."),
        evidence("livy-rhone", "Livy", "XXI", "31", "Hannibal crossed the Rhone."),
    ]
    provider = ScriptedLLMProvider([
        tool("search_historical_evidence", {"query": "Hannibal Alps", "top_k": 5}, "search"),
        tool("build_historical_route", {"event_id": "hannibal-218", "name": "Hannibal into Italy", "period": "218 BCE"}, "route"),
        AgentModelResponse(content="The terrain-aware candidate is available."),
    ])
    _, state = HistoricalGisAgent(provider, Retriever(retrieved), Geography(), max_steps=4).respond(
        "展示汉尼拔翻越阿尔卑斯进入意大利的路线", AgentState(session_id="phase18-hannibal"),
    )
    presentation = state.historical_route_presentation
    assert state.route_intent and state.route_intent.campaign_id == "hannibal_italy_campaign"
    assert {item.author for item in state.historical_evidence} == {"Polybius", "Livy"}
    assert presentation and presentation["route_geojson"]["geometry"]["type"] == "LineString"
    assert presentation["route_geojson"]["properties"]["explanation"]
    point_features = [feature for feature in presentation["geojson"]["features"] if feature["geometry"] and feature["geometry"]["type"] == "Point"]
    corridor_features = [feature for feature in presentation["geojson"]["features"] if feature["properties"].get("layer_type") == "uncertainty_corridor"]
    assert point_features and all(feature["geometry"]["type"] == "Point" for feature in point_features)
    assert corridor_features and all(feature["geometry"]["type"] == "Polygon" for feature in corridor_features)
    assert all(panel["source_references"] for panel in presentation["knowledge_panels"])
    # Provenance IDs remain in the transport DTO; frontend popup helpers expose counts/source labels only.


def test_caesar_reviewed_presentation_keeps_book_i_and_vii_and_alesia():
    presentation = CaesarCampaignPresentationFactory().build()
    books = {waypoint.source_book for waypoint in presentation.waypoints}
    assert {"1", "7"}.issubset(books)
    assert any("Alesia" in waypoint.name for waypoint in presentation.waypoints)
    assert presentation.route_geojson["geometry"]["type"] == "LineString"


def test_unknown_arthur_route_stays_without_campaign_intent_or_presentation():
    provider = ScriptedLLMProvider([AgentModelResponse(content="Current corpus does not contain reliable evidence for this route.")])
    _, state = HistoricalGisAgent(provider, Retriever([]), Geography(), max_steps=1).respond(
        "展示亚瑟王真实行军路线", AgentState(session_id="phase18-arthur"),
    )
    assert state.route_intent is None
    assert state.historical_route is None
    assert state.historical_route_presentation is None


def test_fake_agent_does_not_claim_a_unique_alpine_pass():
    _, state = HistoricalGisAgent(
        RuleBasedFakeLLMProvider(),
        Retriever([evidence("alps", "Polybius", "III", "50", "Hannibal entered the Alps.")]),
        Geography(),
        max_steps=3,
    ).respond("告诉我汉尼拔准确翻越的是哪一个阿尔卑斯山口", AgentState(session_id="phase18-pass"))
    answer = state.final_answer or ""
    assert "unique pass" not in answer.lower()
    assert state.historical_route is None
