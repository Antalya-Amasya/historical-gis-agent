import pytest
from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, PlaceSpatialSemantics
from backend.app.rag.retriever import EmptyHistoricalRetriever
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology
from backend.app.routes.extractor import RouteBuildOutcome
from backend.app.routes.event_route_orchestration import EventRouteOutcome


def road_graph():
    def road(identifier, geometry):
        return RomanRoadSegment(identifier, 0, tuple(geometry), "Main Road", 1, "Certain", identifier, "citation", "bibliography", RoadChronology(None, None, None, None, {}), None, None, None)
    return RomanRoadGraph.from_segments([road("ab", ((0, 0), (1, 0))), road("bc", ((1, 0), (2, 0)))], snap_tolerance_m=5)


def route(names):
    points = []
    for index, (name, semantic) in enumerate(names, 1):
        points.append(HistoricalRoutePoint(sequence=index, historical_place=HistoricalPlace(id=name, canonical_name=name, longitude=float(index - 1), latitude=0, source="fixture", confidence=.9, spatial_semantics=semantic), event_summary="fixture evidence", evidence_refs=[f"e-{name}"], confidence=.9))
    return HistoricalRoute(id="generic-route", event_id="generic", name="generic route", period="test", ordered_points=points, geometry=GeoJsonLineString(coordinates=[(p.historical_place.longitude, p.historical_place.latitude) for p in points]), historical_confidence=.9)


class FixedExtractor:
    def __init__(self, item): self.item = item
    def build_with_diagnostics(self, *_args, **_kwargs): return RouteBuildOutcome(self.item, {"reason_codes": []})


def chat_agent(item):
    provider = ScriptedLLMProvider([AgentModelResponse(tool_calls=[AgentToolCall(id="route", name="build_historical_route", arguments={"event_id":"generic","name":"generic","period":"test"})], finish_reason="tool_calls"), AgentModelResponse(content="Agent answer remains available.")])
    agent = HistoricalGisAgent(provider, EmptyHistoricalRetriever(), max_steps=3)
    agent.tools.route_extractor = FixedExtractor(item)
    agent.tools.roman_road_orchestrator = RomanRoadRouteOrchestrator(RomanRoadCandidateService(road_graph(), max_access_distance_m=500))
    return agent


@pytest.fixture
def client(monkeypatch):
    yield TestClient(main.app), monkeypatch


def invoke(client, monkeypatch, item):
    monkeypatch.setattr(main, "agent", chat_agent(item))
    return client.post("/api/v1/agent/chat", json={"session_id":"generic", "message":"show a route"})


def test_answer_only_chat_response_has_no_presentation(client):
    app, monkeypatch = client
    monkeypatch.setattr(main, "agent", HistoricalGisAgent(ScriptedLLMProvider([AgentModelResponse(content="Ordinary answer.")]), EmptyHistoricalRetriever(), max_steps=1))
    body = app.post("/api/v1/agent/chat", json={"session_id":"answer", "message":"ordinary question"}).json()
    assert body["reply"] and body["state"]["historical_route_presentation"] is None


def test_non_route_output_does_not_force_gis_reconstruction():
    agent = chat_agent(route([("A", PlaceSpatialSemantics.SETTLEMENT), ("B", PlaceSpatialSemantics.SETTLEMENT)]))
    state = AgentState(session_id="non-route", requested_output="answer")
    payload, _ = agent.tools.execute("build_historical_route", {"event_id": "generic", "name": "generic", "period": "test"}, state)
    assert payload["success"] is True
    assert state.historical_route is not None
    assert state.historical_route_presentation is None
    assert "gis_reconstruction" not in state.historical_route_diagnostics


def test_complete_partial_and_unavailable_roman_road_chat_contract(client):
    app, monkeypatch = client
    complete = invoke(app, monkeypatch, route([("A", PlaceSpatialSemantics.SETTLEMENT), ("B", PlaceSpatialSemantics.SETTLEMENT)]))
    assert complete.status_code == 200
    complete_presentation = complete.json()["state"]["historical_route_presentation"]
    complete_state = complete.json()["state"]
    assert complete.json()["reply"] and complete_presentation["route"]["generation_method"] == "ROMAN_ROAD_NETWORK"
    assert complete_state["historical_route_diagnostics"]["gis_reconstruction"] == {"attempted": True, "pipeline": "roman_road_orchestrator", "status": "COMPLETE"}
    assert [point["historical_place"]["canonical_name"] for point in complete_state["historical_route"]["ordered_points"]] == ["A", "B"]
    assert complete_presentation["road_network"]["route_status"] == "COMPLETE"
    assert any(f["properties"].get("segment_role") == "roman_road" and f["geometry"] for f in complete_presentation["geojson"]["features"])
    assert not any(f["properties"].get("segment_role") == "failed_gap" for f in complete_presentation["geojson"]["features"])
    partial = invoke(app, monkeypatch, route([("A", PlaceSpatialSemantics.SETTLEMENT), ("B", PlaceSpatialSemantics.SETTLEMENT), ("C", PlaceSpatialSemantics.RIVER)]))
    partial_presentation = partial.json()["state"]["historical_route_presentation"]
    assert partial.status_code == 200 and partial.json()["reply"] and partial_presentation["road_network"]["route_status"] == "PARTIAL"
    gap = next(f for f in partial_presentation["geojson"]["features"] if f["properties"].get("segment_role") == "failed_gap")
    assert gap["geometry"] is None and gap["properties"]["failure_status"] == "RIVER_GEOMETRY_UNAVAILABLE"
    unavailable = invoke(app, monkeypatch, route([("A", PlaceSpatialSemantics.RIVER), ("C", PlaceSpatialSemantics.RIVER)]))
    unavailable_presentation = unavailable.json()["state"]["historical_route_presentation"]
    assert unavailable.status_code == 200 and unavailable.json()["reply"] and unavailable_presentation["road_network"]["route_status"] == "UNAVAILABLE"
    assert not any(f["properties"].get("segment_role") == "roman_road" for f in unavailable_presentation["geojson"]["features"])
    unavailable_state = unavailable.json()["state"]
    assert unavailable_state["historical_route"] is not None
    assert unavailable_state["historical_route_diagnostics"]["gis_reconstruction"]["status"] == "UNAVAILABLE"
    assert not any(f["geometry"] for f in unavailable_presentation["geojson"]["features"] if f["properties"].get("segment_role") == "failed_gap")


def test_event_first_route_reaches_roman_road_pipeline_once_without_legacy_mix():
    item = route([("A", PlaceSpatialSemantics.SETTLEMENT), ("B", PlaceSpatialSemantics.SETTLEMENT)])
    agent = chat_agent(item)

    class EventFirst:
        def build_with_diagnostics(self, *_args, **_kwargs):
            return EventRouteOutcome(item, (), {"route_source": "event_anchor", "reason_codes": []})

    class LegacyMustNotRun:
        def build_with_diagnostics(self, *_args, **_kwargs):
            raise AssertionError("legacy fallback must not run after an accepted event-first route")

    calls = 0
    original = agent.tools.roman_road_orchestrator.build_roman_road_candidates
    def record(historical_route):
        nonlocal calls
        calls += 1
        return original(historical_route)

    agent.tools.event_route_builder = EventFirst()
    agent.tools.route_extractor = LegacyMustNotRun()
    agent.tools.roman_road_orchestrator.build_roman_road_candidates = record
    _, state = agent.respond("show a route", AgentState(session_id="event-first"))
    assert calls == 1
    assert state.historical_route is item
    assert state.historical_route_diagnostics["route_source"] == "event_anchor"
    assert state.historical_route_diagnostics["gis_reconstruction"]["status"] == "COMPLETE"


def test_repeated_route_tool_call_reuses_existing_gis_result():
    item = route([("A", PlaceSpatialSemantics.SETTLEMENT), ("B", PlaceSpatialSemantics.SETTLEMENT)])
    agent = chat_agent(item)
    state = AgentState(session_id="once", requested_output="historical_route")
    calls = 0
    original = agent.tools.roman_road_orchestrator.build_roman_road_candidates
    def record(historical_route):
        nonlocal calls
        calls += 1
        return original(historical_route)

    agent.tools.roman_road_orchestrator.build_roman_road_candidates = record
    agent.tools.execute("build_historical_route", {"event_id": "generic", "name": "generic", "period": "test"}, state)
    payload, _ = agent.tools.execute("build_historical_route", {"event_id": "different", "name": "different", "period": "test"}, state)
    assert calls == 1
    assert payload["result"]["gis_reconstruction"]["attempted"] is True
