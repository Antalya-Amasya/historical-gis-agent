from backend.app.agent.agent import HistoricalGisAgent
from backend.app.agent.llm.fake import ScriptedLLMProvider
from backend.app.models import AgentModelResponse, AgentState, AgentToolCall, Evidence
from backend.app.candidate_routes.historical_reconstruction import HistoricalRouteReconstructor
from backend.app.route_orchestrator import HistoricalRouteOrchestrator
from backend.app.rag.retriever import HistoricalRetriever


class EvidenceRetriever(HistoricalRetriever):
    def retrieve(self, query, top_k=5, filters=None):
        return [
            Evidence(id="nova", author="Polybius", work="Histories", locator="Book III", book="III", chapter="33", excerpt="Hannibal departed from New Carthage.", text="Hannibal departed from New Carthage."),
            Evidence(id="rhone", author="Polybius", work="Histories", locator="Book III", book="III", chapter="42", excerpt="Hannibal crossed the Rhone and entered the Alps.", text="Hannibal crossed the Rhone and entered the Alps."),
            Evidence(id="padus", author="Polybius", work="Histories", locator="Book III", book="III", chapter="50", excerpt="The Alps would bring him into the plains of the Padus.", text="The Alps would bring him into the plains of the Padus."),
        ]


class Geography:
    coordinates = {
        "Carthago Nova": (37.599896, -0.98452, "265849", "exact_site"),
        "Rhodanus": (43.33167, 4.84861, "148168", "representative_point"),
        "Alpes": (43.74465275, 7.40183905, "783", "regional_centroid"),
        "Padus": (44.952389, 12.432028, "393469", "representative_point"),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        latitude, longitude, source_id, role = self.coordinates[arguments["name"]]
        return {
            "found": True, "id": f"pleiades-{source_id}", "canonical_name": arguments["name"],
            "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places",
            "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}",
            "confidence": 0.9, "uncertain": role != "exact_site", "coordinate_role": role,
        }


def call(name, arguments, identifier):
    return AgentModelResponse(tool_calls=[AgentToolCall(id=identifier, name=name, arguments=arguments)], finish_reason="tool_calls")


def test_hannibal_route_request_returns_terrain_reconstruction_presentation_from_existing_pipeline():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal Alps", "top_k": 5}, "search"),
        call("build_historical_route", {"event_id": "hannibal-alps", "name": "Hannibal into Italy", "period": "218 BCE"}, "route"),
        AgentModelResponse(content="The evidence-backed terrain-aware candidate is available."),
    ])
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=4)
    _, state = agent.respond("展示汉尼拔翻越阿尔卑斯路线", AgentState(session_id="phase16-hannibal"))

    assert state.status == "completed"
    assert state.route_intent is not None
    assert state.route_intent.model_dump() == {"intent": "historical_route", "campaign_id": "hannibal_italy_campaign"}
    assert state.historical_route is not None
    presentation = state.historical_route_presentation
    assert presentation is not None
    assert presentation["route_geojson"]["geometry"]["type"] == "LineString"
    assert presentation["route_geojson"]["properties"]["route_type"] == "terrain_aware_historical_reconstruction"
    coordinates = presentation["route_geojson"]["geometry"]["coordinates"]
    assert len(coordinates) > len(state.historical_route.ordered_points)
    # The default demo's existing offline coastal barrier is traversability data, not a route fact.
    assert all(not (0.0 < longitude < 4.75 and latitude < 41.25) for longitude, latitude in coordinates[1:-1])
    assert all(not (4.75 <= longitude < 7.2 and latitude < 42.75) for longitude, latitude in coordinates[1:-1])
    summary = presentation["presentation_summary"]
    assert summary["campaign_id"] == "second_punic_war"
    assert summary["campaign"] == "Second Punic War"
    assert summary["operation_id"] == "hannibal_invasion_italy"
    assert summary["operation"] == "Hannibal's invasion of Italy (218 BCE)"
    assert summary["route_method"] == "terrain_constrained_reconstruction"
    assert summary["evidence_basis"] == summary["sources"]
    assert summary["geographic_constraints"] and summary["uncertainty_notes"]
    assert [step["order"] for step in summary["timeline"]] == [1, 2, 3, 4]
    assert all("id" not in step for step in summary["timeline"])
    quality = presentation["route_geojson"]["properties"]["route_quality"]
    assert quality["coordinate_count"] == len(coordinates)
    assert quality["intermediate_points"] == len(coordinates) - 2
    assert quality["land_ratio"] == 1.0
    assert quality["waypoint_order_preserved"] is True
    assert quality["terrain_source"] == "offline_mock_terrain"
    assert quality["terrain_constrained"] is True
    assert quality["search_constraint"] == "reviewed_historical_corridor"
    assert quality["elevation_gain"] >= 0
    assert quality["max_slope"] >= 0
    assert quality["mountain_penalty"] >= 0
    assert {source["source_type"] for source in quality["data_sources"]} == {"reviewed_annotation"}
    assert all({"dataset_name", "version", "license", "confidence"} <= source.keys() for source in quality["data_sources"])
    waypoint_coordinates = [
        [point.historical_place.longitude, point.historical_place.latitude]
        for point in state.historical_route.ordered_points
    ]
    assert [coordinates.index(point) for point in waypoint_coordinates] == sorted(coordinates.index(point) for point in waypoint_coordinates)
    assert "#" not in summary["historical_context"]
    assert all("polybius" not in source.lower() or "chunk" not in source.lower() for source in summary["sources"])
    point_features = [feature for feature in presentation["geojson"]["features"] if feature["geometry"] and feature["geometry"]["type"] == "Point"]
    corridor_features = [feature for feature in presentation["geojson"]["features"] if feature["properties"].get("layer_type") == "uncertainty_corridor"]
    assert point_features and all(feature["geometry"]["type"] == "Point" for feature in point_features)
    assert corridor_features and all(feature["geometry"]["type"] == "Polygon" for feature in corridor_features)
    assert all(feature["properties"]["evidence_refs"] for feature in point_features)
    assert all(panel["source_references"] for panel in presentation["knowledge_panels"])
    assert [entry.tool_name for entry in state.tool_history] == ["search_historical_evidence", "build_historical_route"]


def test_hannibal_orchestrator_calls_terrain_reconstructor_and_emits_display_safe_summary():
    class RecordingReconstructor(HistoricalRouteReconstructor):
        called = False

        def reconstruct(self, *args, **kwargs):
            self.called = True
            return super().reconstruct(*args, **kwargs)

    reconstructor = RecordingReconstructor()
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal Alps", "top_k": 5}, "search"),
        call("build_historical_route", {"event_id": "hannibal-alps", "name": "Hannibal into Italy", "period": "218 BCE"}, "route"),
        AgentModelResponse(content="# Internal-looking model prose must not be used by presentation."),
    ])
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=4)
    agent.tools.route_orchestrator = HistoricalRouteOrchestrator(reconstructor=reconstructor)

    _, state = agent.respond("展示汉尼拔进入意大利路线", AgentState(session_id="phase17-5-summary"))

    assert reconstructor.called is True
    assert state.historical_route_presentation is not None
    summary = state.historical_route_presentation["presentation_summary"]
    assert summary["title"] == "Hannibal's invasion of Italy (218 BCE)"
    assert "Internal-looking" not in summary["historical_context"]
    assert all("#" not in value for value in summary.values() if isinstance(value, str))


def test_unrecognized_route_request_does_not_receive_a_campaign_orchestrator_presentation():
    provider = ScriptedLLMProvider([AgentModelResponse(content="Evidence is insufficient to generate a route.")])
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=1)
    _, state = agent.respond("展示未知战役路线", AgentState(session_id="phase16-unknown"))
    assert state.route_intent is None
    assert state.historical_route_presentation is None
