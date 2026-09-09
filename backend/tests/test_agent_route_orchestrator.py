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
            Evidence(id="movement", author="Polybius", work="Histories", locator="Book III", book="III", chapter="42", excerpt="Hannibal marched from Genava to Lutetia.", text="Hannibal marched from Genava to Lutetia."),
            Evidence(id="padus", author="Polybius", work="Histories", locator="Book III", book="III", chapter="50", excerpt="The Alps would bring him into the plains of the Padus.", text="The Alps would bring him into the plains of the Padus."),
        ]


class Geography:
    coordinates = {
        "Carthago Nova": (37.599896, -0.98452, "265849", "exact_site", "port"),
        "Genava": (46.2044, 6.1432, "167901", "exact_site", "settlement"),
        "Lutetia": (48.8566, 2.3522, "108348", "exact_site", "settlement"),
        "Rhodanus": (43.33167, 4.84861, "148168", "representative_point", "river"),
        "Alpes": (43.74465275, 7.40183905, "783", "regional_centroid", "mountain_region"),
        "Padus": (44.952389, 12.432028, "393469", "representative_point", "river"),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        latitude, longitude, source_id, role, semantics = self.coordinates[arguments["name"]]
        return {
            "found": True, "id": f"pleiades-{source_id}", "canonical_name": arguments["name"],
            "latitude": latitude, "longitude": longitude, "source": "Pleiades: A Gazetteer of Past Places",
            "source_id": source_id, "source_url": f"https://pleiades.stoa.org/places/{source_id}",
            "confidence": 0.9, "uncertain": role != "exact_site", "coordinate_role": role,
            "spatial_semantics": semantics,
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
    # This test verifies the documented offline demo constraints, independently
    # of a developer's DEM_HGT_DIR environment setting.
    agent.tools.route_orchestrator = HistoricalRouteOrchestrator()
    _, state = agent.respond("展示汉尼拔翻越阿尔卑斯路线", AgentState(session_id="phase16-hannibal"))

    assert state.status == "completed_with_guardrail"
    assert state.final_grounding_status == "guardrail_fallback"
    assert state.route_intent is not None
    assert state.route_intent.model_dump() == {
        "intent": "historical_route",
        "campaign_id": "hannibal_italy_campaign",
        "entity": "hannibal_alpine_crossing",
        "route_type": "movement",
    }
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
    assert summary["operation_id"] == "hannibal_alpine_crossing"
    assert summary["operation"] == "Hannibal's Alpine Crossing"
    assert summary["route_method"] == "terrain_constrained_reconstruction"
    assert summary["evidence_basis"] == summary["sources"]
    assert summary["geographic_constraints"] and summary["uncertainty_notes"]
    assert [step["order"] for step in summary["timeline"]] == [1, 2]
    assert all("id" not in step for step in summary["timeline"])
    quality = presentation["route_geojson"]["properties"]["route_quality"]
    assert quality["coordinate_count"] == len(coordinates)
    assert quality["intermediate_points"] == len(coordinates) - 2
    assert quality["land_ratio"] == 1.0
    assert quality["waypoint_order_preserved"] is True
    assert quality["terrain_source"] == "offline_mock_terrain"
    assert quality["terrain_constrained"] is True
    assert quality["applied_constraints"] == ["synthetic_mock_terrain", "mock_ocean_blocking", "mock_route_search_bounds", "mock_alpine_terrain_multiplier"]
    assert quality["elevation_gain"] >= 0
    assert quality["max_slope"] >= 0
    assert quality["mountain_penalty"] >= 0
    assert quality["segment_ledger"]
    assert quality["segment_ledger"][0]["geometry_role"] == "algorithmic_candidate"
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
        graph_cell_size_m = None

        def reconstruct(self, *args, **kwargs):
            self.called = True
            self.graph_cell_size_m = args[1].spec.cell_size_m
            return super().reconstruct(*args, **kwargs)

    reconstructor = RecordingReconstructor()
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal Alps", "top_k": 5}, "search"),
        call("build_historical_route", {"event_id": "hannibal-alps", "name": "Hannibal into Italy", "period": "218 BCE"}, "route"),
        AgentModelResponse(content="# Internal-looking model prose must not be used by presentation."),
    ])
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=4)
    agent.tools.route_orchestrator = HistoricalRouteOrchestrator(reconstructor=reconstructor, cell_size_m=5_000)

    _, state = agent.respond("展示汉尼拔进入意大利路线", AgentState(session_id="phase17-5-summary"))

    assert reconstructor.called is True
    assert reconstructor.graph_cell_size_m == 5_000
    assert state.historical_route_presentation is not None
    summary = state.historical_route_presentation["presentation_summary"]
    assert summary["title"] == "Hannibal's invasion of Italy"
    assert "Internal-looking" not in summary["historical_context"]
    assert all("#" not in value for value in summary.values() if isinstance(value, str))


def test_presentation_reports_only_provider_applied_constraints():
    provider = ScriptedLLMProvider([
        call("search_historical_evidence", {"query": "Hannibal Alps", "top_k": 5}, "search"),
        call("build_historical_route", {"event_id": "hannibal-alps", "name": "Hannibal into Italy", "period": "218 BCE"}, "route"),
        AgentModelResponse(content="Candidate available."),
    ])
    from backend.app.candidate_routes.historical_reconstruction import OfflineMockTerrainGraphProvider
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=4)
    agent.tools.route_orchestrator = HistoricalRouteOrchestrator(
        terrain_graph_provider=OfflineMockTerrainGraphProvider(applied_constraints=["fixture_constraint"]),
    )
    _, state = agent.respond("展示汉尼拔翻越阿尔卑斯路线", AgentState(session_id="constraint-truth"))
    route_geojson = state.historical_route_presentation["route_geojson"]
    assert route_geojson["properties"]["applied_constraints"] == ["fixture_constraint"]
    assert state.historical_route_presentation["presentation_summary"]["geographic_constraints"] == ["fixture_constraint"]


def test_unrecognized_route_request_does_not_receive_a_campaign_orchestrator_presentation():
    provider = ScriptedLLMProvider([AgentModelResponse(content="Evidence is insufficient to generate a route.")])
    agent = HistoricalGisAgent(provider, EvidenceRetriever(), Geography(), max_steps=1)
    _, state = agent.respond("展示未知战役路线", AgentState(session_id="phase16-unknown"))
    assert state.route_intent is None
    assert state.historical_route_presentation is None
