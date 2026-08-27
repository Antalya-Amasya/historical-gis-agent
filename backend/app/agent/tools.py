"""Allowlisted Agent-level tools; no filesystem, shell, or arbitrary HTTP access."""
from __future__ import annotations

from time import perf_counter
from pathlib import Path
from backend.app.models import AgentState, HistoricalEvent, HistoricalPlace
from backend.app.core.config import settings
from backend.app.candidate_routes.historical_reconstruction import RealTerrainGraphProvider
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.extractor import HistoricalRouteExtractor
from backend.app.route_orchestrator import (
    HistoricalCampaignIntentRegistry,
    HistoricalRouteOrchestrator,
    RouteOrchestrationError,
)

TOOL_SCHEMAS = [
 {"name":"search_historical_evidence","description":"Search the frozen semantic primary-source retriever. Use it before historical claims or routes. Returns Evidence only; it does not prove an unsupported event.","input_schema":{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer","minimum":1,"maximum":20},"author":{"type":"string"},"book":{"type":"string"}},"required":["query"]}},
 {"name":"resolve_ancient_place","description":"Resolve one evidence-mentioned ancient place through Geography MCP. It does not create or substitute for a HistoricalRoute. Coordinates are audited repository data; never use it to infer an unmentioned place.","input_schema":{"type":"object","properties":{"name":{"type":"string"},"period":{"type":"string"}},"required":["name"]}},
 {"name":"calculate_distance","description":"Calculate local Haversine geodesic distance for supplied points. It is not a road or march distance.","input_schema":{"type":"object","properties":{"point_a":{"type":"object"},"point_b":{"type":"object"}},"required":["point_a","point_b"]}},
 {"name":"get_elevation","description":"Return elevation at an input coordinate. Terrain context does not prove a historical route.","input_schema":{"type":"object","properties":{"latitude":{"type":"number"},"longitude":{"type":"number"}},"required":["latitude","longitude"]}},
 {"name":"get_elevation_profile","description":"Return elevation only for supplied coordinates; it does not generate a route or score one.","input_schema":{"type":"object","properties":{"points":{"type":"array"}},"required":["points"]}},
 {"name":"build_historical_route","description":"Use when the user requests a historical route or map reconstruction. This is the only tool that produces the audited HistoricalRoute from accumulated Evidence; it resolves coordinates via Geography MCP and never generates a candidate terrain route.","input_schema":{"type":"object","properties":{"event_id":{"type":"string"},"name":{"type":"string"},"period":{"type":"string"}},"required":["event_id","name","period"]}},
]

class AgentToolRegistry:
    def __init__(self, retriever: HistoricalRetriever, geography_client, *, route_orchestrator=None, campaign_registry=None):
        self.retriever, self.geography_client = retriever, geography_client
        self.route_extractor = HistoricalRouteExtractor(geography_client)
        self.route_orchestrator = route_orchestrator or HistoricalRouteOrchestrator(
            terrain_graph_provider=self._terrain_graph_provider_from_settings(),
        )
        self.campaign_registry = campaign_registry or HistoricalCampaignIntentRegistry()

    @staticmethod
    def _terrain_graph_provider_from_settings():
        if not settings.dem_hgt_dir:
            return None
        hgt_dir = Path(settings.dem_hgt_dir)
        if not hgt_dir.is_dir():
            return None
        return RealTerrainGraphProvider(MosaicDEMProvider(hgt_dir=hgt_dir))

    def resolve_route_intent(self, message: str):
        return self.campaign_registry.resolve(message)

    @property
    def schemas(self): return TOOL_SCHEMAS
    def execute(self, name: str, arguments: dict, state: AgentState) -> tuple[dict, str]:
        started = perf_counter()
        try:
            result, summary = self._execute(name, arguments, state)
            success = True
        except Exception as exc:
            result, summary, success = {"error": type(exc).__name__}, f"{name} failed: {str(exc)[:240]}", False
        elapsed = int((perf_counter() - started) * 1000)
        return {"success": success, "result": result, "summary": summary, "duration_ms": elapsed}, summary
    def _execute(self, name: str, arguments: dict, state: AgentState) -> tuple[dict, str]:
        if name == "search_historical_evidence":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip(): raise ValueError("query must be a non-empty string")
            top_k = arguments.get("top_k", 5)
            if not isinstance(top_k, int) or not 1 <= top_k <= 20: raise ValueError("top_k must be 1..20")
            filters = {key: arguments[key] for key in ("author", "book") if isinstance(arguments.get(key), str) and arguments[key]}
            evidence = self.retriever.retrieve(query, top_k, filters or None)
            accumulated = {item.id: item for item in state.historical_evidence}
            accumulated.update({item.id: item for item in evidence})
            state.historical_evidence = list(accumulated.values())
            return {"evidence": [item.model_dump(mode="json") for item in evidence], "result_count": len(evidence)}, f"search_historical_evidence evidence_count={len(evidence)} accumulated_evidence_count={len(state.historical_evidence)}"
        if name in {"resolve_ancient_place", "calculate_distance", "get_elevation", "get_elevation_profile"}:
            result = self.geography_client.call(name, arguments)
            if name == "resolve_ancient_place" and result.get("found"):
                state.resolved_places.append(HistoricalPlace.model_validate({k:v for k,v in result.items() if k != "found"}))
            return result, f"{name} completed"
        if name == "build_historical_route":
            for required in ("event_id", "name", "period"):
                if not isinstance(arguments.get(required), str) or not arguments[required].strip(): raise ValueError(f"{required} must be a non-empty string")
            route = self.route_extractor.build(state.historical_evidence, event_id=arguments["event_id"], name=arguments["name"], period=arguments["period"])
            if route is None:
                return {"route": None}, "build_historical_route route_points=0 (insufficient evidence or resolved anchors)"
            state.historical_route = route
            state.current_event = HistoricalEvent(id=arguments["event_id"], name=arguments["name"], period=arguments["period"], summary="Evidence-supported schematic reconstruction.", places=[p.historical_place for p in route.ordered_points], evidence=state.historical_evidence, uncertainty_note="Historical reconstruction only; not an exact march track.")
            if state.route_intent is not None:
                try:
                    presentation = self.route_orchestrator.present(
                        state.route_intent, route, state.historical_evidence,
                    )
                except RouteOrchestrationError as exc:
                    return {"route": route.model_dump(mode="json"), "presentation": None}, f"build_historical_route route_points={len(route.ordered_points)} terrain_presentation_unavailable={type(exc).__name__}"
                state.historical_route_presentation = presentation.model_dump(mode="json")
                return {"route": route.model_dump(mode="json"), "presentation": state.historical_route_presentation}, f"build_historical_route route_points={len(route.ordered_points)} terrain_presentation=ready"
            return {"route": route.model_dump(mode="json")}, f"build_historical_route route_points={len(route.ordered_points)}"
        raise ValueError(f"Unknown agent tool: {name}")
