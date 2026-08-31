"""Allowlisted Agent-level tools; no filesystem, shell, or arbitrary HTTP access."""
from __future__ import annotations

from time import perf_counter
from pathlib import Path
from backend.app.models import AgentState, HistoricalEvent, HistoricalPlace, HistoricalRouteIntent
from backend.app.core.config import settings
from backend.app.candidate_routes.historical_reconstruction import RealTerrainGraphProvider
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.extractor import HistoricalRouteExtractor
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.route_provenance import HistoricalRouteTraceBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.route_orchestrator import (
    HistoricalCampaignIntentRegistry,
    HistoricalRouteOrchestrator,
    RouteOrchestrationError,
)
from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.roads.itiner_e import RomanRoadGraph

TOOL_SCHEMAS = [
 {"name":"submit_grounded_answer","description":"Internal terminal protocol for an evidence-grounded historical answer. Call only after Evidence is supplied. Select only supplied Evidence IDs; do not include source metadata.","input_schema":{"type":"object","properties":{"answer":{"type":"string"},"evidence_ids":{"type":"array","items":{"type":"string"},"maxItems":8},"insufficient_evidence":{"type":"boolean","default":False}},"required":["answer","evidence_ids"]}},
 {"name":"search_historical_evidence","description":"Search the frozen semantic primary-source retriever. Use it before historical claims or routes. Returns Evidence only; it does not prove an unsupported event.","input_schema":{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer","minimum":1,"maximum":20},"author":{"type":"string"},"book":{"type":"string"}},"required":["query"]}},
 {"name":"resolve_ancient_place","description":"Resolve one evidence-mentioned ancient place through Geography MCP. It does not create or substitute for a HistoricalRoute. Coordinates are audited repository data; never use it to infer an unmentioned place.","input_schema":{"type":"object","properties":{"name":{"type":"string"},"period":{"type":"string"}},"required":["name"]}},
 {"name":"calculate_distance","description":"Calculate local Haversine geodesic distance for supplied points. It is not a road or march distance.","input_schema":{"type":"object","properties":{"point_a":{"type":"object"},"point_b":{"type":"object"}},"required":["point_a","point_b"]}},
 {"name":"get_elevation","description":"Return elevation at an input coordinate. Terrain context does not prove a historical route.","input_schema":{"type":"object","properties":{"latitude":{"type":"number"},"longitude":{"type":"number"}},"required":["latitude","longitude"]}},
 {"name":"get_elevation_profile","description":"Return elevation only for supplied coordinates; it does not generate a route or score one.","input_schema":{"type":"object","properties":{"points":{"type":"array"}},"required":["points"]}},
 {"name":"build_historical_route","description":"Use when the user requests a historical route or map reconstruction. This is the only tool that produces the audited HistoricalRoute from accumulated Evidence; when configured GIS capability is available, it may add an explicitly algorithmic Roman-road or terrain candidate between those already-established waypoints.","input_schema":{"type":"object","properties":{"event_id":{"type":"string"},"name":{"type":"string"},"period":{"type":"string"}},"required":["event_id","name","period"]}},
]

class AgentToolRegistry:
    def __init__(self, retriever: HistoricalRetriever, geography_client, *, route_orchestrator=None, campaign_registry=None, roman_road_orchestrator=None):
        self.retriever, self.geography_client = retriever, geography_client
        self.route_extractor = HistoricalRouteExtractor(geography_client)
        self.event_route_builder = EventAnchorRouteBuilder()
        self.event_extractor = EvidenceGroundedHistoricalEventExtractor()
        self.event_consolidator = HistoricalEventConsolidator()
        self.event_place_resolver = HistoricalEventPlaceResolver(geography_client)
        self.route_orchestrator = route_orchestrator or HistoricalRouteOrchestrator(
            terrain_graph_provider=self._terrain_graph_provider_from_settings(),
        )
        self.campaign_registry = campaign_registry or HistoricalCampaignIntentRegistry()
        self.roman_road_orchestrator = roman_road_orchestrator

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
            candidates, extraction_diagnostics = self.event_extractor.extract(state.historical_evidence, query=query)
            consolidated_events, consolidation_diagnostics = self.event_consolidator.consolidate(candidates)
            state.historical_events, place_diagnostics = self.event_place_resolver.resolve(consolidated_events)
            state.historical_event_diagnostics = {
                "extraction": extraction_diagnostics,
                "consolidation": consolidation_diagnostics,
                "place_resolution": place_diagnostics,
                "statement_candidates": [item.model_dump(mode="json") for item in candidates],
            }
            return {"evidence": [item.model_dump(mode="json") for item in evidence], "result_count": len(evidence)}, f"search_historical_evidence evidence_count={len(evidence)} accumulated_evidence_count={len(state.historical_evidence)}"
        if name in {"resolve_ancient_place", "calculate_distance", "get_elevation", "get_elevation_profile"}:
            result = self.geography_client.call(name, arguments)
            if name == "resolve_ancient_place" and result.get("found"):
                state.resolved_places.append(HistoricalPlace.model_validate({k:v for k,v in result.items() if k != "found"}))
            return result, f"{name} completed"
        if name == "build_historical_route":
            for required in ("event_id", "name", "period"):
                if not isinstance(arguments.get(required), str) or not arguments[required].strip(): raise ValueError(f"{required} must be a non-empty string")
            existing_gis = (state.historical_route_diagnostics or {}).get("gis_reconstruction")
            if state.historical_route is not None and existing_gis and existing_gis.get("attempted"):
                return {
                    "route": state.historical_route.model_dump(mode="json"),
                    "presentation": state.historical_route_presentation,
                    "gis_reconstruction": existing_gis,
                }, f"build_historical_route reused_existing_route route_points={len(state.historical_route.ordered_points)}"
            event_first = self.event_route_builder.build_with_diagnostics(
                state.historical_events,
                state.historical_evidence,
                event_id=arguments["event_id"],
                name=arguments["name"],
                period=arguments["period"],
                allow_contextual_related_places=state.requested_output == "historical_route",
            )
            legacy = None
            if event_first.route is not None:
                route, diagnostics = event_first.route, dict(event_first.diagnostics)
            else:
                # Compatibility fallback: strict legacy movement claims, never merged with event-first anchors.
                legacy = self.route_extractor.build_with_diagnostics(state.historical_evidence, event_id=arguments["event_id"], name=arguments["name"], period=arguments["period"])
                route = legacy.route
                diagnostics = {**legacy.diagnostics, "route_source": "legacy_movement_claims" if route is not None else "none", "event_anchor_diagnostics": event_first.diagnostics}
            diagnostics["provenance_trace"] = HistoricalRouteTraceBuilder.build(
                state.historical_events, state.historical_evidence, event_first, legacy, route,
                str(diagnostics.get("route_source", "event_anchor")),
            )
            state.historical_route_diagnostics = diagnostics
            if route is None:
                return {"route": None, "diagnostics": diagnostics}, "build_historical_route route_points=0 diagnostics=" + ",".join(diagnostics.get("reason_codes", []))
            state.historical_route = route
            state.current_event = HistoricalEvent(id=arguments["event_id"], name=arguments["name"], period=arguments["period"], summary="Evidence-supported schematic reconstruction.", places=[p.historical_place for p in route.ordered_points], evidence=state.historical_evidence, uncertainty_note="Historical reconstruction only; not an exact march track.")
            if state.requested_output == "historical_route":
                reconstruction = self._reconstruct_candidate_route(route, state)
                if reconstruction["presentation"] is not None:
                    state.historical_route_presentation = reconstruction["presentation"]
                return {
                    "route": route.model_dump(mode="json"),
                    "presentation": state.historical_route_presentation,
                    "gis_reconstruction": reconstruction["diagnostics"],
                }, f"build_historical_route route_points={len(route.ordered_points)} {reconstruction['summary']}"
            return {"route": route.model_dump(mode="json")}, f"build_historical_route route_points={len(route.ordered_points)}"
        raise ValueError(f"Unknown agent tool: {name}")

    def _reconstruct_candidate_route(self, route, state: AgentState) -> dict:
        """Run one deterministic GIS pass over an already accepted HistoricalRoute."""
        diagnostics = state.historical_route_diagnostics or {}
        if self.roman_road_orchestrator is not None:
            road_result = self.roman_road_orchestrator.build_roman_road_candidates(route)
            presentation = RomanRoadPresentationService().present(route, road_result).model_dump(mode="json")
            diagnostics["gis_reconstruction"] = {
                "attempted": True,
                "pipeline": "roman_road_orchestrator",
                "status": road_result.status.value,
            }
            state.historical_route_diagnostics = diagnostics
            return {
                "presentation": presentation,
                "diagnostics": diagnostics["gis_reconstruction"],
                "summary": f"roman_road_status={road_result.status.value}",
            }

        intent = state.route_intent or HistoricalRouteIntent(
            campaign_id=route.event_id,
            route_type="evidence_route",
        )
        try:
            presentation = self.route_orchestrator.present(intent, route, state.historical_evidence)
        except (RouteOrchestrationError, ValueError, KeyError) as exc:
            diagnostics["gis_reconstruction"] = {
                "attempted": True,
                "pipeline": "terrain_candidate_orchestrator",
                "status": "FAILED",
                "reason_code": type(exc).__name__,
            }
            state.historical_route_diagnostics = diagnostics
            return {
                "presentation": None,
                "diagnostics": diagnostics["gis_reconstruction"],
                "summary": f"terrain_presentation_unavailable={type(exc).__name__}",
            }
        diagnostics["gis_reconstruction"] = {
            "attempted": True,
            "pipeline": "terrain_candidate_orchestrator",
            "status": "COMPLETE",
        }
        state.historical_route_diagnostics = diagnostics
        return {
            "presentation": presentation.model_dump(mode="json"),
            "diagnostics": diagnostics["gis_reconstruction"],
            "summary": "terrain_presentation=ready",
        }
