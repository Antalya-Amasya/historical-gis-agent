"""Offline composition for the reviewed Caesar campaign presentation demo."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.candidate_routes.annotation import HistoricalAnnotation
from backend.app.candidate_routes.evaluation import (
    RouteEvaluationProvenance,
    RouteEvaluationResult,
    build_route_explanation,
)
from backend.app.candidate_routes.geographic import GeographicGridSpec
from backend.app.candidate_routes.grid import GridPoint
from backend.app.candidate_routes.knowledge_panel import KnowledgePanelBuilder
from backend.app.candidate_routes.location import LocationConfidence, ResolvedCoordinate
from backend.app.candidate_routes.models import (
    ArmyProfile,
    CandidateRoute,
    CandidateRouteAnchor,
    RouteCostBreakdown,
    RouteMetrics,
    RouteScore,
)
from backend.app.candidate_routes.planner import HistoricalPlanningRequest, HistoricalRoutePlanner
from backend.app.candidate_routes.presentation import (
    HistoricalRoutePresentation,
    HistoricalRouteResponse,
    LocationAwarePresentationService,
)
from backend.app.candidate_routes.terrain import SyntheticTerrainProvider
from backend.app.models import GeoJsonLineString
from backend.app.rag.campaign_route_adapter import CampaignChainWaypointAdapter, MockHistoricalCoordinateResolver
from backend.app.rag.event_chain_builder import HistoricalCampaignChain, HistoricalEventChainBuilder
from backend.app.rag.event_registry import CorpusEvidenceCatalog, CorpusEvidenceReference, HistoricalEventRecord, HistoricalEventRegistry


class CampaignPresentationBuildError(ValueError):
    pass


class CaesarCampaignPresentationFactory:
    """Builds a read-only local demo from review files and explicit coordinate mappings."""

    _registry_dir = Path(__file__).parent / "rag" / "registries"

    def __init__(self, planner: HistoricalRoutePlanner | None = None) -> None:
        self.planner = planner or HistoricalRoutePlanner()
        self.terrain = SyntheticTerrainProvider()

    def build(self) -> HistoricalRouteResponse:
        chain = self._load_chain()
        presentation_data = self._load_json("caesar_gallic_war_presentation.json")
        resolver = MockHistoricalCoordinateResolver({
            item["event_id"]: ResolvedCoordinate(
                point=GridPoint(*item["grid_point"]),
                confidence=LocationConfidence(item["confidence"]),
                notes=item.get("notes"),
                display_coordinate=(float(item["longitude"]), float(item["latitude"])),
            )
            for item in presentation_data["coordinates"]
        })
        annotations = [HistoricalAnnotation.model_validate(item) for item in presentation_data["annotations"]]
        graph = CampaignChainWaypointAdapter().to_waypoint_graph(chain, annotations=annotations)
        route, score = self._plan_segments(graph, resolver)
        evaluation = RouteEvaluationResult(
            route_id=route.id,
            candidate_route=route,
            army_profile=ArmyProfile(name="caesar_campaign_demo"),
            score=score,
            explanation=build_route_explanation(route, score, ArmyProfile(name="caesar_campaign_demo")),
            provenance=RouteEvaluationProvenance(
                evidence_ids=list(route.evidence_refs),
                source_ids=list(dict.fromkeys(
                    f"{waypoint.source_book}:{waypoint.source_chapter}"
                    for waypoint in graph.waypoints
                    if waypoint.source_book is not None and waypoint.source_chapter is not None
                )),
                algorithmic_factors=["deterministic local-grid candidate segments"],
            ),
        )
        presentation = LocationAwarePresentationService().present(
            graph, route, evaluation, resolver,
            route_name=chain.title,
            period=chain.period,
        )
        panels = [KnowledgePanelBuilder().build(waypoint) for waypoint in presentation.waypoints]
        return HistoricalRouteResponse(
            route=presentation,
            waypoints=presentation.waypoints,
            geojson=presentation.geojson,
            route_geojson=presentation.route_geojson,
            explanations=presentation.explanations,
            location_warnings=presentation.location_warnings,
            knowledge_panels=panels,
        )

    def _plan_segments(self, graph, resolver: MockHistoricalCoordinateResolver) -> tuple[CandidateRoute, RouteScore]:
        if not graph.segments:
            raise CampaignPresentationBuildError("campaign graph requires at least one segment")
        routes: list[CandidateRoute] = []
        scores: list[RouteScore] = []
        profile = ArmyProfile(name="caesar_campaign_demo")
        for segment in graph.segments:
            start_anchor = CandidateRouteAnchor(
                historical_place_id=segment.from_waypoint.id,
                canonical_name=segment.from_waypoint.canonical_name,
                evidence_refs=list(segment.from_waypoint.evidence_refs),
            )
            end_anchor = CandidateRouteAnchor(
                historical_place_id=segment.to_waypoint.id,
                canonical_name=segment.to_waypoint.canonical_name,
                evidence_refs=list(segment.to_waypoint.evidence_refs),
            )
            start = resolver.resolve(start_anchor)
            end = resolver.resolve(end_anchor)
            if start.display_coordinate is None or end.display_coordinate is None:
                raise CampaignPresentationBuildError("configured campaign coordinates must include WGS84 display coordinates")
            spec = GeographicGridSpec.from_anchor_coordinates(
                start.display_coordinate, end.display_coordinate,
                padding_km=50.0, cell_size_m=50_000.0,
            )
            grid = self.terrain.build_grid(spec, resolution_m=spec.cell_size_m)
            result = self.planner.plan(HistoricalPlanningRequest(
                start_anchor=start_anchor,
                end_anchor=end_anchor,
                start_grid_point=spec.geographic_to_grid(*start.display_coordinate),
                end_grid_point=spec.geographic_to_grid(*end.display_coordinate),
                grid=grid,
                army_profile=profile,
            ))
            if result.selected_route is None:
                raise CampaignPresentationBuildError("planner returned no selected candidate route")
            selected = result.selected_route.route
            points = [spec.grid_to_geographic(GridPoint(int(x), int(y))) for x, y in selected.geometry.coordinates]
            routes.append(selected.model_copy(update={
                "geometry": GeoJsonLineString(coordinates=points),
                "coordinate_system": "EPSG:4326",
                "projection_method": spec.projection_method,
                "grid_cell_size_m": spec.cell_size_m,
                "grid_width": spec.width,
                "grid_height": spec.height,
                "terrain_source": self.terrain.source,
                "assumptions": [
                    *selected.assumptions,
                    "WGS84 geometry is the deterministic local-grid candidate connection between explicitly configured display coordinates.",
                    "It is schematic and not a reconstructed historical march track.",
                ],
            }))
            scores.append(result.selected_route.score)
        coordinates: list[tuple[float, float]] = []
        for route in routes:
            part = list(route.geometry.coordinates)
            coordinates.extend(part if not coordinates else part[1:])
        first, last = routes[0], routes[-1]
        breakdown = RouteCostBreakdown(
            distance_cost=sum(route.cost_breakdown.distance_cost for route in routes),
            slope_cost=sum(route.cost_breakdown.slope_cost for route in routes),
            terrain_cost=sum(route.cost_breakdown.terrain_cost for route in routes),
            barrier_cost=sum(route.cost_breakdown.barrier_cost for route in routes),
            historical_cost=sum(route.cost_breakdown.historical_cost for route in routes),
            total_cost=sum(route.cost_breakdown.total_cost for route in routes),
        )
        merged = CandidateRoute(
            id="caesar-gallic-campaign-candidate",
            from_anchor=first.from_anchor,
            to_anchor=last.to_anchor,
            geometry=GeoJsonLineString(coordinates=coordinates),
            metrics=RouteMetrics(
                distance_km=sum(route.metrics.distance_km for route in routes),
                elevation_gain_m=sum(route.metrics.elevation_gain_m for route in routes),
                elevation_loss_m=sum(route.metrics.elevation_loss_m for route in routes),
                estimated_cost=breakdown.total_cost,
                cell_count=sum(route.metrics.cell_count for route in routes),
                segment_count=sum(route.metrics.segment_count for route in routes),
            ),
            cost_breakdown=breakdown,
            confidence=min(waypoint.historical_confidence for waypoint in graph.waypoints if waypoint.historical_confidence is not None),
            assumptions=["Segments retain reviewed evidence anchors and explicit offline display-coordinate mappings.", "The rendered line is an algorithmic schematic candidate, not an asserted historical route."],
            evidence_refs=list(dict.fromkeys(ref for route in routes for ref in route.evidence_refs)),
            coordinate_system="EPSG:4326",
            projection_method="local_equirectangular",
            terrain_source=self.terrain.source,
            generation_method="multi_segment_deterministic_variants",
        )
        score = RouteScore(
            profile_name=profile.name,
            distance_cost=sum(item.distance_cost for item in scores),
            terrain_cost=sum(item.terrain_cost for item in scores),
            historical_cost=sum(item.historical_cost for item in scores),
            total_cost=sum(item.total_cost for item in scores),
            explanation=["Sum of rank-1 route scores for supplied campaign segments."],
        )
        return merged, score

    def _load_chain(self) -> HistoricalCampaignChain:
        raw_records = self._load_json("caesar_events.json")["events"]
        records = [HistoricalEventRecord.model_validate(item) for item in raw_records]
        evidence_catalog = CorpusEvidenceCatalog({
            ref: CorpusEvidenceReference(record.corpus_id, record.source_book, record.source_chapter)
            for record in records for ref in record.evidence_refs
        })
        registry = HistoricalEventRegistry(records, evidence_catalog)
        return HistoricalEventChainBuilder().from_json(registry, self._registry_dir / "caesar_gallic_war_campaign.json")

    def _load_json(self, filename: str) -> dict:
        return json.loads((self._registry_dir / filename).read_text(encoding="utf-8"))
