"""Deterministic bridge from an audited HistoricalRoute to terrain-aware presentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Iterable

from backend.app.candidate_routes.evaluation import evaluate_route
from backend.app.candidate_routes.historical_reconstruction import (
    HistoricalRouteReconstructor,
    OfflineMockTerrainGraphProvider,
    ReviewedHistoricalWaypoint,
)
from backend.app.candidate_routes.knowledge_panel import KnowledgePanelBuilder
from backend.app.candidate_routes.location import LocationConfidence, MockCoordinateResolver, ResolvedCoordinate
from backend.app.candidate_routes.models import CandidateRoute, RouteCostBreakdown, RouteMetrics
from backend.app.candidate_routes.presentation import (
    HistoricalDataSource,
    HistoricalRouteResponse,
    LocationAwarePresentationService,
    PresentationSummary,
    PresentationTimelineStep,
)
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointGraph,
    HistoricalWaypointRole,
    HistoricalWaypointSegment,
)
from backend.app.models import Evidence, HistoricalRoute, HistoricalRouteIntent
from backend.app.candidate_routes.grid import GridPoint
from backend.app.candidate_routes.terrain import TerrainOverride


class RouteOrchestrationError(ValueError):
    """The supplied evidence-grounded route cannot be reconstructed safely."""


@dataclass(frozen=True)
class CampaignIntentDefinition:
    campaign_id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalRouteCorridorRegion:
    """A reviewed broad search region, not a historical geometry or route segment."""

    name: str
    min_longitude: float
    min_latitude: float
    max_longitude: float
    max_latitude: float

    def contains(self, longitude: float, latitude: float) -> bool:
        return self.min_longitude <= longitude <= self.max_longitude and self.min_latitude <= latitude <= self.max_latitude


@dataclass(frozen=True)
class HistoricalRouteCorridor:
    campaign_id: str
    regions: tuple[HistoricalRouteCorridorRegion, ...]

    def allows(self, longitude: float, latitude: float) -> bool:
        return any(region.contains(longitude, latitude) for region in self.regions)


_HANNIBAL_CORRIDOR = HistoricalRouteCorridor(
    campaign_id="hannibal_italy_campaign",
    regions=(
        HistoricalRouteCorridorRegion("Iberian east-coast approach", -1.5, 37.0, 3.5, 43.0),
        HistoricalRouteCorridorRegion("Pyrenees corridor", 0.0, 40.5, 4.0, 44.0),
        HistoricalRouteCorridorRegion("Rhone valley", 3.5, 42.0, 6.0, 45.5),
        HistoricalRouteCorridorRegion("Alpine foothills", 5.0, 42.5, 10.0, 46.0),
        HistoricalRouteCorridorRegion("Po plain approach", 8.0, 43.5, 13.0, 46.5),
    ),
)


class HistoricalCampaignIntentRegistry:
    """Data-only natural-language aliases; it supplies neither facts nor coordinates."""

    def __init__(self, definitions: Iterable[CampaignIntentDefinition] | None = None) -> None:
        self.definitions = tuple(definitions or (
            CampaignIntentDefinition(
                campaign_id="hannibal_italy_campaign",
                aliases=("hannibal", "汉尼拔", "alps", "阿尔卑斯"),
            ),
        ))

    def resolve(self, message: str) -> HistoricalRouteIntent | None:
        normalized = message.lower()
        for definition in self.definitions:
            if any(alias.lower() in normalized for alias in definition.aliases):
                return HistoricalRouteIntent(intent="historical_route", campaign_id=definition.campaign_id)
        return None


class PresentationSummaryBuilder:
    """Builds display-safe text from reviewed route/evidence metadata, never model prose."""

    _CAMPAIGNS = {
        "hannibal_italy_campaign": {
            "campaign_id": "second_punic_war",
            "campaign": "Second Punic War",
            "operation_id": "hannibal_invasion_italy",
            "operation": "Hannibal's invasion of Italy (218 BCE)",
        },
    }

    def build(
        self,
        intent: HistoricalRouteIntent,
        historical_route: HistoricalRoute,
        evidence: list[Evidence],
        *,
        terrain_source: str,
    ) -> PresentationSummary:
        labels = self._CAMPAIGNS.get(intent.campaign_id, {})
        selected_refs = set(historical_route.evidence_refs)
        sources = list(dict.fromkeys(
            self._source_label(item)
            for item in evidence
            if item.id in selected_refs
        ))
        return PresentationSummary(
            title=labels.get("operation", historical_route.name),
            campaign_id=labels.get("campaign_id"),
            campaign=labels.get("campaign"),
            operation_id=labels.get("operation_id"),
            operation=labels.get("operation"),
            date=historical_route.period,
            historical_context="The campaign and its anchors come from reviewed historical evidence metadata.",
            route_method="terrain_constrained_reconstruction",
            evidence_basis=sources,
            route_interpretation="Terrain-constrained reconstruction joins evidence-backed anchors through deterministic A* search; it is not an exact daily march.",
            route_stages=[point.historical_place.canonical_name for point in historical_route.ordered_points],
            sources=sources,
            geographic_constraints=[
                "Reviewed historical corridor restricts the search area.",
                "Coarse offline sea cells are blocked.",
                "Coarse Alpine cells increase terrain movement cost.",
            ],
            uncertainty_notes=[
                "Representative river, regional, and mountain coordinates do not identify an exact passage.",
                "Intermediate geometry is algorithmic rather than direct historical evidence.",
            ],
            limitations=[
                *historical_route.limitations,
                f"Terrain source: {terrain_source}.",
                "Intermediate geometry is algorithmic and is not historical evidence.",
            ],
            timeline=[
                PresentationTimelineStep(
                    order=point.sequence,
                    title=point.historical_place.canonical_name,
                    description=point.event_summary,
                    period=point.date_or_period,
                    evidence_count=len(point.evidence_refs),
                    confidence=point.confidence,
                )
                for point in historical_route.ordered_points
            ],
        )

    @staticmethod
    def _source_label(item: Evidence) -> str:
        locator = item.book or item.locator
        chapter = f", Chapter {item.chapter}" if item.chapter else ""
        return f"{item.author}, {item.work}, {locator}{chapter}"


class HistoricalRouteOrchestrator:
    """Uses only an already-built HistoricalRoute and an offline TerrainGraph provider.

    The registry selects a reviewed campaign label. Coordinates remain the outputs already
    attached to HistoricalRoute points by Geography MCP; this class never resolves names.
    """

    def __init__(
        self,
        terrain_graph_provider: OfflineMockTerrainGraphProvider | None = None,
        reconstructor: HistoricalRouteReconstructor | None = None,
        presentation_service: LocationAwarePresentationService | None = None,
    ) -> None:
        self.terrain_graph_provider = terrain_graph_provider
        self.reconstructor = reconstructor or HistoricalRouteReconstructor()
        self.presentation_service = presentation_service or LocationAwarePresentationService()

    def present(
        self,
        intent: HistoricalRouteIntent,
        historical_route: HistoricalRoute,
        evidence: list[Evidence],
    ) -> HistoricalRouteResponse:
        if intent.intent != "historical_route":
            raise RouteOrchestrationError("only historical_route intents can be reconstructed")
        reviewed = self._reviewed_waypoints(historical_route)
        terrain_graph_provider = self._terrain_graph_provider_for(intent)
        graph = terrain_graph_provider.build_graph(
            reviewed, padding_km=10.0, cell_size_m=25_000.0,
        )
        reconstruction = self.reconstructor.reconstruct(
            reviewed, graph, route_id=f"{intent.campaign_id}-terrain-candidate",
        )
        route = self._merge_candidate_paths(reconstruction.candidate_paths, reconstruction.route_id)
        evaluation = evaluate_route(route, source_ids=self._source_ids(evidence, historical_route))
        waypoint_graph, resolver, summaries = self._waypoint_graph(historical_route, evidence)
        presentation = self.presentation_service.present(
            waypoint_graph,
            route,
            evaluation,
            resolver,
            route_name=historical_route.name,
            period=historical_route.period,
        )
        display_summary = PresentationSummaryBuilder().build(
            intent, historical_route, evidence, terrain_source=reconstruction.terrain_source,
        )
        presentation = presentation.model_copy(update={
            "route_name": display_summary.operation or presentation.route_name,
        })
        route_geojson = dict(presentation.route_geojson)
        route_properties = dict(route_geojson["properties"])
        route_properties.update({
            "route_type": "terrain_aware_historical_reconstruction",
            "campaign_id": intent.campaign_id,
            "terrain_source": reconstruction.terrain_source,
            "explanation": "Terrain-aware candidate connection between evidence-backed, MCP-resolved anchors; not an exact historical march path.",
            "route_quality": self._route_quality(route, graph, reconstruction, reviewed, historical_route),
        })
        route_geojson["properties"] = route_properties
        geojson = dict(presentation.geojson)
        geojson["features"] = [route_geojson, *presentation.geojson["features"][1:], *self._corridor_features(intent)]
        presentation = presentation.model_copy(update={"route_geojson": route_geojson, "geojson": geojson})
        panels = [
            KnowledgePanelBuilder().build(waypoint, summary_provider=lambda view: summaries.get(view.id))
            for waypoint in presentation.waypoints
        ]
        return HistoricalRouteResponse(
            route=presentation,
            waypoints=presentation.waypoints,
            geojson=presentation.geojson,
            route_geojson=presentation.route_geojson,
            explanations=presentation.explanations,
            location_warnings=presentation.location_warnings,
            knowledge_panels=panels,
            presentation_summary=display_summary,
        )

    def _terrain_graph_provider_for(self, intent: HistoricalRouteIntent):
        if self.terrain_graph_provider is not None:
            return self.terrain_graph_provider
        return OfflineMockTerrainGraphProvider(self._terrain_constraints_for(intent))

    @staticmethod
    def _terrain_constraints_for(intent: HistoricalRouteIntent) -> Callable[[GridPoint, float, float], TerrainOverride | None]:
        corridor = _HANNIBAL_CORRIDOR if intent.campaign_id == _HANNIBAL_CORRIDOR.campaign_id else None

        def rule(point: GridPoint, longitude: float, latitude: float) -> TerrainOverride | None:
            # This conservative mask is a local display-demo constraint, not a coastline dataset
            # and not a claim about Hannibal's exact route.
            if 0.0 < longitude < 4.75 and latitude < 41.25:
                return TerrainOverride(terrain="ocean", terrain_multiplier=20.0, blocked=True)
            if 4.75 <= longitude < 7.2 and latitude < 42.75:
                return TerrainOverride(terrain="ocean", terrain_multiplier=20.0, blocked=True)
            if corridor is not None and not corridor.allows(longitude, latitude):
                return TerrainOverride(terrain="outside_reviewed_corridor", terrain_multiplier=20.0, blocked=True)
            # A coarse Alpine terrain classification affects only movement cost; it proves no fact.
            if 6.0 <= longitude <= 9.5 and 43.0 <= latitude <= 46.0:
                return TerrainOverride(terrain="high_mountain", terrain_multiplier=5.0, blocked=False)
            return None

        return rule

    @staticmethod
    def _corridor_features(intent: HistoricalRouteIntent) -> list[dict[str, object]]:
        if intent.campaign_id != _HANNIBAL_CORRIDOR.campaign_id:
            return []
        return [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [region.min_longitude, region.min_latitude],
                    [region.max_longitude, region.min_latitude],
                    [region.max_longitude, region.max_latitude],
                    [region.min_longitude, region.max_latitude],
                    [region.min_longitude, region.min_latitude],
                ]]},
                "properties": {
                    "layer_type": "uncertainty_corridor",
                    "label": region.name,
                    "explanation": "Broad reviewed search area, not an asserted historical corridor or route.",
                },
            }
            for region in _HANNIBAL_CORRIDOR.regions
        ]

    @staticmethod
    def _route_quality(route: CandidateRoute, graph, reconstruction, reviewed: list[ReviewedHistoricalWaypoint], historical_route: HistoricalRoute) -> dict[str, object]:
        coordinates = list(route.geometry.coordinates)
        land_cells = 0
        for longitude, latitude in coordinates:
            cell = graph.grid.cell(graph.spec.geographic_to_grid(longitude, latitude))
            if not cell.blocked and cell.terrain not in {"ocean", "sea", "water"}:
                land_cells += 1
        expected_ids = [item.id for item in reviewed]
        reconstructed_ids = [reconstruction.candidate_paths[0].from_anchor.historical_place_id]
        reconstructed_ids.extend(path.to_anchor.historical_place_id for path in reconstruction.candidate_paths)
        terrain_cost = route.cost_breakdown.terrain_cost
        elevation_gain = route.metrics.elevation_gain_m
        max_slope = 0.0
        for first, second in zip(coordinates, coordinates[1:]):
            first_cell = graph.grid.cell(graph.spec.geographic_to_grid(*first))
            second_cell = graph.grid.cell(graph.spec.geographic_to_grid(*second))
            max_slope = max(max_slope, abs(second_cell.elevation_m - first_cell.elevation_m) / graph.spec.cell_size_m)
        sources = [
            HistoricalDataSource(
                source_type="reviewed_annotation",
                dataset_name="reviewed_historical_route_anchors",
                version="1", license="project-reviewed metadata", confidence=historical_route.historical_confidence,
            ).model_dump(mode="json"),
            HistoricalDataSource(
                source_type="reviewed_annotation",
                dataset_name="hannibal_reconstruction_constraints",
                version="phase-18", license="project configuration", confidence=0.6,
            ).model_dump(mode="json"),
            HistoricalDataSource(
                source_type="reviewed_annotation",
                dataset_name=reconstruction.terrain_source,
                version="offline", license="local test surface", confidence=0.0,
            ).model_dump(mode="json"),
        ]
        return {
            "coordinate_count": len(coordinates),
            "intermediate_points": max(0, len(coordinates) - 2),
            "land_ratio": land_cells / len(coordinates) if coordinates else 0.0,
            "waypoint_order_preserved": reconstructed_ids == expected_ids,
            "terrain_source": reconstruction.terrain_source,
            "terrain_constrained": True,
            "search_constraint": "reviewed_historical_corridor" if len(reviewed) > 1 else "none",
            "elevation_gain": elevation_gain,
            "max_slope": max_slope,
            "mountain_penalty": terrain_cost,
            "data_sources": sources,
        }

    @staticmethod
    def _reviewed_waypoints(route: HistoricalRoute) -> list[ReviewedHistoricalWaypoint]:
        if len(route.ordered_points) < 2:
            raise RouteOrchestrationError("historical route requires at least two evidence-backed anchors")
        result: list[ReviewedHistoricalWaypoint] = []
        for point in route.ordered_points:
            if not point.evidence_refs:
                raise RouteOrchestrationError(f"route point {point.sequence} has no evidence references")
            place = point.historical_place
            result.append(ReviewedHistoricalWaypoint(
                id=place.id,
                canonical_name=place.canonical_name,
                longitude=place.longitude,
                latitude=place.latitude,
                evidence_refs=list(point.evidence_refs),
                confidence=point.confidence,
                coordinate_note=(
                    "Representative coordinate retained from Geography MCP; it is not an inferred passage."
                    if point.coordinate_role != "exact_site" else None
                ),
            ))
        return result

    @staticmethod
    def _merge_candidate_paths(paths: list[CandidateRoute], route_id: str) -> CandidateRoute:
        if not paths:
            raise RouteOrchestrationError("terrain reconstruction returned no candidate paths")
        first, last = paths[0], paths[-1]
        coordinates: list[tuple[float, float]] = []
        for path in paths:
            segment = list(path.geometry.coordinates)
            coordinates.extend(segment if not coordinates else segment[1:])
        breakdown = RouteCostBreakdown(
            distance_cost=sum(path.cost_breakdown.distance_cost for path in paths),
            slope_cost=sum(path.cost_breakdown.slope_cost for path in paths),
            terrain_cost=sum(path.cost_breakdown.terrain_cost for path in paths),
            barrier_cost=sum(path.cost_breakdown.barrier_cost for path in paths),
            historical_cost=sum(path.cost_breakdown.historical_cost for path in paths),
            total_cost=sum(path.cost_breakdown.total_cost for path in paths),
        )
        return CandidateRoute(
            id=route_id,
            from_anchor=first.from_anchor,
            to_anchor=last.to_anchor,
            geometry={"type": "LineString", "coordinates": coordinates},
            metrics=RouteMetrics(
                distance_km=sum(path.metrics.distance_km for path in paths),
                elevation_gain_m=sum(path.metrics.elevation_gain_m for path in paths),
                elevation_loss_m=sum(path.metrics.elevation_loss_m for path in paths),
                estimated_cost=breakdown.total_cost,
                cell_count=sum(path.metrics.cell_count for path in paths),
                segment_count=sum(path.metrics.segment_count for path in paths),
            ),
            cost_breakdown=breakdown,
            confidence=min(path.confidence for path in paths),
            assumptions=[
                "Merged terrain-aware candidates retain only supplied evidence-backed anchors.",
                "No coordinates were generated by the Agent or language model.",
            ],
            evidence_refs=list(dict.fromkeys(ref for path in paths for ref in path.evidence_refs)),
            provenance="terrain_aware_algorithmic_candidate",
            coordinate_system="EPSG:4326",
            projection_method=first.projection_method,
            terrain_source=first.terrain_source,
            generation_method="historical_route_reconstruction",
        )

    @staticmethod
    def _waypoint_graph(route: HistoricalRoute, evidence: list[Evidence]):
        evidence_by_id = {item.id: item for item in evidence}
        waypoints: list[HistoricalWaypoint] = []
        resolver_mapping: dict[str, ResolvedCoordinate] = {}
        summaries: dict[str, str] = {}
        point_count = len(route.ordered_points)
        for index, point in enumerate(route.ordered_points):
            role = HistoricalWaypointRole.START if index == 0 else HistoricalWaypointRole.END if index == point_count - 1 else HistoricalWaypointRole.WAYPOINT
            confidence = LocationConfidence.EXACT if point.coordinate_role == "exact_site" else LocationConfidence.APPROXIMATE
            place = point.historical_place
            primary_evidence = next((evidence_by_id[ref] for ref in point.evidence_refs if ref in evidence_by_id), None)
            waypoint = HistoricalWaypoint(
                id=place.id,
                canonical_name=place.canonical_name,
                order=point.sequence,
                role=role,
                evidence_refs=list(point.evidence_refs),
                location_confidence=confidence,
                location_notes=place.source_url,
                involved_places=[place.canonical_name],
                period=point.date_or_period,
                description=point.event_summary,
                source_book=(primary_evidence.book or primary_evidence.locator) if primary_evidence else None,
                source_chapter=primary_evidence.chapter if primary_evidence else None,
                historical_confidence=point.confidence,
            )
            waypoints.append(waypoint)
            resolver_mapping[waypoint.id] = ResolvedCoordinate(
                point=GridPoint(index, 0),
                confidence=confidence,
                notes=waypoint.location_notes,
                display_coordinate=(place.longitude, place.latitude),
            )
            summaries[waypoint.id] = point.event_summary
        segments = [
            HistoricalWaypointSegment(
                from_waypoint=start,
                to_waypoint=end,
                evidence_refs=list(dict.fromkeys([*start.evidence_refs, *end.evidence_refs])),
            )
            for start, end in zip(waypoints, waypoints[1:])
        ]
        return HistoricalWaypointGraph(waypoints=waypoints, segments=segments), MockCoordinateResolver(resolver_mapping), summaries

    @staticmethod
    def _source_ids(evidence: list[Evidence], route: HistoricalRoute) -> list[str]:
        selected = set(route.evidence_refs)
        return list(dict.fromkeys(
            f"{item.author}:{item.work}:{item.locator}"
            for item in evidence if item.id in selected
        ))
