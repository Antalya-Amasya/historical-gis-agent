"""Deterministic bridge from an audited HistoricalRoute to terrain-aware presentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
from backend.app.rag.campaign_ontology import HistoricalCampaignOntology
from backend.app.candidate_routes.grid import GridPoint
from backend.app.candidate_routes.terrain import TerrainOverride
from backend.app.candidate_routes.barrier_crossings import (
    CrossingCandidateSource,
    barrier_reference_is_between,
    is_broad_mountain_constraint,
    select_crossing_candidate,
)
from backend.app.core.config import settings


class RouteOrchestrationError(ValueError):
    """The supplied evidence-grounded route cannot be reconstructed safely."""


class BarrierCrossingConstraintError(RouteOrchestrationError):
    """A broad mountain constraint lacks a defensible algorithmic crossing."""


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
    """Ontology-backed deterministic matcher; it never supplies facts or coordinates."""

    def __init__(self, ontology: HistoricalCampaignOntology | None = None) -> None:
        self.ontology = ontology or HistoricalCampaignOntology.default()

    def resolve(self, message: str) -> HistoricalRouteIntent | None:
        entity = self.ontology.match(message)
        if entity is None:
            return None
        return HistoricalRouteIntent(
            intent="historical_route",
            campaign_id=entity.route_context_id,
            entity=entity.id,
            route_type=entity.route_type,
        )


class PresentationSummaryBuilder:
    """Builds display-safe text from reviewed route/evidence metadata, never model prose."""

    def __init__(self, ontology: HistoricalCampaignOntology | None = None) -> None:
        self.ontology = ontology or HistoricalCampaignOntology.default()

    def build(
        self,
        intent: HistoricalRouteIntent,
        historical_route: HistoricalRoute,
        evidence: list[Evidence],
        *,
        terrain_source: str,
        applied_constraints: list[str],
    ) -> PresentationSummary:
        entity = self.ontology.get(intent.entity)
        parent = self.ontology.get(entity.parent_campaign) if entity else None
        root = parent
        while root and root.parent_campaign:
            root = self.ontology.get(root.parent_campaign)
        selected_refs = set(historical_route.evidence_refs)
        sources = list(dict.fromkeys(
            self._source_label(item)
            for item in evidence
            if item.id in selected_refs
        ))
        return PresentationSummary(
            title=entity.title if entity else historical_route.name,
            campaign_id=root.id if root else intent.campaign_id,
            campaign=root.title if root else None,
            operation_id=entity.id if entity else None,
            operation=entity.title if entity else historical_route.name,
            date=historical_route.period,
            historical_context="The campaign and its anchors come from reviewed historical evidence metadata.",
            route_method="terrain_constrained_reconstruction",
            evidence_basis=sources,
            route_interpretation="Terrain-constrained reconstruction joins evidence-backed anchors through deterministic A* search; it is not an exact daily march.",
            route_stages=[point.historical_place.canonical_name for point in historical_route.ordered_points],
            sources=sources,
            geographic_constraints=list(applied_constraints),
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
        cell_size_m: float | None = None,
    ) -> None:
        configured_cell_size_m = settings.route_cell_size_m if cell_size_m is None else cell_size_m
        if configured_cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        self.terrain_graph_provider = terrain_graph_provider
        self.reconstructor = reconstructor or HistoricalRouteReconstructor()
        self.presentation_service = presentation_service or LocationAwarePresentationService()
        self.cell_size_m = configured_cell_size_m

    def present(
        self,
        intent: HistoricalRouteIntent,
        historical_route: HistoricalRoute,
        evidence: list[Evidence],
        *,
        cell_size_m: float | None = None,
    ) -> HistoricalRouteResponse:
        if intent.intent != "historical_route":
            raise RouteOrchestrationError("only historical_route intents can be reconstructed")
        reviewed, barrier_context = self._reconstruction_plan(historical_route)
        terrain_graph_provider = self._terrain_graph_provider_for(intent, historical_route)
        effective_cell_size_m = self.cell_size_m if cell_size_m is None else cell_size_m
        if effective_cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        graph = terrain_graph_provider.build_graph(
            reviewed, padding_km=10.0, cell_size_m=effective_cell_size_m,
        )
        reconstruction = self.reconstructor.reconstruct(
            reviewed, graph, route_id=f"{intent.campaign_id}-terrain-candidate",
        )
        route = self._merge_candidate_paths(reconstruction.candidate_paths, reconstruction.route_id)
        crossing = None
        if barrier_context is not None:
            approach, barrier, exit_point = barrier_context
            crossing = select_crossing_candidate(
                list(route.geometry.coordinates),
                approach,
                barrier,
                exit_point,
                source=CrossingCandidateSource.TERRAIN_DERIVED_CROSSING,
                elevation_provider=lambda coordinate: graph.grid.cell(
                    graph.spec.geographic_to_grid(*coordinate)
                ).elevation_m,
            )
            if crossing is None:
                raise BarrierCrossingConstraintError(
                    "terrain path does not provide a crossing within the audited mountain-region search area"
                )
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
            applied_constraints=list(graph.applied_constraints),
        )
        if crossing is not None:
            display_summary = display_summary.model_copy(update={
                "route_method": "terrain_barrier_crossing_reconstruction",
                "route_interpretation": "Terrain A* connects trusted neighboring constraints and selects an algorithmic mountain crossing; the crossing is not a HistoricalWaypoint or an attested pass.",
                "geographic_constraints": [*display_summary.geographic_constraints, "audited_broad_mountain_constraint"],
                "limitations": [*display_summary.limitations, *crossing.limitations],
            })
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
            "applied_constraints": list(graph.applied_constraints),
            "barrier_crossing": crossing.model_dump(mode="json") if crossing is not None else None,
        })
        route_geojson["properties"] = route_properties
        geojson = dict(presentation.geojson)
        crossing_features = [self._crossing_feature(crossing)] if crossing is not None else []
        geojson["features"] = [route_geojson, *presentation.geojson["features"][1:], *crossing_features, *self._corridor_features(historical_route)]
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

    def _terrain_graph_provider_for(self, intent: HistoricalRouteIntent, historical_route: HistoricalRoute):
        if self.terrain_graph_provider is not None:
            return self.terrain_graph_provider
        constraints = [
            "synthetic_mock_terrain",
            "mock_ocean_blocking",
            "mock_route_search_bounds",
            "mock_alpine_terrain_multiplier",
        ]
        return OfflineMockTerrainGraphProvider(
            self._terrain_constraints_for(historical_route),
            applied_constraints=constraints,
        )

    @staticmethod
    def _crossing_feature(crossing) -> dict[str, object]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(crossing.coordinate)},
            "properties": {
                "layer_type": "reconstructed_crossing",
                "name": f"{crossing.barrier_name} algorithmic crossing",
                "barrier_id": crossing.barrier_id,
                "approach_anchor_id": crossing.approach_anchor_id,
                "exit_anchor_id": crossing.exit_anchor_id,
                "candidate_source": crossing.source.value,
                "road_support": crossing.road_support,
                "terrain_support": crossing.terrain_support,
                "reconstruction_cost": crossing.reconstruction_cost,
                "authority": crossing.authority,
                "limitations": list(crossing.limitations),
            },
        }

    @staticmethod
    def _route_search_bounds(historical_route: HistoricalRoute, padding_deg: float = 2.0) -> HistoricalRouteCorridorRegion:
        longitudes = [point.historical_place.longitude for point in historical_route.ordered_points]
        latitudes = [point.historical_place.latitude for point in historical_route.ordered_points]
        return HistoricalRouteCorridorRegion(
            name="route-derived search bounds",
            min_longitude=min(longitudes) - padding_deg,
            min_latitude=min(latitudes) - padding_deg,
            max_longitude=max(longitudes) + padding_deg,
            max_latitude=max(latitudes) + padding_deg,
        )

    @staticmethod
    def _terrain_constraints_for(historical_route: HistoricalRoute) -> Callable[[GridPoint, float, float], TerrainOverride | None]:
        search_bounds = HistoricalRouteOrchestrator._route_search_bounds(historical_route)

        def rule(point: GridPoint, longitude: float, latitude: float) -> TerrainOverride | None:
            # Conservative mock-ocean bands for the offline Mediterranean demo surface.
            if 0.0 < longitude < 4.75 and latitude < 41.25:
                return TerrainOverride(terrain="ocean", terrain_multiplier=20.0, blocked=True)
            if 4.75 <= longitude < 7.2 and latitude < 42.75:
                return TerrainOverride(terrain="ocean", terrain_multiplier=20.0, blocked=True)
            if not search_bounds.contains(longitude, latitude):
                return TerrainOverride(terrain="outside_route_search_bounds", terrain_multiplier=20.0, blocked=True)
            # A coarse Alpine terrain classification affects only movement cost; it proves no fact.
            if 6.0 <= longitude <= 9.5 and 43.0 <= latitude <= 46.0:
                return TerrainOverride(terrain="high_mountain", terrain_multiplier=5.0, blocked=False)
            return None

        return rule

    @staticmethod
    def _corridor_features(historical_route: HistoricalRoute) -> list[dict[str, object]]:
        region = HistoricalRouteOrchestrator._route_search_bounds(historical_route)
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
                    "explanation": "Route-derived search bounds, not an asserted historical corridor or route.",
                },
            }
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
        max_slope = max((ledger.max_slope for ledger in route.segment_ledger), default=route.metrics.max_slope)
        sources = [
            HistoricalDataSource(
                source_type="reviewed_annotation",
                dataset_name="reviewed_historical_route_anchors",
                version="1", license="project-reviewed metadata", confidence=historical_route.historical_confidence,
            ).model_dump(mode="json"),
            HistoricalDataSource(
                source_type="reviewed_annotation",
                dataset_name="mock_terrain_reconstruction_constraints",
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
            "applied_constraints": list(graph.applied_constraints),
            "elevation_gain": elevation_gain,
            "max_slope": max_slope,
            "mountain_penalty": terrain_cost,
            "segment_ledger": [item.model_dump(mode="json") for item in route.segment_ledger],
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

    @classmethod
    def _reconstruction_plan(cls, route: HistoricalRoute):
        reviewed = cls._reviewed_waypoints(route)
        barriers = [
            (index, point)
            for index, point in enumerate(route.ordered_points)
            if is_broad_mountain_constraint(point)
        ]
        if not barriers:
            return reviewed, None
        if len(barriers) > 1:
            raise BarrierCrossingConstraintError("V1 supports one audited mountain constraint per route")
        index, barrier = barriers[0]
        if index == 0:
            raise BarrierCrossingConstraintError("mountain constraint has no trusted approach waypoint")
        if index == len(route.ordered_points) - 1:
            raise BarrierCrossingConstraintError("mountain constraint has no trusted onward waypoint")
        approach = route.ordered_points[index - 1]
        exit_point = route.ordered_points[index + 1]
        if not barrier_reference_is_between(approach, barrier, exit_point):
            raise BarrierCrossingConstraintError(
                "mountain constraint is not between the trusted approach and onward waypoints"
            )
        return [*reviewed[:index], *reviewed[index + 1:]], (approach, barrier, exit_point)

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
            segment_ledger=[ledger for path in paths for ledger in path.segment_ledger],
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
