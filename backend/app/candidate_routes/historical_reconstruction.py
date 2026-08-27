"""Offline, terrain-aware reconstruction from explicitly reviewed historical anchors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from backend.app.models import GeoJsonLineString
from backend.app.gis.srtm import SrtmElevationService, SrtmElevationUnavailableError

from .engine import CandidateRouteEngine
from .geographic import DEFAULT_MAX_GRID_CELLS, GeographicGridSpec, TerrainGrid
from .grid import GridPoint
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor, CandidateRouteSegmentLedger
from .terrain import OfflineMockTerrainProvider, TerrainOverride, TerrainProvider


class HistoricalRouteReconstructionError(ValueError):
    pass


class ReviewedHistoricalWaypoint(BaseModel):
    """A caller-supplied, evidence-backed WGS84 anchor; this model never resolves a name."""

    id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    coordinate_note: str | None = None


@dataclass(frozen=True)
class TerrainGraph:
    """A local projected grid plus WGS84 conversion metadata; it owns no pathfinding logic."""

    spec: GeographicGridSpec
    grid: TerrainGrid
    source: str
    applied_constraints: list[str] = field(default_factory=list)

    def grid_point_for(self, waypoint: ReviewedHistoricalWaypoint) -> GridPoint:
        return self.spec.geographic_to_grid(waypoint.longitude, waypoint.latitude)

    def to_wgs84(self, point: GridPoint) -> tuple[float, float]:
        return self.spec.grid_to_geographic(point)


class RealTerrainGraphProvider:
    """Builds a TerrainGraph from a caller-supplied offline DEM provider; it never downloads data."""

    def __init__(self, terrain_provider: TerrainProvider | SrtmElevationService) -> None:
        self.terrain_provider = terrain_provider

    def build_graph(
        self,
        waypoints: list[ReviewedHistoricalWaypoint],
        *,
        padding_km: float = 10.0,
        cell_size_m: float = 5_000.0,
        max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
    ) -> TerrainGraph:
        source = "srtm3_real_terrain" if isinstance(self.terrain_provider, SrtmElevationService) else self.terrain_provider.source
        spec = _spec_for_waypoints(waypoints, padding_km, cell_size_m, max_grid_cells, source)
        applied_constraints = ["dem_availability_blocking", "dem_nodata_blocking", "slope_cost"]
        if isinstance(self.terrain_provider, SrtmElevationService):
            grid = TerrainGrid(spec)
            for point in tuple(grid._cells):
                longitude, latitude = spec.grid_to_geographic(point)
                try:
                    elevation = self.terrain_provider.get_elevation(latitude, longitude)
                except SrtmElevationUnavailableError as exc:
                    grid.set_cell(point, terrain=f"{exc.status.lower()}_dem", blocked=True, cell_size_m=spec.cell_size_m)
                else:
                    grid.set_cell(point, elevation_m=elevation, terrain="srtm3_dem", terrain_multiplier=1.0, cell_size_m=spec.cell_size_m)
        else:
            grid = self.terrain_provider.build_grid(spec, resolution_m=spec.cell_size_m)
        return TerrainGraph(spec=spec, grid=grid, source=source, applied_constraints=applied_constraints)


class OfflineMockTerrainGraphProvider:
    """Builds deterministic local terrain graphs with an optional caller-supplied terrain rule."""

    source = "offline_mock_terrain"

    def __init__(
        self,
        rule: Callable[[GridPoint, float, float], TerrainOverride | None] | None = None,
        *,
        applied_constraints: list[str] | None = None,
    ) -> None:
        self._terrain = OfflineMockTerrainProvider(rule)
        self.applied_constraints = list(applied_constraints or ["synthetic_mock_terrain"])

    def build_graph(
        self,
        waypoints: list[ReviewedHistoricalWaypoint],
        *,
        padding_km: float = 10.0,
        cell_size_m: float = 5_000.0,
        max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
    ) -> TerrainGraph:
        spec = _spec_for_waypoints(waypoints, padding_km, cell_size_m, max_grid_cells, self.source)
        grid = self._terrain.build_grid(spec, resolution_m=spec.cell_size_m)
        return TerrainGraph(spec=spec, grid=grid, source=self.source, applied_constraints=self.applied_constraints)


def _spec_for_waypoints(
    waypoints: list[ReviewedHistoricalWaypoint],
    padding_km: float,
    cell_size_m: float,
    max_grid_cells: int,
    source: str,
) -> GeographicGridSpec:
    if len(waypoints) < 2:
        raise HistoricalRouteReconstructionError("terrain graph requires at least two reviewed waypoints")
    for waypoint in waypoints:
        if not waypoint.evidence_refs:
            raise HistoricalRouteReconstructionError(f"waypoint {waypoint.id} has no evidence references")
    lower_left = (min(item.longitude for item in waypoints), min(item.latitude for item in waypoints))
    upper_right = (max(item.longitude for item in waypoints), max(item.latitude for item in waypoints))
    return GeographicGridSpec.from_anchor_coordinates(
        lower_left,
        upper_right,
        padding_km=padding_km,
        cell_size_m=cell_size_m,
        max_grid_cells=max_grid_cells,
        source=source,
    )


class ReconstructedHistoricalRoute(BaseModel):
    """Algorithmic terrain-aware candidates, intentionally distinct from historical fact claims."""

    route_id: str
    candidate_paths: list[CandidateRoute] = Field(min_length=1)
    geometry: GeoJsonLineString
    evidence_refs: list[str] = Field(min_length=1)
    terrain_source: str
    provenance: str = "terrain_aware_algorithmic_candidate"
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    def to_geojson(self) -> dict[str, object]:
        return {
            "type": "Feature",
            "geometry": self.geometry.model_dump(mode="json"),
            "properties": {
                "route_id": self.route_id,
                "route_type": "terrain_aware_historical_reconstruction",
                "terrain_source": self.terrain_source,
                "provenance": self.provenance,
                "explanation": "Terrain-aware candidate connections between explicitly supplied evidence-backed waypoints; not an exact historical march path.",
            },
        }


class HistoricalRouteReconstructor:
    """Uses the existing terrain-aware A* engine over a supplied TerrainGraph only."""

    def __init__(self, engine: CandidateRouteEngine | None = None) -> None:
        self.engine = engine or CandidateRouteEngine()

    def reconstruct(
        self,
        waypoints: list[ReviewedHistoricalWaypoint],
        terrain_graph: TerrainGraph,
        *,
        army_profile: ArmyProfile | None = None,
        route_id: str = "historical-reconstruction",
    ) -> ReconstructedHistoricalRoute:
        if len(waypoints) < 2:
            raise HistoricalRouteReconstructionError("reconstruction requires at least two reviewed waypoints")
        profile = army_profile or ArmyProfile(name="historical_reconstruction")
        candidates: list[CandidateRoute] = []
        for first, second in zip(waypoints, waypoints[1:]):
            self._validate_waypoint(first)
            self._validate_waypoint(second)
            candidate = self.engine.build_route(
                from_anchor=CandidateRouteAnchor(
                    historical_place_id=first.id,
                    canonical_name=first.canonical_name,
                    evidence_refs=list(first.evidence_refs),
                ),
                to_anchor=CandidateRouteAnchor(
                    historical_place_id=second.id,
                    canonical_name=second.canonical_name,
                    evidence_refs=list(second.evidence_refs),
                ),
                start=terrain_graph.grid_point_for(first),
                goal=terrain_graph.grid_point_for(second),
                grid=terrain_graph.grid,
                profile=profile,
            )
            geographic_path = [
                terrain_graph.to_wgs84(GridPoint(int(x), int(y)))
                for x, y in candidate.geometry.coordinates
            ]
            # The two endpoint coordinates are reviewed inputs, not derived cell centers.
            geographic_path[0] = (first.longitude, first.latitude)
            geographic_path[-1] = (second.longitude, second.latitude)
            candidates.append(candidate.model_copy(update={
                "id": f"{route_id}-{first.id}-to-{second.id}",
                "geometry": GeoJsonLineString(coordinates=geographic_path),
                "coordinate_system": "EPSG:4326",
                "projection_method": terrain_graph.spec.projection_method,
                "grid_cell_size_m": terrain_graph.spec.cell_size_m,
                "grid_width": terrain_graph.spec.width,
                "grid_height": terrain_graph.spec.height,
                "terrain_source": terrain_graph.source,
                "provenance": "terrain_aware_algorithmic_candidate",
                "assumptions": [
                    *candidate.assumptions,
                    "Endpoints are explicit reviewed coordinates; intermediate cells are offline terrain-aware algorithmic candidates.",
                ],
                "segment_ledger": [self._segment_ledger(candidate, first, second, terrain_graph)],
            }))
        coordinates: list[tuple[float, float]] = []
        for candidate in candidates:
            segment = list(candidate.geometry.coordinates)
            coordinates.extend(segment if not coordinates else segment[1:])
        evidence_refs = list(dict.fromkeys(reference for item in waypoints for reference in item.evidence_refs))
        return ReconstructedHistoricalRoute(
            route_id=route_id,
            candidate_paths=candidates,
            geometry=GeoJsonLineString(coordinates=coordinates),
            evidence_refs=evidence_refs,
            terrain_source=terrain_graph.source,
            assumptions=[
                "Only caller-supplied evidence-backed waypoints are used.",
                "No geocoding, road matching, or historical-fact inference occurs in reconstruction.",
            ],
            limitations=[
                *( ["Offline mock terrain is a deterministic test surface, not a real DEM."]
                   if terrain_graph.source == OfflineMockTerrainGraphProvider.source
                   else ["Terrain elevations are sampled from a caller-supplied offline DEM; coverage and resolution limit the candidate."] ),
                "The output is a terrain-aware candidate, not an asserted historical route or precise march track.",
            ],
        )

    @staticmethod
    def _validate_waypoint(waypoint: ReviewedHistoricalWaypoint) -> None:
        if not waypoint.evidence_refs:
            raise HistoricalRouteReconstructionError(f"waypoint {waypoint.id} has no evidence references")

    @staticmethod
    def _segment_ledger(
        candidate: CandidateRoute,
        first: ReviewedHistoricalWaypoint,
        second: ReviewedHistoricalWaypoint,
        terrain_graph: TerrainGraph,
    ) -> CandidateRouteSegmentLedger:
        missing_or_nodata = sum(
            terrain_graph.grid.cell(GridPoint(int(x), int(y))).terrain in {"missing_dem", "nodata_dem", "no_data"}
            for x, y in candidate.geometry.coordinates
        )
        return CandidateRouteSegmentLedger(
            segment_id=f"{first.id}-to-{second.id}", source_anchor_id=first.id, target_anchor_id=second.id,
            physical_distance_km=candidate.metrics.distance_km,
            elevation_gain_m=candidate.metrics.elevation_gain_m,
            elevation_loss_m=candidate.metrics.elevation_loss_m,
            max_slope=candidate.metrics.max_slope,
            search_cost_total=candidate.metrics.search_cost,
            cost_breakdown=candidate.cost_breakdown,
            terrain_source=terrain_graph.source, grid_resolution_m=terrain_graph.spec.cell_size_m,
            sample_count=candidate.metrics.cell_count, edge_count=candidate.metrics.segment_count,
            nodata_or_missing_count=missing_or_nodata,
            applied_constraints=list(terrain_graph.applied_constraints),
        )
