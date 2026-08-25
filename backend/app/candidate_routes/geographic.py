"""Offline WGS84-to-local-grid adapter for Phase 6.1."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, pi
from typing import Callable

from backend.app.models import GeoJsonLineString, HistoricalRoutePoint

from .engine import CandidateRouteEngine
from .grid import GridCell, GridPoint, SyntheticGrid
from .models import ArmyProfile, CandidateRoute, CandidateRouteAnchor

EARTH_RADIUS_M = 6_371_000.0
DEFAULT_MAX_GRID_CELLS = 100_000
_EPSILON_M = 1e-6


class GridTooLargeError(ValueError):
    pass


class PointOutsideGridError(ValueError):
    pass


@dataclass(frozen=True)
class LocalProjection:
    """Local equirectangular approximation; suitable for small local analysis areas only."""

    reference_lon: float
    reference_lat: float

    def __post_init__(self) -> None:
        if not -180 <= self.reference_lon <= 180 or not -90 < self.reference_lat < 90:
            raise ValueError("projection reference must be a valid non-polar longitude/latitude")

    @property
    def _cos_reference_lat(self) -> float:
        return cos(self.reference_lat * pi / 180.0)

    def to_local(self, lon: float, lat: float) -> tuple[float, float]:
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("longitude/latitude is invalid")
        x = EARTH_RADIUS_M * self._cos_reference_lat * (lon - self.reference_lon) * pi / 180.0
        y = EARTH_RADIUS_M * (lat - self.reference_lat) * pi / 180.0
        return x, y

    def to_geographic(self, x_m: float, y_m: float) -> tuple[float, float]:
        lon = self.reference_lon + (x_m / (EARTH_RADIUS_M * self._cos_reference_lat)) * 180.0 / pi
        lat = self.reference_lat + (y_m / EARTH_RADIUS_M) * 180.0 / pi
        return lon, lat

    def distance_m(self, first: tuple[float, float], second: tuple[float, float]) -> float:
        x1, y1 = self.to_local(*first)
        x2, y2 = self.to_local(*second)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


@dataclass(frozen=True)
class GeographicGridSpec:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    width: int
    height: int
    cell_size_m: float
    projection: LocalProjection
    min_x_m: float
    min_y_m: float
    coordinate_reference: str = "EPSG:4326"
    projection_method: str = "local_equirectangular"
    source: str = "synthetic_geographic"

    @classmethod
    def from_anchor_coordinates(
        cls,
        first: tuple[float, float],
        second: tuple[float, float],
        *,
        padding_km: float = 2.0,
        cell_size_m: float = 1_000.0,
        max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
        source: str = "synthetic_geographic",
    ) -> "GeographicGridSpec":
        if padding_km < 0 or cell_size_m <= 0 or max_grid_cells <= 0:
            raise ValueError("padding, cell size, and grid limit must be positive")
        projection = LocalProjection((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        first_xy, second_xy = projection.to_local(*first), projection.to_local(*second)
        padding_m = padding_km * 1_000.0
        min_x = min(first_xy[0], second_xy[0]) - padding_m
        min_y = min(first_xy[1], second_xy[1]) - padding_m
        raw_max_x = max(first_xy[0], second_xy[0]) + padding_m
        raw_max_y = max(first_xy[1], second_xy[1]) + padding_m
        width = max(1, ceil((raw_max_x - min_x) / cell_size_m))
        height = max(1, ceil((raw_max_y - min_y) / cell_size_m))
        if width * height > max_grid_cells:
            raise GridTooLargeError(f"geographic grid would contain {width * height} cells, above limit {max_grid_cells}")
        max_x = min_x + width * cell_size_m
        max_y = min_y + height * cell_size_m
        min_lon, min_lat = projection.to_geographic(min_x, min_y)
        max_lon, max_lat = projection.to_geographic(max_x, max_y)
        return cls(min_lon, min_lat, max_lon, max_lat, width, height, cell_size_m, projection, min_x, min_y, source=source)

    @property
    def max_x_m(self) -> float:
        return self.min_x_m + self.width * self.cell_size_m

    @property
    def max_y_m(self) -> float:
        return self.min_y_m + self.height * self.cell_size_m

    def geographic_to_grid(self, lon: float, lat: float) -> GridPoint:
        x_m, y_m = self.projection.to_local(lon, lat)
        return GridPoint(self._index(x_m, self.min_x_m, self.max_x_m, self.width), self._index(y_m, self.min_y_m, self.max_y_m, self.height))

    def grid_to_geographic(self, point: GridPoint) -> tuple[float, float]:
        if not 0 <= point.x < self.width or not 0 <= point.y < self.height:
            raise PointOutsideGridError(f"grid point outside bounds: {point}")
        return self.projection.to_geographic(
            self.min_x_m + (point.x + 0.5) * self.cell_size_m,
            self.min_y_m + (point.y + 0.5) * self.cell_size_m,
        )

    def _index(self, value: float, minimum: float, maximum: float, count: int) -> int:
        if value < minimum - _EPSILON_M or value > maximum + _EPSILON_M:
            raise PointOutsideGridError("geographic point lies outside grid bounds")
        if abs(value - maximum) <= _EPSILON_M:
            return count - 1  # explicit inclusive-boundary policy, never a silent clamp
        index = int((value - minimum) // self.cell_size_m)
        if not 0 <= index < count:
            raise PointOutsideGridError("geographic point does not map to a grid cell")
        return index


@dataclass(frozen=True)
class TerrainCell:
    grid_point: GridPoint
    longitude: float
    latitude: float
    elevation_m: float
    terrain: str
    terrain_multiplier: float
    blocked: bool
    provenance: str = "derived_geographic_data"


class TerrainGrid(SyntheticGrid):
    """SyntheticGrid-compatible terrain with WGS84 cell-center metadata."""

    def __init__(self, spec: GeographicGridSpec):
        super().__init__(spec.width, spec.height)
        self.spec = spec
        for point in tuple(self._cells):
            self.set_cell(point, terrain="plain")

    def terrain_cell(self, point: GridPoint) -> TerrainCell:
        cell: GridCell = self.cell(point)
        longitude, latitude = self.spec.grid_to_geographic(point)
        return TerrainCell(point, longitude, latitude, cell.elevation_m, cell.terrain, cell.terrain_multiplier, cell.blocked)


@dataclass(frozen=True)
class TerrainOverride:
    elevation_m: float | None = None
    terrain: str | None = None
    terrain_multiplier: float | None = None
    blocked: bool | None = None


class SyntheticGeographicTerrainProvider:
    """Offline deterministic provider; tests may inject a geographic cell rule."""

    def __init__(self, rule: Callable[[GridPoint, float, float], TerrainOverride | None] | None = None):
        self.rule = rule

    def build(self, spec: GeographicGridSpec) -> TerrainGrid:
        grid = TerrainGrid(spec)
        if self.rule is None:
            return grid
        for point in tuple(grid._cells):
            lon, lat = spec.grid_to_geographic(point)
            override = self.rule(point, lon, lat)
            if override is not None:
                grid.set_cell(point, elevation_m=override.elevation_m, terrain=override.terrain, terrain_multiplier=override.terrain_multiplier, blocked=override.blocked)
        return grid


class GeographicCandidateRouteService:
    """Maps Evidence anchors through an offline terrain grid without changing A*."""

    def __init__(self, terrain_provider: SyntheticGeographicTerrainProvider | None = None, engine: CandidateRouteEngine | None = None):
        self.terrain_provider = terrain_provider or SyntheticGeographicTerrainProvider()
        self.engine = engine or CandidateRouteEngine()

    def build_between(
        self,
        from_point: HistoricalRoutePoint,
        to_point: HistoricalRoutePoint,
        profile: ArmyProfile,
        *,
        padding_km: float = 2.0,
        cell_size_m: float = 1_000.0,
        max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
    ) -> CandidateRoute:
        first = from_point.historical_place
        second = to_point.historical_place
        spec = GeographicGridSpec.from_anchor_coordinates(
            (first.longitude, first.latitude), (second.longitude, second.latitude),
            padding_km=padding_km, cell_size_m=cell_size_m, max_grid_cells=max_grid_cells,
        )
        grid = self.terrain_provider.build(spec)
        candidate = self.engine.build_route(
            from_anchor=CandidateRouteAnchor.from_historical_point(from_point),
            to_anchor=CandidateRouteAnchor.from_historical_point(to_point),
            start=spec.geographic_to_grid(first.longitude, first.latitude),
            goal=spec.geographic_to_grid(second.longitude, second.latitude),
            grid=grid, profile=profile,
        )
        path = [GridPoint(int(x), int(y)) for x, y in candidate.geometry.coordinates]
        geographic_geometry = GeoJsonLineString(coordinates=[spec.grid_to_geographic(point) for point in path])
        metrics = candidate.metrics.model_copy(update={"distance_km": candidate.metrics.segment_count * cell_size_m / 1_000.0})
        return candidate.model_copy(update={
            "geometry": geographic_geometry,
            "metrics": metrics,
            "coordinate_system": spec.coordinate_reference,
            "projection_method": spec.projection_method,
            "grid_cell_size_m": spec.cell_size_m,
            "grid_width": spec.width,
            "grid_height": spec.height,
            "terrain_source": spec.source,
            "assumptions": [*candidate.assumptions, "Terrain is offline synthetic geographic data; geographic cells are algorithmic candidates, not historical facts."],
        })
