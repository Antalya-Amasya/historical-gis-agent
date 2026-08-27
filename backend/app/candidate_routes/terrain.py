"""Offline terrain-provider contracts and SRTM HGT support for Phase 6.2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from backend.app.core.config import settings
from backend.app.gis.dem import ElevationStatus, HgtFormatError, HgtRaster, HgtTileStore
from backend.app.gis.dem_manifest import load_dem_manifest

from .grid import GridPoint

if TYPE_CHECKING:
    from .geographic import GeographicGridSpec, TerrainGrid


class TerrainDataUnavailableError(ValueError):
    """The requested coordinate is outside an offline DEM tile or has no sample."""


UnsupportedDemError = HgtFormatError


class TerrainProvider(ABC):
    """Provides elevations and a geographic terrain grid; never performs routing."""

    source: str

    @abstractmethod
    def get_elevation(self, longitude: float, latitude: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def build_grid(self, bounds: "GeographicGridSpec", resolution_m: float | None = None) -> "TerrainGrid":
        raise NotImplementedError


class RealTerrainProvider(TerrainProvider):
    """Offline DEM contract. Dataset metadata describes terrain data, never historical facts."""

    dataset_id: str
    coordinate_reference: str = "EPSG:4326"
    resolution_m: float | None = None


class SyntheticTerrainProvider(TerrainProvider):
    """Offline deterministic terrain for tests and no-DEM environments."""

    source = "synthetic_geographic"

    def __init__(self, rule: Callable[[GridPoint, float, float], "TerrainOverride | None"] | None = None):
        self.rule = rule

    def get_elevation(self, longitude: float, latitude: float) -> float:
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("longitude/latitude is invalid")
        return 0.0

    def build(self, bounds: "GeographicGridSpec") -> "TerrainGrid":
        """Phase 6.1 compatibility alias for the earlier synthetic-only provider."""
        return self.build_grid(bounds, resolution_m=bounds.cell_size_m)

    def build_grid(self, bounds: "GeographicGridSpec", resolution_m: float | None = None) -> "TerrainGrid":
        if resolution_m is not None and resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if resolution_m is not None and resolution_m != bounds.cell_size_m:
            raise ValueError("terrain resolution must match the geographic grid cell size")
        from .geographic import TerrainGrid

        grid = TerrainGrid(bounds)
        for point in tuple(grid._cells):
            lon, lat = bounds.grid_to_geographic(point)
            override = self.rule(point, lon, lat) if self.rule else None
            if override is not None:
                grid.set_cell(
                    point,
                    elevation_m=override.elevation_m,
                    terrain=override.terrain,
                    terrain_multiplier=override.terrain_multiplier,
                    blocked=override.blocked,
                    cell_size_m=bounds.cell_size_m,
                )
        return grid


class OfflineMockTerrainProvider(SyntheticTerrainProvider):
    """Named deterministic terrain provider retained for offline reconstruction tests."""

    source = "offline_mock_terrain"


@dataclass(frozen=True)
class TerrainOverride:
    elevation_m: float | None = None
    terrain: str | None = None
    terrain_multiplier: float | None = None
    blocked: bool | None = None


class DEMTerrainProvider(RealTerrainProvider):
    """Optional, offline-only SRTM HGT terrain provider; it never downloads DEM data."""

    source = "offline_srtm_hgt"
    coordinate_reference = "EPSG:4326"

    def __init__(self, raster: HgtRaster, *, dataset_id: str = "local_srtm_hgt"):
        self.raster = raster
        self.dataset_id = dataset_id
        # One-degree HGT samples have an approximate latitude-dependent ground spacing;
        # this simple declaration is metadata, not a resampling operation.
        self.resolution_m = 111_320.0 / (raster.size - 1)

    @classmethod
    def from_hgt(cls, path: str | Path) -> "DEMTerrainProvider":
        return cls(HgtRaster.from_file(path))

    def get_elevation(self, longitude: float, latitude: float) -> float:
        try:
            elevation = self.raster.sample_at(longitude, latitude)
        except ValueError as exc:
            raise TerrainDataUnavailableError("coordinate is outside this DEM tile") from exc
        if elevation == HgtRaster.VOID_ELEVATION:
            raise TerrainDataUnavailableError("DEM sample is marked no-data")
        return float(elevation)

    def build_grid(self, bounds: "GeographicGridSpec", resolution_m: float | None = None) -> "TerrainGrid":
        if resolution_m is not None and resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if resolution_m is not None and resolution_m != bounds.cell_size_m:
            raise ValueError("terrain resolution must match the geographic grid cell size")
        from .geographic import TerrainGrid

        grid = TerrainGrid(bounds)
        for point in tuple(grid._cells):
            lon, lat = bounds.grid_to_geographic(point)
            try:
                elevation = self.get_elevation(lon, lat)
            except TerrainDataUnavailableError:
                grid.set_cell(point, terrain="no_data", blocked=True, cell_size_m=bounds.cell_size_m)
            else:
                grid.set_cell(point, elevation_m=elevation, terrain="dem", cell_size_m=bounds.cell_size_m)
        return grid


class MosaicDEMProvider(TerrainProvider):
    """Offline SRTM HGT provider that loads and caches the tile for each query.

    Missing tiles and HGT no-data samples remain explicit unavailable terrain; this
    provider deliberately does not infer water coverage or download data.
    """

    source = "offline_srtm_hgt_mosaic"

    def __init__(self, hgt_dir: str | Path, *, dataset_id: str = "local_srtm_hgt_mosaic", cache_size: int | None = None, samples_per_edge: int | None = 1201, manifest_path: str | Path | None = None) -> None:
        self.hgt_dir = Path(hgt_dir)
        self.dataset_id = dataset_id
        self.provenance_metadata = load_dem_manifest(manifest_path or settings.dem_manifest_path)
        self._store = HgtTileStore(
            self.hgt_dir,
            cache_size=settings.dem_tile_cache_size if cache_size is None else cache_size,
            samples_per_edge=samples_per_edge,
            source=self.source,
        )

    @staticmethod
    def tile_name_for(longitude: float, latitude: float) -> str:
        return HgtTileStore.tile_name_for(latitude, longitude)

    def get_elevation(self, longitude: float, latitude: float) -> float:
        sample = self._store.sample(latitude, longitude)
        if sample.status is not ElevationStatus.VALID:
            raise TerrainDataUnavailableError(f"DEM {sample.status.value.lower()}: {sample.tile_id}")
        assert sample.elevation_m is not None
        return sample.elevation_m

    @property
    def cached_tile_ids(self) -> tuple[str, ...]:
        return self._store.cached_tile_ids

    def build_grid(self, bounds: "GeographicGridSpec", resolution_m: float | None = None) -> "TerrainGrid":
        if resolution_m is not None and resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if resolution_m is not None and resolution_m != bounds.cell_size_m:
            raise ValueError("terrain resolution must match the geographic grid cell size")
        from .geographic import TerrainGrid

        grid = TerrainGrid(bounds)
        for point in tuple(grid._cells):
            lon, lat = bounds.grid_to_geographic(point)
            try:
                elevation = self.get_elevation(lon, lat)
            except TerrainDataUnavailableError:
                grid.set_cell(point, terrain="no_data", blocked=True, cell_size_m=bounds.cell_size_m)
            else:
                grid.set_cell(point, elevation_m=elevation, terrain="dem", cell_size_m=bounds.cell_size_m)
        return grid
