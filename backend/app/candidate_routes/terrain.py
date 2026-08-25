"""Offline terrain-provider contracts and SRTM HGT support for Phase 6.2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isqrt
from pathlib import Path
import re
import struct
from typing import TYPE_CHECKING, Callable

from .grid import GridPoint

if TYPE_CHECKING:
    from .geographic import GeographicGridSpec, TerrainGrid


class TerrainDataUnavailableError(ValueError):
    """The requested coordinate is outside an offline DEM tile or has no sample."""


class UnsupportedDemError(ValueError):
    """The supplied offline DEM does not match the supported HGT format."""


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


@dataclass(frozen=True)
class HgtRaster:
    """One-degree SRTM HGT tile, sampled deterministically with nearest-neighbor lookup."""

    west_lon: int
    south_lat: int
    samples: tuple[tuple[int, ...], ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "HgtRaster":
        dem_path = Path(path)
        match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})\.hgt", dem_path.name, re.IGNORECASE)
        if match is None:
            raise UnsupportedDemError("HGT filename must use the NxxEyyy.hgt convention")
        raw = dem_path.read_bytes()
        if len(raw) == 0 or len(raw) % 2:
            raise UnsupportedDemError("HGT data must contain 16-bit samples")
        size = isqrt(len(raw) // 2)
        if size < 2 or size * size * 2 != len(raw):
            raise UnsupportedDemError("HGT data must be a square raster")
        values = struct.unpack(f">{size * size}h", raw)
        rows = tuple(tuple(values[row * size:(row + 1) * size]) for row in range(size))
        sign_lat = 1 if match.group(1).upper() == "N" else -1
        sign_lon = 1 if match.group(3).upper() == "E" else -1
        return cls(sign_lon * int(match.group(4)), sign_lat * int(match.group(2)), rows)

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def east_lon(self) -> int:
        return self.west_lon + 1

    @property
    def north_lat(self) -> int:
        return self.south_lat + 1

    def elevation_at(self, longitude: float, latitude: float) -> float:
        if not self.west_lon <= longitude <= self.east_lon or not self.south_lat <= latitude <= self.north_lat:
            raise TerrainDataUnavailableError("coordinate is outside this DEM tile")
        x = min(self.size - 1, max(0, round((longitude - self.west_lon) * (self.size - 1))))
        north_index = min(self.size - 1, max(0, round((self.north_lat - latitude) * (self.size - 1))))
        elevation = self.samples[north_index][x]
        if elevation == -32768:
            raise TerrainDataUnavailableError("DEM sample is marked no-data")
        return float(elevation)


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
        return self.raster.elevation_at(longitude, latitude)

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
