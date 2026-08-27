"""Canonical offline HGT sampling contract.

This module models elevation availability only.  It deliberately has no
LAND/WATER semantics: valid sea-level samples, missing tiles, and void samples
are distinct states.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from math import floor
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np


class ElevationStatus(str, Enum):
    VALID = "VALID"
    MISSING = "MISSING"
    NODATA = "NODATA"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ElevationSample:
    elevation_m: float | None
    status: ElevationStatus
    source: str
    tile_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("elevation sample source must be non-empty")
        if self.status is ElevationStatus.VALID and self.elevation_m is None:
            raise ValueError("VALID elevation samples require elevation_m")
        if self.status is not ElevationStatus.VALID and self.elevation_m is not None:
            raise ValueError("unavailable elevation samples must not contain elevation_m")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DEMProvider(Protocol):
    """A source of elevation samples; never a surface classifier."""

    def sample(self, latitude: float, longitude: float) -> ElevationSample: ...


class HgtFormatError(ValueError):
    """A local HGT file cannot be used under the requested contract."""


@dataclass(frozen=True)
class HgtRaster:
    """One-degree big-endian signed-int16 HGT raster with north-origin rows."""

    west_lon: int
    south_lat: int
    samples: np.ndarray

    VOID_ELEVATION = -32768

    @classmethod
    def from_file(cls, path: str | Path, *, samples_per_edge: int | None = None) -> "HgtRaster":
        dem_path = Path(path)
        match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})\.hgt", dem_path.name, re.IGNORECASE)
        if match is None:
            raise HgtFormatError("HGT filename must use the NxxEyyy.hgt convention")
        raw = dem_path.read_bytes()
        if len(raw) == 0 or len(raw) % 2:
            raise HgtFormatError("HGT data must contain 16-bit samples")
        edge = int((len(raw) // 2) ** 0.5)
        if edge < 2 or edge * edge * 2 != len(raw):
            raise HgtFormatError("HGT data must be a square raster")
        if samples_per_edge is not None and edge != samples_per_edge:
            raise HgtFormatError(
                f"invalid HGT raster size for {dem_path.name}: expected {samples_per_edge} samples per edge"
            )
        values = np.frombuffer(raw, dtype=">i2").reshape((edge, edge))
        values.setflags(write=False)
        lat_sign = 1 if match.group(1).upper() == "N" else -1
        lon_sign = 1 if match.group(3).upper() == "E" else -1
        return cls(lon_sign * int(match.group(4)), lat_sign * int(match.group(2)), values)

    @property
    def size(self) -> int:
        return int(self.samples.shape[0])

    def sample_at(self, longitude: float, latitude: float) -> int:
        if not self.west_lon <= longitude <= self.west_lon + 1 or not self.south_lat <= latitude <= self.south_lat + 1:
            raise ValueError("coordinate is outside this DEM tile")
        column = min(self.size - 1, max(0, round((longitude - self.west_lon) * (self.size - 1))))
        row = min(self.size - 1, max(0, round(((self.south_lat + 1) - latitude) * (self.size - 1))))
        return int(self.samples[row, column])


class HgtTileStore:
    """Bounded deterministic HGT loader shared by all local HGT adapters."""

    source = "local_hgt"

    def __init__(
        self,
        hgt_dir: str | Path | None,
        *,
        cache_size: int = 16,
        samples_per_edge: int | None = 1201,
        source: str = source,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        self.hgt_dir = Path(hgt_dir) if hgt_dir is not None else None
        self.cache_size = cache_size
        self.samples_per_edge = samples_per_edge
        self.source = source
        self._cache: OrderedDict[str, HgtRaster] = OrderedDict()

    @staticmethod
    def tile_name_for(latitude: float, longitude: float) -> str:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("latitude/longitude is invalid")
        south_lat, west_lon = floor(latitude), floor(longitude)
        return f"{'N' if south_lat >= 0 else 'S'}{abs(south_lat):02d}{'E' if west_lon >= 0 else 'W'}{abs(west_lon):03d}.hgt"

    @property
    def cached_tile_ids(self) -> tuple[str, ...]:
        return tuple(self._cache)

    def sample(self, latitude: float, longitude: float) -> ElevationSample:
        tile_id = self.tile_name_for(latitude, longitude)
        raster_or_status = self._raster_for(tile_id)
        if isinstance(raster_or_status, ElevationStatus):
            return ElevationSample(None, raster_or_status, self.source, tile_id)
        try:
            elevation = raster_or_status.sample_at(longitude, latitude)
        except ValueError:
            return ElevationSample(None, ElevationStatus.INVALID, self.source, tile_id)
        if elevation == HgtRaster.VOID_ELEVATION:
            return ElevationSample(None, ElevationStatus.NODATA, self.source, tile_id)
        return ElevationSample(float(elevation), ElevationStatus.VALID, self.source, tile_id)

    def _raster_for(self, tile_id: str) -> HgtRaster | ElevationStatus:
        cached = self._cache.get(tile_id)
        if cached is not None:
            self._cache.move_to_end(tile_id)
            return cached
        if self.hgt_dir is None:
            return ElevationStatus.MISSING
        path = self.hgt_dir / tile_id
        if not path.is_file():
            return ElevationStatus.MISSING
        try:
            raster = HgtRaster.from_file(path, samples_per_edge=self.samples_per_edge)
        except (OSError, HgtFormatError):
            return ElevationStatus.INVALID
        self._cache[tile_id] = raster
        self._cache.move_to_end(tile_id)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return raster
