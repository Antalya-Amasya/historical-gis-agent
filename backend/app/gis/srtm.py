"""Bounded-memory, on-demand access to local SRTM3 HGT elevation tiles."""
from __future__ import annotations

from functools import lru_cache
from math import floor
from pathlib import Path

import numpy as np

from backend.app.core.config import settings


class SrtmElevationUnavailableError(ValueError):
    """A sample is unavailable; it is never an elevation of zero metres."""

    def __init__(self, status: str, tile_name: str):
        super().__init__(f"SRTM elevation unavailable: {status} ({tile_name})")
        self.status = status
        self.tile_name = tile_name


class SrtmElevationService:
    """Read local SRTM3 tiles through a 16-entry LRU tile buffer.

    HGT rows run north-to-south, while columns run west-to-east. Missing coverage
    and void samples raise explicit availability errors; neither is sea level.
    """

    SAMPLES_PER_EDGE = 1201
    VOID_ELEVATION = -32768

    def __init__(self, hgt_dir: str | Path | None = None) -> None:
        configured_dir = hgt_dir if hgt_dir is not None else settings.dem_hgt_dir
        self.hgt_dir = Path(configured_dir) if configured_dir else None

    @staticmethod
    def tile_name_for(lat: float, lon: float) -> str:
        """Return the HGT name for the tile whose southwest corner contains a point."""
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("latitude/longitude is invalid")
        south_lat = floor(lat)
        west_lon = floor(lon)
        lat_prefix = "N" if south_lat >= 0 else "S"
        lon_prefix = "E" if west_lon >= 0 else "W"
        return f"{lat_prefix}{abs(south_lat):02d}{lon_prefix}{abs(west_lon):03d}.hgt"

    @lru_cache(maxsize=16)
    def _load_tile(self, tile_name: str) -> np.ndarray | None:
        """Load one tile on a cache miss; missing coverage intentionally maps to None."""
        if self.hgt_dir is None:
            return None
        path = self.hgt_dir / tile_name
        if not path.is_file():
            return None
        raw = path.read_bytes()
        expected_size = self.SAMPLES_PER_EDGE * self.SAMPLES_PER_EDGE * 2
        if len(raw) != expected_size:
            raise ValueError(f"invalid SRTM3 tile size for {tile_name}: expected {expected_size} bytes")
        tile = np.frombuffer(raw, dtype=">i2").reshape((self.SAMPLES_PER_EDGE, self.SAMPLES_PER_EDGE))
        tile.setflags(write=False)
        return tile

    def get_elevation(self, lat: float, lon: float) -> float:
        """Return elevation in metres, resolving tile seams transparently.

        The sampled coordinate is assigned with floor() to one southwest-based
        tile. The HGT row is flipped because array row zero is the north edge.
        """
        tile_name = self.tile_name_for(lat, lon)
        tile = self._load_tile(tile_name)
        if tile is None:
            raise SrtmElevationUnavailableError("MISSING", tile_name)

        south_lat, west_lon = floor(lat), floor(lon)
        column = round((lon - west_lon) * (self.SAMPLES_PER_EDGE - 1))
        row = round(((south_lat + 1) - lat) * (self.SAMPLES_PER_EDGE - 1))
        column = min(self.SAMPLES_PER_EDGE - 1, max(0, column))
        row = min(self.SAMPLES_PER_EDGE - 1, max(0, row))
        elevation = int(tile[row, column])
        if elevation == self.VOID_ELEVATION:
            raise SrtmElevationUnavailableError("NODATA", tile_name)
        return float(elevation)
