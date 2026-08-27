"""Compatibility adapter for the canonical local HGT sample contract."""
from __future__ import annotations

from pathlib import Path

from backend.app.core.config import settings

from .dem import ElevationSample, ElevationStatus, HgtTileStore


class SrtmElevationUnavailableError(ValueError):
    """Compatibility error for callers that still expect exception semantics."""

    def __init__(self, status: str, tile_name: str):
        super().__init__(f"SRTM elevation unavailable: {status} ({tile_name})")
        self.status = status
        self.tile_name = tile_name


class SrtmElevationService:
    """Strict SRTM3 adapter over :class:`HgtTileStore`."""

    SAMPLES_PER_EDGE = 1201
    VOID_ELEVATION = -32768

    def __init__(self, hgt_dir: str | Path | None = None, *, cache_size: int | None = None) -> None:
        configured_dir = hgt_dir if hgt_dir is not None else settings.dem_hgt_dir
        self.hgt_dir = Path(configured_dir) if configured_dir else None
        self._store = HgtTileStore(
            self.hgt_dir,
            cache_size=settings.dem_tile_cache_size if cache_size is None else cache_size,
            samples_per_edge=self.SAMPLES_PER_EDGE,
            source="srtm3_hgt",
        )

    @staticmethod
    def tile_name_for(lat: float, lon: float) -> str:
        return HgtTileStore.tile_name_for(lat, lon)

    def sample(self, latitude: float, longitude: float) -> ElevationSample:
        return self._store.sample(latitude, longitude)

    @property
    def cached_tile_ids(self) -> tuple[str, ...]:
        return self._store.cached_tile_ids

    def get_elevation(self, lat: float, lon: float) -> float:
        sample = self.sample(lat, lon)
        if sample.status is not ElevationStatus.VALID:
            raise SrtmElevationUnavailableError(sample.status.value, sample.tile_id or "UNKNOWN")
        assert sample.elevation_m is not None
        return sample.elevation_m
