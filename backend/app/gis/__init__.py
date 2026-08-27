"""Low-level GIS data services used by terrain-aware route planning."""

from .srtm import SrtmElevationService, SrtmElevationUnavailableError

__all__ = ["SrtmElevationService", "SrtmElevationUnavailableError"]
