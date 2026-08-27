"""Low-level GIS data services used by terrain-aware route planning."""

from .srtm import SrtmElevationService, SrtmElevationUnavailableError
from .surface import (
    UNKNOWN_SURFACE,
    MockSurfaceClassifier,
    NoSurfaceClassifier,
    SurfaceClassification,
    SurfaceClassifier,
    SurfaceType,
)

__all__ = [
    "MockSurfaceClassifier",
    "NoSurfaceClassifier",
    "SrtmElevationService",
    "SrtmElevationUnavailableError",
    "SurfaceClassification",
    "SurfaceClassifier",
    "SurfaceType",
    "UNKNOWN_SURFACE",
]
