"""Low-level GIS data services used by terrain-aware route planning."""

from .srtm import SrtmElevationService, SrtmElevationUnavailableError
from .natural_earth_surface import NaturalEarthDatasetAudit, NaturalEarthSurfaceClassifier
from .natural_earth_preprocess import NaturalEarthPreprocessAudit, normalize_natural_earth_geojson
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
    "NaturalEarthDatasetAudit",
    "NaturalEarthSurfaceClassifier",
    "NaturalEarthPreprocessAudit",
    "NoSurfaceClassifier",
    "SrtmElevationService",
    "SrtmElevationUnavailableError",
    "SurfaceClassification",
    "SurfaceClassifier",
    "SurfaceType",
    "UNKNOWN_SURFACE",
    "normalize_natural_earth_geojson",
]
