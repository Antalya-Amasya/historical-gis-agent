"""Low-level GIS data services used by terrain-aware route planning."""

from .srtm import SrtmElevationService, SrtmElevationUnavailableError
from .dem import DEMProvider, ElevationSample, ElevationStatus, HgtFormatError, HgtRaster, HgtTileStore
from .dem_manifest import DEMInventory, DEMProvenance, build_hgt_inventory, load_dem_manifest
from .ports import HistoricalPort, InMemoryPortRegistry, PortEvidenceStatus, PortRegistry, PortStatus, PortSurfaceContext, PortSurfaceDiagnostic, validate_port_surface_context
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
    "DEMInventory",
    "DEMProvider",
    "DEMProvenance",
    "ElevationSample",
    "ElevationStatus",
    "HgtFormatError",
    "HgtRaster",
    "HgtTileStore",
    "HistoricalPort",
    "InMemoryPortRegistry",
    "NaturalEarthDatasetAudit",
    "NaturalEarthSurfaceClassifier",
    "NaturalEarthPreprocessAudit",
    "NoSurfaceClassifier",
    "PortRegistry",
    "PortEvidenceStatus",
    "PortStatus",
    "PortSurfaceContext",
    "PortSurfaceDiagnostic",
    "SrtmElevationService",
    "SrtmElevationUnavailableError",
    "SurfaceClassification",
    "SurfaceClassifier",
    "SurfaceType",
    "UNKNOWN_SURFACE",
    "normalize_natural_earth_geojson",
    "build_hgt_inventory",
    "load_dem_manifest",
    "validate_port_surface_context",
]
