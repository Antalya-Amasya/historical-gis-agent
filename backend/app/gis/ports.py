"""Evidence- and provenance-aware historical port assertions.

This module is intentionally an index, not a coastal inference or routing
system.  A registry entry states a reviewed assertion supplied by its caller;
it never creates ports from surface, DEM, city, or route data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import asin, cos, radians, sin, sqrt
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol

from .surface import SurfaceClassification, SurfaceClassifier, SurfaceType


class PortStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


# Semantic alias for callers that name the field after the evidence assertion.
PortEvidenceStatus = PortStatus


class PortSurfaceContext(str, Enum):
    COMPATIBLE = "compatible"
    AMBIGUOUS = "ambiguous"
    INCOMPATIBLE = "incompatible"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HistoricalPort:
    """A supplied historical transport-node assertion with separate provenance."""

    port_id: str
    canonical_name: str
    latitude: float
    longitude: float
    historical_names: tuple[str, ...] = ()
    valid_from: str | None = None
    valid_to: str | None = None
    status: PortStatus = PortStatus.UNKNOWN
    confidence: float | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    source_documents: tuple[str, ...] = ()
    historical_status_source: str = "UNKNOWN"
    coordinate_source: str = "UNKNOWN"
    coordinate_role: str = "UNKNOWN"
    coordinate_uncertainty_km: float | None = None
    supports_land_access: bool = False
    supports_sea_access: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.port_id or not self.canonical_name:
            raise ValueError("port_id and canonical_name must be non-empty")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("port coordinates must be valid WGS84 latitude/longitude")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.coordinate_uncertainty_km is not None and self.coordinate_uncertainty_km < 0:
            raise ValueError("coordinate_uncertainty_km must be non-negative")
        object.__setattr__(self, "historical_names", tuple(self.historical_names))
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "source_documents", tuple(self.source_documents))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class PortSurfaceDiagnostic:
    port_id: str
    context: PortSurfaceContext
    surface: SurfaceClassification
    note: str


class PortRegistry(Protocol):
    """Read-only lookup contract.  It deliberately has no route mutation API."""

    def get_by_id(self, port_id: str) -> HistoricalPort | None: ...

    def find_by_name(self, name: str) -> tuple[HistoricalPort, ...]: ...

    def find_near(self, latitude: float, longitude: float, tolerance_km: float) -> tuple[HistoricalPort, ...]: ...

    def list_ports(self, *, status: PortStatus | None = None) -> tuple[HistoricalPort, ...]: ...


class InMemoryPortRegistry:
    """Explicit in-memory registry for reviewed records and synthetic fixtures."""

    def __init__(self, ports: Iterable[HistoricalPort] = ()) -> None:
        self._ports: dict[str, HistoricalPort] = {}
        for port in ports:
            self.register(port)

    def register(self, port: HistoricalPort) -> None:
        """Explicit setup operation; no inference or external lookup occurs."""
        if port.port_id in self._ports:
            raise ValueError(f"duplicate port_id: {port.port_id}")
        self._ports[port.port_id] = port

    def get_by_id(self, port_id: str) -> HistoricalPort | None:
        return self._ports.get(port_id)

    def find_by_name(self, name: str) -> tuple[HistoricalPort, ...]:
        needle = name.strip().casefold()
        if not needle:
            return ()
        return tuple(
            port for port in self.list_ports()
            if needle == port.canonical_name.casefold() or needle in {item.casefold() for item in port.historical_names}
        )

    def find_near(self, latitude: float, longitude: float, tolerance_km: float) -> tuple[HistoricalPort, ...]:
        if tolerance_km < 0:
            raise ValueError("tolerance_km must be non-negative")
        matches = [
            (self._distance_km(latitude, longitude, port.latitude, port.longitude), port)
            for port in self.list_ports()
        ]
        return tuple(
            port for distance, port in sorted(matches, key=lambda item: (item[0], item[1].port_id))
            if distance <= tolerance_km
        )

    def list_ports(self, *, status: PortStatus | None = None) -> tuple[HistoricalPort, ...]:
        return tuple(port for _, port in sorted(self._ports.items()) if status is None or port.status is status)

    @staticmethod
    def _distance_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
        lat_a, lon_a, lat_b, lon_b = map(radians, (latitude_a, longitude_a, latitude_b, longitude_b))
        return 2 * 6_371.0088 * asin(sqrt(
            sin((lat_b - lat_a) / 2) ** 2
            + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
        ))


def validate_port_surface_context(port: HistoricalPort, classifier: SurfaceClassifier) -> PortSurfaceDiagnostic:
    """Return a modern-surface diagnostic without modifying the port assertion."""
    surface = classifier.classify(port.latitude, port.longitude)
    if surface.surface_type is SurfaceType.WATER:
        context = PortSurfaceContext.COMPATIBLE
    elif surface.surface_type is SurfaceType.LAND:
        context = PortSurfaceContext.INCOMPATIBLE
    elif surface.status in {"unavailable", "dataset_unavailable", "invalid_coordinate"}:
        context = PortSurfaceContext.UNAVAILABLE
    else:
        context = PortSurfaceContext.AMBIGUOUS
    return PortSurfaceDiagnostic(
        port_id=port.port_id,
        context=context,
        surface=surface,
        note="Modern surface context is diagnostic only; it neither proves nor changes the historical port assertion.",
    )
