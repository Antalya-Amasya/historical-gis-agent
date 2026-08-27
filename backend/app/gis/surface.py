"""Independent land/water surface semantics for future GIS data adapters.

Surface classification deliberately has no dependency on elevation sampling,
historical evidence, routing, or movement costs.  Callers must apply a
classifier explicitly; there is no implicit world-wide land or water default.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class SurfaceType(str, Enum):
    """Known surface semantics, separate from elevation availability."""

    LAND = "land"
    WATER = "water"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SurfaceClassification:
    """An auditable surface result returned by a classifier."""

    surface_type: SurfaceType
    source: str
    confidence: float | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("surface classification source must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("surface classification confidence must be between 0 and 1")


UNKNOWN_SURFACE = SurfaceClassification(
    surface_type=SurfaceType.UNKNOWN,
    source="surface_classification_unavailable",
    status="unavailable",
)


class SurfaceClassifier(Protocol):
    """Read-only coordinate classifier; adapters may later wrap real datasets."""

    def classify(self, latitude: float, longitude: float) -> SurfaceClassification:
        """Return an auditable surface result without sampling elevation."""


class NoSurfaceClassifier:
    """Explicit absence of a surface dataset; never infers land or water."""

    def classify(self, latitude: float, longitude: float) -> SurfaceClassification:
        return UNKNOWN_SURFACE


class MockSurfaceClassifier:
    """Deterministic, explicitly registered classifier for tests and development."""

    source = "mock_surface_classifier"

    def __init__(self, default: SurfaceClassification = UNKNOWN_SURFACE) -> None:
        self._default = default
        self._entries: dict[tuple[float, float], SurfaceClassification] = {}

    def register(
        self,
        latitude: float,
        longitude: float,
        surface_type: SurfaceType,
        *,
        source: str | None = None,
        confidence: float | None = 1.0,
        status: str | None = "registered",
    ) -> None:
        self._entries[(latitude, longitude)] = SurfaceClassification(
            surface_type=surface_type,
            source=source or self.source,
            confidence=confidence,
            status=status,
        )

    def classify(self, latitude: float, longitude: float) -> SurfaceClassification:
        return self._entries.get((latitude, longitude), self._default)
