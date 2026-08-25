"""Deterministic location-confidence contracts at the planning input boundary."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .grid import GridPoint
from .models import CandidateRouteAnchor


class LocationConfidence(str, Enum):
    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"
    DISPUTED = "DISPUTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ResolvedCoordinate:
    point: GridPoint
    confidence: LocationConfidence = LocationConfidence.UNKNOWN
    notes: str | None = None
    # Explicit WGS84 display coordinate; GridPoint itself is never treated as longitude/latitude.
    display_coordinate: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.display_coordinate is None:
            return
        longitude, latitude = self.display_coordinate
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("display_coordinate must be valid longitude/latitude")


class CoordinateResolutionError(ValueError):
    pass


class RouteCoordinateResolver(ABC):
    @abstractmethod
    def resolve(self, anchor: CandidateRouteAnchor) -> ResolvedCoordinate:
        raise NotImplementedError


class MockCoordinateResolver(RouteCoordinateResolver):
    """Fixed offline mapping. Bare GridPoint values remain backwards-compatible exact mappings."""

    def __init__(self, mapping: dict[str, GridPoint | ResolvedCoordinate]):
        self.mapping = dict(mapping)

    def resolve(self, anchor: CandidateRouteAnchor) -> ResolvedCoordinate:
        value = self.mapping.get(anchor.historical_place_id) or self.mapping.get(anchor.canonical_name)
        if value is None:
            raise CoordinateResolutionError(f"no configured grid coordinate for anchor {anchor.historical_place_id}")
        return value if isinstance(value, ResolvedCoordinate) else ResolvedCoordinate(value, LocationConfidence.EXACT)


def validate_resolved_coordinate(
    anchor: CandidateRouteAnchor,
    resolved: ResolvedCoordinate,
    *,
    allow_disputed_locations: bool,
) -> list[str]:
    if resolved.confidence is LocationConfidence.UNKNOWN:
        raise CoordinateResolutionError(f"location confidence is unknown for anchor {anchor.historical_place_id}")
    if resolved.confidence is LocationConfidence.DISPUTED and not allow_disputed_locations:
        raise CoordinateResolutionError(f"disputed location requires explicit allowance for anchor {anchor.historical_place_id}")
    if resolved.confidence is LocationConfidence.APPROXIMATE:
        detail = resolved.notes or "location is approximate"
        return [f"{anchor.canonical_name}: {detail}"]
    if resolved.confidence is LocationConfidence.DISPUTED:
        detail = resolved.notes or "location is disputed and explicitly allowed"
        return [f"{anchor.canonical_name}: {detail}"]
    return []
