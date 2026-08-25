"""Phase 6 candidate-route contract, deliberately separate from HistoricalRoute."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from backend.app.models import GeoJsonLineString, HistoricalRoutePoint


class CandidateRouteAnchor(BaseModel):
    """An Evidence-grounded anchor supplied by HistoricalRoute, never selected by the engine."""

    historical_place_id: str
    canonical_name: str
    evidence_refs: list[str] = Field(default_factory=list)
    provenance: str = "evidence_grounded_anchor"

    @classmethod
    def from_historical_point(cls, point: HistoricalRoutePoint) -> "CandidateRouteAnchor":
        return cls(
            historical_place_id=point.historical_place.id,
            canonical_name=point.historical_place.canonical_name,
            evidence_refs=list(point.evidence_refs),
        )


class ArmyProfile(BaseModel):
    """Minimal generic movement weighting; no historical army simulation is implied."""

    movement_mode: str = "ground"
    distance_weight: float = Field(default=1.0, ge=0)
    slope_weight: float = Field(default=1.0, ge=0)
    terrain_weight: float = Field(default=1.0, ge=0)
    barrier_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def has_some_cost(self) -> "ArmyProfile":
        if self.distance_weight + self.slope_weight + self.terrain_weight + self.barrier_weight <= 0:
            raise ValueError("at least one movement cost weight must be positive")
        return self


class RouteCostBreakdown(BaseModel):
    distance_cost: float = Field(ge=0)
    slope_cost: float = Field(ge=0)
    terrain_cost: float = Field(ge=0)
    barrier_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)


class RouteMetrics(BaseModel):
    distance_km: float = Field(ge=0)
    elevation_gain_m: float = Field(ge=0)
    elevation_loss_m: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    cell_count: int = Field(ge=1)
    segment_count: int = Field(ge=0)


class CandidateRoute(BaseModel):
    """An algorithmic connection between supplied historical anchors, not a historical claim."""

    id: str
    from_anchor: CandidateRouteAnchor
    to_anchor: CandidateRouteAnchor
    geometry: GeoJsonLineString
    metrics: RouteMetrics
    cost_breakdown: RouteCostBreakdown
    confidence: float = Field(ge=0, le=1)
    assumptions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    provenance: str = "algorithmic_candidate"
    coordinate_system: str = "synthetic_grid"
    projection_method: str | None = None
    grid_cell_size_m: float | None = Field(default=None, gt=0)
    grid_width: int | None = Field(default=None, ge=1)
    grid_height: int | None = Field(default=None, ge=1)
    terrain_source: str | None = None
