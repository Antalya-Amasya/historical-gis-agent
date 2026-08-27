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
    """Parameterized movement preferences, not a historical-facts database."""

    name: str = "generic_ground"
    movement_type: str = "ground"
    # Legacy compatibility for Phase 6.0 callers. New code should use movement_type.
    movement_mode: str | None = None
    mountain_tolerance: float = Field(default=0.5, ge=0, le=1)
    river_crossing_penalty: float = Field(default=1.0, ge=0)
    rough_terrain_penalty: float = Field(default=1.0, ge=0)
    supply_range: float = Field(default=30.0, gt=0)
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
    historical_cost: float = Field(default=0.0, ge=0)
    total_cost: float = Field(ge=0)


class RouteScore(BaseModel):
    """A deterministic score for one existing candidate route and one profile."""

    profile_name: str
    distance_cost: float = Field(ge=0)
    terrain_cost: float = Field(ge=0)
    historical_cost: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    explanation: list[str] = Field(default_factory=list)


class RouteMetrics(BaseModel):
    distance_km: float = Field(ge=0)
    elevation_gain_m: float = Field(ge=0)
    elevation_loss_m: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    cell_count: int = Field(ge=1)
    segment_count: int = Field(ge=0)
    # Search cost is intentionally separate from physical geographic distance.
    search_cost: float = Field(default=0.0, ge=0)
    max_slope: float = Field(default=0.0, ge=0)


class CandidateRouteSegmentLedger(BaseModel):
    """Auditable GIS computation for one evidence-grounded anchor segment."""

    segment_id: str
    source_anchor_id: str
    target_anchor_id: str
    physical_distance_km: float = Field(ge=0)
    elevation_gain_m: float = Field(ge=0)
    elevation_loss_m: float = Field(ge=0)
    max_slope: float = Field(ge=0)
    search_cost_total: float = Field(ge=0)
    cost_breakdown: RouteCostBreakdown
    terrain_source: str | None = None
    grid_resolution_m: float | None = Field(default=None, gt=0)
    sample_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodata_or_missing_count: int = Field(default=0, ge=0)
    applied_constraints: list[str] = Field(default_factory=list)
    geometry_role: str = "algorithmic_candidate"


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
    generation_method: str = "single_astar"
    segment_ledger: list[CandidateRouteSegmentLedger] = Field(default_factory=list)


    def evaluate(self, army_profile: ArmyProfile | None = None) -> RouteScore:
        """Score this fixed route; it neither searches nor changes its geometry."""
        if army_profile is None:
            terrain_cost = self.cost_breakdown.slope_cost + self.cost_breakdown.terrain_cost
            total = self.cost_breakdown.distance_cost + terrain_cost + self.cost_breakdown.barrier_cost
            return RouteScore(
                profile_name="baseline",
                distance_cost=self.cost_breakdown.distance_cost,
                terrain_cost=terrain_cost,
                historical_cost=0.0,
                total_cost=total,
                explanation=["Baseline Phase 6.2 terrain score; no historical movement profile applied."],
            )
        from .historical_costs import HistoricalCostModel

        return HistoricalCostModel().score_route(self, army_profile)


class CandidateRouteConstraints(BaseModel):
    """Deterministic strategy selection; these are routing inputs, not historical claims."""

    include_shortest_distance: bool = True
    include_terrain_optimized: bool = True
    include_historical_profile: bool = True

    @model_validator(mode="after")
    def includes_at_least_one_strategy(self) -> "CandidateRouteConstraints":
        if not (self.include_shortest_distance or self.include_terrain_optimized or self.include_historical_profile):
            raise ValueError("at least one candidate generation strategy must be enabled")
        return self


class CandidateRouteSet(BaseModel):
    routes: list[CandidateRoute] = Field(default_factory=list)
    generation_method: str = "deterministic_variants"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RankingProfile(BaseModel):
    """Weights for deterministic minimization; lower weighted cost ranks first."""

    distance_weight: float = Field(default=1.0, ge=0)
    terrain_weight: float = Field(default=1.0, ge=0)
    historical_weight: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def has_some_weight(self) -> "RankingProfile":
        if self.distance_weight + self.terrain_weight + self.historical_weight <= 0:
            raise ValueError("at least one ranking weight must be positive")
        return self


class RankedRoute(BaseModel):
    route: CandidateRoute
    rank: int = Field(ge=1)
    score: RouteScore
    ranking_score: float = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)


class RankedRouteSet(BaseModel):
    routes: list[RankedRoute] = Field(default_factory=list)
    ranking_profile: RankingProfile
    generation_method: str
