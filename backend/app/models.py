from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PlaceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HistoricalPlace(BaseModel):
    id: str
    canonical_name: str
    modern_name: str | None = None
    latitude: float
    longitude: float
    period: str | None = None
    source: str
    source_id: str | None = None
    source_url: str | None = None
    confidence: float = Field(ge=0, le=1)
    uncertain: bool = False
    alternatives: list["HistoricalPlace"] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    author: str
    work: str
    locator: str
    excerpt: str
    period: str | None = None
    topic: str | None = None
    reliability_note: str | None = None
    text: str | None = None
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_file: str | None = None
    source_type: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalEvent(BaseModel):
    id: str
    name: str
    period: str
    summary: str
    places: list[HistoricalPlace] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    uncertainty_note: str | None = None


class RouteMetrics(BaseModel):
    distance_km: float | None = Field(default=None, ge=0)
    elevation_gain_m: float | None = Field(default=None, ge=0)
    max_elevation_m: float | None = Field(default=None, ge=0)
    slope_risk: float | None = Field(default=None, ge=0)
    river_crossing_risk: float | None = Field(default=None, ge=0)
    historical_mismatch_cost: float | None = Field(default=None, ge=0)
    water_risk: str | None = None


class ArmyProfile(BaseModel):
    name: str
    infantry: int = Field(ge=0)
    cavalry: int = Field(ge=0)
    elephants: int = Field(ge=0)
    baggage: str
    mobility: dict[str, float] = Field(default_factory=dict)


class RouteCandidate(BaseModel):
    id: str
    name: str
    coordinates: list[tuple[float, float]] = Field(default_factory=list)
    metrics: RouteMetrics
    evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    historical_confidence: float = Field(ge=0, le=1)
    geographic_cost: float = Field(ge=0)


class AgentState(BaseModel):
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    current_event: HistoricalEvent | None = None
    historical_period: str | None = None
    selected_route: str | None = None
    candidate_routes: list[RouteCandidate] = Field(default_factory=list)
    army_profile: ArmyProfile | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)
    historical_evidence: list[Evidence] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    map_state: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    state: AgentState


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] = Field(default_factory=dict)


class RagSearchResponse(BaseModel):
    query: str
    evidence: list[Evidence]
