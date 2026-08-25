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
    coordinate_role: str = "exact_site"
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


class GeoJsonLineString(BaseModel):
    type: str = "LineString"
    coordinates: list[tuple[float, float]] = Field(default_factory=list)


class ExtractedHistoricalPlaceMention(BaseModel):
    raw_name: str
    normalized_name: str
    sequence_hint: int
    date_or_period: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    context_excerpt: str
    confidence: float = Field(ge=0, le=1)
    unresolved_reason: str | None = None


class HistoricalRoutePoint(BaseModel):
    sequence: int = Field(ge=1)
    historical_place: HistoricalPlace
    event_summary: str
    date_or_period: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    coordinate_role: str = "exact_site"
    source_support: list[str] = Field(default_factory=list)


class HistoricalRoute(BaseModel):
    id: str
    event_id: str
    name: str
    period: str
    ordered_points: list[HistoricalRoutePoint] = Field(default_factory=list)
    geometry: GeoJsonLineString
    evidence_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    historical_confidence: float = Field(ge=0, le=1)
    unresolved_mentions: list[ExtractedHistoricalPlaceMention] = Field(default_factory=list)
    source_disagreements: list[str] = Field(default_factory=list)


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


class AgentToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    http_status: int | None = None


class HistoricalRouteIntent(BaseModel):
    """Structured campaign selection; it contains no coordinates or inferred places."""

    intent: str = "historical_route"
    campaign_id: str


class AgentToolHistoryEntry(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result_summary: str
    duration_ms: int = Field(ge=0)
    outcome: str = "success"
    budget_source: str = "general"


class AgentState(BaseModel):
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    current_event: HistoricalEvent | None = None
    historical_route: HistoricalRoute | None = None
    historical_period: str | None = None
    selected_route: str | None = None
    candidate_routes: list[RouteCandidate] = Field(default_factory=list)
    army_profile: ArmyProfile | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)
    historical_evidence: list[Evidence] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    map_state: dict[str, Any] = Field(default_factory=dict)
    user_query: str | None = None
    intent: str | None = None
    route_intent: HistoricalRouteIntent | None = None
    historical_route_presentation: dict[str, Any] | None = None
    requested_output: str = "answer"
    selected_model_tier: str | None = None
    selected_model_id: str | None = None
    model_policy: str | None = None
    quality_mode: str | None = None
    evidence_support_status: str | None = None
    relevant_evidence_count: int = 0
    total_evidence_count: int = 0
    matched_subject_terms: list[str] = Field(default_factory=list)
    missing_subject_terms: list[str] = Field(default_factory=list)
    grounding_corrections: int = 0
    detected_phrase_count: int = 0
    detected_entity_count: int = 0
    evidence_grounded_entity_count: int = 0
    query_context_entity_count: int = 0
    detected_work_titles: list[str] = Field(default_factory=list)
    evidence_grounded_claim_count: int = 0
    unverified_suggestion_count: int = 0
    unverified_suggestion_terms: list[str] = Field(default_factory=list)
    unsupported_fact_claim_count: int = 0
    unsupported_fact_terms: list[str] = Field(default_factory=list)
    ignored_non_entity_terms: list[str] = Field(default_factory=list)
    final_grounding_status: str | None = None
    resolved_places: list[HistoricalPlace] = Field(default_factory=list)
    tool_history: list[AgentToolHistoryEntry] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    status: str = "idle"
    warnings: list[str] = Field(default_factory=list)
    final_answer: str | None = None
    tool_execution_stats: dict[str, int] = Field(default_factory=dict)


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
