from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlaceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlaceSpatialSemantics(str, Enum):
    """Audited spatial meaning of a HistoricalPlace, separate from its name."""

    SETTLEMENT = "settlement"
    RIVER = "river"
    MOUNTAIN_REGION = "mountain_region"
    REGION = "region"
    ISLAND = "island"
    UNKNOWN = "unknown"


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
    spatial_semantics: PlaceSpatialSemantics = PlaceSpatialSemantics.UNKNOWN
    spatial_semantics_provenance: str | None = None
    authoritative_geometry_available: bool = False
    authoritative_geometry_reference: str | None = None
    authority_metadata: dict[str, Any] = Field(default_factory=dict)
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


class HistoricalEventType(str, Enum):
    POLITICAL = "POLITICAL"
    MILITARY = "MILITARY"
    BATTLE = "BATTLE"
    SIEGE = "SIEGE"
    REFORM = "REFORM"
    ASSASSINATION = "ASSASSINATION"
    TREATY = "TREATY"
    ELECTION = "ELECTION"
    REBELLION = "REBELLION"
    MOVEMENT = "MOVEMENT"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class TemporalPrecision(str, Enum):
    DAY = "DAY"
    MONTH = "MONTH"
    YEAR = "YEAR"
    YEAR_RANGE = "YEAR_RANGE"
    APPROXIMATE = "APPROXIMATE"
    UNKNOWN = "UNKNOWN"


class TemporalGroundingStatus(str, Enum):
    EVIDENCE_GROUNDED = "EVIDENCE_GROUNDED"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"


class EventPlaceRole(str, Enum):
    EVENT_SITE = "EVENT_SITE"
    ORIGIN = "ORIGIN"
    DESTINATION = "DESTINATION"
    RELATED_PLACE = "RELATED_PLACE"
    UNKNOWN = "UNKNOWN"


class EventPlaceResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    TEXT_ONLY = "TEXT_ONLY"
    NORMALIZED_TEXT_ONLY = "NORMALIZED_TEXT_ONLY"
    UNRESOLVED = "UNRESOLVED"
    UNLOCATED = "UNLOCATED"
    UNAVAILABLE = "UNAVAILABLE"
    AMBIGUOUS = "AMBIGUOUS"


class PlaceMentionValidationClass(str, Enum):
    GEOGRAPHIC_PLACE_CANDIDATE = "GEOGRAPHIC_PLACE_CANDIDATE"
    NON_PLACE_HIGH_CONFIDENCE = "NON_PLACE_HIGH_CONFIDENCE"
    UNKNOWN = "UNKNOWN"


class EventGroundingStatus(str, Enum):
    EVIDENCE_GROUNDED = "EVIDENCE_GROUNDED"
    INSUFFICIENT_GROUNDING = "INSUFFICIENT_GROUNDING"


class HistoricalEventTemporalGrounding(BaseModel):
    """Evidence-only historical time using signed historical years (no year zero).

    BCE years are negative (1 BCE == -1); CE years are positive (1 CE == 1).
    This deliberately is not astronomical year numbering.
    """
    raw_expression: str | None = None
    normalized_start: str | None = None
    normalized_end: str | None = None
    precision: TemporalPrecision = TemporalPrecision.UNKNOWN
    evidence_refs: list[str] = Field(default_factory=list)
    status: TemporalGroundingStatus = TemporalGroundingStatus.UNRESOLVED


class HistoricalEventPlaceMention(BaseModel):
    raw_text: str
    canonical_hint: str | None = None
    role: EventPlaceRole = EventPlaceRole.UNKNOWN
    evidence_refs: list[str] = Field(default_factory=list)
    resolution_status: EventPlaceResolutionStatus = EventPlaceResolutionStatus.TEXT_ONLY
    alias_provenance: str | None = None
    validation_class: PlaceMentionValidationClass = PlaceMentionValidationClass.UNKNOWN
    validation_reason: str | None = None


class HistoricalEventPlaceBinding(BaseModel):
    """A resolver-backed event/place relationship, not an event-site assertion by default."""

    mention: HistoricalEventPlaceMention
    place: HistoricalPlace | None = None
    role: EventPlaceRole = EventPlaceRole.UNKNOWN
    resolution_status: EventPlaceResolutionStatus = EventPlaceResolutionStatus.UNRESOLVED
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    resolver_provenance: str | None = None
    limitations: list[str] = Field(default_factory=list)


class HistoricalEvent(BaseModel):
    id: str
    name: str
    period: str | None = None
    summary: str
    places: list[HistoricalPlace] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    uncertainty_note: str | None = None
    event_type: HistoricalEventType = HistoricalEventType.UNKNOWN
    temporal_grounding: HistoricalEventTemporalGrounding = Field(default_factory=HistoricalEventTemporalGrounding)
    place_mentions: list[HistoricalEventPlaceMention] = Field(default_factory=list)
    place_bindings: list[HistoricalEventPlaceBinding] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    grounding_status: EventGroundingStatus = EventGroundingStatus.EVIDENCE_GROUNDED
    limitations: list[str] = Field(default_factory=list)
    identity_key: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    source_statements: list[str] = Field(default_factory=list)
    temporal_groundings: list[HistoricalEventTemporalGrounding] = Field(default_factory=list)


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
    alias_provenance: str | None = None


class HistoricalClaim(BaseModel):
    """Minimal, evidence-backed statement used by route orchestration only."""
    id: str
    claim_type: str
    text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    status: str = "accepted"
    source_place: str | None = None
    destination_place: str | None = None
    traversed_place: str | None = None
    movement_relation: str | None = None
    sequence_status: str = "unordered"
    textual_basis: str | None = None


class HistoricalRoutePoint(BaseModel):
    sequence: int = Field(ge=1)
    historical_place: HistoricalPlace
    event_summary: str
    date_or_period: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    coordinate_role: str = "exact_site"
    source_support: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)


class HistoricalRouteBranchRelation(BaseModel):
    """A proven ordering relation that cannot be placed in a unique linear traversal."""

    earlier: str
    later: str
    rule: str
    event_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    branch_kind: Literal["outgoing_branch", "incoming_hub", "isolated"] = "isolated"


class HistoricalRouteComponent(BaseModel):
    """One evidence-backed linear route fragment. Components are not ordered relative to each other."""

    component_id: str
    ordered_points: list[HistoricalRoutePoint] = Field(default_factory=list)
    relation_claim_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    status: str = "PROVEN_LINEAR"


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
    claims: list[HistoricalClaim] = Field(default_factory=list)
    route_components: list[HistoricalRouteComponent] = Field(default_factory=list)
    branch_relations: list[HistoricalRouteBranchRelation] = Field(default_factory=list)


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
    """Ontology-selected historical entity; it contains no coordinates or inferred places."""

    intent: str = "historical_route"
    campaign_id: str
    entity: str | None = None
    route_type: str | None = None


class AgentToolHistoryEntry(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result_summary: str
    duration_ms: int = Field(ge=0)
    outcome: str = "success"
    budget_source: str = "general"


class AgentProviderCallTiming(BaseModel):
    """Request-relative, content-free timing for one provider invocation."""
    call_number: int = Field(ge=1)
    agent_step: int = Field(ge=1)
    started_ms: int = Field(ge=0)
    finished_ms: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)
    status: Literal["SUCCESS", "TIMEOUT", "ERROR"]


class AgentState(BaseModel):
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    current_event: HistoricalEvent | None = None
    historical_events: list[HistoricalEvent] = Field(default_factory=list)
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
    historical_route_diagnostics: dict[str, Any] | None = None
    historical_event_diagnostics: dict[str, Any] | None = None
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
    provider_call_timing: list[AgentProviderCallTiming] = Field(default_factory=list)


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
