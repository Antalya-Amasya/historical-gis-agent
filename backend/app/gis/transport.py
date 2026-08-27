"""Future multimodal transition contracts without graph-search integration."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .ports import HistoricalPort, PortSurfaceDiagnostic


class TransportMode(str, Enum):
    LAND = "LAND"
    SEA = "SEA"


@dataclass(frozen=True)
class SearchState:
    """A future graph state; mode is part of identity, even at one location."""

    location_id: str
    latitude: float
    longitude: float
    mode: TransportMode

    def __post_init__(self) -> None:
        if not self.location_id:
            raise ValueError("location_id must be non-empty")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("search-state coordinates must be valid WGS84 latitude/longitude")


class TransitionEvidenceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    UNKNOWN = "UNKNOWN"


class TransitionEligibilityStatus(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PortTransitionRequest:
    """An explicit claimed mode switch at one supplied historical port."""

    port: HistoricalPort | None
    from_mode: TransportMode
    to_mode: TransportMode
    transition_evidence_status: TransitionEvidenceStatus = TransitionEvidenceStatus.UNKNOWN
    supporting_evidence_ids: tuple[str, ...] = ()
    event_id: str | None = None
    surface_diagnostic: PortSurfaceDiagnostic | None = None


@dataclass(frozen=True)
class TransitionEligibility:
    status: TransitionEligibilityStatus
    reason: str
    port_id: str | None
    from_mode: TransportMode
    to_mode: TransportMode
    transition_evidence_status: TransitionEvidenceStatus
    supporting_evidence_ids: tuple[str, ...]
    port_evidence_ids: tuple[str, ...]
    event_id: str | None
    surface_diagnostic: PortSurfaceDiagnostic | None


def evaluate_port_transition(request: PortTransitionRequest) -> TransitionEligibility:
    """Evaluate explicit evidence permission; never infer it from geography."""
    port = request.port
    common = dict(
        port_id=port.port_id if port else None,
        from_mode=request.from_mode,
        to_mode=request.to_mode,
        transition_evidence_status=request.transition_evidence_status,
        supporting_evidence_ids=tuple(request.supporting_evidence_ids),
        port_evidence_ids=tuple(port.supporting_evidence_ids) if port else (),
        event_id=request.event_id,
        surface_diagnostic=request.surface_diagnostic,
    )
    if port is None:
        return TransitionEligibility(TransitionEligibilityStatus.DENIED, "an explicit HistoricalPort is required", **common)
    if request.from_mode is request.to_mode:
        return TransitionEligibility(TransitionEligibilityStatus.DENIED, "a transition requires distinct modes", **common)
    if not port.supports_land_access or not port.supports_sea_access:
        return TransitionEligibility(TransitionEligibilityStatus.DENIED, "port record does not explicitly support both modes", **common)
    if request.transition_evidence_status in {TransitionEvidenceStatus.CONFIRMED, TransitionEvidenceStatus.SUPPORTED}:
        return TransitionEligibility(TransitionEligibilityStatus.ALLOWED, "explicit route-transition evidence permits this port switch", **common)
    return TransitionEligibility(
        TransitionEligibilityStatus.UNKNOWN,
        "historical port existence does not establish route-transition permission",
        **common,
    )


class TransitionCostModel(Protocol):
    """Configuration-level cost interface; it encodes no historical facts."""

    def cost_for(self, transition: TransitionEligibility) -> float: ...


@dataclass(frozen=True)
class FixedTransitionCostModel:
    """Deterministic non-negative placeholder for a future explicit transition edge."""

    fixed_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.fixed_cost < 0:
            raise ValueError("fixed transition cost must be non-negative")

    def cost_for(self, transition: TransitionEligibility) -> float:
        if transition.status is not TransitionEligibilityStatus.ALLOWED:
            raise ValueError("transition cost is unavailable unless the transition is allowed")
        return self.fixed_cost
