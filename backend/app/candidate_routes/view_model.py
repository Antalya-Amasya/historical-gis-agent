"""Stable display DTOs for historical waypoints; no map SDK or enrichment involved."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .annotation import HistoricalEventType, HistoricalExternalReference
from .location import LocationConfidence

if TYPE_CHECKING:
    from .waypoint_graph import HistoricalWaypoint


class HistoricalWaypointViewModel(BaseModel):
    id: str
    name: str
    annotation_title: str | None = None
    event_type: HistoricalEventType | None = None
    period: str | None = None
    description: str | None = None
    location_confidence: LocationConfidence
    location_notes: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    source_book: str | None = None
    source_chapter: str | None = None
    historical_confidence: float | None = Field(default=None, ge=0, le=1)
    external_references: list[HistoricalExternalReference] = Field(default_factory=list)


def to_waypoint_view_model(waypoint: "HistoricalWaypoint") -> HistoricalWaypointViewModel:
    """Convert only already-supplied waypoint/annotation fields into a frontend-neutral DTO."""
    annotation = waypoint.annotation
    return HistoricalWaypointViewModel(
        id=waypoint.id,
        name=waypoint.canonical_name,
        annotation_title=annotation.title if annotation else None,
        event_type=annotation.event_type if annotation else None,
        period=annotation.period if annotation else waypoint.period,
        description=annotation.description if annotation else waypoint.description,
        location_confidence=waypoint.location_confidence,
        location_notes=waypoint.location_notes,
        evidence_refs=list(waypoint.evidence_refs),
        # Internal evidence IDs remain in evidence_refs for programmatic provenance,
        # but the user-facing source labels expose only reviewed human-readable locators.
        source_references=[
            *( [f"Book {waypoint.source_book}"] if waypoint.source_book else []),
            *( [f"Chapter {waypoint.source_chapter}"] if waypoint.source_chapter else []),
        ],
        source_book=waypoint.source_book,
        source_chapter=waypoint.source_chapter,
        historical_confidence=waypoint.historical_confidence,
        external_references=list(annotation.external_references) if annotation else [],
    )
