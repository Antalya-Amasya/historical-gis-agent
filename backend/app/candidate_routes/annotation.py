"""Input-supplied historical event annotations; no external enrichment occurs here."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class HistoricalExternalReferenceType(str, Enum):
    WIKIPEDIA = "WIKIPEDIA"
    GOOGLE_MAPS = "GOOGLE_MAPS"
    ENCYCLOPEDIA = "ENCYCLOPEDIA"
    PAPER = "PAPER"
    OTHER = "OTHER"


class HistoricalExternalReference(BaseModel):
    id: str
    reference_type: HistoricalExternalReferenceType
    title: str
    url: str
    language: str | None = None
    description: str | None = None


class HistoricalEventType(str, Enum):
    BATTLE = "BATTLE"
    SIEGE = "SIEGE"
    CAMPAIGN = "CAMPAIGN"
    CROSSING = "CROSSING"
    CITY = "CITY"
    OTHER = "OTHER"


class HistoricalAnnotation(BaseModel):
    id: str
    title: str
    event_type: HistoricalEventType
    period: str | None = None
    description: str
    source_refs: list[str] = Field(min_length=1)
    external_links: list[str] = Field(default_factory=list)
    external_references: list[HistoricalExternalReference] = Field(default_factory=list)
