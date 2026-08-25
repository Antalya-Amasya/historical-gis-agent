"""Offline knowledge-panel DTOs built from supplied waypoint display data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from pydantic import BaseModel, Field

from .annotation import HistoricalEventType, HistoricalExternalReference
from .location import LocationConfidence
from .view_model import HistoricalWaypointViewModel


class SummaryResult(BaseModel):
    text: str | None = None
    used_evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class EvidenceBackedSummary(BaseModel):
    summary_text: str
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class SummaryEvidenceError(ValueError):
    pass


class HistoricalSummaryProvider(ABC):
    @abstractmethod
    def get_summary(self, waypoint_id: str, evidence_refs: list[str]) -> SummaryResult:
        raise NotImplementedError


class EvidenceBackedSummaryProvider(HistoricalSummaryProvider):
    """Fixed local summaries for tests/demos; it never infers, fetches, or generates text."""

    def __init__(self, summaries: dict[str, EvidenceBackedSummary]):
        self.summaries = dict(summaries)

    def get_summary(self, waypoint_id: str, evidence_refs: list[str]) -> SummaryResult:
        summary = self.summaries.get(waypoint_id)
        if summary is None:
            return SummaryResult(text=None, used_evidence_refs=[], confidence=0.0)
        return SummaryResult(
            text=summary.summary_text,
            used_evidence_refs=list(summary.evidence_refs),
            confidence=summary.confidence,
        )


class HistoricalKnowledgePanel(BaseModel):
    waypoint_id: str
    title: str
    period: str | None = None
    event_type: HistoricalEventType | None = None
    summary: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    external_references: list[HistoricalExternalReference] = Field(default_factory=list)
    confidence: LocationConfidence
    evidence_backed_summary: EvidenceBackedSummary | None = None


class KnowledgePanelBuilder:
    """Creates display data only; it neither enriches, infers, nor fetches historical content."""

    def build(
        self,
        waypoint: HistoricalWaypointViewModel,
        *,
        summary: str | None = None,
        summary_provider: Callable[[HistoricalWaypointViewModel], str | None] | None = None,
        evidence_summary_provider: HistoricalSummaryProvider | None = None,
    ) -> HistoricalKnowledgePanel:
        if sum(value is not None for value in (summary, summary_provider, evidence_summary_provider)) > 1:
            raise ValueError("provide only one summary source")
        evidence_backed_summary = None
        if evidence_summary_provider is not None:
            if not waypoint.evidence_refs:
                raise SummaryEvidenceError("cannot request an evidence-backed summary without waypoint evidence")
            result = evidence_summary_provider.get_summary(waypoint.id, list(waypoint.evidence_refs))
            if not set(result.used_evidence_refs).issubset(waypoint.evidence_refs):
                raise SummaryEvidenceError("summary provider used evidence not attached to waypoint")
            supplied_summary = result.text
            if result.text is not None:
                if not result.used_evidence_refs:
                    raise SummaryEvidenceError("non-empty summary must declare used evidence references")
                evidence_backed_summary = EvidenceBackedSummary(
                    summary_text=result.text,
                    evidence_refs=list(result.used_evidence_refs),
                    confidence=result.confidence,
                )
        else:
            supplied_summary = summary_provider(waypoint) if summary_provider else summary
        return HistoricalKnowledgePanel(
            waypoint_id=waypoint.id,
            title=waypoint.annotation_title or waypoint.name,
            period=waypoint.period,
            event_type=waypoint.event_type,
            summary=supplied_summary,
            evidence_refs=list(waypoint.evidence_refs),
            source_references=list(waypoint.source_references),
            external_references=list(waypoint.external_references),
            confidence=waypoint.location_confidence,
            evidence_backed_summary=evidence_backed_summary,
        )
