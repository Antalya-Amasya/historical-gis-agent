"""Explicit-input bridge from retrieved evidence to a provenance-preserving event chain."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedEvidenceChunk(BaseModel):
    """A retrieval result plus caller-supplied event labels; text is never interpreted here."""

    corpus_id: str = Field(min_length=1)
    book: str = Field(min_length=1)
    chapter: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    event_id: str = Field(min_length=1)
    period: str | None = None
    description: str = Field(min_length=1)
    involved_places: list[str] = Field(default_factory=list)
    sequence: int = Field(ge=1)


class EventStepProvenance(BaseModel):
    source_corpus: str
    book: str
    chapter: str
    evidence_refs: list[str] = Field(min_length=1)


class HistoricalEventStep(BaseModel):
    id: str
    sequence: int = Field(ge=1)
    description: str
    involved_places: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    provenance: EventStepProvenance


class HistoricalEventChain(BaseModel):
    event_id: str
    period: str | None = None
    description: str
    involved_places: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(min_length=1)
    steps: list[HistoricalEventStep] = Field(min_length=1)


class EventExtractionError(ValueError):
    pass


class HistoricalEventExtractor:
    """Binds only explicit retrieval fields; it performs no historical interpretation or ordering."""

    def extract(self, chunks: list[RetrievedEvidenceChunk]) -> HistoricalEventChain:
        if not chunks:
            raise EventExtractionError("at least one retrieved evidence chunk is required")
        event_ids = {chunk.event_id for chunk in chunks}
        if len(event_ids) != 1:
            raise EventExtractionError("all chunks must declare the same explicit event_id")
        periods = {chunk.period for chunk in chunks}
        if len(periods) != 1:
            raise EventExtractionError("all chunks must declare the same explicit period")
        descriptions = {chunk.description for chunk in chunks}
        if len(descriptions) != 1:
            raise EventExtractionError("all chunks must declare the same explicit description")
        sequences = [chunk.sequence for chunk in chunks]
        if len(sequences) != len(set(sequences)):
            raise EventExtractionError("explicit evidence sequence values must be unique")
        steps = [
            HistoricalEventStep(
                id=f"{chunk.event_id}:{chunk.sequence}", sequence=chunk.sequence,
                description=chunk.description, involved_places=list(chunk.involved_places),
                evidence_refs=list(chunk.evidence_refs),
                provenance=EventStepProvenance(source_corpus=chunk.corpus_id, book=chunk.book, chapter=chunk.chapter, evidence_refs=list(chunk.evidence_refs)),
            )
            for chunk in sorted(chunks, key=lambda item: item.sequence)
        ]
        return HistoricalEventChain(
            event_id=chunks[0].event_id, period=chunks[0].period, description=chunks[0].description,
            involved_places=list(dict.fromkeys(place for chunk in chunks for place in chunk.involved_places)),
            evidence_refs=list(dict.fromkeys(reference for chunk in chunks for reference in chunk.evidence_refs)),
            steps=steps,
        )
