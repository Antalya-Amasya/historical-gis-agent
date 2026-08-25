"""Manually reviewed historical-event registry, independent of retrieval and route planning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .ingestion.models import TextChunk


class HistoricalEventRecord(BaseModel):
    event_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    period: str = Field(min_length=1)
    description: str = Field(min_length=1)
    involved_places: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    source_book: str = Field(min_length=1)
    source_chapter: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class CorpusEvidenceReference:
    corpus_id: str
    book: str
    chapter: str


class CorpusEvidenceCatalog:
    """Explicit chunk inventory supplied by ingestion/review; it does not query a RAG store."""

    def __init__(self, references: dict[str, CorpusEvidenceReference]):
        self.references = dict(references)

    @classmethod
    def from_chunks(cls, chunks: list[TextChunk]) -> "CorpusEvidenceCatalog":
        return cls({chunk.id: CorpusEvidenceReference(str(chunk.metadata["corpus_id"]), str(chunk.metadata["book"]), str(chunk.metadata["chapter"])) for chunk in chunks})


class EventRegistryValidationError(ValueError):
    pass


class HistoricalEventRegistry:
    def __init__(self, records: list[HistoricalEventRecord], evidence_catalog: CorpusEvidenceCatalog):
        self._records = {record.event_id: record for record in records}
        if len(self._records) != len(records):
            raise EventRegistryValidationError("event_id values must be unique")
        for record in records:
            self._validate(record, evidence_catalog)

    @classmethod
    def from_json(cls, path: Path, evidence_catalog: CorpusEvidenceCatalog) -> "HistoricalEventRegistry":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([HistoricalEventRecord.model_validate(item) for item in data["events"]], evidence_catalog)

    def get(self, event_id: str) -> HistoricalEventRecord:
        try:
            return self._records[event_id]
        except KeyError as exc:
            raise KeyError(f"event not registered: {event_id}") from exc

    def records(self) -> list[HistoricalEventRecord]:
        return list(self._records.values())

    @staticmethod
    def _validate(record: HistoricalEventRecord, evidence_catalog: CorpusEvidenceCatalog) -> None:
        matched = []
        for evidence_ref in record.evidence_refs:
            evidence = evidence_catalog.references.get(evidence_ref)
            if evidence is None:
                raise EventRegistryValidationError(f"unknown evidence_ref: {evidence_ref}")
            if evidence.corpus_id != record.corpus_id:
                raise EventRegistryValidationError(f"evidence_ref belongs to a different corpus: {evidence_ref}")
            matched.append(evidence)
        if not any(item.book == record.source_book and item.chapter == record.source_chapter for item in matched):
            raise EventRegistryValidationError("source_book/source_chapter must match a declared evidence_ref")
