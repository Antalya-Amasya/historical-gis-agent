"""Conservative, deterministic matching of primary-evidence events to chronology records.

This module deliberately produces enrichment *candidates*.  It neither mutates
``HistoricalEvent.temporal_grounding`` nor joins modern-chronology provenance
to primary evidence.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import defaultdict
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from backend.app.models import HistoricalEvent, TemporalGroundingStatus


class ChronologyMatchStatus(str, Enum):
    MATCH = "MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    NO_MATCH = "NO_MATCH"


class ChronologyRecord(BaseModel):
    record_id: str
    canonical_event_label: str | None = None
    raw_event_text: str
    normalized_start: str | None = None
    normalized_end: str | None = None
    precision: str
    raw_temporal_expression: str | None = None
    source_id: str
    source_locator: dict[str, object] = Field(default_factory=dict)
    source_excerpt: str


class ChronologyEnrichmentCandidate(BaseModel):
    event_id: str
    chronology_record_id: str | None = None
    match_status: ChronologyMatchStatus
    match_confidence: float = Field(ge=0, le=1)
    matched_signals: list[str] = Field(default_factory=list)
    rejected_signals: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    normalized_start: str | None = None
    normalized_end: str | None = None
    precision: str | None = None
    chronology_source_id: str | None = None
    chronology_source_locator: dict[str, object] = Field(default_factory=dict)
    chronology_excerpt: str | None = None
    event_evidence_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: [
        "Candidate enrichment only; modern chronology provenance is separate from primary evidence.",
        "No HistoricalEvent temporal field was modified.",
    ])


_STOP = frozenset("""a an and are as at be been by for from in into is it of on or that the to was were with
    this these those he she they his her their its while after before during then than also not no only both
    roman rome republic history historian source chapter book volume footnotes army war battle military campaign
    event events action actions""".split())
_GENERIC = frozenset({"caesar", "cæsar", "rome", "roman", "carthage", "scipio", "hannibal", "pompey"})


def _tokens(value: str | None) -> set[str]:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = value.replace("æ", "ae").replace("œ", "oe")
    return {token for token in re.findall(r"[a-z][a-z']{2,}", value) if token not in _STOP}


def _ordered_tokens(value: str | None) -> list[str]:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = value.replace("æ", "ae").replace("œ", "oe")
    return [token for token in re.findall(r"[a-z][a-z']{2,}", value) if token not in _STOP]


def _phrases(value: str | None) -> set[tuple[str, str]]:
    tokens = _ordered_tokens(value)
    return set(zip(tokens, tokens[1:]))


def _proper_tokens(value: str | None) -> set[str]:
    """Capitalized names/places are a gate, never an inferred entity resolver."""
    raw = re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}", value or "")
    return {token for part in raw for token in _tokens(part)}


def _lead_proper_tokens(value: str | None) -> set[str]:
    """Named terms near statement start are a conservative event-subject proxy."""
    words = " ".join((value or "").split()[:16])
    return _proper_tokens(words)


def _event_text(event: HistoricalEvent) -> str:
    # Consolidation can retain several statement candidates.  Using them as one
    # lexical bag would let an unrelated co-consolidated statement manufacture
    # identity evidence, so matching stays anchored to the event's own summary.
    return event.summary


def _event_places(event: HistoricalEvent) -> set[str]:
    values = {(item.canonical_hint or item.raw_text).casefold() for item in event.place_mentions}
    values |= {binding.place.canonical_name.casefold() for binding in event.place_bindings if binding.place}
    return {value for value in values if value}


class ChronologyMatcher:
    """Small in-memory index with conservative identity gates, not semantic search."""

    def __init__(self, records: list[ChronologyRecord]) -> None:
        started = time.perf_counter()
        self.records = records
        self.by_token: dict[str, set[int]] = defaultdict(set)
        self.document_frequency: dict[str, int] = defaultdict(int)
        for index, record in enumerate(records):
            record_tokens = _tokens(f"{record.canonical_event_label or ''} {record.raw_event_text}")
            for token in record_tokens:
                self.by_token[token].add(index)
                self.document_frequency[token] += 1
        self.index_build_ms = (time.perf_counter() - started) * 1000

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChronologyMatcher":
        return cls([ChronologyRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    def _candidate(self, event: HistoricalEvent, record: ChronologyRecord, shared: set[str], phrases: set[tuple[str, str]]) -> ChronologyEnrichmentCandidate:
        event_places = _event_places(event)
        record_text = f"{record.canonical_event_label or ''} {record.raw_event_text}"
        place_overlap = sorted(place for place in event_places if place in record_text.casefold())
        distinctive = sorted(shared - _GENERIC)
        proper_shared = _proper_tokens(_event_text(event)) & _proper_tokens(record_text)
        subject_shared = proper_shared & _lead_proper_tokens(_event_text(event))
        signals = [f"distinctive_lexical_overlap:{token}" for token in distinctive]
        signals.extend(f"shared_named_term:{token}" for token in sorted(proper_shared))
        rare_phrases = sorted(" ".join(phrase) for phrase in phrases
                              if min(self.document_frequency[phrase[0]], self.document_frequency[phrase[1]]) <= 10
                              and set(phrase) & proper_shared)
        signals.extend(f"rare_phrase_overlap:{phrase}" for phrase in rare_phrases)
        if place_overlap:
            signals.extend(f"place_overlap:{place}" for place in place_overlap)
        primary = event.temporal_grounding
        temporal_conflict = bool(primary.normalized_start and record.normalized_start and
                                 (primary.normalized_start, primary.normalized_end) != (record.normalized_start, record.normalized_end))
        temporal_agreement = bool(primary.normalized_start and record.normalized_start and not temporal_conflict)
        if temporal_agreement:
            signals.append("primary_temporal_agreement")
        rejected: list[str] = []
        conflicts: list[str] = []
        if temporal_conflict:
            conflicts.append("primary_temporal_conflict")
            status, confidence = ChronologyMatchStatus.CONFLICT, 0.0
        elif rare_phrases and subject_shared and any(set(phrase.split()) & subject_shared for phrase in rare_phrases) and len(distinctive) >= 2 and (len(proper_shared) >= 2 or (proper_shared and (place_overlap or temporal_agreement))):
            status, confidence = ChronologyMatchStatus.MATCH, min(0.95, 0.60 + 0.08 * len(distinctive) + (0.10 if place_overlap else 0) + (0.10 if temporal_agreement else 0))
        else:
            if shared & _GENERIC:
                rejected.append("generic_entity_overlap_insufficient")
            rejected.append("missing_subject_identity_plus_rare_phrase_signals")
            status, confidence = ChronologyMatchStatus.AMBIGUOUS, 0.0
        return ChronologyEnrichmentCandidate(
            event_id=event.id, chronology_record_id=record.record_id, match_status=status, match_confidence=confidence,
            matched_signals=signals, rejected_signals=rejected, conflicts=conflicts,
            normalized_start=record.normalized_start, normalized_end=record.normalized_end, precision=record.precision,
            chronology_source_id=record.source_id, chronology_source_locator=record.source_locator,
            chronology_excerpt=record.source_excerpt, event_evidence_refs=list(event.evidence_refs),
        )

    def match(self, event: HistoricalEvent) -> list[ChronologyEnrichmentCandidate]:
        event_tokens = _tokens(_event_text(event))
        candidate_indexes: set[int] = set()
        for token in event_tokens:
            candidate_indexes |= self.by_token.get(token, set())
        if not candidate_indexes:
            return [ChronologyEnrichmentCandidate(event_id=event.id, match_status=ChronologyMatchStatus.NO_MATCH,
                match_confidence=0.0, rejected_signals=["no_deterministic_candidate"], event_evidence_refs=list(event.evidence_refs))]
        event_phrases = _phrases(_event_text(event))
        candidates = []
        for index in sorted(candidate_indexes, key=lambda item: self.records[item].record_id):
            record = self.records[index]
            record_text = f"{record.canonical_event_label or ''} {record.raw_event_text}"
            shared = event_tokens & _tokens(record_text)
            distinctive = shared - _GENERIC
            proper_shared = _proper_tokens(_event_text(event)) & _proper_tokens(record_text)
            if len(distinctive) >= 2 and proper_shared:
                candidates.append(self._candidate(event, record, shared, event_phrases & _phrases(record_text)))
        strong = [item for item in candidates if item.match_status is ChronologyMatchStatus.MATCH]
        years = {(item.normalized_start, item.normalized_end) for item in strong}
        if len(years) > 1:
            for item in strong:
                item.match_status = ChronologyMatchStatus.CONFLICT
                item.match_confidence = 0.0
                item.conflicts.append("cross_source_temporal_conflict")
        if strong:
            return strong
        return candidates[:20] or [ChronologyEnrichmentCandidate(event_id=event.id, match_status=ChronologyMatchStatus.NO_MATCH,
            match_confidence=0.0, rejected_signals=["no_deterministic_candidate"], event_evidence_refs=list(event.evidence_refs))]
