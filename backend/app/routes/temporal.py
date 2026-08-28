"""Bounded, evidence-text-only historical temporal normalization.

Normalized values use signed *historical* years: BCE is negative, CE is
positive, and there is no year zero.  The parser intentionally rejects bare
numbers and relative chronology rather than supplying historical knowledge.
"""
from __future__ import annotations

import re

from backend.app.models import (
    HistoricalEventTemporalGrounding,
    TemporalGroundingStatus,
    TemporalPrecision,
)


class EvidenceTemporalResolver:
    """Parse only explicitly marked low-risk year expressions from a statement."""

    _RANGE_PATTERNS = (
        re.compile(r"\b(?P<first>\d{1,4})\s*(?:–|—|-)\s*(?P<second>\d{1,4})\s*(?P<era>BCE|BC)\b", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s*(?P<era>BCE|BC)\s+to\s+(?P<second>\d{1,4})\s*(?:BCE|BC)\b", re.I),
        re.compile(r"公元前\s*(?P<first>\d{1,4})\s*年?\s*(?:至|到|[-–—])\s*公元前?\s*(?P<second>\d{1,4})\s*年", re.I),
    )
    _YEAR_PATTERNS = (
        re.compile(r"\b(?:B\.\s*C\.\s*|BC\s*|BCE\s*)(?P<year>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*(?P<era>BCE|BC)\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*B\.\s*C\.", re.I),
        re.compile(r"\b(?:A\.\s*D\.\s*|AD\s*|CE\s*)(?P<year>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*(?P<era>CE|AD)\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*A\.\s*D\.", re.I),
        re.compile(r"公元前\s*(?P<year>\d{1,4})\s*年", re.I),
        re.compile(r"公元\s*(?P<year>\d{1,4})\s*年", re.I),
    )
    _APPROXIMATE = re.compile(r"\b(?:about|circa|c\.)\s+|约\s*", re.I)
    _CENTURY_PATTERNS = (
        re.compile(r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth)\s+century\s+(?:BCE|BC|CE|AD)\b", re.I),
        re.compile(r"公元前\s*\d+\s*世纪|公元\s*\d+\s*世纪", re.I),
    )

    @staticmethod
    def _year(value: str, era: str | None, raw: str) -> int:
        number = int(value)
        # The patterns require an era marker; zero is never a valid historical year.
        if number == 0:
            raise ValueError("year zero is not valid in the historical-year convention")
        bce = "公元前" in raw or (era or "").upper() in {"BC", "BCE"} or bool(re.search(r"\bB\.\s*C\.", raw, re.I))
        return -number if bce else number

    @staticmethod
    def _grounding(raw: str, evidence_ref: str, start: int | None = None, end: int | None = None,
                   precision: TemporalPrecision = TemporalPrecision.YEAR,
                   status: TemporalGroundingStatus = TemporalGroundingStatus.EVIDENCE_GROUNDED) -> HistoricalEventTemporalGrounding:
        return HistoricalEventTemporalGrounding(
            raw_expression=raw, normalized_start=str(start) if start is not None else None,
            normalized_end=str(end) if end is not None else None, precision=precision,
            evidence_refs=[evidence_ref], status=status,
        )

    def resolve(self, text: str, evidence_ref: str) -> tuple[list[HistoricalEventTemporalGrounding], list[str]]:
        """Return explicit temporal readings and diagnostic codes; never infer a relative date."""
        values: list[HistoricalEventTemporalGrounding] = []
        consumed: list[tuple[int, int]] = []
        for pattern in self._RANGE_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(0)
                try:
                    first = self._year(match.group("first"), match.groupdict().get("era"), raw)
                    second = self._year(match.group("second"), match.groupdict().get("era"), raw)
                except ValueError:
                    continue
                values.append(self._grounding(raw, evidence_ref, min(first, second), max(first, second), TemporalPrecision.YEAR_RANGE))
                consumed.append(match.span())
        for pattern in self._YEAR_PATTERNS:
            for match in pattern.finditer(text):
                if any(start <= match.start() and match.end() <= end for start, end in consumed):
                    continue
                raw = match.group(0)
                try:
                    year = self._year(match.group("year"), match.groupdict().get("era"), raw)
                except ValueError:
                    continue
                prefix = text[max(0, match.start() - 12):match.start()]
                approximate = bool(self._APPROXIMATE.search(prefix))
                expression = (prefix + raw).strip() if approximate else raw
                values.append(self._grounding(expression, evidence_ref, year, year,
                                               TemporalPrecision.APPROXIMATE if approximate else TemporalPrecision.YEAR))
        if not values:
            for pattern in self._CENTURY_PATTERNS:
                match = pattern.search(text)
                if match:
                    values.append(self._grounding(match.group(0), evidence_ref, precision=TemporalPrecision.UNKNOWN))
                    return values, ["TEMPORAL_CENTURY_UNRESOLVED"]
            return [], ["TEMPORAL_UNRESOLVED"]
        unique = {item.model_dump_json(): item for item in values}
        resolved = list(unique.values())
        normalized = {(item.normalized_start, item.normalized_end) for item in resolved}
        if len(normalized) > 1:
            return resolved, ["TEMPORAL_CONFLICT"]
        return resolved, ["TEMPORAL_RESOLVED"]

    def primary(self, readings: list[HistoricalEventTemporalGrounding], evidence_ref: str) -> HistoricalEventTemporalGrounding:
        if not readings:
            return HistoricalEventTemporalGrounding(evidence_refs=[evidence_ref])
        normalized = {(item.normalized_start, item.normalized_end) for item in readings}
        if len(normalized) > 1:
            return HistoricalEventTemporalGrounding(
                raw_expression="; ".join(item.raw_expression or "" for item in readings),
                evidence_refs=[evidence_ref], status=TemporalGroundingStatus.CONFLICT,
            )
        rank = {TemporalPrecision.YEAR: 0, TemporalPrecision.YEAR_RANGE: 1, TemporalPrecision.APPROXIMATE: 2, TemporalPrecision.UNKNOWN: 3}
        return min(readings, key=lambda item: rank[item.precision])
