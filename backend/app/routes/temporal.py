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


class TemporalResolutionContext:
    """Evidence-local explicit YEAR anchor for bounded relative normalization."""

    def __init__(self) -> None:
        self._anchor_year: int | None = None
        self._relative_consumed: bool = False

    @property
    def anchor_year(self) -> int | None:
        return self._anchor_year

    @property
    def has_anchor(self) -> bool:
        return self._anchor_year is not None

    def reset(self) -> None:
        self._anchor_year = None
        self._relative_consumed = False

    def observe_explicit(
        self,
        readings: list[HistoricalEventTemporalGrounding],
        codes: list[str],
    ) -> None:
        """Record only unambiguous explicit YEAR anchors; never relative-derived years."""
        if "TEMPORAL_CONFLICT" in codes:
            return
        explicit_years = [
            int(item.normalized_start)
            for item in readings
            if item.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
            and item.precision is TemporalPrecision.YEAR
            and item.normalized_start is not None
        ]
        if not explicit_years:
            return
        unique = set(explicit_years)
        if len(unique) == 1:
            self._anchor_year = explicit_years[-1]
            self._relative_consumed = False

    def mark_relative_consumed(self) -> None:
        self._relative_consumed = True


class EvidenceTemporalResolver:
    """Parse only explicitly marked low-risk year expressions from a statement."""

    _RANGE_PATTERNS = (
        re.compile(r"\b(?P<first>\d{1,4})\s*(?:–|—|-)\s*(?P<second>\d{1,4})\s*(?P<era>BCE|BC)\b", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s*(?P<era>BCE|BC)\s+to\s+(?P<second>\d{1,4})\s*(?:BCE|BC)\b", re.I),
        re.compile(r"公元前\s*(?P<first>\d{1,4})\s*年?\s*(?:至|到|[-–—])\s*公元前?\s*(?P<second>\d{1,4})\s*年", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s*(?:–|—|-)\s*(?P<second>\d{1,4})\s*(?P<era>CE|AD)\b", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s+to\s+(?P<second>\d{1,4})\s*(?P<era>CE|AD)\b", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s*(?P<era>CE|AD)\s+to\s+(?P<second>\d{1,4})\s*(?:CE|AD)\b", re.I),
        re.compile(r"\b(?P<era>BCE|BC|B\.\s*C\.)\s*(?P<first>\d{1,4})\s*(?:–|—|-)\s*(?P<second>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<era>BCE|BC|B\.\s*C\.)\s*(?P<first>\d{1,4})\s+to\s+(?P<second>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<era>CE|AD|A\.\s*D\.)\s*(?P<first>\d{1,4})\s*(?:–|—|-)\s*(?P<second>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<first>\d{1,4})\s*--\s*(?P<second>\d{1,4})\s*(?P<era>BCE|BC|CE|AD)\b", re.I),
        re.compile(r"\bbetween\s+(?P<first>\d{1,4})\s+and\s+(?P<second>\d{1,4})\s*(?P<era>BCE|BC|CE|AD)\b", re.I),
        re.compile(r"\bbetween\s+(?P<era>BCE|BC|B\.\s*C\.)\s+(?P<first>\d{1,4})\s+and\s+(?P<second>\d{1,4})\b", re.I),
    )
    _RANGE_LIKE = re.compile(
        r"\b(?:"
        r"(?:(?:BCE|BC|CE|AD|B\.\s*C\.|A\.\s*D\.)\s*)?"
        r"\d{1,4}\s*"
        r"(?:–|—|--|-|\bto\b|\buntil\b|/)"
        r"\s*"
        r"\d{1,4}\s*"
        r"(?:BCE|BC|CE|AD|B\.\s*C\.|A\.\s*D\.)?"
        r"|between\s+(?:(?:BCE|BC|CE|AD|B\.\s*C\.|A\.\s*D\.)?\s*)?"
        r"\d{1,4}\s+and\s+\d{1,4}\s*(?:BCE|BC|CE|AD|B\.\s*C\.|A\.\s*D\.)?"
        r")\b",
        re.I,
    )
    _YEAR_PATTERNS = (
        re.compile(r"\b(?P<era>BCE|BC|B\.\s*C\.)\s*(?P<year>\d{1,4})\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*(?P<era>BCE|BC)\b", re.I),
        re.compile(r"\b(?P<year>\d{1,4})\s*B\.\s*C\.", re.I),
        re.compile(r"\b(?P<era>AD|CE|A\.\s*D\.)\s*(?P<year>\d{1,4})\b", re.I),
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
    _RELATIVE_YEAR_PATTERNS = (
        (re.compile(r"\bthe\s+following\s+year\b", re.I), 1),
        (re.compile(r"\b(?:the\s+)?next\s+year\b", re.I), 1),
        (re.compile(r"\b(?:in\s+)?(?:the\s+)?same\s+year\b", re.I), 0),
        (re.compile(r"\b(?:the\s+)?previous\s+year\b", re.I), -1),
        (re.compile(r"\b(?:the\s+)?preceding\s+year\b", re.I), -1),
    )
    _UNSUPPORTED_RELATIVE = re.compile(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:days?|weeks?|months?)\s+later\b"
        r"|\b(?:during|in)\s+the\s+winter\b|\bin\s+spring\b"
        r"|\bafter\s+this\b|\bthereupon\b|\bthen\b",
        re.I,
    )

    @staticmethod
    def _year(value: str, era: str | None, raw: str) -> int:
        number = int(value)
        # The patterns require an era marker; zero is never a valid historical year.
        if number == 0:
            raise ValueError("year zero is not valid in the historical-year convention")
        era_token = re.sub(r"\s+", "", (era or "").upper())
        bce = (
            "公元前" in raw
            or era_token in {"BC", "BCE"}
            or bool(re.search(r"B\.\s*C\.", era or "", re.I))
            or bool(re.search(r"\bB\.\s*C\.", raw, re.I))
        )
        return -number if bce else number

    @staticmethod
    def _shift_year(year: int, delta: int) -> int:
        result = year + delta
        if result == 0:
            return 1 if delta > 0 else -1
        return result

    @staticmethod
    def _grounding(raw: str, evidence_ref: str, start: int | None = None, end: int | None = None,
                   precision: TemporalPrecision = TemporalPrecision.YEAR,
                   status: TemporalGroundingStatus = TemporalGroundingStatus.EVIDENCE_GROUNDED) -> HistoricalEventTemporalGrounding:
        return HistoricalEventTemporalGrounding(
            raw_expression=raw, normalized_start=str(start) if start is not None else None,
            normalized_end=str(end) if end is not None else None, precision=precision,
            evidence_refs=[evidence_ref], status=status,
        )

    @staticmethod
    def _inside_unresolved_range(text: str, start: int, end: int, consumed: list[tuple[int, int]]) -> bool:
        for match in EvidenceTemporalResolver._RANGE_LIKE.finditer(text):
            if start < match.start() or end > match.end():
                continue
            if any(range_start <= match.start() and match.end() <= range_end for range_start, range_end in consumed):
                continue
            return True
        return False

    def _resolve_explicit(self, text: str, evidence_ref: str) -> tuple[list[HistoricalEventTemporalGrounding], list[str]]:
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
                if self._inside_unresolved_range(text, match.start(), match.end(), consumed):
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
            return [], []
        unique = {item.model_dump_json(): item for item in values}
        resolved = list(unique.values())
        normalized = {(item.normalized_start, item.normalized_end) for item in resolved}
        if len(normalized) > 1:
            return resolved, ["TEMPORAL_CONFLICT"]
        return resolved, ["TEMPORAL_RESOLVED"]

    def _resolve_relative(
        self,
        text: str,
        evidence_ref: str,
        context: TemporalResolutionContext,
    ) -> tuple[list[HistoricalEventTemporalGrounding], list[str]]:
        if not context.has_anchor or context._relative_consumed or self._UNSUPPORTED_RELATIVE.search(text):
            return [], ["TEMPORAL_UNRESOLVED"]
        for pattern, delta in self._RELATIVE_YEAR_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            year = context.anchor_year if delta == 0 else self._shift_year(context.anchor_year, delta)
            raw = match.group(0)
            context.mark_relative_consumed()
            return [self._grounding(raw, evidence_ref, year, year)], ["TEMPORAL_RESOLVED"]
        return [], ["TEMPORAL_UNRESOLVED"]

    def resolve(
        self,
        text: str,
        evidence_ref: str,
        *,
        context: TemporalResolutionContext | None = None,
    ) -> tuple[list[HistoricalEventTemporalGrounding], list[str]]:
        """Return explicit or evidence-anchored relative temporal readings and diagnostic codes."""
        readings, codes = self._resolve_explicit(text, evidence_ref)
        if readings:
            if context is not None:
                context.observe_explicit(readings, codes)
            return readings, codes
        if context is not None and context.has_anchor:
            relative_readings, relative_codes = self._resolve_relative(text, evidence_ref, context)
            if relative_readings:
                return relative_readings, relative_codes
            return [], relative_codes
        if not codes:
            for pattern in self._CENTURY_PATTERNS:
                if pattern.search(text):
                    return [], ["TEMPORAL_CENTURY_UNRESOLVED"]
            return [], ["TEMPORAL_UNRESOLVED"]
        return [], codes

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
