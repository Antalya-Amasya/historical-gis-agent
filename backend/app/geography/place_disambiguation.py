"""Context compatibility filters for historical place disambiguation.

CONTEXT MAY EXCLUDE — never prove a candidate from context alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

_ANCIENT_END = 640
_ROMAN_REPUBLIC_END = -27
_MODERN_START = 1500

_BASINS: tuple[tuple[str, float, float, float, float], ...] = (
    ("greek_aegean", 34.0, 19.0, 41.8, 30.0),
    ("italian_peninsula", 36.5, 6.5, 46.8, 19.0),
    ("western_anatolia", 35.5, 25.5, 42.5, 32.0),
    ("hispania", 35.5, -10.0, 44.0, 4.5),
    ("north_africa", 24.0, -12.0, 38.0, 25.0),
    ("near_east", 24.0, 30.0, 42.0, 55.0),
    ("bactria_india", 23.0, 60.0, 42.0, 85.0),
)

_ENDPOINT_INCOMPATIBLE_TYPES = frozenset({"plaza", "label", "labeled feature"})
_TRAVERSAL_INCOMPATIBLE_TYPES = frozenset({"province", "province-2", "diocese-roman", "plaza", "label", "labeled feature"})
_REGION_TYPES = frozenset({"region", "province", "province-2", "people", "ethnic-region", "peninsula"})


@dataclass(frozen=True)
class ResolvedCoMention:
    name: str
    latitude: float
    longitude: float
    spatial_semantics: str | None = None


@dataclass(frozen=True)
class PlaceResolutionContext:
    period: str | None = None
    source_statement: str | None = None
    place_role: str | None = None
    co_mentions: tuple[str, ...] = ()
    resolved_co_mentions: tuple[ResolvedCoMention, ...] = ()


@dataclass
class CandidateFilterResult:
    passed: bool
    failed_filters: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _basins_for_point(latitude: float, longitude: float) -> frozenset[str]:
    matched: set[str] = set()
    for name, min_lat, min_lon, max_lat, max_lon in _BASINS:
        if min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon:
            matched.add(name)
    return frozenset(matched)


def _parse_event_year_window(period: str | None) -> tuple[int, int] | None:
    if not period:
        return None
    years: list[int] = []
    for match in re.finditer(r"(-?\d{1,4})\s*(BCE|BC|CE|AD)?", period, flags=re.IGNORECASE):
        value = int(match.group(1))
        era = (match.group(2) or "").upper()
        if era in {"BCE", "BC"} or (era == "" and value < 0):
            years.append(-abs(value))
        elif era in {"CE", "AD"}:
            years.append(abs(value))
        elif value < 0:
            years.append(value)
        else:
            years.append(value)
    if not years:
        if "roman republic" in period.casefold():
            return (-509, _ROMAN_REPUBLIC_END)
        return None
    start, end = min(years), max(years)
    return (start, end) if start <= end else (end, start)


def _attestation_years(detail: dict[str, Any]) -> list[tuple[int | None, int | None]]:
    spans: list[tuple[int | None, int | None]] = []
    start, end = detail.get("name_start"), detail.get("name_end")
    if start is not None or end is not None:
        spans.append((start, end))
    for att in detail.get("attestations") or []:
        period = (att.get("time_period") or "").casefold()
        if period == "modern":
            spans.append((_MODERN_START, 2100))
        elif period in {"archaic", "classical", "hellenistic-republican", "roman", "late-antique"}:
            spans.append((_ANCIENT_END * -1, _ANCIENT_END))
    return spans


def _period_incompatible(candidate: dict[str, Any], event_window: tuple[int, int] | None) -> bool:
    if event_window is None:
        return False
    event_start, event_end = event_window
    if event_end > _MODERN_START:
        return False
    spans: list[tuple[int | None, int | None]] = []
    for detail in candidate.get("matched_name_details") or []:
        spans.extend(_attestation_years(detail))
    if not spans:
        return False
    for start, end in spans:
        if start is None and end is None:
            continue
        cand_start = start if start is not None else end
        cand_end = end if end is not None else start
        if cand_start is None or cand_end is None:
            continue
        if cand_end < event_start or cand_start > event_end:
            continue
        return False
    return True


def _place_types(candidate: dict[str, Any], place: HistoricalPlace | None) -> frozenset[str]:
    types = {value.casefold() for value in candidate.get("place_types") or []}
    if place and place.authority_metadata:
        types.update(value.casefold() for value in place.authority_metadata.get("place_types") or [])
    return frozenset(types)


def _type_incompatible(candidate: dict[str, Any], place: HistoricalPlace | None, context: PlaceResolutionContext) -> bool:
    types = _place_types(candidate, place)
    if not types:
        return False
    role = (context.place_role or "").upper()
    if role == "TRAVERSAL" and types & _TRAVERSAL_INCOMPATIBLE_TYPES:
        return True
    if role in {"ORIGIN", "DESTINATION"} and types <= _ENDPOINT_INCOMPATIBLE_TYPES:
        return True
    if role in {"ORIGIN", "DESTINATION", "EVENT_SITE"} and types == {"unlocated"}:
        return True
    if role == "RELATED_PLACE" and types <= _ENDPOINT_INCOMPATIBLE_TYPES:
        if context.resolved_co_mentions:
            return True
    return False


def _geo_incompatible(
    candidate: dict[str, Any],
    place: HistoricalPlace | None,
    context: PlaceResolutionContext,
) -> bool:
    if place is None or place.latitude is None or place.longitude is None:
        return False
    if not context.resolved_co_mentions:
        return False
    candidate_basins = _basins_for_point(place.latitude, place.longitude)
    if not candidate_basins:
        return False
    for co_mention in context.resolved_co_mentions:
        co_basins = _basins_for_point(co_mention.latitude, co_mention.longitude)
        if not co_basins:
            continue
        if candidate_basins.isdisjoint(co_basins):
            return True
    return False


def _requires_located_anchor(context: PlaceResolutionContext) -> bool:
    role = (context.place_role or "").upper()
    return role in {"ORIGIN", "DESTINATION", "EVENT_SITE"}


def _located_incompatible(candidate: dict[str, Any], context: PlaceResolutionContext) -> bool:
    if not _requires_located_anchor(context):
        return False
    if candidate.get("coordinate_available"):
        return False
    types = _place_types(candidate, None)
    return "unlocated" in types or not candidate.get("coordinate_available", True)


def evaluate_candidate(
    candidate: dict[str, Any],
    place: HistoricalPlace | None,
    context: PlaceResolutionContext,
    *,
    event_window: tuple[int, int] | None,
) -> CandidateFilterResult:
    failed: list[str] = []
    if _period_incompatible(candidate, event_window):
        failed.append("period_incompatible")
    if _type_incompatible(candidate, place, context):
        failed.append("type_incompatible")
    if _geo_incompatible(candidate, place, context):
        failed.append("geo_basin_incompatible")
    if _located_incompatible(candidate, context):
        failed.append("unlocated_anchor_ineligible")
    return CandidateFilterResult(passed=not failed, failed_filters=tuple(failed))


def filter_resolution_candidates(
    *,
    status: str,
    places: tuple[HistoricalPlace, ...],
    candidates: tuple[dict[str, Any], ...],
    context: PlaceResolutionContext | None,
) -> tuple[str, tuple[HistoricalPlace, ...], tuple[dict[str, Any], ...], list[dict[str, Any]]]:
    if context is None:
        return status, places, candidates, []
    event_window = _parse_event_year_window(context.period)
    diagnostics: list[dict[str, Any]] = []

    if status == "CURATED":
        surviving_places: list[HistoricalPlace] = []
        for place in places:
            candidate = {
                "pleiades_id": place.source_id,
                "canonical_name": place.canonical_name,
                "place_types": [place.spatial_semantics.value if place.spatial_semantics else "unknown"],
                "coordinate_available": place.latitude is not None and place.longitude is not None,
                "matched_name_details": (place.authority_metadata or {}).get("matched_name_details", []),
            }
            result = evaluate_candidate(candidate, place, context, event_window=event_window)
            diagnostics.append({
                "pleiades_id": place.source_id,
                "canonical_name": place.canonical_name,
                "passed": result.passed,
                "failed_filters": list(result.failed_filters),
            })
            if result.passed:
                surviving_places.append(place)
        surviving_candidates = [
            {
                "pleiades_id": place.source_id,
                "canonical_name": place.canonical_name,
                "place_types": [place.spatial_semantics.value if place.spatial_semantics else "unknown"],
                "coordinate_available": place.latitude is not None,
            }
            for place in surviving_places
        ]
    else:
        place_by_id = {str(place.source_id): place for place in places if place.source_id}
        surviving_places = []
        surviving_candidates = []
        for candidate in candidates:
            pleiades_id = str(candidate.get("pleiades_id"))
            place = place_by_id.get(pleiades_id)
            result = evaluate_candidate(candidate, place, context, event_window=event_window)
            diagnostics.append({
                "pleiades_id": pleiades_id,
                "canonical_name": candidate.get("canonical_name"),
                "passed": result.passed,
                "failed_filters": list(result.failed_filters),
            })
            if result.passed:
                surviving_candidates.append(candidate)
                if place is not None:
                    surviving_places.append(place)

    original_count = len(places) if status == "CURATED" else len(candidates)
    count = len(surviving_candidates)
    excluded_filters = {
        filter_name
        for item in diagnostics
        if not item.get("passed")
        for filter_name in item.get("failed_filters", [])
    }
    survivor_by_contextual_exclusion = (
        original_count >= 2
        and count == 1
        and "geo_basin_incompatible" in excluded_filters
    )
    if count == 0:
        return ("AMBIGUOUS" if original_count else status), (), (), diagnostics
    if count == 1 and len(surviving_places) == 1 and not survivor_by_contextual_exclusion:
        return "UNIQUE", tuple(surviving_places), tuple(surviving_candidates), diagnostics
    if count == 1 and not surviving_places:
        return "UNLOCATED", (), tuple(surviving_candidates), diagnostics
    return "AMBIGUOUS", tuple(surviving_places), tuple(surviving_candidates), diagnostics
