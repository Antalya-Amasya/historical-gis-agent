"""Diagnostic-only positive authority candidate evaluation over Pleiades v3 locations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from backend.app.geography.coordinate_authority import (
    MULTI_LOCATION_CONFLICT,
    NON_POINT_GEOMETRY,
    REPRESENTATIVE_ONLY,
    SOURCE_POINT_UNCERTAIN,
    UNKNOWN,
    CoordinateAuthorityDiagnostic,
    _collect_authority_facts,
    classify_coordinate_authority,
)

POSITIVE_POLICY_VERSION = "g6dh-positive-v1"
STRENGTH_NONE = "NONE"
STRENGTH_SUPPORTING = "SUPPORTING"
STRENGTH_STRONG = "STRONG_CANDIDATE"
_BLOCK_TYPES = frozenset({"representative", "central_point", "approximate", "associated_modern", "relocated_modern"})
_BLOCK_G6DD = frozenset({REPRESENTATIVE_ONLY, MULTI_LOCATION_CONFLICT, NON_POINT_GEOMETRY, UNKNOWN})
_MATCH_REPR = frozenset({"MATCHES_SINGLE_SOURCE_POINT", "MATCHES_MULTIPLE_AGREEING_POINTS"})
_STRONG = frozenset({"EXCAVATION_SITE_LOCATION", "SURVEYED_SITE_LOCATION", "MEASURED_MONUMENT_LOCATION"})
_POSITIVE = (
    ("EXCAVATION_SITE_LOCATION", re.compile(r"\bexcavation(?:s)?\b", re.I)),
    ("SURVEYED_SITE_LOCATION", re.compile(r"\b(?:archaeological )?survey(?:ed|s)?\b", re.I)),
    ("MEASURED_MONUMENT_LOCATION", re.compile(r"\b(?:measured|georeferenced)\b", re.I)),
    ("SITE_SPECIFIC_DESCRIPTION", re.compile(r"\b(?:on site|uncovered|revealed|visible remains|ancient remains|hellenistic remains|roman remains|archaeological site)\b", re.I)),
)
_GENERIC_REF = re.compile(r"\b(?:wikipedia|geohack|openstreetmap|barrington|batlas)\b", re.I)
_ARCH_REF = re.compile(r"(?:excavation|hadashot|arch[a-z\u00e9]*olog|survey|cag\b|seg\b)", re.I)


@dataclass(frozen=True)
class PositiveAuthorityCandidateDiagnostic:
    is_candidate: bool
    strength: str
    positive_reasons: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    source_location_ids: tuple[str, ...] = ()
    source_policy_family: str = "UNKNOWN"
    policy_version: str = POSITIVE_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _location_text(location: dict[str, Any]) -> str:
    parts = [location.get("title") or "", location.get("description") or ""]
    for reference in location.get("references") or []:
        if isinstance(reference, dict):
            parts.extend(
                str(reference.get(key) or "")
                for key in ("formattedCitation", "shortTitle", "citationDetail", "accessURI")
            )
    return " ".join(parts)


def _positive_reasons(location: dict[str, Any]) -> tuple[str, ...]:
    text = _location_text(location)
    reasons = [code for code, pattern in _POSITIVE if pattern.search(text)]
    for reference in location.get("references") or []:
        if not isinstance(reference, dict):
            continue
        ref_text = " ".join(
            str(reference.get(key) or "") for key in ("formattedCitation", "shortTitle", "citationDetail", "accessURI")
        )
        if "hadashot-esi.org.il" in ref_text or (_ARCH_REF.search(ref_text) and not _GENERIC_REF.search(ref_text)):
            reasons.append("ARCHAEOLOGICAL_SITE_REFERENCE")
            break
    return tuple(dict.fromkeys(reasons))


def evaluate_positive_authority_candidate(
    *,
    representative_longitude: float | None,
    representative_latitude: float | None,
    locations: list[dict[str, Any]],
    authority_diagnostic: CoordinateAuthorityDiagnostic | None = None,
) -> PositiveAuthorityCandidateDiagnostic:
    diagnostic = authority_diagnostic or classify_coordinate_authority(
        representative_longitude=representative_longitude,
        representative_latitude=representative_latitude,
        locations=locations,
    )
    facts = _collect_authority_facts(
        representative_longitude=representative_longitude,
        representative_latitude=representative_latitude,
        locations=locations,
    )
    repr_coord = (
        (representative_longitude, representative_latitude)
        if representative_longitude is not None and representative_latitude is not None
        else None
    )
    matched = tuple(
        location for _, coords, location in facts["point_records"] if repr_coord is not None and coords == repr_coord
    )
    blockers: list[str] = []
    if diagnostic.authority_class in _BLOCK_G6DD:
        blockers.append(f"G6DD_{diagnostic.authority_class}")
    if facts["any_representative_semantics"]:
        blockers.append("EXPLICIT_REPRESENTATIVE_SEMANTICS")
    for location in facts["locations"]:
        for value in location.get("location_types") or []:
            normalized = str(value).casefold().strip()
            if normalized in _BLOCK_TYPES:
                blockers.append(f"BLOCKING_LOCATION_TYPE_{normalized.upper()}")
    if not facts["point_records"]:
        blockers.append("NO_SOURCE_POINT")
    if diagnostic.repr_point_relation not in _MATCH_REPR:
        blockers.append("REPRPOINT_MISMATCH")
    if diagnostic.source_policy_family == "OSM" and not any(_positive_reasons(location) for location in matched):
        blockers.append("OSM_SOURCE_ONLY")
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        return PositiveAuthorityCandidateDiagnostic(
            is_candidate=False,
            strength=STRENGTH_NONE,
            blocking_reasons=tuple(blockers),
            source_location_ids=tuple(item.get("location_id") or "" for item in matched if item.get("location_id")),
            source_policy_family=diagnostic.source_policy_family,
        )

    positive = ["SOURCE_POINT_MATCHES_REPRPOINT", *(_ for location in matched for _ in _positive_reasons(location))]
    positive = list(dict.fromkeys(positive))
    strength = STRENGTH_STRONG if any(item in _STRONG for item in positive) else (
        STRENGTH_SUPPORTING if len(positive) > 1 else STRENGTH_NONE
    )
    return PositiveAuthorityCandidateDiagnostic(
        is_candidate=strength != STRENGTH_NONE,
        strength=strength,
        positive_reasons=tuple(positive) if strength != STRENGTH_NONE else (),
        source_location_ids=tuple(item.get("location_id") or "" for item in matched if item.get("location_id")),
        source_policy_family=diagnostic.source_policy_family,
    )
