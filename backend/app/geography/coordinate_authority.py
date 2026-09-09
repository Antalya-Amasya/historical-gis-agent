"""Diagnostic-only coordinate authority classification for Pleiades v3 locations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

POLICY_VERSION = "g6dd-v1"

AUTHORITATIVE_SITE_POINT = "AUTHORITATIVE_SITE_POINT"
SOURCE_POINT_UNCERTAIN = "SOURCE_POINT_UNCERTAIN"
REPRESENTATIVE_ONLY = "REPRESENTATIVE_ONLY"
MULTI_LOCATION_CONFLICT = "MULTI_LOCATION_CONFLICT"
NON_POINT_GEOMETRY = "NON_POINT_GEOMETRY"
UNKNOWN = "UNKNOWN"

_REPRESENTATIVE_LOCATION_TYPES = frozenset({"representative", "central_point", "approximate"})
_TEXT_NEGATIVE_PATTERNS = (
    re.compile(r"\brepresentative\b", re.I),
    re.compile(r"\bcentral[_ -]?point\b", re.I),
    re.compile(r"\bapproximate\b", re.I),
    re.compile(r"\bmap[- ]scale\b", re.I),
    re.compile(r"\bdigitized representative\b", re.I),
    re.compile(r"\blabel point\b", re.I),
    re.compile(r"\bderived center\b", re.I),
)
_FAMILY_PRIORITY = ("OSM", "DARE", "DARMC", "BARRINGTON", "PLEIADES", "OTHER")
_EMPTY_GEOMETRY_SUMMARY = {"point_count": 0, "non_point_count": 0, "geometry_types": []}
_EMPTY_MULTIPLICITY_SUMMARY = {"unique_point_count": 0, "conflicting": False}


@dataclass(frozen=True)
class CoordinateAuthorityDiagnostic:
    authority_class: str
    reason_codes: tuple[str, ...] = ()
    source_location_ids: tuple[str, ...] = ()
    repr_point_relation: str = "UNKNOWN"
    geometry_summary: dict[str, Any] = field(default_factory=dict)
    multiplicity_summary: dict[str, Any] = field(default_factory=dict)
    explicit_negative_signals: tuple[str, ...] = ()
    source_policy_family: str = "UNKNOWN"
    accuracy_uri: str | None = None
    temporal_applicability: str = "UNKNOWN"
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _point_coordinates(geometry: dict[str, Any] | None) -> tuple[float, float] | None:
    if not geometry or geometry.get("type") != "Point":
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    lon, lat = coordinates[0], coordinates[1]
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        return None
    return float(lon), float(lat)


def _source_policy_family(location: dict[str, Any]) -> str:
    haystack = " ".join(
        part
        for part in (
            location.get("provenance") or "",
            location.get("title") or "",
            location.get("description") or "",
            " ".join(str(item.get("shortTitle", "")) for item in (location.get("references") or []) if isinstance(item, dict)),
        )
        if part
    ).casefold()
    accuracy = (location.get("accuracy") or "").casefold()
    if "openstreetmap" in haystack or "osm" in haystack or "openstreetmap" in accuracy:
        return "OSM"
    if "digital atlas of the roman empire" in haystack or "dare" in haystack or "/dare-" in accuracy:
        return "DARE"
    if "darmc" in haystack or "/darmc-" in accuracy:
        return "DARMC"
    if "barrington" in haystack or "batlas" in haystack:
        return "BARRINGTON"
    if (location.get("provenance") or "").casefold() == "pleiades":
        return "PLEIADES"
    if haystack:
        return "OTHER"
    return "UNKNOWN"


def _explicit_negative_signals(location: dict[str, Any], family: str) -> list[str]:
    signals: list[str] = []
    for value in location.get("location_types") or []:
        normalized = str(value).casefold().strip()
        if normalized in _REPRESENTATIVE_LOCATION_TYPES:
            signals.append(normalized)
        elif normalized:
            signals.append(normalized)
    for field_name in ("title", "description"):
        text = location.get(field_name) or ""
        for pattern in _TEXT_NEGATIVE_PATTERNS:
            if pattern.search(text):
                signals.append(pattern.pattern)
    if family == "DARE":
        signals.append("dare_source")
    if family == "DARMC":
        signals.append("darmc_source")
    return list(dict.fromkeys(signals))


def _has_explicit_representative_semantics(location: dict[str, Any]) -> bool:
    for value in location.get("location_types") or []:
        if str(value).casefold().strip() in _REPRESENTATIVE_LOCATION_TYPES:
            return True
    text = " ".join(filter(None, [location.get("title"), location.get("description")]))
    return any(pattern.search(text) for pattern in _TEXT_NEGATIVE_PATTERNS)


def _repr_point_relation(
    representative_longitude: float | None,
    representative_latitude: float | None,
    point_coordinates: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> str:
    if representative_longitude is None or representative_latitude is None:
        return "UNKNOWN"
    if not point_coordinates:
        return "NO_SOURCE_POINT"
    unique_points = list(dict.fromkeys(point_coordinates))
    repr_coord = (representative_longitude, representative_latitude)
    matches = [coord for coord in unique_points if coord == repr_coord]
    if len(unique_points) == 1:
        return "MATCHES_SINGLE_SOURCE_POINT" if matches else "MATCHES_NO_SOURCE_POINT"
    if len(matches) == len(unique_points) and matches:
        return "MATCHES_MULTIPLE_AGREEING_POINTS"
    if matches:
        return "MATCHES_ONE_OF_CONFLICTING_POINTS"
    return "MATCHES_NO_SOURCE_POINT"


def _primary_source_policy_family(families: tuple[str, ...]) -> str:
    for candidate in _FAMILY_PRIORITY:
        if candidate in families:
            return candidate
    return "UNKNOWN"


def _collect_authority_facts(
    *,
    representative_longitude: float | None,
    representative_latitude: float | None,
    locations: list[dict[str, Any]],
) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    geometry_types: list[str | None] = []
    point_records: list[tuple[str, tuple[float, float], dict[str, Any]]] = []
    non_point_count = 0
    for location in locations:
        family = _source_policy_family(location)
        geometry = location.get("geometry") or {}
        point_coords = _point_coordinates(geometry if geometry else None)
        parsed.append(
            {
                "location": location,
                "location_id": str(location.get("location_id") or ""),
                "family": family,
                "negative_signals": _explicit_negative_signals(location, family),
                "representative_semantics": _has_explicit_representative_semantics(location),
                "point_coords": point_coords,
            }
        )
        geometry_types.append(location.get("geometry_type") or geometry.get("type"))
        if point_coords is not None:
            point_records.append((parsed[-1]["location_id"], point_coords, location))
        elif geometry.get("type") or location.get("geometry_type"):
            non_point_count += 1

    point_coords = tuple(coords for _, coords, _ in point_records)
    unique_points = tuple(dict.fromkeys(point_coords))
    negative_signals = tuple(signal for item in parsed for signal in item["negative_signals"])
    families = tuple(item["family"] for item in parsed)
    return {
        "locations": locations,
        "source_location_ids": tuple(item["location_id"] for item in parsed if item["location_id"]),
        "point_records": tuple(point_records),
        "point_coords": point_coords,
        "unique_points": unique_points,
        "non_point_count": non_point_count,
        "geometry_summary": {
            "point_count": len(point_records),
            "non_point_count": non_point_count,
            "geometry_types": list(dict.fromkeys(filter(None, geometry_types))),
        },
        "multiplicity_summary": {
            "unique_point_count": len(unique_points),
            "conflicting": len(unique_points) > 1,
            "location_count": len(locations),
        },
        "repr_relation": _repr_point_relation(representative_longitude, representative_latitude, point_coords),
        "negative_signals": negative_signals,
        "primary_family": _primary_source_policy_family(families),
        "accuracy_uri": next((location.get("accuracy") for location in locations if location.get("accuracy")), None),
        "any_representative_semantics": any(item["representative_semantics"] for item in parsed),
    }


def _build_diagnostic(facts: dict[str, Any], authority_class: str, reason_codes: tuple[str, ...] | list[str]) -> CoordinateAuthorityDiagnostic:
    return CoordinateAuthorityDiagnostic(
        authority_class=authority_class,
        reason_codes=tuple(reason_codes),
        source_location_ids=facts["source_location_ids"],
        repr_point_relation=facts["repr_relation"],
        geometry_summary=facts["geometry_summary"],
        multiplicity_summary=facts["multiplicity_summary"],
        explicit_negative_signals=facts["negative_signals"],
        source_policy_family=facts["primary_family"],
        accuracy_uri=facts["accuracy_uri"],
        temporal_applicability="UNKNOWN",
        policy_version=POLICY_VERSION,
    )


def _representative_reason_codes(facts: dict[str, Any]) -> tuple[str, ...]:
    reason_codes: list[str] = []
    locations = facts["locations"]
    if any("central_point" in (location.get("location_types") or []) for location in locations):
        reason_codes.append("CENTRAL_POINT_ONLY")
    if any("representative" in (location.get("location_types") or []) for location in locations):
        reason_codes.append("EXPLICIT_REPRESENTATIVE_LOCATION")
    if facts["primary_family"] == "DARE":
        reason_codes.append("DARE_REPRESENTATIVE_SOURCE")
    if facts["primary_family"] == "DARMC":
        reason_codes.append("DARMC_MAP_SCALE_SOURCE")
    if facts["primary_family"] == "OSM":
        reason_codes.append("OSM_SOURCE_UNVERIFIED")
    if len(facts["unique_points"]) > 1:
        reason_codes.append("MULTIPLE_SOURCE_POINTS")
    if not reason_codes:
        reason_codes.append("EXPLICIT_REPRESENTATIVE_LOCATION")
    return tuple(dict.fromkeys(reason_codes))


def classify_coordinate_authority(
    *,
    representative_longitude: float | None,
    representative_latitude: float | None,
    locations: list[dict[str, Any]],
    period: str | None = None,
) -> CoordinateAuthorityDiagnostic:
    del period  # temporal diagnostics remain UNKNOWN in g6dd-v1
    if not locations:
        return CoordinateAuthorityDiagnostic(
            authority_class=UNKNOWN,
            reason_codes=("MISSING_SOURCE_AUTHORITY",),
            source_location_ids=(),
            repr_point_relation="NO_SOURCE_POINT",
            geometry_summary=_EMPTY_GEOMETRY_SUMMARY,
            multiplicity_summary=_EMPTY_MULTIPLICITY_SUMMARY,
            temporal_applicability="UNKNOWN",
            policy_version=POLICY_VERSION,
        )

    facts = _collect_authority_facts(
        representative_longitude=representative_longitude,
        representative_latitude=representative_latitude,
        locations=locations,
    )

    if not facts["point_records"] and facts["non_point_count"] > 0:
        return _build_diagnostic(facts, NON_POINT_GEOMETRY, ("NON_POINT_SOURCE_GEOMETRY",))

    if not facts["point_records"]:
        return _build_diagnostic(facts, UNKNOWN, ("MISSING_SOURCE_AUTHORITY",))

    if len(facts["unique_points"]) >= 3:
        return _build_diagnostic(facts, MULTI_LOCATION_CONFLICT, ("MULTIPLE_CONFLICTING_SOURCE_POINTS",))

    if facts["any_representative_semantics"]:
        return _build_diagnostic(facts, REPRESENTATIVE_ONLY, _representative_reason_codes(facts))

    if len(facts["unique_points"]) >= 2:
        return _build_diagnostic(facts, MULTI_LOCATION_CONFLICT, ("MULTIPLE_CONFLICTING_SOURCE_POINTS",))

    reason_codes = [
        "SOURCE_POINT_MATCHES_REPRPOINT" if facts["repr_relation"] == "MATCHES_SINGLE_SOURCE_POINT" else "NO_SOURCE_POINT"
    ]
    if facts["primary_family"] == "OSM":
        reason_codes.append("OSM_SOURCE_UNVERIFIED")
    if facts["primary_family"] == "PLEIADES":
        reason_codes.append("MISSING_SOURCE_AUTHORITY")
    return _build_diagnostic(facts, SOURCE_POINT_UNCERTAIN, tuple(dict.fromkeys(reason_codes)))
