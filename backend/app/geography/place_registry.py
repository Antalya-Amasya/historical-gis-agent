"""Curated Roman Republican places with an offline Pleiades fallback."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"
_DATA = Path(__file__).with_name("data") / "roman_republic_places.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INDEX = _REPOSITORY_ROOT / "data" / "pleiades_v4_1" / "pleiades_v4_1.sqlite3"
_INDEX_ENV = "PLEIADES_GAZETTEER_PATH"
_SUPPORTED_INDEX_SCHEMA = "1"


@dataclass(frozen=True)
class GazetteerResolution:
    status: str
    places: tuple[HistoricalPlace, ...] = ()
    candidate_count: int = 0
    candidates: tuple[dict[str, Any], ...] = ()
    reason: str | None = None


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)

@lru_cache(maxsize=1)
def records() -> list[dict]:
    return json.loads(_DATA.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def places() -> tuple[HistoricalPlace, ...]:
    return tuple(HistoricalPlace(id=f"pleiades-{r['pleiades_id']}", canonical_name=r["canonical_name"], modern_name=r.get("modern_name"), latitude=r["latitude"], longitude=r["longitude"], period=r.get("period"), source=PLEIADES_SOURCE, source_id=str(r["pleiades_id"]), source_url=f"https://pleiades.stoa.org/places/{r['pleiades_id']}", confidence=r["confidence"], uncertain=r.get("uncertain", False), coordinate_role=r["coordinate_role"], spatial_semantics=PlaceSpatialSemantics(r["spatial_semantics"]), spatial_semantics_provenance=r["spatial_semantics_provenance"]) for r in records())

@lru_cache(maxsize=1)
def aliases() -> dict[str, tuple[HistoricalPlace, ...]]:
    by_id = {place.id: place for place in places()}; result = {}
    for r in records():
        for alias in r["aliases"]: result.setdefault(normalize_name(alias), []).append(by_id[f"pleiades-{r['pleiades_id']}"])
    return {key: tuple(value) for key, value in result.items()}

def _index_path() -> tuple[Path | None, bool]:
    configured = os.getenv(_INDEX_ENV)
    if configured:
        return Path(configured), True
    return _DEFAULT_INDEX, False


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    values = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")}
    if values.get("index_schema_version") != _SUPPORTED_INDEX_SCHEMA:
        raise RuntimeError(
            f"Unsupported Pleiades index schema: {values.get('index_schema_version')!r}"
        )
    return values


def _spatial_semantics(place_types: tuple[str, ...]) -> PlaceSpatialSemantics:
    values = {value.casefold().replace("_", "-") for value in place_types}
    if "island" in values:
        return PlaceSpatialSemantics.ISLAND
    if "river" in values:
        return PlaceSpatialSemantics.RIVER
    if "mountain" in values:
        return PlaceSpatialSemantics.MOUNTAIN_REGION
    if values & {"region", "province", "province-2", "people", "ethnic-region"}:
        return PlaceSpatialSemantics.REGION
    if values & {"settlement", "urban", "fort", "fort-2", "station", "fortified-settlement"}:
        return PlaceSpatialSemantics.SETTLEMENT
    return PlaceSpatialSemantics.UNKNOWN


def _lookup_index(path: Path, name: str) -> GazetteerResolution:
    with _connect_read_only(path) as connection:
        metadata = _metadata(connection)
        rows = connection.execute(
            """SELECT n.pleiades_id, n.original_name, n.name_resource_id,
                      n.language, n.name_type, n.provenance AS name_provenance,
                      p.title, p.place_types_json, p.representative_lon,
                      p.representative_lat, p.uri, p.provenance AS place_provenance
                 FROM names AS n
                 JOIN places AS p USING (pleiades_id)
                WHERE n.normalized_name = ?
                ORDER BY n.pleiades_id, n.original_name, n.name_resource_id""",
            (normalize_name(name),),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["pleiades_id"], []).append(row)
        if not grouped:
            return GazetteerResolution(status="NOT_FOUND")

        places_found: list[HistoricalPlace] = []
        candidate_summaries: list[dict[str, Any]] = []
        for pleiades_id, name_rows in grouped.items():
            first = name_rows[0]
            place_types = tuple(json.loads(first["place_types_json"] or "[]"))
            locations = connection.execute(
                """SELECT location_id, geometry_type, accuracy, accuracy_value, provenance
                     FROM locations WHERE pleiades_id = ? ORDER BY location_id""",
                (pleiades_id,),
            ).fetchall()
            coordinate_available = (
                first["representative_lon"] is not None
                and first["representative_lat"] is not None
            )
            candidate_summaries.append(
                {
                    "pleiades_id": pleiades_id,
                    "canonical_name": first["title"],
                    "place_types": list(place_types),
                    "coordinate_available": coordinate_available,
                    "matched_names": list(dict.fromkeys(row["original_name"] for row in name_rows)),
                }
            )
            if not coordinate_available:
                continue
            semantics = _spatial_semantics(place_types)
            coordinate_role = (
                "regional_centroid" if semantics is PlaceSpatialSemantics.REGION else "representative_point"
            )
            authority_metadata = {
                "dataset_version": metadata.get("dataset_version"),
                "dataset_release_date": metadata.get("dataset_release_date"),
                "index_schema_version": metadata.get("index_schema_version"),
                "official_source": metadata.get("official_source"),
                "dataset_sha256": metadata.get("sha256"),
                "matched_name": first["original_name"],
                "matched_name_id": first["name_resource_id"],
                "matched_name_language": first["language"],
                "matched_name_type": first["name_type"],
                "name_provenance": first["name_provenance"],
                "place_types": list(place_types),
                "place_provenance": first["place_provenance"],
                "locations": [dict(location) for location in locations],
            }
            places_found.append(
                HistoricalPlace(
                    id=f"pleiades-{pleiades_id}",
                    canonical_name=first["title"],
                    latitude=first["representative_lat"],
                    longitude=first["representative_lon"],
                    source=f"{PLEIADES_SOURCE} Dataset {metadata.get('dataset_version', '')}".strip(),
                    source_id=pleiades_id,
                    source_url=first["uri"] or f"https://pleiades.stoa.org/places/{pleiades_id}",
                    confidence=0.55 if semantics is PlaceSpatialSemantics.REGION else 0.65,
                    uncertain=True,
                    coordinate_role=coordinate_role,
                    spatial_semantics=semantics,
                    spatial_semantics_provenance="Pleiades generic fallback; representative coordinates are not exact historical event locations.",
                    authoritative_geometry_available=any(location["geometry_type"] for location in locations),
                    authoritative_geometry_reference=(
                        f"Pleiades locations for place {pleiades_id}" if locations else None
                    ),
                    authority_metadata=authority_metadata,
                )
            )
        status = "UNIQUE" if len(grouped) == 1 else "AMBIGUOUS"
        if status == "UNIQUE" and not places_found:
            status = "UNLOCATED"
        return GazetteerResolution(
            status=status,
            places=tuple(places_found),
            candidate_count=len(grouped),
            candidates=tuple(candidate_summaries),
        )


def resolve_with_status(name: str) -> GazetteerResolution:
    curated = aliases().get(normalize_name(name), ())
    if curated:
        return GazetteerResolution(
            status="CURATED", places=curated, candidate_count=len(curated)
        )
    path, explicitly_configured = _index_path()
    if path is None:
        return GazetteerResolution(status="NOT_FOUND")
    if not path.is_file():
        reason = f"Configured Pleiades index does not exist: {path}"
        return GazetteerResolution(status="UNAVAILABLE", reason=reason)
    try:
        return _lookup_index(path, name)
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        if explicitly_configured or path == _DEFAULT_INDEX:
            return GazetteerResolution(status="UNAVAILABLE", reason=str(exc))
        return GazetteerResolution(status="NOT_FOUND")


def resolve(name: str) -> tuple[HistoricalPlace, ...]:
    return resolve_with_status(name).places


def alias_records() -> tuple[tuple[str, tuple[str, ...], str], ...]: return tuple((r["canonical_name"], tuple(r["aliases"]), "pleiades_registry_snapshot_2026_08") for r in records())
