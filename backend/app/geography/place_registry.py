"""Curated Roman Republican places with an offline Pleiades fallback."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.app.geography.feature_semantics import coordinate_role_for_semantics
from backend.app.geography.normalization import normalize_name
from backend.app.geography.place_disambiguation import PlaceResolutionContext, filter_resolution_candidates
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"
_DATA = Path(__file__).with_name("data") / "roman_republic_places.json"
_PHYSICAL_DATA = Path(__file__).with_name("data") / "physical_features.json"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_INDEX = _REPOSITORY_ROOT / "data" / "pleiades_v4_1" / "pleiades_v4_1.sqlite3"
_INDEX_ENV = "PLEIADES_GAZETTEER_PATH"
_SUPPORTED_INDEX_SCHEMA = "3"


@dataclass(frozen=True)
class GazetteerResolution:
    status: str
    places: tuple[HistoricalPlace, ...] = ()
    candidate_count: int = 0
    candidates: tuple[dict[str, Any], ...] = ()
    reason: str | None = None
    disambiguation_diagnostics: tuple[dict[str, Any], ...] = ()


@lru_cache(maxsize=1)
def records() -> list[dict]:
    return json.loads(_DATA.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def physical_records() -> list[dict]:
    if not _PHYSICAL_DATA.is_file():
        return []
    return json.loads(_PHYSICAL_DATA.read_text(encoding="utf-8"))


def _record_to_place(record: dict) -> HistoricalPlace:
    return HistoricalPlace(
        id=f"pleiades-{record['pleiades_id']}",
        canonical_name=record["canonical_name"],
        modern_name=record.get("modern_name"),
        latitude=record["latitude"],
        longitude=record["longitude"],
        period=record.get("period"),
        source=PLEIADES_SOURCE,
        source_id=str(record["pleiades_id"]),
        source_url=f"https://pleiades.stoa.org/places/{record['pleiades_id']}",
        confidence=record["confidence"],
        uncertain=record.get("uncertain", False),
        coordinate_role=record["coordinate_role"],
        spatial_semantics=PlaceSpatialSemantics(record["spatial_semantics"]),
        spatial_semantics_provenance=record["spatial_semantics_provenance"],
    )


@lru_cache(maxsize=1)
def places() -> tuple[HistoricalPlace, ...]:
    return tuple(_record_to_place(record) for record in [*records(), *physical_records()])


@lru_cache(maxsize=1)
def aliases() -> dict[str, tuple[HistoricalPlace, ...]]:
    by_id = {place.id: place for place in places()}
    result: dict[str, list[HistoricalPlace]] = {}
    for record in [*records(), *physical_records()]:
        for alias in record["aliases"]:
            result.setdefault(normalize_name(alias), []).append(by_id[f"pleiades-{record['pleiades_id']}"])
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
            f"Unsupported Pleiades index schema: {values.get('index_schema_version')!r}; "
            f"expected {_SUPPORTED_INDEX_SCHEMA!r}"
        )
    return values


def _spatial_semantics(place_types: tuple[str, ...], *, title: str = "") -> PlaceSpatialSemantics:
    values = {value.casefold().replace("_", "-") for value in place_types}
    title_lower = title.casefold()
    if "pass" in values:
        return PlaceSpatialSemantics.PASS
    if "island" in values:
        return PlaceSpatialSemantics.ISLAND
    if "river" in values:
        return PlaceSpatialSemantics.RIVER
    if "mountain" in values:
        return PlaceSpatialSemantics.MOUNTAIN_REGION
    if "water-open" in values:
        if "strait" in title_lower:
            return PlaceSpatialSemantics.STRAIT
        return PlaceSpatialSemantics.SEA
    if values & {"port"} and values & {"settlement", "urban", "archaeological-site"}:
        return PlaceSpatialSemantics.PORT
    if values & {"region", "province", "province-2", "people", "ethnic-region"}:
        return PlaceSpatialSemantics.REGION
    if values & {"settlement", "urban", "fort", "fort-2", "station", "fortified-settlement"}:
        return PlaceSpatialSemantics.SETTLEMENT
    return PlaceSpatialSemantics.UNKNOWN


def _name_attestations(connection: sqlite3.Connection, name_row_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    if not name_row_ids:
        return {}
    placeholders = ",".join("?" for _ in name_row_ids)
    rows = connection.execute(
        f"""SELECT name_row_id, time_period, time_period_uri, confidence, confidence_uri
              FROM name_attestations
             WHERE name_row_id IN ({placeholders})
             ORDER BY name_row_id, row_id""",
        name_row_ids,
    ).fetchall()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["name_row_id"], []).append(
            {
                "time_period": row["time_period"],
                "time_period_uri": row["time_period_uri"],
                "confidence": row["confidence"],
                "confidence_uri": row["confidence_uri"],
            }
        )
    return grouped


def _location_metadata(row: sqlite3.Row) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "location_id": row["location_id"],
        "geometry_type": row["geometry_type"],
        "accuracy": row["accuracy"],
        "accuracy_value": row["accuracy_value"],
        "provenance": row["provenance"],
    }
    if row["geometry_json"]:
        payload["geometry"] = json.loads(row["geometry_json"])
    if row["title"] is not None:
        payload["title"] = row["title"]
    if row["description"] is not None:
        payload["description"] = row["description"]
    if row["start"] is not None:
        payload["start"] = row["start"]
    if row["end"] is not None:
        payload["end"] = row["end"]
    if row["attestations_json"]:
        payload["attestations"] = json.loads(row["attestations_json"])
    if row["feature_types_json"]:
        payload["feature_types"] = json.loads(row["feature_types_json"])
    if row["location_types_json"]:
        payload["location_types"] = json.loads(row["location_types_json"])
    if row["references_json"]:
        payload["references"] = json.loads(row["references_json"])
    return payload


def _lookup_index(path: Path, name: str) -> GazetteerResolution:
    connection = _connect_read_only(path)
    try:
        metadata = _metadata(connection)
        rows = connection.execute(
            """SELECT n.row_id, n.pleiades_id, n.original_name, n.name_resource_id,
                      n.language, n.name_type, n.name_start, n.name_end,
                      n.provenance AS name_provenance,
                      p.title, p.place_types_json, p.representative_lon,
                      p.representative_lat, p.bbox_min_lon, p.bbox_min_lat,
                      p.bbox_max_lon, p.bbox_max_lat, p.uri, p.provenance AS place_provenance
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

        attestations_by_name = _name_attestations(
            connection,
            [row["row_id"] for name_rows in grouped.values() for row in name_rows],
        )
        places_found: list[HistoricalPlace] = []
        candidate_summaries: list[dict[str, Any]] = []
        for pleiades_id, name_rows in grouped.items():
            first = name_rows[0]
            place_types = tuple(json.loads(first["place_types_json"] or "[]"))
            locations = connection.execute(
                """SELECT location_id, geometry_type, accuracy, accuracy_value, provenance,
                          geometry_json, title, description, start, end,
                          attestations_json, feature_types_json, location_types_json, references_json
                     FROM locations WHERE pleiades_id = ? ORDER BY location_id""",
                (pleiades_id,),
            ).fetchall()
            coordinate_available = (
                first["representative_lon"] is not None
                and first["representative_lat"] is not None
            )
            matched_name_details = []
            for row in name_rows:
                matched_name_details.append(
                    {
                        "original_name": row["original_name"],
                        "name_type": row["name_type"],
                        "name_resource_id": row["name_resource_id"],
                        "language": row["language"],
                        "name_start": row["name_start"],
                        "name_end": row["name_end"],
                        "attestations": attestations_by_name.get(row["row_id"], []),
                    }
                )
            candidate_summaries.append(
                {
                    "pleiades_id": pleiades_id,
                    "canonical_name": first["title"],
                    "place_types": list(place_types),
                    "coordinate_available": coordinate_available,
                    "matched_names": list(dict.fromkeys(row["original_name"] for row in name_rows)),
                    "matched_name_details": matched_name_details,
                }
            )
            if not coordinate_available:
                continue
            semantics = _spatial_semantics(place_types, title=first["title"] or "")
            coordinate_role = coordinate_role_for_semantics(
                semantics, place_types=place_types, title=first["title"] or ""
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
                "matched_name_start": first["name_start"],
                "matched_name_end": first["name_end"],
                "matched_name_attestations": attestations_by_name.get(first["row_id"], []),
                "matched_name_details": matched_name_details,
                "name_provenance": first["name_provenance"],
                "place_types": list(place_types),
                "place_provenance": first["place_provenance"],
                "bbox": {
                    "min_lon": first["bbox_min_lon"],
                    "min_lat": first["bbox_min_lat"],
                    "max_lon": first["bbox_max_lon"],
                    "max_lat": first["bbox_max_lat"],
                },
                "locations": [_location_metadata(location) for location in locations],
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
    finally:
        connection.close()


def _apply_context(
    resolution: GazetteerResolution,
    context: PlaceResolutionContext | None,
) -> GazetteerResolution:
    if context is None:
        return resolution
    status, places, candidates, diagnostics = filter_resolution_candidates(
        status=resolution.status,
        places=resolution.places,
        candidates=resolution.candidates,
        context=context,
    )
    return GazetteerResolution(
        status=status,
        places=places,
        candidate_count=len(candidates) or resolution.candidate_count,
        candidates=candidates,
        reason="context_filtered" if diagnostics else resolution.reason,
        disambiguation_diagnostics=tuple(diagnostics),
    )


def resolve_with_status(name: str, context: PlaceResolutionContext | None = None) -> GazetteerResolution:
    curated = aliases().get(normalize_name(name), ())
    if curated:
        resolution = GazetteerResolution(
            status="CURATED", places=curated, candidate_count=len(curated)
        )
        return _apply_context(resolution, context)
    path, explicitly_configured = _index_path()
    if path is None:
        return GazetteerResolution(status="NOT_FOUND")
    if not path.is_file():
        reason = f"Configured Pleiades index does not exist: {path}"
        return GazetteerResolution(status="UNAVAILABLE", reason=reason)
    try:
        return _apply_context(_lookup_index(path, name), context)
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        if explicitly_configured or path == _DEFAULT_INDEX:
            return GazetteerResolution(status="UNAVAILABLE", reason=str(exc))
        return GazetteerResolution(status="NOT_FOUND")


def resolve(name: str) -> tuple[HistoricalPlace, ...]:
    return resolve_with_status(name).places


def alias_records() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    return tuple(
        (r["canonical_name"], tuple(r["aliases"]), "pleiades_registry_snapshot_2026_08")
        for r in records()
    )
