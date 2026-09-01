"""Build the pinned Pleiades 4.1 read-only gazetteer index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.geography.normalization import normalize_name

DATASET_VERSION = "4.1"
DATASET_RELEASE_DATE = "2025-05-28"
OFFICIAL_SOURCE = "https://zenodo.org/records/15540082"
LICENSE = "CC BY 3.0"
EXPECTED_SHA256 = "94c5c337d27a07f1a5fa231b6513e40e6d3cf7d5ad7b8c010f8e8bfd9159cd85"
EXPECTED_PLACE_COUNT = 41_480
INDEX_SCHEMA_VERSION = "2"
IMPORTER_VERSION = "2"
_DERIVED_TITLE_DISAMBIGUATOR = re.compile(r"^(.+?)\s+\([^)]+\)\s*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE places (
            pleiades_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            place_types_json TEXT NOT NULL,
            representative_lon REAL,
            representative_lat REAL,
            bbox_min_lon REAL,
            bbox_min_lat REAL,
            bbox_max_lon REAL,
            bbox_max_lat REAL,
            uri TEXT,
            provenance TEXT,
            review_state TEXT
        );
        CREATE TABLE names (
            row_id INTEGER PRIMARY KEY,
            normalized_name TEXT NOT NULL,
            original_name TEXT NOT NULL,
            pleiades_id TEXT NOT NULL,
            name_resource_id TEXT,
            language TEXT,
            name_type TEXT,
            name_start INTEGER,
            name_end INTEGER,
            provenance TEXT
        );
        CREATE TABLE name_attestations (
            row_id INTEGER PRIMARY KEY,
            name_row_id INTEGER NOT NULL,
            time_period TEXT,
            time_period_uri TEXT,
            confidence TEXT,
            confidence_uri TEXT
        );
        CREATE TABLE locations (
            row_id INTEGER PRIMARY KEY,
            pleiades_id TEXT NOT NULL,
            location_id TEXT,
            geometry_type TEXT,
            accuracy TEXT,
            accuracy_value REAL,
            provenance TEXT
        );
        CREATE INDEX names_normalized_name_idx ON names(normalized_name);
        CREATE INDEX locations_pleiades_id_idx ON locations(pleiades_id);
        CREATE INDEX name_attestations_name_row_id_idx ON name_attestations(name_row_id);
        """
    )


def _romanized_forms(romanized: object) -> list[str]:
    values: list[str] = []
    if isinstance(romanized, str):
        parts = [part.strip() for part in romanized.split(",")] if "," in romanized else [romanized]
        values.extend(part for part in parts if part)
    elif isinstance(romanized, list):
        for value in romanized:
            if isinstance(value, str):
                parts = [part.strip() for part in value.split(",")] if "," in value else [value]
                values.extend(part for part in parts if part)
    return values


def _name_forms(name: dict) -> list[str]:
    values: list[str] = []
    attested = name.get("attested")
    if isinstance(attested, str) and attested.strip():
        values.append(attested)
    values.extend(_romanized_forms(name.get("romanized")))
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _derived_title_search_forms(title: str) -> list[str]:
    forms = [title.strip()]
    match = _DERIVED_TITLE_DISAMBIGUATOR.match(title.strip())
    if match:
        clean = match.group(1).strip()
        if clean and clean not in forms:
            forms.append(clean)
    return forms


def _insert_name(
    connection: sqlite3.Connection,
    *,
    pleiades_id: str,
    original: str,
    name: dict,
    name_type: str | None,
    seen_names: set[tuple[str, str | None]],
) -> int:
    resource_id = str(name.get("id")) if name.get("id") is not None else None
    key = (original, resource_id)
    if key in seen_names:
        return 0
    seen_names.add(key)
    cursor = connection.execute(
        """INSERT INTO names
           (normalized_name, original_name, pleiades_id, name_resource_id,
            language, name_type, name_start, name_end, provenance)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            normalize_name(original),
            original,
            pleiades_id,
            resource_id,
            name.get("language"),
            name_type or name.get("nameType"),
            name.get("start"),
            name.get("end"),
            name.get("provenance"),
        ),
    )
    name_row_id = cursor.lastrowid
    for attestation in name.get("attestations") or []:
        connection.execute(
            """INSERT INTO name_attestations
               (name_row_id, time_period, time_period_uri, confidence, confidence_uri)
               VALUES (?, ?, ?, ?, ?)""",
            (
                name_row_id,
                attestation.get("timePeriod"),
                attestation.get("timePeriodURI"),
                attestation.get("confidence"),
                attestation.get("confidenceURI"),
            ),
        )
    return 1


def _insert_derived_title_names(
    connection: sqlite3.Connection,
    *,
    pleiades_id: str,
    title: str,
    provenance: str | None,
    indexed_forms: set[str],
    seen_names: set[tuple[str, str | None]],
) -> int:
    inserted = 0
    synthetic = {
        "id": None,
        "language": None,
        "nameType": "derived_title",
        "provenance": provenance,
        "attestations": [],
        "start": None,
        "end": None,
    }
    for form in _derived_title_search_forms(title):
        normalized = normalize_name(form)
        if normalized in indexed_forms:
            continue
        indexed_forms.add(normalized)
        inserted += _insert_name(
            connection,
            pleiades_id=pleiades_id,
            original=form,
            name={**synthetic, "attested": form},
            name_type="derived_title",
            seen_names=seen_names,
        )
    return inserted


def build_index(
    source: Path,
    output: Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    expected_place_count: int = EXPECTED_PLACE_COUNT,
) -> dict[str, object]:
    actual_sha256 = sha256(source)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"Pleiades archive SHA-256 mismatch: {actual_sha256}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    place_count = name_count = location_count = attestation_count = 0
    try:
        connection = sqlite3.connect(temporary)
        try:
            _schema(connection)
            with zipfile.ZipFile(source) as archive:
                entries = sorted(
                    name for name in archive.namelist()
                    if "/data/json/" in name and name.endswith(".json")
                )
                for entry in entries:
                    with archive.open(entry) as stream:
                        place = json.load(stream)
                    pleiades_id = str(place["id"])
                    point = place.get("reprPoint") or []
                    longitude = point[0] if len(point) >= 2 else None
                    latitude = point[1] if len(point) >= 2 else None
                    bbox = place.get("bbox") or []
                    bbox_min_lon = bbox[0] if len(bbox) >= 4 else None
                    bbox_min_lat = bbox[1] if len(bbox) >= 4 else None
                    bbox_max_lon = bbox[2] if len(bbox) >= 4 else None
                    bbox_max_lat = bbox[3] if len(bbox) >= 4 else None
                    place_types = place.get("placeTypes") or []
                    connection.execute(
                        """INSERT INTO places
                           (pleiades_id, title, place_types_json, representative_lon,
                            representative_lat, bbox_min_lon, bbox_min_lat, bbox_max_lon,
                            bbox_max_lat, uri, provenance, review_state)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            pleiades_id,
                            place.get("title") or pleiades_id,
                            json.dumps(place_types, ensure_ascii=False, separators=(",", ":")),
                            longitude,
                            latitude,
                            bbox_min_lon,
                            bbox_min_lat,
                            bbox_max_lon,
                            bbox_max_lat,
                            place.get("uri"),
                            place.get("provenance"),
                            place.get("review_state"),
                        ),
                    )
                    seen_names: set[tuple[str, str | None]] = set()
                    indexed_forms = {
                        normalize_name(form)
                        for item in (place.get("names") or [])
                        for form in _name_forms(item)
                    }
                    for name in place.get("names") or []:
                        for original in _name_forms(name):
                            name_count += _insert_name(
                                connection,
                                pleiades_id=pleiades_id,
                                original=original,
                                name=name,
                                name_type=None,
                                seen_names=seen_names,
                            )
                    title = place.get("title")
                    if isinstance(title, str) and title.strip():
                        name_count += _insert_derived_title_names(
                            connection,
                            pleiades_id=pleiades_id,
                            title=title,
                            provenance=place.get("provenance"),
                            indexed_forms=indexed_forms,
                            seen_names=seen_names,
                        )
                    for location in place.get("locations") or []:
                        geometry = location.get("geometry") or {}
                        connection.execute(
                            """INSERT INTO locations
                               (pleiades_id, location_id, geometry_type, accuracy,
                                accuracy_value, provenance)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                pleiades_id,
                                str(location.get("id") or ""),
                                geometry.get("type"),
                                location.get("accuracy"),
                                location.get("accuracy_value"),
                                location.get("provenance"),
                            ),
                        )
                        location_count += 1
                    place_count += 1
            if place_count != expected_place_count:
                raise ValueError(f"Pleiades place-count mismatch: {place_count}")
            attestation_count = connection.execute(
                "SELECT COUNT(*) FROM name_attestations"
            ).fetchone()[0]
            metadata = {
                "dataset_version": DATASET_VERSION,
                "dataset_release_date": DATASET_RELEASE_DATE,
                "official_source": OFFICIAL_SOURCE,
                "license": LICENSE,
                "sha256": actual_sha256,
                "build_timestamp": datetime.now(timezone.utc).isoformat(),
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "importer_version": IMPORTER_VERSION,
                "place_count": str(place_count),
                "name_count": str(name_count),
                "attestation_count": str(attestation_count),
                "location_count": str(location_count),
            }
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.commit()
        finally:
            connection.close()
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "output": str(output),
        "place_count": place_count,
        "name_count": name_count,
        "attestation_count": attestation_count,
        "location_count": location_count,
        "sha256": actual_sha256,
        "size": output.stat().st_size,
        "index_schema_version": INDEX_SCHEMA_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_index(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
