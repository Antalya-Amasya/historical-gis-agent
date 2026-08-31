"""Build the pinned Pleiades 4.1 read-only gazetteer index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_VERSION = "4.1"
DATASET_RELEASE_DATE = "2025-05-28"
OFFICIAL_SOURCE = "https://zenodo.org/records/15540082"
LICENSE = "CC BY 3.0"
EXPECTED_SHA256 = "94c5c337d27a07f1a5fa231b6513e40e6d3cf7d5ad7b8c010f8e8bfd9159cd85"
EXPECTED_PLACE_COUNT = 41_480
INDEX_SCHEMA_VERSION = "1"
IMPORTER_VERSION = "1"


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold().strip())


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
            provenance TEXT
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
        """
    )


def _name_forms(name: dict) -> list[str]:
    values: list[str] = []
    attested = name.get("attested")
    if isinstance(attested, str) and attested.strip():
        values.append(attested)
    romanized = name.get("romanized")
    if isinstance(romanized, str):
        values.append(romanized)
    elif isinstance(romanized, list):
        values.extend(value for value in romanized if isinstance(value, str))
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


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
    place_count = name_count = location_count = 0
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
                    place_types = place.get("placeTypes") or []
                    connection.execute(
                        "INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            pleiades_id, place.get("title") or pleiades_id,
                            json.dumps(place_types, ensure_ascii=False, separators=(",", ":")),
                            longitude, latitude, place.get("uri"), place.get("provenance"),
                            place.get("review_state"),
                        ),
                    )
                    seen_names: set[tuple[str, str | None]] = set()
                    raw_names = list(place.get("names") or [])
                    indexed_forms = {
                        normalize_name(form)
                        for item in raw_names for form in _name_forms(item)
                    }
                    title = place.get("title")
                    if isinstance(title, str) and normalize_name(title) not in indexed_forms:
                        raw_names.append({
                            "attested": title, "id": None, "language": None,
                            "nameType": "canonical", "provenance": place.get("provenance"),
                        })
                    for name in raw_names:
                        for original in _name_forms(name):
                            resource_id = str(name.get("id")) if name.get("id") is not None else None
                            key = (original, resource_id)
                            if key in seen_names:
                                continue
                            seen_names.add(key)
                            connection.execute(
                                """INSERT INTO names
                                   (normalized_name, original_name, pleiades_id, name_resource_id,
                                    language, name_type, provenance) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (normalize_name(original), original, pleiades_id, resource_id,
                                 name.get("language"), name.get("nameType"), name.get("provenance")),
                            )
                            name_count += 1
                    for location in place.get("locations") or []:
                        geometry = location.get("geometry") or {}
                        connection.execute(
                            """INSERT INTO locations
                               (pleiades_id, location_id, geometry_type, accuracy, accuracy_value, provenance)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (pleiades_id, str(location.get("id") or ""), geometry.get("type"),
                             location.get("accuracy"), location.get("accuracy_value"),
                             location.get("provenance")),
                        )
                        location_count += 1
                    place_count += 1
            if place_count != expected_place_count:
                raise ValueError(f"Pleiades place-count mismatch: {place_count}")
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
        "output": str(output), "place_count": place_count,
        "name_count": name_count, "location_count": location_count,
        "sha256": actual_sha256, "size": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_index(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
