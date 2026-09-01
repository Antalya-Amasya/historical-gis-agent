"""G3 regression tests for Pleiades candidate generation and resolver semantics."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from backend.app.geography.normalization import normalize_name
from backend.app.models import PlaceSpatialSemantics
from geography_mcp.service import GeographyService
from scripts.build_pleiades_index import (
    INDEX_SCHEMA_VERSION,
    _derived_title_search_forms,
    _insert_derived_title_names,
    _insert_name,
    _name_forms,
    _schema,
    build_index,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _REPO_ROOT / "scripts" / "build_pleiades_index.py"


def test_normalization_is_shared_between_builder_and_runtime():
    assert normalize_name("  Hellas,   Hellás  ") == place_registry.normalize_name("  Hellas,   Hellás  ")
    assert _BUILDER.read_text(encoding="utf-8").count("def normalize_name") == 0
    assert "from backend.app.geography.normalization import normalize_name" in _BUILDER.read_text(encoding="utf-8")


def test_romanized_comma_splitting_creates_individual_forms():
    forms = _name_forms({"romanized": "Hellas, Hellás"})
    assert forms == ["Hellas", "Hellás"]


def test_derived_title_search_forms_keep_full_title_and_clean_form():
    assert _derived_title_search_forms("Italia (region)") == ["Italia (region)", "Italia"]
    assert _derived_title_search_forms("Achaia (region?)") == ["Achaia (region?)", "Achaia"]


def _build_fixture_db(tmp_path: Path, *, schema_version: str = INDEX_SCHEMA_VERSION) -> Path:
    path = tmp_path / "pleiades.sqlite3"
    connection = sqlite3.connect(path)
    try:
        if schema_version == "2":
            _schema(connection)
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                {
                    "index_schema_version": "2",
                    "dataset_version": "4.1",
                    "dataset_release_date": "2025-05-28",
                    "official_source": "fixture",
                    "sha256": "abc",
                }.items(),
            )
            connection.execute(
                """INSERT INTO places
                   (pleiades_id, title, place_types_json, representative_lon, representative_lat,
                    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, uri, provenance, review_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "452346", "Italia (region)", json.dumps(["region", "label"]),
                    12.5, 41.5, 12.0, 41.0, 13.0, 42.0,
                    "https://pleiades.stoa.org/places/452346", "fixture", "published",
                ),
            )
            seen: set[tuple[str, str | None]] = set()
            indexed: set[str] = set()
            _insert_derived_title_names(
                connection,
                pleiades_id="452346",
                title="Italia (region)",
                provenance="fixture",
                indexed_forms=indexed,
                seen_names=seen,
            )
            romanized_name = {
                "romanized": "Italia, Italía",
                "id": "italia-region",
                "language": "la",
                "nameType": "geographic",
                "attestations": [{
                    "timePeriod": "roman",
                    "timePeriodURI": "https://example.invalid/roman",
                    "confidence": "confident",
                    "confidenceURI": "https://example.invalid/confident",
                }],
                "start": -30,
                "end": 300,
                "provenance": "fixture",
            }
            for original in _name_forms(romanized_name):
                _insert_name(
                    connection,
                    pleiades_id="452346",
                    original=original,
                    name=romanized_name,
                    name_type=None,
                    seen_names=seen,
                )
            connection.execute(
                """INSERT INTO places
                   (pleiades_id, title, place_types_json, representative_lon, representative_lat,
                    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, uri, provenance, review_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "900010", "Italia", json.dumps(["region"]),
                    11.0, 43.0, None, None, None, None,
                    "https://pleiades.stoa.org/places/900010", "fixture", "published",
                ),
            )
            connection.execute(
                """INSERT INTO names
                   (normalized_name, original_name, pleiades_id, name_resource_id,
                    language, name_type, name_start, name_end, provenance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    normalize_name("Italia"), "Italia", "900010", "italia-alt",
                    "", "geographic", None, None, "fixture",
                ),
            )
            connection.execute(
                """INSERT INTO places
                   (pleiades_id, title, place_types_json, representative_lon, representative_lat,
                    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, uri, provenance, review_state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "900001", "Unlocated Place", json.dumps(["settlement"]),
                    None, None, None, None, None, None,
                    "https://pleiades.stoa.org/places/900001", "fixture", "published",
                ),
            )
            connection.execute(
                """INSERT INTO names
                   (normalized_name, original_name, pleiades_id, name_resource_id,
                    language, name_type, name_start, name_end, provenance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    normalize_name("Unlocated Place"),
                    "Unlocated Place",
                    "900001",
                    "unlocated",
                    "",
                    "geographic",
                    None,
                    None,
                    "fixture",
                ),
            )
            for pleiades_id, title, lon, lat in (
                ("501596", "Samothrace", 25.5, 40.5),
                ("501597", "Samothrace", 25.6, 40.5),
                ("606283", "Chios", None, None),
            ):
                connection.execute(
                    """INSERT INTO places
                       (pleiades_id, title, place_types_json, representative_lon, representative_lat,
                        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, uri, provenance, review_state)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pleiades_id, title, json.dumps(["settlement"]),
                        lon, lat, None, None, None, None,
                        f"https://pleiades.stoa.org/places/{pleiades_id}", "fixture", "published",
                    ),
                )
                connection.execute(
                    """INSERT INTO names
                       (normalized_name, original_name, pleiades_id, name_resource_id,
                        language, name_type, name_start, name_end, provenance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        normalize_name(title), title, pleiades_id, f"name-{pleiades_id}",
                        "la", "geographic", None, None, "fixture",
                    ),
                )
            connection.commit()
        else:
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE places (
                    pleiades_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    place_types_json TEXT NOT NULL, representative_lon REAL,
                    representative_lat REAL, uri TEXT, provenance TEXT, review_state TEXT
                );
                CREATE TABLE names (
                    row_id INTEGER PRIMARY KEY, normalized_name TEXT NOT NULL,
                    original_name TEXT NOT NULL, pleiades_id TEXT NOT NULL,
                    name_resource_id TEXT, language TEXT, name_type TEXT, provenance TEXT
                );
                CREATE TABLE locations (
                    row_id INTEGER PRIMARY KEY, pleiades_id TEXT NOT NULL,
                    location_id TEXT, geometry_type TEXT, accuracy TEXT,
                    accuracy_value REAL, provenance TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                ("index_schema_version", "1"),
            )
            connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture()
def g3_fixture_db(tmp_path, monkeypatch):
    path = _build_fixture_db(tmp_path)
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    return path


def test_italia_hidden_candidate_and_derived_title_semantics(g3_fixture_db):
    rows = sqlite3.connect(g3_fixture_db).execute(
        "SELECT original_name, name_type FROM names WHERE pleiades_id = '452346' ORDER BY original_name"
    ).fetchall()
    assert ("Italia (region)", "derived_title") in rows
    assert ("Italia", "derived_title") in rows
    assert ("Italia", "geographic") in rows
    assert all(name_type != "canonical" for _, name_type in rows)

    lookup = place_registry._lookup_index(g3_fixture_db, "Italia")
    assert lookup.status == "AMBIGUOUS"
    assert lookup.candidate_count >= 2
    assert "452346" in {candidate["pleiades_id"] for candidate in lookup.candidates}

    located = next(
        candidate for candidate in lookup.candidates if candidate["pleiades_id"] == "452346"
    )
    detail = next(
        item for item in located["matched_name_details"]
        if item["original_name"] == "Italia" and item["name_type"] == "geographic"
    )
    assert detail["name_start"] == -30
    assert detail["name_end"] == 300
    assert detail["attestations"] == [{
        "time_period": "roman",
        "time_period_uri": "https://example.invalid/roman",
        "confidence": "confident",
        "confidence_uri": "https://example.invalid/confident",
    }]
    bbox_place = sqlite3.connect(g3_fixture_db).execute(
        "SELECT bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat FROM places WHERE pleiades_id = '452346'"
    ).fetchone()
    assert bbox_place == (12.0, 41.0, 13.0, 42.0)


def test_schema_v1_index_is_unavailable_with_explicit_reason(tmp_path, monkeypatch):
    path = _build_fixture_db(tmp_path, schema_version="1")
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    result = place_registry.resolve_with_status("Roma")
    assert result.status == "UNAVAILABLE"
    assert "Unsupported Pleiades index schema" in (result.reason or "")
    payload = GeographyService().resolve_ancient_place_payload("Roma")
    assert payload["status"] == "UNAVAILABLE"


def test_unlocated_and_ambiguous_semantics(g3_fixture_db):
    unlocated = place_registry.resolve_with_status("Unlocated Place")
    assert unlocated.status == "UNLOCATED"
    assert unlocated.candidate_count == 1

    ambiguous = place_registry.resolve_with_status("Samothrace")
    assert ambiguous.status == "AMBIGUOUS"
    assert ambiguous.candidate_count == 2
    assert GeographyService().resolve_ancient_place("Samothrace") is None


def test_multiple_unlocated_candidates_remain_ambiguous(g3_fixture_db):
    connection = sqlite3.connect(g3_fixture_db)
    connection.execute(
        """INSERT INTO places
           (pleiades_id, title, place_types_json, representative_lon, representative_lat,
            bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, uri, provenance, review_state)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "900002", "Samothrace", json.dumps(["label"]), None, None,
            None, None, None, None, "https://pleiades.stoa.org/places/900002", "fixture", "published",
        ),
    )
    connection.execute(
        """INSERT INTO names
           (normalized_name, original_name, pleiades_id, name_resource_id,
            language, name_type, name_start, name_end, provenance)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            normalize_name("Samothrace"), "Samothrace", "900002", "samothrace-label",
            "", "label", None, None, "fixture",
        ),
    )
    connection.commit()
    connection.close()

    result = place_registry.resolve_with_status("Samothrace")
    assert result.status == "AMBIGUOUS"
    assert result.candidate_count == 3
    assert GeographyService().resolve_ancient_place("Samothrace") is None


def test_no_candidate_ranking_or_first_wins_promotion(g3_fixture_db):
    result = place_registry._lookup_index(g3_fixture_db, "Samothrace")
    assert result.status == "AMBIGUOUS"
    assert result.candidate_count >= 2
    assert len(result.places) <= result.candidate_count
    assert GeographyService().resolve_ancient_place("Samothrace") is None


def test_greece_and_italy_curated_overrides_remain_unique():
    greece = place_registry.resolve_with_status("Greece")
    assert greece.status == "CURATED"
    assert greece.places[0].canonical_name == "Hellas"
    assert greece.places[0].source_id == "1001896"
    assert greece.places[0].spatial_semantics is PlaceSpatialSemantics.REGION
    assert greece.places[0].coordinate_role == "regional_centroid"
    assert greece.places[0].uncertain is True

    italy = place_registry.resolve_with_status("Italy")
    assert italy.status == "CURATED"
    assert italy.places[0].canonical_name == "Italia"
    assert italy.places[0].source_id == "1052"


@pytest.mark.skipif(
    not Path(r"C:\D\python\historical-gis-codex\data\pleiades_v4_1\pleiades.datasets-v4.1.zip").is_file(),
    reason="Pleiades source archive not available",
)
def test_builder_round_trip_on_real_archive(tmp_path):
    source = Path(r"C:\D\python\historical-gis-codex\data\pleiades_v4_1\pleiades.datasets-v4.1.zip")
    output = tmp_path / "pleiades.sqlite3"
    summary = build_index(source, output)
    connection = sqlite3.connect(output)
    metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    assert metadata["index_schema_version"] == "2"
    assert int(summary["name_count"]) > 0
    split_rows = connection.execute(
        "SELECT COUNT(*) FROM names WHERE original_name = 'Hellas'"
    ).fetchone()[0]
    assert split_rows >= 1
    derived_rows = connection.execute(
        "SELECT COUNT(*) FROM names WHERE name_type = 'derived_title'"
    ).fetchone()[0]
    assert derived_rows > 0
    attestation_rows = connection.execute("SELECT COUNT(*) FROM name_attestations").fetchone()[0]
    assert attestation_rows > 0
    bbox_rows = connection.execute(
        "SELECT COUNT(*) FROM places WHERE bbox_min_lon IS NOT NULL"
    ).fetchone()[0]
    assert bbox_rows > 0
    connection.close()
