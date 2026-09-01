import json
import sqlite3

import pytest

from backend.app.geography import place_registry
from backend.app.models import PlaceSpatialSemantics
from geography_mcp.service import GeographyService


@pytest.fixture()
def gazetteer(tmp_path, monkeypatch):
    path = tmp_path / "pleiades.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE places (pleiades_id TEXT PRIMARY KEY, title TEXT NOT NULL,
          place_types_json TEXT NOT NULL, representative_lon REAL, representative_lat REAL,
          uri TEXT, provenance TEXT, review_state TEXT);
        CREATE TABLE names (row_id INTEGER PRIMARY KEY, normalized_name TEXT NOT NULL,
          original_name TEXT NOT NULL, pleiades_id TEXT NOT NULL, name_resource_id TEXT,
          language TEXT, name_type TEXT, provenance TEXT);
        CREATE TABLE locations (row_id INTEGER PRIMARY KEY, pleiades_id TEXT NOT NULL,
          location_id TEXT, geometry_type TEXT, accuracy TEXT, accuracy_value REAL,
          provenance TEXT);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", {
            "index_schema_version": "1", "dataset_version": "4.1",
            "dataset_release_date": "2025-05-28", "official_source": "official",
            "sha256": "abc",
        }.items())
        places = [
            ("423025", "Roma", ["settlement"], 12.5, 41.9),
            ("540987", "Orchomenus", ["settlement"], 22.9, 38.5),
            ("857287", "Pontus", ["region"], 36.0, 41.0),
            ("501596", "Samothrace", ["settlement"], 25.5, 40.5),
            ("501597", "Samothrace", ["island"], 25.6, 40.5),
            ("550496", "Chios", ["settlement"], 26.1, 38.4),
            ("550497", "Chios", ["island"], 26.0, 38.4),
            ("606283", "Chios", ["unknown"], None, None),
        ]
        for identifier, title, types, lon, lat in places:
            connection.execute("INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (identifier, title, json.dumps(types), lon, lat,
                 f"https://pleiades.stoa.org/places/{identifier}", "fixture", "published"))
            connection.execute("INSERT INTO names (normalized_name, original_name, pleiades_id, name_resource_id, language, name_type, provenance) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (place_registry.normalize_name(title), title, identifier, f"name-{identifier}", "la", "geographic", "fixture"))
            if lon is not None:
                connection.execute("INSERT INTO locations (pleiades_id, location_id, geometry_type, accuracy, accuracy_value, provenance) VALUES (?, ?, ?, ?, ?, ?)",
                    (identifier, f"loc-{identifier}", "Point", "rough", 1000, "fixture"))
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    return path


def test_curated_override_wins_and_preserves_semantics(gazetteer):
    result = place_registry.resolve_with_status("Bituriges")
    assert result.status == "CURATED"
    assert result.places[0].id == "pleiades-138225"
    assert result.places[0].spatial_semantics is PlaceSpatialSemantics.REGION
    assert result.places[0].coordinate_role == "regional_centroid"
    assert result.places[0].uncertain is True


@pytest.mark.parametrize("name, identifier", [("Roma", "423025"), ("Orchomenus", "540987")])
def test_unique_fallback(name, identifier, gazetteer):
    result = place_registry.resolve_with_status(name)
    assert result.status == "UNIQUE"
    assert result.places[0].source_id == identifier
    assert result.places[0].coordinate_role == "representative_point"
    assert result.places[0].uncertain is True
    assert result.places[0].authority_metadata["dataset_version"] == "4.1"
    assert result.places[0].authority_metadata["matched_name_id"] == f"name-{identifier}"


def test_region_is_conservative(gazetteer):
    place = place_registry.resolve_with_status("Pontus").places[0]
    assert place.spatial_semantics is PlaceSpatialSemantics.REGION
    assert place.coordinate_role == "regional_centroid"
    assert place.coordinate_role != "exact_site"


@pytest.mark.parametrize("name, count", [("Samothrace", 2), ("Chios", 3)])
def test_ambiguity_fails_closed(name, count, gazetteer):
    result = place_registry.resolve_with_status(name)
    assert result.status == "AMBIGUOUS"
    assert result.candidate_count == count
    assert GeographyService().resolve_ancient_place(name) is None
    payload = GeographyService().resolve_ancient_place_payload(name)
    assert payload["ambiguous"] is True
    assert payload["candidate_count"] == count


def test_unknown_and_unavailable_are_explicit(gazetteer, monkeypatch, tmp_path):
    assert place_registry.resolve_with_status("No Such Ancient Place").status == "NOT_FOUND"
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(tmp_path / "missing.sqlite3"))
    result = place_registry.resolve_with_status("Roma")
    assert result.status == "UNAVAILABLE"
    assert "does not exist" in result.reason


def test_generic_fallback_never_claims_exact_site(gazetteer):
    for name in ("Roma", "Orchomenus", "Pontus"):
        assert place_registry.resolve_with_status(name).places[0].coordinate_role != "exact_site"


@pytest.mark.parametrize("name", ["Greece", "Graecia"])
def test_greece_exonyms_resolve_to_audited_hellas_authority(name):
    result = place_registry.resolve_with_status(name)
    place = result.places[0]
    assert result.status == "CURATED"
    assert place.canonical_name == "Hellas"
    assert place.source_id == "1001896"
    assert place.spatial_semantics is PlaceSpatialSemantics.REGION
    assert place.coordinate_role == "regional_centroid"
    assert place.uncertain is True


@pytest.mark.parametrize("name", ["Italy", "Italia"])
def test_italy_exonyms_resolve_to_audited_italia_authority(name):
    result = place_registry.resolve_with_status(name)
    place = result.places[0]
    assert result.status == "CURATED"
    assert place.canonical_name == "Italia"
    assert place.source_id == "1052"
    assert place.spatial_semantics is PlaceSpatialSemantics.REGION
    assert place.coordinate_role == "regional_centroid"
    assert place.uncertain is True
    assert place.source_id != "992073"


def test_italia_curated_disambiguation_beats_pleiades_diocese(gazetteer):
    assert place_registry.resolve_with_status("Italia").status == "CURATED"
    assert place_registry.resolve_with_status("Italia").places[0].source_id == "1052"
