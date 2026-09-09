"""G6DC: canonical Pleiades locationType / featureType source-key ingestion fix."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from scripts.build_pleiades_index import build_index

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODEX_ZIP = Path(r"C:\D\python\historical-gis-codex\data\pleiades_v4_1\pleiades.datasets-v4.1.zip")
_CANONICAL_ZIP = _REPO_ROOT / "data" / "pleiades_v4_1" / "pleiades.datasets-v4.1.zip"


def _source_zip() -> Path:
    if _CANONICAL_ZIP.is_file():
        return _CANONICAL_ZIP
    if _CODEX_ZIP.is_file():
        return _CODEX_ZIP
    pytest.skip("Pleiades source archive not available")


def _build_fixture_index(tmp_path: Path, places: list[dict]) -> Path:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for place in places:
            archive.writestr(
                f"root/data/json/{place['id']}.json",
                json.dumps(place, ensure_ascii=False, separators=(",", ":")),
            )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "index.sqlite3"
    build_index(source, output, expected_sha256=digest, expected_place_count=len(places))
    return output


def test_canonical_singular_locationType_and_featureType_are_ingested(tmp_path):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "177434",
            "title": "Alesia",
            "placeTypes": ["settlement"],
            "reprPoint": [4.503884, 47.536622],
            "locations": [{
                "id": "alesia-point",
                "geometry": {"type": "Point", "coordinates": [4.503884, 47.536622]},
                "locationType": ["representative"],
                "featureType": ["unknown"],
            }],
        }],
    )
    row = sqlite3.connect(path).execute(
        "SELECT location_types_json, feature_types_json FROM locations WHERE pleiades_id = ?",
        ("177434",),
    ).fetchone()
    assert json.loads(row[0]) == ["representative"]
    assert json.loads(row[1]) == ["unknown"]


def test_missing_locationType_and_featureType_remain_empty_arrays(tmp_path):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "999002",
            "title": "Plain Place",
            "placeTypes": ["settlement"],
            "reprPoint": [1.0, 2.0],
            "locations": [{
                "id": "plain-point",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "provenance": "fixture",
            }],
        }],
    )
    row = sqlite3.connect(path).execute(
        "SELECT location_types_json, feature_types_json FROM locations WHERE pleiades_id = ?",
        ("999002",),
    ).fetchone()
    assert json.loads(row[0]) == []
    assert json.loads(row[1]) == []


def test_plural_source_keys_are_not_ingested(tmp_path):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "999003",
            "title": "Wrong Keys",
            "placeTypes": ["settlement"],
            "reprPoint": [1.0, 2.0],
            "locations": [{
                "id": "wrong-keys",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "locationTypes": ["central_point"],
                "featureTypes": ["settlement"],
            }],
        }],
    )
    row = sqlite3.connect(path).execute(
        "SELECT location_types_json, feature_types_json FROM locations WHERE pleiades_id = ?",
        ("999003",),
    ).fetchone()
    assert json.loads(row[0]) == []
    assert json.loads(row[1]) == []


def test_runtime_authority_metadata_exposes_corrected_location_types(tmp_path, monkeypatch):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "727192",
            "title": "Pelusium",
            "placeTypes": ["settlement", "port"],
            "reprPoint": [32.5473404, 31.0427587],
            "names": [{"attested": "Pelusium", "romanized": "Pelusium"}],
            "locations": [{
                "id": "pelusium-point",
                "geometry": {"type": "Point", "coordinates": [32.5473404, 31.0427587]},
                "locationType": ["central_point"],
                "featureType": ["settlement", "archaeological-site"],
            }],
        }],
    )
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    result = place_registry.resolve_with_status("Pelusium")
    assert result.status == "UNIQUE"
    location = result.places[0].authority_metadata["locations"][0]
    assert location["location_types"] == ["central_point"]
    assert location["feature_types"] == ["settlement", "archaeological-site"]
