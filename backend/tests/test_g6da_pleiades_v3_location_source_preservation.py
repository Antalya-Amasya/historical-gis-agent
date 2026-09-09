"""G6DA: Pleiades index schema v3 location source-preservation tests."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from backend.app.geography.feature_semantics import coordinate_role_for_semantics
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics
from scripts.build_pleiades_index import (
    EXPECTED_SHA256,
    INDEX_SCHEMA_VERSION,
    build_index,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_ZIP = _REPO_ROOT / "data" / "pleiades_v4_1" / "pleiades.datasets-v4.1.zip"
_CODEX_ZIP = Path(r"C:\D\python\historical-gis-codex\data\pleiades_v4_1\pleiades.datasets-v4.1.zip")
_V2_BASELINE = _REPO_ROOT / "backend" / "tests" / "fixtures" / "pleiades_index_v2_baseline.sqlite3"


def _source_zip() -> Path:
    if _CANONICAL_ZIP.is_file():
        return _CANONICAL_ZIP
    if _CODEX_ZIP.is_file():
        return _CODEX_ZIP
    pytest.skip("Pleiades source archive not available")


def _build_zip(tmp_path: Path, places: list[dict]) -> Path:
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for place in places:
            archive.writestr(
                f"root/data/json/{place['id']}.json",
                json.dumps(place, ensure_ascii=False, separators=(",", ":")),
            )
    return source


def _location_rows(path: Path, pleiades_id: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM locations WHERE pleiades_id = ? ORDER BY location_id",
        (pleiades_id,),
    ).fetchall()
    connection.close()
    return rows


def _build_fixture_index(tmp_path: Path, places: list[dict]) -> Path:
    source = _build_zip(tmp_path, places)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "index.sqlite3"
    build_index(source, output, expected_sha256=digest, expected_place_count=len(places))
    return output


def test_v3_schema_columns_exist(tmp_path):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "1",
            "title": "Fixture",
            "placeTypes": ["settlement"],
            "reprPoint": [1.0, 2.0],
            "locations": [],
        }],
    )
    columns = {
        row[1]
        for row in sqlite3.connect(path).execute("PRAGMA table_info(locations)")
    }
    assert columns >= {
        "geometry_json",
        "title",
        "description",
        "start",
        "end",
        "attestations_json",
        "feature_types_json",
        "location_types_json",
        "references_json",
    }
    metadata = dict(sqlite3.connect(path).execute("SELECT key, value FROM metadata"))
    assert metadata["index_schema_version"] == "3"


def test_point_geometry_is_preserved_exactly(tmp_path):
    place = {
        "id": "177434",
        "title": "Alesia",
        "placeTypes": ["settlement"],
        "reprPoint": [4.503884, 47.536622],
        "locations": [{
            "id": "center-point",
            "title": "Center point",
            "description": "Exact fixture point",
            "geometry": {"type": "Point", "coordinates": [4.503884, 47.536622]},
            "accuracy": "https://example.invalid/accuracy",
            "accuracy_value": 5.0,
            "provenance": "Pleiades",
            "locationType": ["representative"],
            "references": [{"shortTitle": "FixtureRef", "type": "citesAsDataSource"}],
            "attestations": [],
            "featureType": ["unknown"],
        }],
    }
    path = _build_fixture_index(tmp_path, [place])
    row = _location_rows(path, "177434")[0]
    geometry = json.loads(row["geometry_json"])
    assert row["geometry_type"] == "Point"
    assert geometry == {"type": "Point", "coordinates": [4.503884, 47.536622]}


def test_non_point_geometry_remains_non_point(tmp_path):
    place = {
        "id": "442523",
        "title": "Cannae",
        "placeTypes": ["settlement"],
        "reprPoint": [16.0, 41.0],
        "locations": [{
            "id": "cannae-line",
            "geometry": {
                "type": "LineString",
                "coordinates": [[16.0, 41.0], [16.1, 41.1], [16.2, 41.0]],
            },
            "accuracy": "rough",
            "provenance": "fixture",
        }],
    }
    path = _build_fixture_index(tmp_path, [place])
    row = _location_rows(path, "442523")[0]
    geometry = json.loads(row["geometry_json"])
    assert row["geometry_type"] == "LineString"
    assert geometry["type"] == "LineString"
    assert len(geometry["coordinates"]) == 3


def test_multiple_locations_preserved_independently(tmp_path):
    place = {
        "id": "138373",
        "title": "Gergovia?",
        "placeTypes": ["settlement"],
        "reprPoint": [3.1241435, 45.718167],
        "locations": [
            {
                "id": "darmc-location-12528",
                "title": "DARMC location 12528",
                "geometry": {"type": "Point", "coordinates": [3.129429, 45.716242]},
                "provenance": "DARMC",
            },
            {
                "id": "dare-location",
                "title": "DARE Location",
                "geometry": {"type": "Point", "coordinates": [3.118858, 45.720092]},
                "provenance": "DARE",
            },
        ],
    }
    path = _build_fixture_index(tmp_path, [place])
    rows = _location_rows(path, "138373")
    assert [row["location_id"] for row in rows] == ["dare-location", "darmc-location-12528"]
    assert json.loads(rows[0]["geometry_json"])["coordinates"] == [3.118858, 45.720092]
    assert json.loads(rows[1]["geometry_json"])["coordinates"] == [3.129429, 45.716242]


def test_title_description_start_end_attestations_types_and_references_preserved(tmp_path):
    place = {
        "id": "727192",
        "title": "Pelusium",
        "placeTypes": ["settlement"],
        "reprPoint": [32.5473404, 31.0427587],
        "locations": [{
            "id": "ancient-roman-city-of-pelusium",
            "title": "OSM Location: Ancient Roman City of Pelusium",
            "description": "Location based on OpenStreetMap. Dates after BAtlas.",
            "start": -750,
            "end": 640,
            "geometry": {"type": "Point", "coordinates": [32.5473404, 31.0427587]},
            "accuracy": "https://pleiades.stoa.org/features/metadata/generic-osm-accuracy-assessment",
            "provenance": "OpenStreetMap (Node 6069103273)",
            "locationType": ["central_point"],
            "featureType": ["settlement", "port"],
            "attestations": [{
                "confidence": "confident",
                "confidenceURI": "https://pleiades.stoa.org/vocabularies/attestation-confidence/confident",
                "timePeriod": "roman",
                "timePeriodURI": "https://pleiades.stoa.org/vocabularies/time-periods/roman",
            }],
            "references": [{
                "shortTitle": "OSM",
                "type": "citesAsDataSource",
                "formattedCitation": "osm:node=6069103273",
            }],
        }],
    }
    path = _build_fixture_index(tmp_path, [place])
    row = _location_rows(path, "727192")[0]
    assert row["title"] == "OSM Location: Ancient Roman City of Pelusium"
    assert row["description"] == "Location based on OpenStreetMap. Dates after BAtlas."
    assert row["start"] == -750
    assert row["end"] == 640
    assert json.loads(row["attestations_json"]) == [{
        "confidence": "confident",
        "confidenceURI": "https://pleiades.stoa.org/vocabularies/attestation-confidence/confident",
        "timePeriod": "roman",
        "timePeriodURI": "https://pleiades.stoa.org/vocabularies/time-periods/roman",
    }]
    assert json.loads(row["feature_types_json"]) == ["settlement", "port"]
    assert json.loads(row["location_types_json"]) == ["central_point"]
    assert json.loads(row["references_json"]) == [{
        "shortTitle": "OSM",
        "type": "citesAsDataSource",
        "formattedCitation": "osm:node=6069103273",
    }]


def test_schema_v2_index_is_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "v2.sqlite3"
    shutil.copy2(_V2_BASELINE, path)
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    result = place_registry.resolve_with_status("Roma")
    assert result.status == "UNAVAILABLE"
    assert "Unsupported Pleiades index schema: '2'" in (result.reason or "")


def test_runtime_exposes_preserved_location_metadata(tmp_path, monkeypatch):
    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "999001",
            "title": "Fixture Port",
            "placeTypes": ["settlement", "port"],
            "reprPoint": [12.28, 41.76],
            "names": [{"attested": "Fixture Port", "romanized": "Fixture Port"}],
            "locations": [{
                "id": "fixture-port-point",
                "title": "Harbor point",
                "geometry": {"type": "Point", "coordinates": [12.28, 41.76]},
                "accuracy": "exact",
                "provenance": "fixture",
                "references": [{"shortTitle": "Ref"}],
            }],
        }],
    )
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    result = place_registry.resolve_with_status("Fixture Port")
    assert result.status == "UNIQUE"
    locations = result.places[0].authority_metadata["locations"]
    assert locations[0]["geometry"]["coordinates"] == [12.28, 41.76]
    assert locations[0]["references"] == [{"shortTitle": "Ref"}]


def _resolution_signature(path: Path, name: str, *, schema: str) -> dict:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")}
        if metadata.get("index_schema_version") != schema:
            raise RuntimeError(f"unexpected schema {metadata.get('index_schema_version')}")
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
            (place_registry.normalize_name(name),),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["pleiades_id"], []).append(row)
        if not grouped:
            return {"status": "NOT_FOUND", "candidate_count": 0, "candidates": [], "places": []}

        resolved_places = []
        candidate_summaries = []
        for pleiades_id, name_rows in grouped.items():
            first = name_rows[0]
            place_types = tuple(json.loads(first["place_types_json"] or "[]"))
            coordinate_available = (
                first["representative_lon"] is not None
                and first["representative_lat"] is not None
            )
            candidate_summaries.append(
                {
                    "pleiades_id": pleiades_id,
                    "coordinate_available": coordinate_available,
                    "matched_names": list(dict.fromkeys(row["original_name"] for row in name_rows)),
                }
            )
            if not coordinate_available:
                continue
            semantics = place_registry._spatial_semantics(place_types, title=first["title"] or "")
            coordinate_role = coordinate_role_for_semantics(
                semantics, place_types=place_types, title=first["title"] or ""
            )
            resolved_places.append(
                {
                    "pleiades_id": pleiades_id,
                    "latitude": first["representative_lat"],
                    "longitude": first["representative_lon"],
                    "coordinate_role": coordinate_role,
                    "spatial_semantics": semantics.value,
                }
            )
        status = "UNIQUE" if len(grouped) == 1 else "AMBIGUOUS"
        if status == "UNIQUE" and not resolved_places:
            status = "UNLOCATED"
        return {
            "status": status,
            "candidate_count": len(grouped),
            "candidates": candidate_summaries,
            "places": resolved_places,
        }
    finally:
        connection.close()


@pytest.mark.parametrize("name", ["Pelusium", "Cilicia", "Lutetia", "Alexandria"])
def test_v3_index_resolution_parity_with_v2_baseline(name, tmp_path):
    if not _V2_BASELINE.is_file():
        pytest.skip("v2 baseline index missing")
    v3_path = tmp_path / "v3.sqlite3"
    build_index(_source_zip(), v3_path)
    v2_signature = _resolution_signature(_V2_BASELINE, name, schema="2")
    v3_signature = _resolution_signature(v3_path, name, schema="3")
    assert v3_signature == v2_signature


@pytest.mark.parametrize("name", ["Cyprus", "Alesia"])
def test_curated_controls_remain_unchanged(name):
    result = place_registry.resolve_with_status(name)
    assert result.status == "CURATED"
    assert result.places
    assert result.places[0].coordinate_role in {"feature_centroid", "representative_point", "regional_centroid", "exact_site"}


def test_g6cx_guards_remain_unchanged():
    from backend.app.geography.feature_semantics import exact_anchor_eligible

    blocked = [
        (PlaceSpatialSemantics.REGION, "regional_centroid"),
        (PlaceSpatialSemantics.ISLAND, "feature_centroid"),
        (PlaceSpatialSemantics.SETTLEMENT, "representative_point"),
    ]
    for semantics, role in blocked:
        place = HistoricalPlace(
            id="x",
            canonical_name="X",
            latitude=1.0,
            longitude=2.0,
            source="fixture",
            confidence=0.8,
            spatial_semantics=semantics,
            coordinate_role=role,
        )
        assert exact_anchor_eligible(place, strong_role=True) is False
    allowed = HistoricalPlace(
        id="port",
        canonical_name="Port",
        latitude=1.0,
        longitude=2.0,
        source="fixture",
        confidence=0.8,
        spatial_semantics=PlaceSpatialSemantics.PORT,
        coordinate_role="exact_site",
    )
    assert exact_anchor_eligible(allowed, strong_role=True) is True


@pytest.mark.skipif(not _source_zip().is_file(), reason="Pleiades source archive not available")
def test_real_controls_from_source_archive(tmp_path):
    output = tmp_path / "controls.sqlite3"
    summary = build_index(_source_zip(), output)
    assert summary["index_schema_version"] == "3"
    metadata = dict(sqlite3.connect(output).execute("SELECT key, value FROM metadata"))
    assert metadata["sha256"] == EXPECTED_SHA256
    assert metadata["dataset_version"] == "4.1"
    assert metadata["dataset_release_date"] == "2025-05-28"
    assert int(metadata["place_count"]) == 41480
    assert int(summary["location_count"]) == int(metadata["location_count"])

    alesia = _location_rows(output, "177434")[0]
    assert alesia["geometry_type"] == "Point"
    assert json.loads(alesia["geometry_json"])["coordinates"] == [4.503884, 47.536622]
    assert alesia["location_id"] == "center-point-of-the-site-alesia-commune-alise-sainte-reine"
    assert alesia["accuracy"] == "https://pleiades.stoa.org/features/metadata/google-geoeye-2011"
    assert alesia["provenance"] == "Pleiades"
    assert alesia["description"]
    assert json.loads(alesia["location_types_json"]) == ["representative"]
    assert json.loads(alesia["feature_types_json"]) == ["unknown"]

    pelusium = _location_rows(output, "727192")[0]
    assert json.loads(pelusium["geometry_json"])["coordinates"] == [32.5473404, 31.0427587]
    assert pelusium["start"] == -750
    assert pelusium["end"] == 640
    assert json.loads(pelusium["attestations_json"])
    assert json.loads(pelusium["references_json"])
    assert json.loads(pelusium["location_types_json"]) == ["central_point"]
    assert json.loads(pelusium["feature_types_json"]) == ["settlement", "archaeological-site"]

    gergovia = _location_rows(output, "138373")
    assert len(gergovia) == 2
    assert {row["location_id"] for row in gergovia} == {"darmc-location-12528", "dare-location"}
    assert all(json.loads(row["location_types_json"]) == ["representative"] for row in gergovia)

    lutetia = _location_rows(output, "109126")
    genava = _location_rows(output, "177528")
    assert len(lutetia) == 2
    assert len(genava) == 2

    cannae = _location_rows(output, "442523")[0]
    assert cannae["geometry_type"] == "LineString"
    assert json.loads(cannae["geometry_json"])["type"] == "LineString"

    ostia = _location_rows(output, "422995")[0]
    assert ostia["geometry_type"] == "MultiLineString"
    assert json.loads(ostia["geometry_json"])["type"] == "MultiLineString"

    repr_a = sqlite3.connect(output).execute(
        "SELECT representative_lon, representative_lat FROM places WHERE pleiades_id = ?",
        ("177434",),
    ).fetchone()
    assert repr_a == (4.503884, 47.536622)
