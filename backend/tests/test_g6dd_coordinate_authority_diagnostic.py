"""G6DD: diagnostic-only coordinate authority classifier over Pleiades v3 locations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from backend.app.geography.coordinate_authority import (
    AUTHORITATIVE_SITE_POINT,
    MULTI_LOCATION_CONFLICT,
    NON_POINT_GEOMETRY,
    REPRESENTATIVE_ONLY,
    SOURCE_POINT_UNCERTAIN,
    UNKNOWN,
    classify_coordinate_authority,
)
from backend.app.geography.feature_semantics import coordinate_role_for_semantics

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_DB = _REPO_ROOT / "data" / "pleiades_v4_1" / "pleiades_v4_1.sqlite3"


def _location(
    location_id: str,
    geometry: dict,
    *,
    location_types: list[str] | None = None,
    provenance: str | None = None,
    accuracy: str | None = None,
    title: str | None = None,
    description: str | None = None,
) -> dict:
    payload = {
        "location_id": location_id,
        "geometry": geometry,
        "geometry_type": geometry.get("type"),
    }
    if location_types is not None:
        payload["location_types"] = location_types
    if provenance is not None:
        payload["provenance"] = provenance
    if accuracy is not None:
        payload["accuracy"] = accuracy
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    return payload


def _classify(
    *,
    repr_lon: float | None,
    repr_lat: float | None,
    locations: list[dict],
):
    return classify_coordinate_authority(
        representative_longitude=repr_lon,
        representative_latitude=repr_lat,
        locations=locations,
    )


def _load_control(pleiades_id: str):
    if not _CANONICAL_DB.is_file():
        pytest.skip("canonical Pleiades v3 index missing")
    connection = sqlite3.connect(_CANONICAL_DB)
    connection.row_factory = sqlite3.Row
    try:
        place = connection.execute(
            "SELECT representative_lon, representative_lat FROM places WHERE pleiades_id = ?",
            (pleiades_id,),
        ).fetchone()
        locations = [
            place_registry._location_metadata(row)
            for row in connection.execute(
                """SELECT location_id, geometry_type, accuracy, accuracy_value, provenance,
                          geometry_json, title, description, start, end,
                          attestations_json, feature_types_json, location_types_json, references_json
                     FROM locations WHERE pleiades_id = ? ORDER BY location_id""",
                (pleiades_id,),
            )
        ]
    finally:
        connection.close()
    return place, locations


@pytest.mark.parametrize(
    ("case_id", "repr_lon", "repr_lat", "locations", "expected_class"),
    [
        (
            "A_single_point_no_proof",
            1.0,
            2.0,
            [_location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}, provenance="pleiades")],
            SOURCE_POINT_UNCERTAIN,
        ),
        (
            "B_representative_location_type",
            4.5,
            47.5,
            [_location("p1", {"type": "Point", "coordinates": [4.5, 47.5]}, location_types=["representative"])],
            REPRESENTATIVE_ONLY,
        ),
        (
            "C_central_point_location_type",
            32.5,
            31.0,
            [_location("p1", {"type": "Point", "coordinates": [32.5, 31.0]}, location_types=["central_point"])],
            REPRESENTATIVE_ONLY,
        ),
        (
            "D_two_conflicting_points",
            1.0,
            2.0,
            [
                _location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}),
                _location("p2", {"type": "Point", "coordinates": [3.0, 4.0]}),
            ],
            MULTI_LOCATION_CONFLICT,
        ),
        (
            "E_identical_points_not_conflict",
            1.0,
            2.0,
            [
                _location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}),
                _location("p2", {"type": "Point", "coordinates": [1.0, 2.0]}),
            ],
            SOURCE_POINT_UNCERTAIN,
        ),
        (
            "F_linestring_only",
            None,
            None,
            [_location("l1", {"type": "LineString", "coordinates": [[1.0, 2.0], [1.1, 2.1]]})],
            NON_POINT_GEOMETRY,
        ),
        (
            "G_multilinestring_only",
            None,
            None,
            [
                _location(
                    "ml1",
                    {"type": "MultiLineString", "coordinates": [[[1.0, 2.0], [1.1, 2.1]]]},
                )
            ],
            NON_POINT_GEOMETRY,
        ),
        (
            "I_missing_metadata",
            None,
            None,
            [],
            UNKNOWN,
        ),
        (
            "J_osm_point_alone",
            12.0,
            41.0,
            [
                _location(
                    "osm1",
                    {"type": "Point", "coordinates": [12.0, 41.0]},
                    provenance="OpenStreetMap",
                    accuracy="https://pleiades.stoa.org/places/accuracy/openstreetmap",
                )
            ],
            SOURCE_POINT_UNCERTAIN,
        ),
    ],
)
def test_generic_matrix(case_id, repr_lon, repr_lat, locations, expected_class):
    del case_id
    diagnostic = _classify(repr_lon=repr_lon, repr_lat=repr_lat, locations=locations)
    assert diagnostic.authority_class == expected_class
    assert diagnostic.authority_class != AUTHORITATIVE_SITE_POINT


def test_reprpoint_matches_one_conflicting_point_is_not_authoritative():
    diagnostic = _classify(
        repr_lon=1.0,
        repr_lat=2.0,
        locations=[
            _location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}),
            _location("p2", {"type": "Point", "coordinates": [3.0, 4.0]}),
        ],
    )
    assert diagnostic.authority_class == MULTI_LOCATION_CONFLICT
    assert diagnostic.repr_point_relation == "MATCHES_ONE_OF_CONFLICTING_POINTS"
    assert diagnostic.authority_class != AUTHORITATIVE_SITE_POINT


@pytest.mark.parametrize(
    ("pleiades_id", "expected_class"),
    [
        ("177434", REPRESENTATIVE_ONLY),
        ("727192", REPRESENTATIVE_ONLY),
        ("138373", REPRESENTATIVE_ONLY),
        ("109126", REPRESENTATIVE_ONLY),
        ("177528", REPRESENTATIVE_ONLY),
        ("422995", NON_POINT_GEOMETRY),
        ("442523", NON_POINT_GEOMETRY),
        ("29457", MULTI_LOCATION_CONFLICT),
        ("727070", REPRESENTATIVE_ONLY),
    ],
)
def test_real_controls(pleiades_id, expected_class):
    place, locations = _load_control(pleiades_id)
    diagnostic = _classify(
        repr_lon=place["representative_lon"],
        repr_lat=place["representative_lat"],
        locations=locations,
    )
    assert diagnostic.authority_class == expected_class
    assert diagnostic.policy_version == "g6dd-v1"


def test_false_positive_sample_has_zero_authoritative_promotions():
    if not _CANONICAL_DB.is_file():
        pytest.skip("canonical Pleiades v3 index missing")
    connection = sqlite3.connect(_CANONICAL_DB)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT p.pleiades_id, p.representative_lon, p.representative_lat
                 FROM places AS p
                 JOIN locations AS l USING (pleiades_id)
                WHERE p.representative_lon IS NOT NULL
                  AND p.representative_lat IS NOT NULL
                  AND l.geometry_type = 'Point'
                  AND l.accuracy IS NOT NULL
                  AND l.provenance IS NOT NULL
                GROUP BY p.pleiades_id
               HAVING COUNT(*) = 1
                LIMIT 250"""
        ).fetchall()
        distribution: dict[str, int] = {}
        for row in rows:
            _, locations = _load_control(row["pleiades_id"])
            diagnostic = _classify(
                repr_lon=row["representative_lon"],
                repr_lat=row["representative_lat"],
                locations=locations,
            )
            distribution[diagnostic.authority_class] = distribution.get(diagnostic.authority_class, 0) + 1
            assert diagnostic.authority_class != AUTHORITATIVE_SITE_POINT
    finally:
        connection.close()
    assert distribution.get(AUTHORITATIVE_SITE_POINT, 0) == 0


@pytest.mark.parametrize(
    "pleiades_id",
    ["177434", "727192", "138373", "109126", "177528", "422995", "442523", "29457", "727070"],
)
def test_coordinate_role_unchanged_for_real_controls(pleiades_id):
    place, locations = _load_control(pleiades_id)
    connection = sqlite3.connect(_CANONICAL_DB)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT title, place_types_json FROM places WHERE pleiades_id = ?",
            (pleiades_id,),
        ).fetchone()
    finally:
        connection.close()
    place_types = tuple(json.loads(row["place_types_json"] or "[]"))
    semantics = place_registry._spatial_semantics(place_types, title=row["title"] or "")
    expected_role = coordinate_role_for_semantics(
        semantics, place_types=place_types, title=row["title"] or ""
    )
    diagnostic = _classify(
        repr_lon=place["representative_lon"],
        repr_lat=place["representative_lat"],
        locations=locations,
    )
    assert diagnostic.authority_class != AUTHORITATIVE_SITE_POINT
    assert expected_role == coordinate_role_for_semantics(
        semantics, place_types=place_types, title=row["title"] or ""
    )


def test_place_registry_exposes_diagnostic_without_changing_coordinate_role(tmp_path, monkeypatch):
    from backend.tests.test_g6dc_pleiades_location_type_source_key_fix import _build_fixture_index

    path = _build_fixture_index(
        tmp_path,
        [{
            "id": "999901",
            "title": "Diagnostic Fixture",
            "placeTypes": ["settlement"],
            "reprPoint": [1.0, 2.0],
            "names": [{"attested": "DiagFixture", "romanized": "DiagFixture"}],
            "locations": [{
                "id": "diag-point",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "locationType": ["representative"],
                "provenance": "fixture",
            }],
        }],
    )
    place_registry.records.cache_clear()
    place_registry.places.cache_clear()
    place_registry.aliases.cache_clear()
    monkeypatch.setenv("PLEIADES_GAZETTEER_PATH", str(path))
    result = place_registry.resolve_with_status("DiagFixture")
    assert result.status == "UNIQUE"
    place = result.places[0]
    diagnostic = place.authority_metadata["coordinate_authority_diagnostic"]
    assert diagnostic["authority_class"] == REPRESENTATIVE_ONLY
    assert diagnostic["policy_version"] == "g6dd-v1"
    assert place.coordinate_role == "representative_point"
