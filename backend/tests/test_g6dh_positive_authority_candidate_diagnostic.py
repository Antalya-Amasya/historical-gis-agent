"""G6DH: diagnostic-only positive authority candidate evaluation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from backend.app.geography.coordinate_authority import (
    MULTI_LOCATION_CONFLICT,
    NON_POINT_GEOMETRY,
    REPRESENTATIVE_ONLY,
    SOURCE_POINT_UNCERTAIN,
    classify_coordinate_authority,
)
from backend.app.geography.positive_authority_candidate import (
    STRENGTH_NONE,
    STRENGTH_STRONG,
    STRENGTH_SUPPORTING,
    evaluate_positive_authority_candidate,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_DB = _REPO_ROOT / "data" / "pleiades_v4_1" / "pleiades_v4_1.sqlite3"


def _location(location_id: str, geometry: dict, **kwargs) -> dict:
    payload = {"location_id": location_id, "geometry": geometry, "geometry_type": geometry.get("type")}
    payload.update(kwargs)
    return payload


def _evaluate(*, repr_lon, repr_lat, locations):
    diagnostic = classify_coordinate_authority(
        representative_longitude=repr_lon,
        representative_latitude=repr_lat,
        locations=locations,
    )
    return diagnostic, evaluate_positive_authority_candidate(
        representative_longitude=repr_lon,
        representative_latitude=repr_lat,
        locations=locations,
        authority_diagnostic=diagnostic,
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


def test_excavation_address_control_is_strong_candidate():
    place, locations = _load_control("534036732")
    diagnostic, candidate = _evaluate(
        repr_lon=place["representative_lon"],
        repr_lat=place["representative_lat"],
        locations=locations,
    )
    assert diagnostic.authority_class == SOURCE_POINT_UNCERTAIN
    assert candidate.is_candidate is True
    assert candidate.strength == STRENGTH_STRONG
    assert "EXCAVATION_SITE_LOCATION" in candidate.positive_reasons


def test_representative_plus_excavation_description_is_blocked():
    diagnostic, candidate = _evaluate(
        repr_lon=4.5,
        repr_lat=47.5,
        locations=[
            _location(
                "p1",
                {"type": "Point", "coordinates": [4.5, 47.5]},
                location_types=["representative"],
                description="Excavations on site revealed Hellenistic remains.",
            )
        ],
    )
    assert diagnostic.authority_class == REPRESENTATIVE_ONLY
    assert candidate.is_candidate is False
    assert candidate.strength == STRENGTH_NONE
    assert candidate.blocking_reasons


def test_central_point_plus_archaeology_type_is_blocked():
    diagnostic, candidate = _evaluate(
        repr_lon=32.5,
        repr_lat=31.0,
        locations=[
            _location(
                "p1",
                {"type": "Point", "coordinates": [32.5, 31.0]},
                location_types=["central_point"],
                feature_types=["archaeological-site"],
                provenance="OpenStreetMap",
            )
        ],
    )
    assert diagnostic.authority_class == REPRESENTATIVE_ONLY
    assert candidate.is_candidate is False
    assert "BLOCKING_LOCATION_TYPE_CENTRAL_POINT" in candidate.blocking_reasons


def test_archaeological_site_type_only_is_not_strong():
    diagnostic, candidate = _evaluate(
        repr_lon=1.0,
        repr_lat=2.0,
        locations=[
            _location(
                "p1",
                {"type": "Point", "coordinates": [1.0, 2.0]},
                feature_types=["archaeological-site"],
            )
        ],
    )
    assert diagnostic.authority_class == SOURCE_POINT_UNCERTAIN
    assert candidate.strength != STRENGTH_STRONG


def test_osm_point_only_is_none():
    diagnostic, candidate = _evaluate(
        repr_lon=12.0,
        repr_lat=41.0,
        locations=[
            _location(
                "osm1",
                {"type": "Point", "coordinates": [12.0, 41.0]},
                provenance="OpenStreetMap",
                accuracy="https://pleiades.stoa.org/places/accuracy/openstreetmap",
            )
        ],
    )
    assert candidate.is_candidate is False
    assert candidate.strength == STRENGTH_NONE
    assert "OSM_SOURCE_ONLY" in candidate.blocking_reasons


def test_non_point_geometry_is_none():
    diagnostic, candidate = _evaluate(
        repr_lon=None,
        repr_lat=None,
        locations=[_location("l1", {"type": "LineString", "coordinates": [[1.0, 2.0], [1.1, 2.1]]})],
    )
    assert diagnostic.authority_class == NON_POINT_GEOMETRY
    assert candidate.is_candidate is False


def test_multi_location_conflict_is_none():
    diagnostic, candidate = _evaluate(
        repr_lon=1.0,
        repr_lat=2.0,
        locations=[
            _location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}),
            _location("p2", {"type": "Point", "coordinates": [3.0, 4.0]}),
        ],
    )
    assert diagnostic.authority_class == MULTI_LOCATION_CONFLICT
    assert candidate.is_candidate is False


def test_generic_scholarly_reference_is_not_strong():
    diagnostic, candidate = _evaluate(
        repr_lon=1.0,
        repr_lat=2.0,
        locations=[
            _location(
                "p1",
                {"type": "Point", "coordinates": [1.0, 2.0]},
                references=[{"formattedCitation": "Barrington Atlas Map 44", "shortTitle": "BAtlas"}],
            )
        ],
    )
    assert candidate.strength != STRENGTH_STRONG


@pytest.mark.parametrize(
    ("pleiades_id",),
    [
        ("177434",),
        ("727192",),
        ("100447491",),
        ("102024695",),
        ("103018178",),
        ("103328498",),
    ],
)
def test_negative_controls_remain_non_candidates(pleiades_id: str):
    place, locations = _load_control(pleiades_id)
    _, candidate = _evaluate(
        repr_lon=place["representative_lon"],
        repr_lat=place["representative_lat"],
        locations=locations,
    )
    assert candidate.is_candidate is False
    assert candidate.strength == STRENGTH_NONE


def test_bounded_sample_has_few_strong_candidates():
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
        distribution = {STRENGTH_NONE: 0, STRENGTH_SUPPORTING: 0, STRENGTH_STRONG: 0}
        for row in rows:
            place, locations = _load_control(row["pleiades_id"])
            _, candidate = _evaluate(
                repr_lon=place["representative_lon"],
                repr_lat=place["representative_lat"],
                locations=locations,
            )
            distribution[candidate.strength] = distribution.get(candidate.strength, 0) + 1
    finally:
        connection.close()
    assert distribution[STRENGTH_STRONG] <= 2
