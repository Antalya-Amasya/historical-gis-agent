"""G6DF: full diagnostic output parity harness for coordinate authority cleanup."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.app.geography import place_registry
from backend.app.geography.coordinate_authority import (
    AUTHORITATIVE_SITE_POINT,
    classify_coordinate_authority,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "g6dd_coordinate_authority_snapshots.json"
_CANONICAL_DB = _REPO_ROOT / "data" / "pleiades_v4_1" / "pleiades_v4_1.sqlite3"
_SNAPSHOTS = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _location(location_id: str, geometry: dict, **kwargs) -> dict:
    payload = {
        "location_id": location_id,
        "geometry": geometry,
        "geometry_type": geometry.get("type"),
    }
    payload.update(kwargs)
    return payload


_GENERIC_INPUTS = {
    "A": (1.0, 2.0, [_location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}, provenance="pleiades")]),
    "B": (4.5, 47.5, [_location("p1", {"type": "Point", "coordinates": [4.5, 47.5]}, location_types=["representative"])]),
    "C": (32.5, 31.0, [_location("p1", {"type": "Point", "coordinates": [32.5, 31.0]}, location_types=["central_point"])]),
    "D": (1.0, 2.0, [_location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}), _location("p2", {"type": "Point", "coordinates": [3.0, 4.0]})]),
    "E": (1.0, 2.0, [_location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}), _location("p2", {"type": "Point", "coordinates": [1.0, 2.0]})]),
    "F": (None, None, [_location("l1", {"type": "LineString", "coordinates": [[1.0, 2.0], [1.1, 2.1]]})]),
    "G": (None, None, [_location("ml1", {"type": "MultiLineString", "coordinates": [[[1.0, 2.0], [1.1, 2.1]]]})]),
    "H": (1.0, 2.0, [_location("p1", {"type": "Point", "coordinates": [1.0, 2.0]}), _location("p2", {"type": "Point", "coordinates": [3.0, 4.0]})]),
    "I": (None, None, []),
    "J": (
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
    ),
}


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


def _normalize_diagnostic(payload: dict) -> dict:
    normalized = dict(payload)
    for key in ("reason_codes", "source_location_ids", "explicit_negative_signals"):
        value = normalized.get(key)
        if isinstance(value, tuple):
            normalized[key] = list(value)
    return normalized


def _assert_full_parity(actual: dict, expected: dict, label: str) -> None:
    assert _normalize_diagnostic(actual) == expected, (
        f"{label} diagnostic output changed: {json.dumps(_normalize_diagnostic(actual), sort_keys=True)}"
    )


@pytest.mark.parametrize("case_id", sorted(_GENERIC_INPUTS))
def test_generic_matrix_full_output_parity(case_id: str):
    repr_lon, repr_lat, locations = _GENERIC_INPUTS[case_id]
    diagnostic = classify_coordinate_authority(
        representative_longitude=repr_lon,
        representative_latitude=repr_lat,
        locations=locations,
    ).to_dict()
    _assert_full_parity(diagnostic, _SNAPSHOTS["generic_cases"][case_id], case_id)


@pytest.mark.parametrize("pleiades_id", sorted(_SNAPSHOTS["real_controls"]))
def test_real_controls_full_output_parity(pleiades_id: str):
    place, locations = _load_control(pleiades_id)
    diagnostic = classify_coordinate_authority(
        representative_longitude=place["representative_lon"],
        representative_latitude=place["representative_lat"],
        locations=locations,
    ).to_dict()
    _assert_full_parity(diagnostic, _SNAPSHOTS["real_controls"][pleiades_id], pleiades_id)


def test_bounded_sample_distribution_parity():
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
            diagnostic = classify_coordinate_authority(
                representative_longitude=row["representative_lon"],
                representative_latitude=row["representative_lat"],
                locations=locations,
            )
            distribution[diagnostic.authority_class] = distribution.get(diagnostic.authority_class, 0) + 1
            assert diagnostic.authority_class != AUTHORITATIVE_SITE_POINT
    finally:
        connection.close()
    assert distribution == _SNAPSHOTS["sample_distribution"]
