import hashlib
import json
import sqlite3
import zipfile

import pytest

from scripts.build_pleiades_index import INDEX_SCHEMA_VERSION, build_index, normalize_name


def test_streaming_builder_schema_normalization_and_metadata(tmp_path):
    source = tmp_path / "source.zip"
    place = {
        "id": "1", "title": " Rōma  Nova ", "placeTypes": ["settlement"],
        "reprPoint": [12.5, 41.9], "uri": "https://example/1", "provenance": "fixture",
        "review_state": "published",
        "names": [{"id": "n1", "attested": "Rōma\u00a0 Nova", "romanized": ["Roma Nova"],
                   "language": "la", "nameType": "geographic", "provenance": "name fixture"}],
        "locations": [{"id": "l1", "title": "Site point", "description": "fixture point",
                       "geometry": {"type": "Point", "coordinates": [12.5, 41.9]},
                       "accuracy": "rough", "accuracy_value": 1000, "provenance": "location fixture",
                       "attestations": [], "featureType": ["settlement"], "locationType": ["central_point"],
                       "references": [{"shortTitle": "Fixture"}]}],
    }
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("root/data/json/1.json", json.dumps(place))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "index.sqlite3"
    summary = build_index(source, output, expected_sha256=digest, expected_place_count=1)
    assert summary["place_count"] == 1
    with sqlite3.connect(output) as connection:
        names = {row[0] for row in connection.execute("SELECT normalized_name FROM names")}
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert normalize_name(" RŌMA\u00a0 NOVA ") in names
        assert "roma nova" in names
        assert metadata["dataset_version"] == "4.1"
        assert metadata["sha256"] == digest
        assert metadata["index_schema_version"] == INDEX_SCHEMA_VERSION
        row = connection.execute(
            "SELECT location_id, geometry_json, title, references_json FROM locations"
        ).fetchone()
        assert row[0] == "l1"
        assert json.loads(row[1]) == {"type": "Point", "coordinates": [12.5, 41.9]}
        assert row[2] == "Site point"
        assert json.loads(row[3]) == [{"shortTitle": "Fixture"}]
        types = sqlite3.connect(output).execute(
            "SELECT location_types_json, feature_types_json FROM locations"
        ).fetchone()
        assert json.loads(types[0]) == ["central_point"]
        assert json.loads(types[1]) == ["settlement"]


def test_builder_rejects_wrong_identity(tmp_path):
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("root/data/json/1.json", "{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_index(source, tmp_path / "index.sqlite3", expected_sha256="0" * 64, expected_place_count=1)
