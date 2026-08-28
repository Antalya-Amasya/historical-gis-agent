import hashlib
import importlib.util
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "acquire_itiner_e.py"
SPEC = importlib.util.spec_from_file_location("acquire_itiner_e", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class Response(BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def opener(payload: bytes):
    return lambda _url, timeout: Response(payload)


def test_streamed_download_writes_atomically_with_hash_and_metadata(tmp_path):
    payload = b'{"type":"Feature"}\n'
    output = tmp_path / "itiner_e_latest.ndjson"
    result = module.download_export(output, opener=opener(payload), calculate_sha256=True)
    assert result.skipped is False and output.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    metadata = tmp_path / "metadata.json"
    module.write_metadata(result, url=module.ITINER_E_EXPORT_URL, metadata_path=metadata, data_format="geojson", accessed_at="2026-08-28T00:00:00+00:00")
    assert 'CC BY 4.0' in metadata.read_text(encoding="utf-8")
    assert '"format": "geojson"' in metadata.read_text(encoding="utf-8")


def test_http_failure_preserves_existing_file(tmp_path):
    output = tmp_path / "itiner.ndjson"
    output.write_bytes(b"valid")

    def failed(_url, timeout):
        raise HTTPError(_url, 503, "offline", {}, None)

    with pytest.raises(RuntimeError, match="Itiner-e download failed"):
        module.download_export(output, force=True, opener=failed)
    assert output.read_bytes() == b"valid"


def test_interrupted_download_removes_partial_and_preserves_existing_file(tmp_path):
    output = tmp_path / "itiner.ndjson"
    output.write_bytes(b"existing")

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell() == 0:
                return super().read(2)
            raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        module.download_export(output, force=True, opener=lambda _url, timeout: Interrupted(b"new-data"))
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".itiner.ndjson.*.part"))


def test_existing_file_skips_unless_forced(tmp_path):
    output = tmp_path / "itiner.ndjson"
    output.write_bytes(b"existing")
    skipped = module.download_export(output, opener=opener(b"new"))
    replaced = module.download_export(output, force=True, opener=opener(b"new"))
    assert skipped.skipped is True and output.read_bytes() == b"new"
    assert replaced.skipped is False


def test_ndjson_parser_and_malformed_line_handling(tmp_path):
    path = tmp_path / "sample.ndjson"
    path.write_text('{"type":"Feature"}\n\n{"properties":{"Type":"Main Road"}}\n', encoding="utf-8")
    assert list(module.iter_ndjson(path))[1]["properties"]["Type"] == "Main Road"
    path.write_text('{"ok":true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(module.iter_ndjson(path))


def test_geojson_audit_reports_schema_and_endpoint_tolerance(tmp_path):
    audit_spec = importlib.util.spec_from_file_location("audit_itiner_e", SCRIPT.parent / "audit_itiner_e.py")
    assert audit_spec and audit_spec.loader
    audit_module = importlib.util.module_from_spec(audit_spec)
    sys.modules[audit_spec.name] = audit_module
    audit_spec.loader.exec_module(audit_module)
    path = tmp_path / "roads.geojson"
    path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"LineString","coordinates":[[1,2],[3,4]]},"properties":{"Type":"Main Road","Lower_Date":-100,"pleiadesPlaces":["x"]}},{"type":"Feature","geometry":{"type":"LineString","coordinates":[[3.000001,4],[5,6]]},"properties":{"Type":"Secondary Road","Lower_Date":9999}}]}', encoding="utf-8")
    result = audit_module.audit_geojson(path)
    assert result["record_count"] == 2
    assert result["route_type_fields"]["Type"]["unique_non_null"] == 2
    assert result["chronology_fields"]["Lower_Date"]["common_values"] == [(-100, 1), (9999, 1)]
    assert result["connectivity"]["near_1e5_shared_endpoint_keys"] == 1
