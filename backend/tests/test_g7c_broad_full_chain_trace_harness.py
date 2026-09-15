"""G7C: deterministic smoke tests for broad full-chain trace harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "g7c_broad_full_chain_trace.py"
FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "g7_broad_full_chain_evaluation.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("g7c_broad_full_chain_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harness_module_imports():
    module = _load_module()
    assert hasattr(module, "main")
    assert hasattr(module, "run_case")
    assert hasattr(module, "verify_offline_embedding_ready")
    assert hasattr(module, "PROHIBITED_CHECKS")
    assert len(module.PROHIBITED_CHECKS) >= 10


def test_fixture_loads_twenty_cases():
    module = _load_module()
    payload = module.load_fixture()
    assert payload["schema_version"] == "g7-broad-v1"
    assert len(payload["cases"]) == 20
    origins = {case["case_origin"] for case in payload["cases"]}
    assert origins == {"REAL_CORPUS", "SYNTHETIC_CONTRACT"}
    tiers = {case["tier"] for case in payload["cases"]}
    assert tiers == {"A", "B", "C"}


def test_local_model_path_resolution():
    module = _load_module()
    from backend.app.core.config import settings

    local_path = module.resolve_local_embedding_snapshot(settings.rag_embedding_model)
    assert local_path.exists()
    complete, missing = module.snapshot_is_complete(local_path)
    assert complete, missing
    assert local_path.name


def test_verify_offline_embedding_ready_without_network():
    module = _load_module()
    from backend.app.core.config import settings

    network_calls: list[str] = []

    def _block_request(*args, **kwargs):
        network_calls.append(str(args[0]) if args else "request")
        raise AssertionError("network/API call attempted during offline pre-flight")

    with patch("requests.sessions.Session.request", side_effect=_block_request):
        with patch("urllib.request.urlopen", side_effect=_block_request):
            result = module.verify_offline_embedding_ready(
                settings.rag_embedding_model,
                "cpu",
                settings.rag_embedding_batch_size,
            )
    assert result["local_path"]
    assert result["dimensions"] > 0
    assert network_calls == []


def test_incomplete_local_cache_fails_fast(tmp_path):
    module = _load_module()
    from backend.app.core.config import settings

    incomplete = tmp_path / "incomplete-snapshot"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")

    def _fake_resolve(_model_name: str):
        return incomplete

    with patch.object(module, "resolve_local_embedding_snapshot", _fake_resolve):
        with pytest.raises(module.OfflineEmbeddingError) as error:
            module.verify_offline_embedding_ready(settings.rag_embedding_model, "cpu", 16)
    assert error.value.code == "OFFLINE_EMBEDDING_UNAVAILABLE"


def test_stale_output_head_is_not_current(tmp_path, monkeypatch):
    module = _load_module()
    trace_path = tmp_path / "g7c_case_traces.json"
    summary_path = tmp_path / "g7c_summary.json"
    monkeypatch.setattr(module, "TRACE_PATH", trace_path)
    monkeypatch.setattr(module, "SUMMARY_PATH", summary_path)

    summary_path.write_text(json.dumps({"head": "old-head", "stale": False}), encoding="utf-8")
    stale = module.check_stale_outputs("current-head")
    assert stale["stale"] is True
    assert stale["accepted_as_current"] is False
    assert module.output_is_current("current-head") is False


def test_current_output_head_is_accepted(tmp_path, monkeypatch):
    module = _load_module()
    summary_path = tmp_path / "g7c_summary.json"
    monkeypatch.setattr(module, "SUMMARY_PATH", summary_path)
    summary_path.write_text(json.dumps({"head": "same-head", "stale": False}), encoding="utf-8")
    assert module.check_stale_outputs("same-head")["accepted_as_current"] is True
    assert module.output_is_current("same-head") is True


def test_synthetic_retrieval_bypass():
    module = _load_module()
    payload = module.load_fixture()
    synthetic = next(case for case in payload["cases"] if case["case_id"] == "G7-C03")
    trace = module.synthetic_retrieval_trace(synthetic)
    assert trace["mode"] == "SYNTHETIC_BYPASS"
    assert trace["raw_semantic_ids"] == []
    assert trace["final_evidence_ids"] == synthetic["gold_evidence_ids"]


def test_trace_schema_shape_for_synthetic_case():
    module = _load_module()
    payload = module.load_fixture()
    synthetic = next(case for case in payload["cases"] if case["case_id"] == "G7-C03")
    tools = module.AgentToolRegistry(module.EmptyHistoricalRetriever(), module.OfflineGeography())
    trace = module.run_case(synthetic, tools, module.EmptyHistoricalRetriever())

    assert trace["case_id"] == "G7-C03"
    assert trace["retrieval"]["mode"] == "SYNTHETIC_BYPASS"
    assert trace["diagnostics"]["first_divergence"]
    assert isinstance(trace["safety"]["prohibited_claim_hits"], dict)
    for claim, status in trace["safety"]["prohibited_claim_hits"].items():
        assert status in {"PASS", "FAIL", "NOT_CHECKED"}

    stage_keys = {
        "RAW_SEMANTIC",
        "RAW_LEXICAL",
        "LOCAL_PROPOSAL",
        "UNION",
        "COMMON_RERANK",
        "COVERAGE",
        "FINAL_EVIDENCE",
        "EVENTS",
        "RESOLVED_PLACES",
        "RELATIONS",
        "COMPONENTS",
        "ROUTE",
        "GIS_PRESENTATION",
    }
    assert stage_keys <= set(trace["stages"].keys())
    assert "admitted_relations" in trace["relations"]
    assert "first_divergence" in trace["diagnostics"]
    json.dumps(trace)


def test_prohibited_claim_result_shape():
    module = _load_module()
    payload = module.load_fixture()
    synthetic = next(case for case in payload["cases"] if case["case_id"] == "G7-C03")
    tools = module.AgentToolRegistry(module.EmptyHistoricalRetriever(), module.OfflineGeography())
    trace = module.run_case(synthetic, tools, module.EmptyHistoricalRetriever())
    listed = set(synthetic.get("prohibited_claims") or [])
    hits = trace["safety"]["prohibited_claim_hits"]
    assert set(hits.keys()) == listed
    assert all(status in {"PASS", "FAIL", "NOT_CHECKED"} for status in hits.values())


def test_no_network_required_for_synthetic_unit_tests(monkeypatch):
    module = _load_module()

    def _fail_network(*args, **kwargs):
        raise AssertionError("network/API call attempted in unit test")

    monkeypatch.setattr(module, "build_retriever", _fail_network)
    payload = module.load_fixture()
    synthetic = next(case for case in payload["cases"] if case["case_id"] == "G7-C03")
    tools = module.AgentToolRegistry(module.EmptyHistoricalRetriever(), module.OfflineGeography())
    trace = module.run_case(synthetic, tools, module.EmptyHistoricalRetriever())
    assert trace["retrieval"]["mode"] == "SYNTHETIC_BYPASS"
