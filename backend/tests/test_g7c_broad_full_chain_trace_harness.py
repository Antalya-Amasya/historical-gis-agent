"""G7C: deterministic smoke tests for broad full-chain trace harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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


def test_no_network_required_for_unit_tests(monkeypatch):
    module = _load_module()

    def _fail_network(*args, **kwargs):
        raise AssertionError("network/API call attempted in unit test")

    monkeypatch.setattr(module, "build_retriever", _fail_network)
    payload = module.load_fixture()
    synthetic = next(case for case in payload["cases"] if case["case_id"] == "G7-C03")
    tools = module.AgentToolRegistry(module.EmptyHistoricalRetriever(), module.OfflineGeography())
    trace = module.run_case(synthetic, tools, module.EmptyHistoricalRetriever())
    assert trace["retrieval"]["mode"] == "SYNTHETIC_BYPASS"
