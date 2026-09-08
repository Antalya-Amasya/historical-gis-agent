"""G6BU: smoke tests for offline full-chain acceptance harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from backend.tests.g5r_trusted_benchmark import hard_benchmark_queries

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "g6bu_full_chain_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("g6bu_full_chain_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_harness_module_imports():
    module = _load_module()
    assert hasattr(module, "main")
    assert hasattr(module, "STAGE_ORDER")
    assert len(module.STAGE_ORDER) == 8


def test_g5r_fixture_loads_five_cases():
    queries = hard_benchmark_queries()
    assert len(queries) == 5
    assert {query["query_id"] for query in queries} == {
        "caesar_adriatic",
        "pompey_post_pharsalus_egypt",
        "mithridates_first_war",
        "lucullus_mithridatic",
        "alexander_hydaspes",
    }


def test_stage_schema_helpers():
    module = _load_module()
    stages = {
        "query": {"status": "PASS"},
        "retrieval": {"status": "PASS"},
        "evidence": {"status": "PARTIAL"},
        "event": {"status": "FAIL"},
        "geography": {"status": "SKIPPED"},
        "relation": {"status": "SKIPPED"},
        "route": {"status": "SKIPPED"},
        "presentation": {"status": "SKIPPED"},
    }
    assert module.first_divergence(stages) == "EVIDENCE"
    assert module.deepest_successful_stage(stages) == "RETRIEVAL"


@pytest.mark.integration
def test_one_case_executes_offline_without_api():
    module = _load_module()
    retriever = module.build_retriever()
    tools = module.AgentToolRegistry(retriever, module.OfflineGeography())
    case = module.run_case(retriever, tools, hard_benchmark_queries()[0])
    assert case["stages"]["query"]["status"] == "PASS"
    assert case["route_status"] in {"PASS", "PARTIAL", "FAIL"}
    assert case["first_divergence"]
    assert json.dumps(case)
