"""G7H: span-aware harness evidence scoring contract tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from backend.app.models import Evidence

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "g7c_broad_full_chain_trace.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("g7c_broad_full_chain_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _case(case_id: str) -> dict:
    module = _load_module()
    payload = module.load_fixture()
    return next(case for case in payload["cases"] if case["case_id"] == case_id)


def _evidence(module, case: dict, evidence_id: str, text: str):
    source = case["source"]
    parent, start, end = module.parse_evidence_span(evidence_id)
    return Evidence(
        id=evidence_id,
        author=source["author"],
        work=source["work"],
        locator="0",
        excerpt=text[:500],
        text=text,
        metadata={
            "document_id": source["document_id"],
            "parent_id": source["parent_id"],
            "source_chunk_id": source["parent_id"],
            "passage_start": start,
            "passage_end": end,
        },
    )


def test_exact_id_classifies_as_exact_match():
    module = _load_module()
    case = _case("G7-A01")
    gold_id = case["gold_evidence_ids"][0]
    evidence = _evidence(module, case, gold_id, case["exact_bounded_passage"])
    status = module.classify_gold_evidence_match(case, gold_id, gold_id, {gold_id: evidence})
    assert status == "EXACT_MATCH"


def test_overlapping_same_assertion_classifies_as_equivalent():
    module = _load_module()
    case = _case("G7-A04")
    gold_id = case["gold_evidence_ids"][0]
    overlap_id = "e2698fa32c836a515191464e3bf62397:100:300"
    overlap_text = case["exact_bounded_passage"]
    evidence_by_id = {
        gold_id: _evidence(module, case, gold_id, case["exact_bounded_passage"]),
        overlap_id: _evidence(module, case, overlap_id, overlap_text),
    }
    status = module.classify_gold_evidence_match(case, gold_id, overlap_id, evidence_by_id)
    assert status == "ASSERTION_EQUIVALENT_OVERLAP"


def test_overlap_missing_destination_is_missing():
    module = _load_module()
    case = _case("G7-A01")
    gold_id = case["gold_evidence_ids"][0]
    overlap_id = f"{case['source']['parent_id']}:661:750"
    wrong_text = "When Eumenes was returning to Asia he went up from Cirrha to sacrifice."
    evidence_by_id = {
        gold_id: _evidence(module, case, gold_id, case["exact_bounded_passage"]),
        overlap_id: _evidence(module, case, overlap_id, wrong_text),
    }
    status = module.classify_gold_evidence_match(case, gold_id, overlap_id, evidence_by_id)
    assert status == "MISSING"


def test_unrelated_same_parent_sibling_is_missing():
    module = _load_module()
    case = _case("G7-B04")
    gold_id = case["gold_evidence_ids"][0]
    sibling_id = "6dee1ba6ddc4ea4b84e50e9ad28704f2:0:378"
    unrelated = "Theseus was born in Troezen and raised by his mother Aethra."
    evidence_by_id = {
        gold_id: _evidence(module, case, gold_id, case["exact_bounded_passage"]),
        sibling_id: _evidence(module, case, sibling_id, unrelated),
    }
    status = module.classify_gold_evidence_match(case, gold_id, sibling_id, evidence_by_id)
    assert status == "MISSING"


def test_raw_availability_differs_from_exact_final_recall():
    module = _load_module()
    case = _case("G7-A01")
    gold_id = case["gold_evidence_ids"][0]
    evidence = _evidence(module, case, gold_id, case["exact_bounded_passage"])
    stage_ids = {
        "raw_semantic_ids": {gold_id},
        "raw_lexical_ids": set(),
        "proposal_ids": set(),
        "coverage_input_ids": set(),
        "canonical_union_rerank_ids": set(),
        "coverage_output_ids": set(),
        "final_evidence_ids": set(),
    }
    report = module.evaluate_gold_matches(case, stage_ids, {gold_id: evidence})
    assert report["raw_gold_available"] is True
    assert report["exact_present_final"] is False
    assert report["assertion_preserving_present_final"] is False


def test_assertion_preserving_final_differs_from_exact_when_overlap_sibling_selected():
    module = _load_module()
    case = _case("G7-B04")
    gold_id = case["gold_evidence_ids"][0]
    sibling_id = "6dee1ba6ddc4ea4b84e50e9ad28704f2:2082:2650"
    sibling_text = (
        "Theseus sent away his children privately to Euboea, commending them to the care of Elephenor; "
        "and he himself sailed to Scyros, where he had lands left him by his father."
    )
    evidence_by_id = {
        gold_id: _evidence(module, case, gold_id, case["exact_bounded_passage"]),
        sibling_id: _evidence(module, case, sibling_id, sibling_text),
    }
    stage_ids = {
        "raw_semantic_ids": set(),
        "raw_lexical_ids": set(),
        "proposal_ids": set(),
        "coverage_input_ids": set(),
        "canonical_union_rerank_ids": set(),
        "coverage_output_ids": {sibling_id},
        "final_evidence_ids": {sibling_id},
    }
    report = module.evaluate_gold_matches(case, stage_ids, evidence_by_id)
    assert report["exact_present_final"] is False
    assert report["assertion_preserving_present_final"] is True


def test_first_divergence_moves_downstream_when_equivalent_evidence_survives():
    module = _load_module()
    case = _case("G7-A01")
    gold_comparison = {
        "gold_present_final": False,
        "assertion_preserving_present_final": True,
        "gold_events_represented": False,
        "gold_relations_represented": True,
        "route_status_matched": False,
        "actual_route_status": "NONE",
    }
    assert module.determine_first_divergence(case, gold_comparison, {}) == "EVENT_EXTRACTION"


def test_detailed_trace_reuses_coverage_output_for_final_evidence(monkeypatch):
    module = _load_module()

    class FakeRetriever:
        def _collect_candidates(self, query, semantic_k=60, lexical_k=60):
            return [
                Evidence(
                    id="parent:1:10",
                    author="A",
                    work="W",
                    locator="0",
                    excerpt="Caesar crossed to Tauromenium.",
                    text="Caesar crossed to Tauromenium.",
                    metadata={"semantic_candidate": True, "parent_id": "parent", "source_chunk_id": "parent", "passage_start": 1, "passage_end": 10},
                )
            ]

        def retrieve_candidates(self, query, observation_k, filters=None):
            return self._collect_candidates(query)

    coverage_item = Evidence(
        id="parent:1:10",
        author="A",
        work="W",
        locator="0",
        excerpt="Caesar crossed to Tauromenium.",
        text="Caesar crossed to Tauromenium.",
        metadata={"parent_id": "parent", "source_chunk_id": "parent", "passage_start": 1, "passage_end": 10},
    )

    def _fake_merge(query, intent_results, budget):
        return [coverage_item]

    monkeypatch.setattr(module, "merge_coverage_results", _fake_merge)
    trace = module.detailed_retrieval_trace(FakeRetriever(), "Trace Caesar's crossing.")
    assert trace["final_evidence_ids"] == trace["coverage_output_ids"]
    assert trace["canonical_union_rerank_ids"] == ["parent:1:10"]


def test_polarity_mismatch_overlap_is_missing():
    module = _load_module()
    case = _case("G7-C01")
    gold_id = case["gold_evidence_ids"][0]
    overlap_id = f"{case['source']['parent_id']}:1367:1450"
    wrong_text = "The relieving force entered Setovia during the siege."
    evidence_by_id = {
        gold_id: _evidence(module, case, gold_id, case["exact_bounded_passage"]),
        overlap_id: _evidence(module, case, overlap_id, wrong_text),
    }
    status = module.classify_gold_evidence_match(case, gold_id, overlap_id, evidence_by_id)
    assert status == "MISSING"


def test_aggregate_metrics_expose_distinct_recall_names():
    module = _load_module()
    traces = [
        {
            "case_origin": "REAL_CORPUS",
            "tier": "A",
            "gold_comparison": {
                "gold_present_final": False,
                "assertion_preserving_present_final": True,
                "raw_gold_available": True,
                "gold_relations_represented": True,
                "expected_route_status": "FULL",
            },
            "route": {"status": "NONE"},
            "diagnostics": {"first_divergence": "ROUTE_CONSTRUCTION"},
        }
    ]
    metrics = module.aggregate_metrics(traces)
    assert "Trusted Evidence Recall" not in metrics
    assert metrics["Exact Final Evidence Recall"] == 0.0
    assert metrics["Assertion-Preserving Final Evidence Recall"] == 1.0
    assert metrics["Raw Gold Availability"] == 1.0
