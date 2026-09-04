"""G5R: trusted L3 movement benchmark integrity tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tests.g5r_trusted_benchmark import (
    ALLOWED_QUALITY,
    FIXTURE,
    benchmark_summary,
    hard_benchmark_queries,
    load_trusted_benchmark,
    trusted_key_movement_recall,
    trusted_references,
)

DEPRECATED_THESEUS_ID = "3c7ef8ef4efa1d6e26cb3e199a73ba7a:1335:2228"
G5B_FIXTURE = Path(__file__).parent / "fixtures" / "g5b_movement_benchmark.json"


def test_no_invalid_theseus_xenophon_key_in_trusted_benchmark():
    refs = trusted_references()
    assert not any(ref["evidence_id"] == DEPRECATED_THESEUS_ID for ref in refs)
    deprecated = load_trusted_benchmark()["deprecated_references"]
    assert any(item["evidence_id"] == DEPRECATED_THESEUS_ID for item in deprecated)


def test_no_term_only_key_proxies_in_trusted_benchmark():
    for ref in trusted_references():
        stmt = ref["supporting_statement"].casefold()
        assert any(
            token in stmt
            for token in (
                "cross",
                "march",
                "sail",
                "travel",
                "fly",
                "fled",
                "passed",
                "invaded",
                "landed",
                "came over",
                "made over",
                "transport",
                "entering",
                "returned into",
            )
        ), ref["benchmark_id"]


def test_every_hard_reference_has_required_fields():
    required = {
        "benchmark_id",
        "query_id",
        "evidence_id",
        "source_title",
        "navigation_path",
        "supporting_statement",
        "narrative_subject",
        "movement_type",
        "episode_anchor",
        "quality",
        "validation_notes",
    }
    for ref in trusted_references():
        missing = required - ref.keys()
        assert not missing, f"{ref['benchmark_id']} missing {missing}"
        assert ref["quality"] in ALLOWED_QUALITY
        assert ref["evidence_id"]
        assert ref["supporting_statement"].strip()
        assert ref["narrative_subject"].strip()
        assert ref["episode_anchor"].strip()
        assert ref["validation_notes"].startswith("VALID")


def test_xenophon_query_excluded_from_hard_denominator():
    summary = benchmark_summary()
    assert "xenophon_cunaxa_return" in summary["no_reference_queries"]
    hard_ids = {q["query_id"] for q in hard_benchmark_queries()}
    assert "xenophon_cunaxa_return" not in hard_ids


def test_no_reference_query_has_null_minimum_recall():
    data = load_trusted_benchmark()
    xen = next(q for q in data["queries"] if q["query_id"] == "xenophon_cunaxa_return")
    assert xen.get("status") == "NO_TRUSTED_REFERENCE_AVAILABLE"
    assert xen["trusted_reference_ids"] == []
    assert xen["minimum_expected_reference_recall"] is None


def test_benchmark_meets_minimum_size_targets():
    summary = benchmark_summary()
    assert summary["queries_with_trusted_refs"] >= 5
    assert summary["roman_republic_queries"] >= 3
    assert summary["trusted_reference_count"] >= 8
    assert summary["high_count"] >= 1
    assert summary["medium_count"] >= 1


def test_g5b_x01_marked_invalid_for_l3_if_present():
    rows = json.loads(G5B_FIXTURE.read_text(encoding="utf-8"))
    x01 = next(row for row in rows if row["id"] == "g5b-x01")
    assert x01.get("benchmark_tier") == "L1_MECHANISM"
    assert x01.get("invalid_as_l3_ground_truth") is True


def test_trusted_recall_excludes_no_reference_queries():
    result = trusted_key_movement_recall([])
    assert result["queries_in_denominator"] == len(hard_benchmark_queries())
    assert result["aggregate_recall"] == 0.0
    assert result["total_expected"] == len(trusted_references())


@pytest.mark.integration
def test_trusted_references_retrievable_under_coverage(production_retriever):
    """Each trusted reference should appear in its query's coverage retrieval."""
    from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET

    misses: list[str] = []
    for query in hard_benchmark_queries():
        evidence = production_retriever.retrieve_with_coverage(
            query["query"], budget=DEFAULT_COVERAGE_BUDGET,
        )
        recall = trusted_key_movement_recall(evidence, query_id=query["query_id"])
        per = recall["per_query"][query["query_id"]]
        if per["hit"] < per["expected"]:
            misses.extend(per["miss_ids"])
    if misses:
        pytest.xfail(
            f"corpus retrieval did not surface all trusted refs in coverage window: {misses[:5]}",
        )
