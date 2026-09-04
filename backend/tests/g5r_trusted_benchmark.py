"""G5R: L3 trusted historical movement benchmark loader and metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.models import Evidence

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "g5r_trusted_movement_benchmark.json"
ALLOWED_QUALITY = frozenset({"HIGH", "MEDIUM"})


def load_trusted_benchmark() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def trusted_queries(include_no_reference: bool = False) -> list[dict[str, Any]]:
    data = load_trusted_benchmark()
    queries = data["queries"]
    if include_no_reference:
        return queries
    return [q for q in queries if q.get("trusted_reference_ids")]


def trusted_references() -> list[dict[str, Any]]:
    data = load_trusted_benchmark()
    return [ref for ref in data["references"] if ref["quality"] in ALLOWED_QUALITY]


def references_for_query(query_id: str) -> list[dict[str, Any]]:
    return [ref for ref in trusted_references() if ref["query_id"] == query_id]


def hard_benchmark_queries() -> list[dict[str, Any]]:
    return trusted_queries(include_no_reference=False)


def trusted_key_movement_recall(
    evidence: list[Evidence],
    *,
    query_id: str | None = None,
) -> dict[str, Any]:
    """TRUSTED_KEY_MOVEMENT_RECALL for one query or all hard queries."""
    evidence_ids = {item.id for item in evidence}
    queries = hard_benchmark_queries()
    if query_id is not None:
        queries = [q for q in queries if q["query_id"] == query_id]

    per_query: dict[str, Any] = {}
    total_expected = 0
    total_hit = 0
    for query in queries:
        refs = references_for_query(query["query_id"])
        expected_ids = {ref["evidence_id"] for ref in refs}
        hit_ids = sorted(expected_ids & evidence_ids)
        expected_count = len(expected_ids)
        hit_count = len(hit_ids)
        total_expected += expected_count
        total_hit += hit_count
        per_query[query["query_id"]] = {
            "expected": expected_count,
            "hit": hit_count,
            "recall": (hit_count / expected_count) if expected_count else None,
            "hit_ids": hit_ids,
            "miss_ids": sorted(expected_ids - evidence_ids),
        }

    return {
        "per_query": per_query,
        "aggregate_recall": (total_hit / total_expected) if total_expected else None,
        "total_expected": total_expected,
        "total_hit": total_hit,
        "queries_in_denominator": len(queries),
    }


def benchmark_summary() -> dict[str, Any]:
    data = load_trusted_benchmark()
    refs = trusted_references()
    queries = data["queries"]
    hard = hard_benchmark_queries()
    return {
        "trusted_reference_count": len(refs),
        "high_count": sum(1 for ref in refs if ref["quality"] == "HIGH"),
        "medium_count": sum(1 for ref in refs if ref["quality"] == "MEDIUM"),
        "queries_total": len(queries),
        "queries_with_trusted_refs": len(hard),
        "roman_republic_queries": sum(1 for q in hard if q["domain"] == "ROMAN_REPUBLIC"),
        "stress_control_queries": sum(1 for q in hard if q["domain"] == "STRESS_CONTROL"),
        "no_reference_queries": [q["query_id"] for q in queries if not q.get("trusted_reference_ids")],
    }
