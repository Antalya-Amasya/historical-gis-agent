"""G7F: bounded source-family diversity in coverage global fill."""
from __future__ import annotations

from collections import Counter

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    _coverage_facets,
    _coverage_family_key,
    merge_coverage_results,
    passages_overlap,
)
from backend.app.rag.retrieval_intents import RetrievalIntent

QUERY = "Trace the route from Alpha to Beta through Gamma."


def _item(
    family: str,
    index: int,
    *,
    score: float,
    text: str | None = None,
) -> Evidence:
    passage_text = text or f"{family} marched from Alpha to Beta in segment {index}."
    start = index * 120
    end = start + 80
    return Evidence(
        id=f"{family}:{start}:{end}",
        author="Plutarch",
        work="Lives",
        locator=str(index),
        excerpt=passage_text[:500],
        text=passage_text,
        score=score,
        metadata={
            "parent_id": family,
            "source_chunk_id": family,
            "document_id": family,
            "passage_start": start,
            "passage_end": end,
            "passage_index": index,
            "retrieval_ranking": {"final_score": score},
        },
    )


def _family_counts(items: list[Evidence]) -> Counter[str]:
    return Counter(_coverage_family_key(item) for item in items)


def _merge_reason_counts(items: list[Evidence]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        reason = (item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason")
        if reason:
            counts[str(reason)] += 1
    return counts


def test_global_fill_prefers_distinct_families_before_repeats():
    items = [_item("family-a", index, score=1.0 - index * 0.01) for index in range(18)]
    items.extend(_item(family, 0, score=0.2 - idx * 0.01) for idx, family in enumerate(("family-b", "family-c", "family-d")))
    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), items)],
        budget=DEFAULT_COVERAGE_BUDGET,
    )
    counts = _family_counts(merged)
    assert len(merged) == DEFAULT_COVERAGE_BUDGET
    assert counts["family-b"] >= 1
    assert counts["family-c"] >= 1
    assert counts["family-d"] >= 1
    assert len(counts) >= 4


def test_second_pass_allows_repeated_families_when_budget_requires():
    items = [_item("family-a", index, score=1.0 - index * 0.01) for index in range(25)]
    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), items)],
        budget=DEFAULT_COVERAGE_BUDGET,
    )
    assert len(merged) == DEFAULT_COVERAGE_BUDGET
    assert _family_counts(merged)["family-a"] > 1


def _merge_reason(item: Evidence) -> str:
    return str((item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason"))


def test_distinct_channel_facet_retains_reserved_protection():
    movement_items = [
        _item("family-a", 0, score=0.95, text="Subject marched from Alpha to Beta."),
    ]
    distinct_fragment = _item("family-b", 0, score=0.90, text="During the campaign the convoy put to sea.")
    distinct_fragment.metadata["heading"] = "THROUGH"
    empty_facet = _item("family-empty", 0, score=0.85, text="Annual census rolls listed population totals.")
    subject_items = [distinct_fragment, empty_facet]
    merged = merge_coverage_results(
        QUERY,
        [
            (RetrievalIntent("SUBJECT", "subject route"), subject_items),
            (RetrievalIntent("MOVEMENT", "movement route"), movement_items),
        ],
        budget=DEFAULT_COVERAGE_BUDGET,
    )
    by_family = {item.metadata["parent_id"]: item for item in merged}
    reasons = _merge_reason_counts(merged)

    assert not _coverage_facets(QUERY, empty_facet)
    assert "ROUTE_FRAGMENT" in _coverage_facets(QUERY, by_family["family-b"])
    assert reasons["intent_movement_coverage"] >= 1
    assert reasons["intent_coverage_slot"] >= 1
    assert _merge_reason(by_family["family-a"]) == "intent_movement_coverage"
    assert _merge_reason(by_family["family-b"]) == "intent_coverage_slot"
    assert "family-empty" not in by_family or _merge_reason(by_family["family-empty"]) != "intent_coverage_slot"


def test_overlap_suppression_unchanged():
    left = _item("family-a", 0, score=0.95)
    right = _item("family-a", 1, score=0.90)
    right.metadata["passage_start"] = 10
    right.metadata["passage_end"] = 70
    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), [left, right])],
        budget=5,
    )
    assert len(merged) == 1
    assert passages_overlap(left, right)


def test_continuation_behavior_unchanged():
    from backend.app.rag.evidence_ranking import rerank_evidence
    from backend.app.rag.lexical_index import derive_passages

    query = (
        "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
        "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
    )
    parent = "parent-a"
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He sailed from there to Pelusium in Egypt."
    )

    def passage(index: int, score: float) -> Evidence:
        passage_obj = derive_passages(parent, parent_text, {"source_chunk_id": parent, "document_id": parent})[index]
        return rerank_evidence(
            query,
            [
                Evidence(
                    id=passage_obj.id,
                    author="Plutarch",
                    work="Lives",
                    locator="1",
                    excerpt=passage_obj.text[:500],
                    text=passage_obj.text,
                    metadata=dict(passage_obj.metadata),
                )
            ],
        )[0]

    merged = merge_coverage_results(
        query,
        [(RetrievalIntent("MOVEMENT", query), [passage(1, 0.9), passage(2, 0.7)])],
        budget=5,
    )
    pelusium = [item for item in merged if "Pelusium" in (item.text or "")]
    assert len(merged) == 2
    assert len(pelusium) == 1


def test_budget_remains_exactly_twenty():
    items = [_item(f"family-{index % 5}", index, score=1.0 - index * 0.01) for index in range(40)]
    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), items)],
        budget=DEFAULT_COVERAGE_BUDGET,
    )
    assert len(merged) == DEFAULT_COVERAGE_BUDGET
