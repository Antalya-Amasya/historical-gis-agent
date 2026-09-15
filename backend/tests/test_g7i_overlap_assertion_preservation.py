"""G7I: preserve qualified movement assertions during overlap arbitration."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    merge_coverage_results,
    movement_bearing_text,
    passages_overlap,
)
from backend.app.rag.retrieval_intents import RetrievalIntent

QUERY = "Trace Commander Alpha's route from Origin to Beta toward Gamma."


def _item(
    parent: str,
    *,
    start: int,
    end: int,
    score: float,
    text: str,
) -> Evidence:
    return Evidence(
        id=f"{parent}:{start}:{end}",
        author="Plutarch",
        work="Lives",
        locator=str(start),
        excerpt=text[:500],
        text=text,
        score=score,
        metadata={
            "parent_id": parent,
            "source_chunk_id": parent,
            "document_id": parent,
            "passage_start": start,
            "passage_end": end,
            "passage_index": start // 120,
            "retrieval_ranking": {"final_score": score},
        },
    )


def test_overlap_supersession_preserves_stronger_movement_assertion():
    parent = "parent-x"
    movement_text = (
        "Commander Alpha marched from Origin to Beta; "
        "then he turned course with his fleet toward Gamma."
    )
    context_text = (
        "Commander Alpha received reports from aides about Gamma; "
        "friends discussed policy in Beta afterward."
    )
    context = _item(
        parent,
        start=80,
        end=200,
        score=0.95,
        text=context_text,
    )
    movement = _item(
        parent,
        start=0,
        end=160,
        score=0.85,
        text=movement_text,
    )
    assert passages_overlap(context, movement)

    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), [context, movement])],
        budget=5,
    )

    assert len(merged) == 1
    assert merged[0].id == movement.id
    assert movement_bearing_text(merged[0].text or "")


def test_equivalent_overlap_sibling_still_suppressed():
    parent = "parent-y"
    left_text = "Commander Alpha marched from Origin to Beta through Gamma."
    right_text = "Commander Alpha marched from Origin toward Beta and Gamma."
    left = _item(parent, start=0, end=120, score=0.95, text=left_text)
    right = _item(parent, start=10, end=110, score=0.90, text=right_text)
    assert passages_overlap(left, right)

    merged = merge_coverage_results(
        QUERY,
        [(RetrievalIntent("CANONICAL", QUERY), [left, right])],
        budget=5,
    )

    assert len(merged) == 1
