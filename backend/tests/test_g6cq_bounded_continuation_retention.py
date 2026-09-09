"""G6CQ: preserve bounded same-parent movement continuations across overlap merge."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import merge_coverage_results
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.lexical_index import derive_passages
from backend.app.rag.retrieval_intents import RetrievalIntent

POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)
ARISTON_QUERY = "Trace Ariston's movements from Rhodes to Cyprus."


def _meta(parent: str) -> dict:
    return {
        "document_id": parent,
        "source_chunk_id": parent,
        "author": "Plutarch",
        "work": "Lives",
    }


def _passage_evidence(
    parent: str,
    parent_text: str,
    passage_index: int,
    *,
    score: float = 0.8,
) -> Evidence:
    passage = derive_passages(parent, parent_text, _meta(parent))[passage_index]
    item = Evidence(
        id=passage.id,
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt=passage.text[:500],
        text=passage.text,
        score=score,
        metadata=dict(passage.metadata),
    )
    ranked = rerank_evidence(POMPEY_QUERY, [item])[0]
    return ranked


def _merged(parent: str, parent_text: str, selected_index: int, candidate_index: int, *, query: str = POMPEY_QUERY):
    selected = _passage_evidence(parent, parent_text, selected_index, score=0.9)
    candidate = _passage_evidence(parent, parent_text, candidate_index, score=0.7)
    return merge_coverage_results(
        query,
        [(RetrievalIntent("MOVEMENT", query), [selected, candidate])],
        budget=5,
    )


def _text_hits(items: list[Evidence], needle: str) -> list[str]:
    return [item.text for item in items if needle.casefold() in (item.text or "").casefold()]


def test_useful_movement_suffix_retained_once():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He sailed from there to Pelusium in Egypt."
    )
    merged = _merged("parent-a", parent_text, 1, 2)
    pelusium = _text_hits(merged, "Pelusium")
    assert len(merged) == 2
    assert len(pelusium) == 1
    assert "sailed from there to Pelusium" in pelusium[0]


def test_commentary_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "The court later debated grain supplies at Rome."
    )
    merged = _merged("parent-b", parent_text, 1, 2)
    assert len(merged) == 1


def test_modal_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He should sail from there to Pelusium in Egypt."
    )
    merged = _merged("parent-c", parent_text, 1, 2)
    assert len(merged) == 1


def test_wrong_actor_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "Brutus marched from Asia to Bithynia."
    )
    merged = _merged("parent-d", parent_text, 1, 2)
    assert len(merged) == 1


def test_duplicate_suffix_does_not_duplicate_authority():
    parent_text = (
        "Pompey sailed from Rhodes to Cyprus. "
        "He sailed from there to Pelusium in Egypt. "
        "He sailed from there to Pelusium in Egypt."
    )
    selected = _passage_evidence("parent-e", parent_text, 0)
    duplicate = _passage_evidence("parent-e", parent_text, 1, score=0.7)
    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), [selected, duplicate])],
        budget=5,
    )
    assert len(merged) == 1


def test_different_parent_overlap_like_offsets_not_retained():
    text_a = "Alpha marched from Rome to Capua. Beta marched from Capua to Neapolis."
    text_b = "Gamma marched from Rome to Capua. Delta marched from Capua to Neapolis."
    selected = _passage_evidence("parent-f", text_a, 1)
    candidate = _passage_evidence("parent-g", text_b, 1, score=0.7)
    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), [selected, candidate])],
        budget=5,
    )
    assert len(merged) == 2


def test_over_bounded_extension_fails_closed():
    parent = "parent-h"
    selected = Evidence(
        id=f"{parent}:0:70",
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt="A. B.",
        text="Sentence A about camp. Sentence B about coasting.",
        score=0.9,
        metadata={
            **_meta(parent),
            "passage_start": 0,
            "passage_end": 70,
            "passage_index": 1,
            "retrieval_ranking": {"final_score": 0.9},
        },
    )
    candidate = Evidence(
        id=f"{parent}:40:160",
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt="B. C. D.",
        text="Sentence B about coasting. Sentence C sailed to Pelusium. Sentence D about court.",
        score=0.8,
        metadata={
            **_meta(parent),
            "passage_start": 40,
            "passage_end": 160,
            "passage_index": 2,
            "retrieval_ranking": {"final_score": 0.8},
        },
    )
    selected = rerank_evidence(POMPEY_QUERY, [selected])[0]
    candidate = rerank_evidence(POMPEY_QUERY, [candidate])[0]
    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), [selected, candidate])],
        budget=5,
    )
    assert len(merged) == 1


def test_pompey_control_offsets_retain_pelusium():
    parent = "0e7e813af196915fa9ae7245dc4a069e"
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He sailed from there to Pelusium in Egypt."
    )
    passages = derive_passages(parent, parent_text, _meta(parent))
    selected = next(item for item in passages if item.metadata["passage_index"] == 1)
    candidate = next(item for item in passages if item.metadata["passage_index"] == 2)

    def as_evidence(passage, score: float) -> Evidence:
        item = Evidence(
            id=passage.id,
            author="Plutarch",
            work="Lives",
            locator="1",
            excerpt=passage.text[:500],
            text=passage.text,
            score=score,
            metadata=dict(passage.metadata),
        )
        return rerank_evidence(POMPEY_QUERY, [item])[0]

    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), [as_evidence(selected, 0.9), as_evidence(candidate, 0.8)])],
        budget=5,
    )
    pelusium_hits = _text_hits(merged, "Pelusium")
    assert len(pelusium_hits) == 1
    assert "Cyprus" not in pelusium_hits[0]
