"""G6CS: qualify steer-family verbs for bounded continuation retention."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    _continuation_asserted_movement,
    merge_coverage_results,
)
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.lexical_index import derive_passages
from backend.app.rag.retrieval_intents import RetrievalIntent

POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


def _meta(parent: str) -> dict:
    return {
        "document_id": parent,
        "source_chunk_id": parent,
        "author": "Plutarch",
        "work": "Lives",
    }


def _merge_with_suffix(suffix_sentence: str, *, parent: str = "parent-steer") -> list[Evidence]:
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        f"{suffix_sentence}"
    )
    passages = derive_passages(parent, parent_text, _meta(parent))
    selected = passages[1]
    candidate = passages[2]

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

    return merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), [as_evidence(selected, 0.9), as_evidence(candidate, 0.8)])],
        budget=5,
    )


def test_steered_his_course_that_way_qualifies():
    assert _continuation_asserted_movement("he steered his course that way.")
    merged = _merge_with_suffix("He steered his course that way.")
    assert len(merged) == 2
    assert any("steered his course that way" in (item.text or "") for item in merged)


def test_steered_toward_cyprus_qualifies():
    assert _continuation_asserted_movement("Ariston steered toward Cyprus.")
    merged = _merge_with_suffix("Ariston steered toward Cyprus.")
    assert len(merged) == 2


def test_was_steering_toward_rhodes_qualifies():
    assert _continuation_asserted_movement("Ariston was steering toward Rhodes.")
    merged = _merge_with_suffix("Ariston was steering toward Rhodes.")
    assert len(merged) == 2


def test_should_steer_rejected():
    assert not _continuation_asserted_movement("Ariston should steer toward Cyprus.")
    assert len(_merge_with_suffix("Ariston should steer toward Cyprus.")) == 1


def test_discussed_steering_rejected():
    assert not _continuation_asserted_movement("Ariston discussed steering toward Cyprus.")
    assert len(_merge_with_suffix("Ariston discussed steering toward Cyprus.")) == 1


def test_did_not_steer_rejected():
    assert not _continuation_asserted_movement("Ariston did not steer toward Cyprus.")
    assert len(_merge_with_suffix("Ariston did not steer toward Cyprus.")) == 1


def test_steered_conversation_rejected():
    assert not _continuation_asserted_movement("He steered the conversation toward Rome.")
    assert len(_merge_with_suffix("He steered the conversation toward Rome.")) == 1


def test_pompey_real_steering_suffix_retained():
    parent = "0e7e813af196915fa9ae7245dc4a069e"
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He steered his course that way toward Pelusium."
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
    assert len(merged) == 2
    assert any("steered his course that way" in (item.text or "") for item in merged)
    assert not any("Cyprus" in (item.text or "") for item in merged if item.metadata.get("continuation_suffix"))
