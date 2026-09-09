"""G6CU: discover bounded continuations when anchor is selected, before sibling reaches add()."""
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
PARENT = "0e7e813af196915fa9ae7245dc4a069e"
PARENT_TEXT = (
    "Pompey marched from Rhodes toward the coast. "
    "He coasted along the shore as far as Cilicia. "
    "He steered his course that way toward Pelusium."
)


def _meta(parent: str) -> dict:
    return {
        "document_id": parent,
        "source_chunk_id": parent,
        "author": "Plutarch",
        "work": "Lives",
    }


def _passage_item(parent: str, parent_text: str, passage_index: int) -> Evidence:
    passage = derive_passages(parent, parent_text, _meta(parent))[passage_index]
    return Evidence(
        id=passage.id,
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt=passage.text[:500],
        text=passage.text,
        score=0.5,
        metadata=dict(passage.metadata),
    )


def _filler(identifier: str, text: str, *, lexical_score: float) -> Evidence:
    return Evidence(
        id=identifier,
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt=text[:500],
        text=text,
        score=0.5,
        metadata={
            **_meta(f"filler-{identifier}"),
            "lexical_candidate": True,
            "lexical_score": lexical_score,
            "semantic_candidate": True,
            "vector_rank": 99,
        },
    )


def _ranked(items: list[Evidence]) -> list[Evidence]:
    return rerank_evidence(POMPEY_QUERY, items)


def _merge(items: list[Evidence], *, budget: int = 20) -> list[Evidence]:
    ranked = _ranked(items)
    return merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), ranked)],
        budget=budget,
    )


def _anchor_and_sibling() -> tuple[Evidence, Evidence]:
    anchor = _passage_item(PARENT, PARENT_TEXT, 1)
    sibling = _passage_item(PARENT, PARENT_TEXT, 2)
    anchor = anchor.model_copy(update={
        "metadata": {
            **anchor.metadata,
            "lexical_candidate": True,
            "lexical_score": 100.0,
            "semantic_candidate": True,
            "vector_rank": 1,
        },
    })
    sibling = sibling.model_copy(update={
        "metadata": {
            **sibling.metadata,
            "lexical_candidate": True,
            "lexical_score": 0.01,
            "semantic_candidate": True,
            "vector_rank": 999,
        },
    })
    return anchor, sibling


def test_below_budget_sibling_discovered_when_anchor_selected():
    anchor, sibling = _anchor_and_sibling()
    fillers = [
        _filler(
            f"noise-{index}",
            "The senate debated grain supplies in Rome throughout the winter.",
            lexical_score=40.0 - index * 0.01,
        )
        for index in range(30)
    ]
    pool = _ranked([*fillers, anchor, sibling])
    ranks = {item.id: int(item.metadata.get("rank") or 999) for item in pool}
    assert ranks[anchor.id] <= 20
    assert ranks[sibling.id] > 20

    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), pool)],
        budget=20,
    )
    assert anchor.id in {item.id for item in merged}
    assert sibling.id not in {item.id for item in merged}
    suffix_ids = [item.id for item in merged if item.metadata.get("continuation_suffix")]
    assert len(suffix_ids) == 1
    assert "steered his course that way" in next(item.text for item in merged if item.metadata.get("continuation_suffix"))


def test_within_budget_sibling_retained_once():
    anchor, sibling = _anchor_and_sibling()
    pool = _ranked([anchor, sibling])
    merged = _merge(pool, budget=5)
    suffix_items = [item for item in merged if item.metadata.get("continuation_suffix")]
    assert len(suffix_items) == 1
    assert sibling.id not in {item.id for item in merged}


def test_non_adjacent_same_parent_not_retained():
    anchor = _passage_item(PARENT, PARENT_TEXT, 1)
    non_adjacent = _passage_item(PARENT, PARENT_TEXT, 0)
    merged = _merge(_ranked([anchor, non_adjacent]), budget=5)
    assert len(merged) == 1
    assert not any(item.metadata.get("continuation_suffix") for item in merged)


def test_modal_adjacent_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "He should steer his course that way toward Pelusium."
    )
    anchor, sibling = _passage_item(PARENT, parent_text, 1), _passage_item(PARENT, parent_text, 2)
    merged = _merge(_ranked([anchor, sibling]), budget=5)
    assert len(merged) == 1


def test_wrong_actor_adjacent_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "Brutus marched from Asia to Bithynia."
    )
    anchor, sibling = _passage_item(PARENT, parent_text, 1), _passage_item(PARENT, parent_text, 2)
    merged = _merge(_ranked([anchor, sibling]), budget=5)
    assert len(merged) == 1


def test_commentary_adjacent_suffix_not_retained():
    parent_text = (
        "Pompey marched from Rhodes toward the coast. "
        "He coasted along the shore as far as Cilicia. "
        "The court later debated grain supplies at Rome."
    )
    anchor, sibling = _passage_item(PARENT, parent_text, 1), _passage_item(PARENT, parent_text, 2)
    merged = _merge(_ranked([anchor, sibling]), budget=5)
    assert len(merged) == 1


def test_multiple_continuation_siblings_choose_one_deterministically():
    anchor = _passage_item(PARENT, PARENT_TEXT, 1)
    sibling_a = _passage_item(PARENT, PARENT_TEXT, 2)
    sibling_b = _passage_item(PARENT, PARENT_TEXT, 2)
    sibling_b = sibling_b.model_copy(update={"id": sibling_b.id.replace(":2:", ":2b:")})
    merged = _merge(_ranked([anchor, sibling_b, sibling_a]), budget=5)
    assert len([item for item in merged if item.metadata.get("continuation_suffix")]) == 1


def test_budget_boundary_never_exceeded():
    anchor, sibling = _anchor_and_sibling()
    fillers = [
        _filler(
            f"budget-{index}",
            "Pompey marched from Greece through the eastern Mediterranean toward Egypt in 48 BCE.",
            lexical_score=35.0 - index,
        )
        for index in range(25)
    ]
    merged = _merge(_ranked([*fillers, anchor, sibling]), budget=20)
    assert len(merged) <= 20


def test_different_parent_not_retained():
    anchor = _passage_item(PARENT, PARENT_TEXT, 1)
    other_parent_text = (
        "Alpha marched from Rome to Capua. "
        "Beta marched from Capua to Neapolis. "
        "Gamma sailed from Neapolis to Tarentum."
    )
    sibling = _passage_item("other-parent", other_parent_text, 2)
    merged = _merge(_ranked([anchor, sibling]), budget=5)
    assert len(merged) <= 2
    assert not any(item.metadata.get("continuation_suffix") for item in merged)


def test_pompey_production_like_control():
    anchor, sibling = _anchor_and_sibling()
    fillers = [
        _filler(
            f"prod-{index}",
            "The senate debated grain supplies in Rome throughout the winter.",
            lexical_score=50.0 - index * 0.05,
        )
        for index in range(130)
    ]
    pool = _ranked([*fillers, anchor, sibling])
    ranks = {item.id: int(item.metadata.get("rank") or 999) for item in pool}
    assert ranks[anchor.id] <= 20
    assert ranks[sibling.id] > 20

    merged = merge_coverage_results(
        POMPEY_QUERY,
        [(RetrievalIntent("MOVEMENT", POMPEY_QUERY), pool)],
        budget=20,
    )
    assert anchor.id in {item.id for item in merged}
    assert sibling.id not in {item.id for item in merged}
    suffix = next(item for item in merged if item.metadata.get("continuation_suffix"))
    assert suffix.id.endswith(f":{anchor.metadata['passage_end']}:{sibling.metadata['passage_end']}")
    assert "steered his course that way" in suffix.text
