"""R4-D2B: after lexical reserve, remaining proposal seats compete globally by rank."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_RAW_OBSERVATION_K,
    _qualifies_lexical_proposal,
    passages_overlap,
    select_qualified_local_proposals,
)


def _item(
    identifier: str,
    *,
    rank: int,
    parent: str,
    start: int,
    end: int,
    lexical: bool = False,
    qualified: bool = True,
    lexical_score: float = 10.0,
) -> Evidence:
    ranking: dict = {
        "navigation_penalty": 0.0,
        "final_score": 1.0 / max(rank, 1),
    }
    metadata = {
        "document_id": "doc",
        "parent_id": parent,
        "source_chunk_id": parent,
        "passage_start": start,
        "passage_end": end,
        "rank": rank,
        "retrieval_ranking": ranking,
    }
    if lexical:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
        ranking.update(
            {
                "lexical_support": 0.9,
                "lexical_rank_relevance": min(1.0, lexical_score / 40.0),
                "lexical_score_relevance": min(1.0, lexical_score / 40.0),
            }
        )
    if qualified:
        metadata["semantic_candidate"] = True
        ranking.update(
            {
                "semantic_relevance": 0.2,
                "parent_semantic_prior": 0.2,
                "passage_local_support": 0.1,
            }
        )
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="1",
        excerpt="excerpt",
        text="The army marched from Alpha to Beta.",
        score=1.0 / max(rank, 1),
        metadata=metadata,
    )


def test_lexical_reserve_remains_protected():
    lexical = [
        _item(
            f"lex-{index}:0:10",
            rank=100 + index,
            parent=f"lex-{index}",
            start=0,
            end=10,
            lexical=True,
            lexical_score=40.0 - index,
        )
        for index in range(1, 25)
    ]
    semantic = [
        _item(f"sem-{index}:0:10", rank=index, parent=f"sem-{index}", start=0, end=10)
        for index in range(1, 50)
    ]
    selected = select_qualified_local_proposals(lexical + semantic, DEFAULT_RAW_OBSERVATION_K)
    reserved = [item for item in selected if _qualifies_lexical_proposal(item)]
    assert len(reserved) == 20
    assert len(selected) == DEFAULT_RAW_OBSERVATION_K


def test_overlapping_same_parent_windows_do_not_flood():
    parent = "parent-a"
    overlapping = [
        _item(
            f"{parent}:{start}:{start + 120}",
            rank=index,
            parent=parent,
            start=start,
            end=start + 120,
        )
        for index, start in enumerate(range(0, 80, 10), start=1)
    ]
    others = [
        _item(f"other-{index}:0:10", rank=50 + index, parent=f"other-{index}", start=0, end=10)
        for index in range(1, 40)
    ]
    selected = select_qualified_local_proposals(overlapping + others, 20)
    from_parent = [item for item in selected if item.metadata["parent_id"] == parent]
    assert len(from_parent) == 1
    assert from_parent[0].id == overlapping[0].id
    assert not any(
        passages_overlap(left, right)
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
        if left.metadata["parent_id"] == right.metadata["parent_id"] == parent
    )


def test_disjoint_same_parent_statements_may_both_survive_by_rank():
    first = _item("parent-a:0:50", rank=1, parent="parent-a", start=0, end=50)
    second = _item("parent-a:200:250", rank=2, parent="parent-a", start=200, end=250)
    others = [
        _item(f"other-{index}:0:10", rank=10 + index, parent=f"other-{index}", start=0, end=10)
        for index in range(1, 20)
    ]
    selected = select_qualified_local_proposals([first, second, *others], 10)
    ids = {item.id for item in selected}
    assert first.id in ids
    assert second.id in ids
    assert not passages_overlap(first, second)


def test_worse_ranked_other_family_first_rep_does_not_displace_better_disjoint():
    seated = _item("parent-a:0:40", rank=1, parent="parent-a", start=0, end=40)
    disjoint = _item("parent-a:200:240", rank=5, parent="parent-a", start=200, end=240)
    other_firsts = [
        _item(f"family-{index}:0:10", rank=10 + index, parent=f"family-{index}", start=0, end=10)
        for index in range(1, 40)
    ]
    selected = select_qualified_local_proposals([seated, disjoint, *other_firsts], 20)
    ids = {item.id for item in selected}
    assert seated.id in ids
    assert disjoint.id in ids
    late_first = next(item for item in other_firsts if item.metadata["rank"] == 49)
    assert late_first.id not in ids


def test_additional_disjoint_member_must_win_by_rank():
    first = _item("parent-a:0:40", rank=1, parent="parent-a", start=0, end=40)
    weak_disjoint = _item("parent-a:400:440", rank=80, parent="parent-a", start=400, end=440)
    others = [
        _item(f"family-{index}:0:10", rank=index + 1, parent=f"family-{index}", start=0, end=10)
        for index in range(1, 40)
    ]
    selected = select_qualified_local_proposals([first, weak_disjoint, *others], 20)
    ids = {item.id for item in selected}
    assert first.id in ids
    assert weak_disjoint.id not in ids
    assert len(selected) == 20


def test_proposal_budget_unchanged():
    items = [
        _item(f"p-{index}:0:10", rank=index, parent=f"p-{index}", start=0, end=10)
        for index in range(1, 120)
    ]
    selected = select_qualified_local_proposals(items, DEFAULT_RAW_OBSERVATION_K)
    assert len(selected) == DEFAULT_RAW_OBSERVATION_K
