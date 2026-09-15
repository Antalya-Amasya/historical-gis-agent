"""G7G: bounded local proposal preservation contract tests."""
from __future__ import annotations

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_RAW_OBSERVATION_K,
    _coverage_family_key,
    _qualifies_lexical_proposal,
    _qualifies_proposal,
    select_qualified_local_proposals,
)
from backend.app.rag.evidence_ranking import rerank_evidence

QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)


def _evidence(
    identifier: str,
    text: str,
    *,
    score: float = 0.5,
    lexical_score: float | None = None,
    semantic: bool = True,
    vector_rank: int = 10,
    source: str | None = None,
    ranking: dict | None = None,
) -> Evidence:
    metadata = {
        "document_id": "doc",
        "parent_id": source or identifier.split(":", 1)[0],
        "source_chunk_id": source or identifier.split(":", 1)[0],
        "semantic_candidate": semantic,
        "vector_rank": vector_rank,
        "retrieval_ranking": ranking or {"final_score": score},
    }
    if lexical_score is not None:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
        metadata["retrieval_ranking"] = {
            **metadata["retrieval_ranking"],
            "lexical_support": 0.9,
            "lexical_rank_relevance": 0.9,
        }
    if semantic:
        metadata["retrieval_ranking"] = {
            **metadata["retrieval_ranking"],
            "semantic_relevance": 0.2,
            "parent_semantic_prior": 0.2,
            "passage_local_support": 0.1,
        }
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text[:500],
        text=text,
        score=score,
        metadata=metadata,
    )


def test_qualified_lexical_preserved_despite_family_representatives():
    flood = [
        _evidence(
            f"family-a:{index * 10}:{index * 10 + 80}",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.95 - index * 0.0005,
            vector_rank=index,
            source="family-a",
        )
        for index in range(1, 80)
    ]
    target = _evidence(
        "family-b:800:880",
        "Julius Caesar put to sea in winter and passed the Ionian Sea to Oricum.",
        score=0.05,
        lexical_score=42.0,
        semantic=False,
        vector_rank=55,
        source="family-b",
    )
    ranked = rerank_evidence(QUERY, flood + [target])
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert target.id in {item.id for item in preserved}


def test_qualified_semantic_from_independent_family_preserved():
    target = _evidence(
        "family-target:1391:1640",
        "Caesar crossed the Adriatic from Brundisium in the winter and opened his campaign.",
        score=0.08,
        vector_rank=58,
        source="family-target",
    )
    flood = [
        _evidence(
            f"family-{index}:{index * 10}:{index * 10 + 80}",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.9 - index * 0.0005,
            vector_rank=index,
            source=f"family-{index}",
        )
        for index in range(1, 70)
    ]
    ranked = rerank_evidence(QUERY, [target, *flood])
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert target.id in {item.id for item in preserved}


def test_unqualified_candidate_not_promoted_for_diversity():
    qualified = _evidence(
        "strong:10:20",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.9,
        lexical_score=30.0,
        vector_rank=5,
        source="strong-parent",
    )
    weak = _evidence(
        "weak:10:20",
        "The senate debated grain supplies in Rome that winter.",
        score=0.01,
        lexical_score=0.5,
        semantic=False,
        vector_rank=999,
        source="weak-parent",
    )
    ranked = rerank_evidence(QUERY, [qualified, weak])
    preserved = select_qualified_local_proposals(ranked, 1)
    assert preserved[0].id == qualified.id


def test_bounded_rescue_replaces_worse_unqualified_candidate():
    qualified = _evidence(
        "family-b:800:880",
        "Julius Caesar put to sea in winter and passed the Ionian Sea to Oricum.",
        score=0.05,
        lexical_score=42.0,
        semantic=False,
        vector_rank=38,
        source="family-b",
    )
    tail = [
        _evidence(
            f"tail:{index}:0:10",
            "Generic winter camp notes without a crossing.",
            score=0.01,
            semantic=False,
            lexical_score=None,
            vector_rank=100 + index,
            source=f"tail-{index}",
        )
        for index in range(1, 80)
    ]
    flood = [
        _evidence(
            f"family-a:{index * 10}:{index * 10 + 80}",
            f"Julius Caesar marched from Italy toward Epirus in passage {index}.",
            score=0.95 - index * 0.0005,
            vector_rank=index,
            source="family-a",
        )
        for index in range(1, 80)
    ]
    ranked = rerank_evidence(QUERY, [qualified, *flood, *tail])
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert qualified.id in {item.id for item in preserved}


def test_qualified_rescue_cannot_evict_another_qualified_candidate():
    better = _evidence(
        "family-a:10:20",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.85,
        vector_rank=22,
        source="family-a",
    )
    preserved = _evidence(
        "family-b:800:880",
        "Julius Caesar put to sea in winter and passed the Ionian Sea to Oricum.",
        score=0.05,
        lexical_score=42.0,
        semantic=False,
        vector_rank=38,
        source="family-b",
    )
    tail = [
        _evidence(
            f"tail:{index}:0:10",
            "Generic winter camp notes without a crossing.",
            score=0.01,
            semantic=False,
            lexical_score=None,
            vector_rank=100 + index,
            source=f"tail-{index}",
        )
        for index in range(1, 80)
    ]
    ranked = rerank_evidence(QUERY, [better, preserved, *tail])
    selected = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert preserved.id in {item.id for item in selected}


def test_family_diversity_does_not_hard_cap_family_to_one():
    first = _evidence("family-a:0:10", "Caesar marched toward Epirus.", vector_rank=1, source="family-a")
    second = _evidence("family-a:20:30", "Caesar crossed the Adriatic toward Epirus.", vector_rank=2, source="family-a")
    filler = [
        _evidence(f"other-{index}:0:10", f"Other passage {index}.", vector_rank=10 + index, source=f"other-{index}")
        for index in range(1, 80)
    ]
    ranked = rerank_evidence(QUERY, [first, second, *filler])
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    family_ids = [item.id for item in preserved if _coverage_family_key(item) == "family-a"]
    assert len(family_ids) >= 2


def test_canonical_fill_allows_repeats_after_unique_families_exhausted():
    items = [
        _evidence("a:0:10", "Caesar marched toward Epirus.", vector_rank=1, source="a"),
        _evidence("b:0:10", "Caesar marched toward Epirus again.", vector_rank=2, source="b"),
        _evidence("c:0:10", "Generic winter camp notes.", vector_rank=3, source="c", semantic=False, lexical_score=None),
    ]
    ranked = rerank_evidence(QUERY, items)
    preserved = select_qualified_local_proposals(ranked, 2)
    assert [item.id for item in preserved] == ["a:0:10", "b:0:10"]


def test_mandatory_lexical_reserve_preserved():
    lexical = [
        _evidence(
            f"lex-{index}:0:10",
            "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
            score=0.9 - index * 0.01,
            lexical_score=40.0 - index,
            vector_rank=50 + index,
            source=f"lex-{index}",
        )
        for index in range(1, 25)
    ]
    ranked = rerank_evidence(QUERY, lexical)
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    lexical_selected = [item for item in preserved if _qualifies_lexical_proposal(item)]
    assert len(lexical_selected) >= 6


def test_proposal_count_stays_bounded():
    candidates = [
        _evidence(f"id:{index}:0:10", f"Caesar marched toward Epirus passage {index}.", vector_rank=index)
        for index in range(1, 200)
    ]
    ranked = rerank_evidence(QUERY, candidates)
    preserved = select_qualified_local_proposals(ranked, DEFAULT_RAW_OBSERVATION_K)
    assert len(preserved) <= DEFAULT_RAW_OBSERVATION_K


def test_duplicate_candidate_appears_once():
    dual = _evidence(
        "dup:10:20",
        "Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.4,
        lexical_score=20.0,
        vector_rank=15,
        source="dup-parent",
    )
    ranked = rerank_evidence(QUERY, [dual])
    preserved = select_qualified_local_proposals(ranked, 5)
    assert len(preserved) == 1


def test_unqualified_cannot_displace_better_ranked_candidate_via_diversity():
    top = _evidence("top:0:10", "Caesar marched toward Epirus.", vector_rank=1, source="top")
    weak = _evidence(
        "weak:0:10",
        "The senate debated grain supplies in Rome that winter.",
        score=0.01,
        semantic=False,
        lexical_score=None,
        vector_rank=999,
        source="weak",
    )
    ranked = rerank_evidence(QUERY, [top, weak])
    preserved = select_qualified_local_proposals(ranked, 1)
    assert preserved[0].id == top.id
    assert not any(not _qualifies_proposal(item) for item in preserved if item.id == weak.id)
