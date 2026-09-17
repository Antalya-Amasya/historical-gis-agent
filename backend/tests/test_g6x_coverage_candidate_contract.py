"""G6X: coverage union before truncation; comparable global scores."""
from __future__ import annotations

from unittest.mock import patch

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_PER_INTENT_K, _coverage_facets, merge_coverage_results
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent
from backend.app.rag.retriever import ChromaHistoricalRetriever

CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
EPISODE_QUERY = "Pharsalus 48 BCE campaign Caesar Epirus"


def _evidence(
    identifier: str,
    text: str,
    *,
    score: float = 0.5,
    lexical_score: float | None = None,
    semantic: bool = True,
    vector_rank: int = 10,
    source: str | None = None,
) -> Evidence:
    metadata = {
        "document_id": "doc",
        "source_chunk_id": source or identifier.split(":", 1)[0],
        "semantic_candidate": semantic,
        "vector_rank": vector_rank,
        "retrieval_ranking": {"final_score": score},
    }
    if lexical_score is not None:
        metadata["lexical_candidate"] = True
        metadata["lexical_score"] = lexical_score
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


def _final_score(item: Evidence) -> float:
    ranking = item.metadata.get("retrieval_ranking") or {}
    return float(ranking.get("final_score", item.score or 0.0))


def _provenance_channels(item: Evidence) -> set[tuple[str, str]]:
    provenance = item.metadata.get("retrieval_provenance") or {}
    channels = provenance.get("matched_channels") or []
    if channels:
        return {(str(entry["kind"]), str(entry["query"])) for entry in channels}
    intents = provenance.get("matched_intents") or []
    query = provenance.get("retrieval_query")
    return {(str(kind), str(query)) for kind in intents}


def test_a_pre_union_cutoff_does_not_hide_rank7_from_coverage_union():
    """GPT-6 F03: a globally strong rank-7 child must reach the coverage union."""
    weak = "Generic winter camp notes without a crossing."
    strong = (
        "Julius Caesar crossed from Italy across the Adriatic into Epirus "
        "and marched toward Pharsalus in 48 BCE."
    )
    intent_pool = [
        _evidence(f"A{index}", weak, score=0.9 - index * 0.01, vector_rank=index)
        for index in range(1, 7)
    ]
    a7 = _evidence("A7", strong, score=0.1, vector_rank=90, lexical_score=40.0)
    intent_pool.append(a7)
    canonical_pool = [
        _evidence(f"C{index}", weak, score=0.5, vector_rank=index)
        for index in range(1, 8)
    ]

    def pool_for(query: str) -> list[Evidence]:
        return canonical_pool if query == CAESAR_QUERY else intent_pool

    retriever = ChromaHistoricalRetriever(object(), None)
    captured: list[list[tuple[RetrievalIntent, list[Evidence]]]] = []

    def fake_retrieve(query: str, top_k: int = 5, filters=None):
        return pool_for(query)[:top_k]

    def fake_candidates(query: str, observation_k: int = 60, filters=None):
        return pool_for(query)

    def capturing_merge(user_query, intent_results, budget):
        captured.append(list(intent_results))
        return merge_coverage_results(user_query, intent_results, budget)

    retriever.retrieve = fake_retrieve  # type: ignore[method-assign]
    retriever.retrieve_candidates = fake_candidates  # type: ignore[method-assign]
    with patch("backend.app.rag.retriever.merge_coverage_results", capturing_merge):
        retriever.retrieve_with_coverage(CAESAR_QUERY, budget=20, per_intent_k=DEFAULT_PER_INTENT_K)

    union_ids = {item.id for _intent, items in captured[0] for item in items}
    assert "A7" in union_ids


def test_b_pool_size_does_not_change_global_coverage_score():
    """GPT-6 F02: unrelated low-score pool mates must not move comparable scores."""
    target = _evidence(
        "target",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        lexical_score=10.0,
        vector_rank=5,
    )
    competitor = _evidence(
        "competitor",
        "The senate debated grain supplies in Rome that winter.",
        lexical_score=9.0,
        vector_rank=6,
    )
    junk = [
        _evidence(f"junk-{index}", "Unrelated agricultural remark.", lexical_score=0.1, vector_rank=80 + index)
        for index in range(12)
    ]
    local_small = rerank_evidence(CAESAR_QUERY, [target, competitor])
    local_large = rerank_evidence(CAESAR_QUERY, [target, competitor, *junk])
    small_local = next(item for item in local_small if item.id == "target")
    large_local = next(item for item in local_large if item.id == "target")
    assert abs(_final_score(small_local) - _final_score(large_local)) > 0.05

    merged_small = merge_coverage_results(
        CAESAR_QUERY,
        [(RetrievalIntent("CANONICAL", CAESAR_QUERY), [target, competitor])],
        budget=20,
    )
    merged_large = merge_coverage_results(
        CAESAR_QUERY,
        [(RetrievalIntent("CANONICAL", CAESAR_QUERY), [target, competitor, *junk])],
        budget=20,
    )
    score_small = _final_score(next(item for item in merged_small if item.id == "target"))
    score_large = _final_score(next(item for item in merged_large if item.id == "target"))
    assert abs(score_small - score_large) < 1e-6


def test_c_channel_order_does_not_change_final_ids_or_provenance():
    shared = _evidence(
        "shared",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        lexical_score=12.0,
    )
    canon_only = _evidence("canon-only", "Caesar landed near Epirus after the crossing.")
    episode_only = _evidence("episode-only", "After Pharsalus the campaign continued in 48 BCE.")
    order_a = [
        (RetrievalIntent("CANONICAL", CAESAR_QUERY), [shared, canon_only]),
        (RetrievalIntent("EPISODE", EPISODE_QUERY), [shared, episode_only]),
    ]
    order_b = list(reversed(order_a))
    merged_a = merge_coverage_results(CAESAR_QUERY, order_a, budget=20)
    merged_b = merge_coverage_results(CAESAR_QUERY, order_b, budget=20)
    assert [item.id for item in merged_a] == [item.id for item in merged_b]
    shared_a = next(item for item in merged_a if item.id == "shared")
    shared_b = next(item for item in merged_b if item.id == "shared")
    assert _provenance_channels(shared_a) == _provenance_channels(shared_b)
    assert ("CANONICAL", CAESAR_QUERY) in _provenance_channels(shared_a)
    assert ("EPISODE", EPISODE_QUERY) in _provenance_channels(shared_a)


def test_d_duplicate_id_merges_all_channel_provenance():
    semantic = _evidence(
        "shared-id",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        semantic=True,
        lexical_score=None,
        vector_rank=3,
    )
    lexical = _evidence(
        "shared-id",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        semantic=False,
        lexical_score=18.0,
        vector_rank=40,
    )
    merged = merge_coverage_results(
        CAESAR_QUERY,
        [
            (RetrievalIntent("CANONICAL", CAESAR_QUERY), [semantic]),
            (RetrievalIntent("EPISODE", EPISODE_QUERY), [lexical]),
        ],
        budget=5,
    )
    assert sum(1 for item in merged if item.id == "shared-id") == 1
    item = next(entry for entry in merged if entry.id == "shared-id")
    assert item.metadata.get("semantic_candidate") is True
    assert item.metadata.get("lexical_candidate") is True
    assert item.metadata.get("lexical_score") == 18.0
    channels = _provenance_channels(item)
    assert ("CANONICAL", CAESAR_QUERY) in channels
    assert ("EPISODE", EPISODE_QUERY) in channels


def _merge_reason(item: Evidence) -> str:
    return str((item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason"))


def test_distinct_channel_facet_retains_reserved_protection():
    shared_movement = _evidence("X", "Caesar marched from Italy into Epirus across the Adriatic.")
    same_facet_channel_next = _evidence("Y", "Caesar marched from Brundisium toward Epirus.")
    high_rank = _evidence(
        "Z",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus and toward Pharsalus.",
        lexical_score=50.0,
        vector_rank=1,
    )
    distinct_fragment = _evidence(
        "U",
        "During the campaign in 48 BCE the convoy put to sea.",
        vector_rank=40,
    )
    distinct_fragment = distinct_fragment.model_copy(
        update={"metadata": {**distinct_fragment.metadata, "heading": "CAESAR"}},
    )
    merged = merge_coverage_results(
        CAESAR_QUERY,
        [
            (RetrievalIntent("CANONICAL", CAESAR_QUERY), [shared_movement]),
            (RetrievalIntent("EPISODE", EPISODE_QUERY), [shared_movement, same_facet_channel_next, distinct_fragment]),
            (RetrievalIntent("MOVEMENT", "Caesar Italy Epirus march"), [high_rank]),
        ],
        budget=4,
    )
    by_id = {item.id: item for item in merged}
    shared_facets = _coverage_facets(CAESAR_QUERY, shared_movement)
    assert shared_facets
    assert _coverage_facets(CAESAR_QUERY, same_facet_channel_next) == shared_facets
    assert "ROUTE_FRAGMENT" in _coverage_facets(CAESAR_QUERY, by_id["U"])

    assert _merge_reason(by_id["X"]) == "intent_movement_coverage"
    assert _merge_reason(by_id["U"]) == "intent_coverage_slot"
    assert "Y" in by_id
    assert _merge_reason(by_id["Y"]) == "global_rank_fill"
    assert _merge_reason(by_id["Y"]) not in {"intent_movement_coverage", "intent_coverage_slot"}


def test_i7_final_budget_remains_20():
    items = [
        _evidence(f"id-{index}", f"Caesar marched from camp {index} into Epirus.", score=1 - index * 0.01)
        for index in range(40)
    ]
    merged = merge_coverage_results(
        CAESAR_QUERY,
        [(RetrievalIntent("CANONICAL", CAESAR_QUERY), items)],
        budget=20,
    )
    assert len(merged) == 20
