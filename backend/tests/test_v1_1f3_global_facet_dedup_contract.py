"""V1.1F3R3: feasible reservation-only facet dedup contract (RED on b56ec85)."""
from __future__ import annotations

from collections import Counter

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, merge_coverage_results
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent

QUERY = (
    "Trace Commander Alpha's route from Harbor One through Cape Two "
    "to Region Delta during the Northern Delta Campaign."
)
INTENTS = (
    RetrievalIntent("CANONICAL", QUERY),
    RetrievalIntent("SUBJECT", "Commander Alpha Harbor One march"),
    RetrievalIntent("EPISODE", "Northern Delta Campaign Alpha campaign"),
    RetrievalIntent("MOVEMENT", "Commander Alpha marched Harbor One Cape Two Region Delta"),
    RetrievalIntent("ENDPOINT", "Harbor One Cape Two Region Delta route"),
    RetrievalIntent("FEATURE", "Commander Alpha cross Harbor One Cape Two"),
)
COMPLETE_ROUTE_ID = "family-complete:0:120"
UNIQUE_CHANNEL_ID = "family-unique:0:120"
EMPTY_FACET_ID = "family-empty:0:120"
EPISODE_CONTROL_ID = "family-episode:0:120"
ROUTE_CORE = (
    "Commander Alpha marched from Harbor One through Cape Two toward Region Delta "
    "during the Northern Delta Campaign."
)
REDUNDANT_MOVEMENT_IDS = frozenset(f"family-{kind}:0:120" for kind in (
    "canonical", "subject", "episode", "movement", "endpoint", "feature",
))
REDUNDANT_COMPLEMENTARY_IDS = frozenset(f"family-{kind}:0:120" for kind in ("bio", "loc", "episode-prose"))
RESERVED_REASONS = frozenset({"intent_movement_coverage", "intent_coverage_slot"})
GLOBAL_REASON = "global_rank_fill"
CORRECTED_REACHABLE_DEPTH = 19
BASELINE_REACHABLE_DEPTH = 13


def _reason(item: Evidence) -> str:
    return str((item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason"))


def _evidence(
    identifier: str,
    text: str,
    *,
    source: str,
    lexical_score: float,
    vector_rank: int,
    ranking: dict[str, float] | None = None,
    navigation_path: str | None = None,
) -> Evidence:
    metadata = {
        "document_id": "doc",
        "source_chunk_id": source,
        "semantic_candidate": True,
        "lexical_candidate": True,
        "lexical_score": lexical_score,
        "vector_rank": vector_rank,
    }
    if navigation_path:
        metadata["navigation_path"] = navigation_path
    if ranking:
        metadata["retrieval_ranking"] = ranking
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text[:500],
        text=text,
        score=0.5,
        metadata=metadata,
    )


def _merge_stats(merged: list[Evidence]) -> dict[str, int]:
    reasons = Counter(_reason(item) for item in merged)
    global_ranks = [
        int(item.metadata.get("rank") or 999)
        for item in merged
        if _reason(item) == GLOBAL_REASON
    ]
    return {
        "reserved": sum(reasons[r] for r in RESERVED_REASONS),
        "global_fill": reasons[GLOBAL_REASON],
        "depth": max(global_ranks, default=0),
    }


def _build_fixture() -> tuple[list[tuple[RetrievalIntent, list[Evidence]]], dict[str, Evidence], int]:
    redundant_specs = (
        ("canonical", ROUTE_CORE, 24.0, 1),
        ("subject", "Commander Alpha marched from Harbor One through Cape Two toward Region Delta.", 23.0, 2),
        ("episode", f"During the Northern Delta Campaign, {ROUTE_CORE}", 22.0, 3),
        ("movement", "Commander Alpha marched from Harbor One through Cape Two toward Region Delta.", 21.0, 4),
        ("endpoint", "Commander Alpha advanced from Harbor One through Cape Two toward Region Delta.", 20.0, 5),
        ("feature", "Commander Alpha marched from Harbor One across Cape Two toward Region Delta.", 19.0, 6),
    )
    complementary_specs = (
        ("bio", "Commander Alpha was born in a distant province and later entered politics.", 18.0, 7),
        ("loc", "Harbor One, Cape Two, and Region Delta were often discussed in geographic surveys.", 17.0, 8),
        ("episode-prose", "The Northern Delta Campaign opened with supply gathering near Harbor One.", 16.0, 9),
        ("empty", "Annual provincial census rolls listed population totals without route detail.", 15.0, 10),
    )
    pool = {
        f"family-{kind}:0:120": _evidence(f"family-{kind}:0:120", text, source=f"family-{kind}", lexical_score=lex, vector_rank=vr)
        for kind, text, lex, vr in redundant_specs + complementary_specs
    }
    pool[UNIQUE_CHANNEL_ID] = _evidence(
        UNIQUE_CHANNEL_ID,
        "During the Northern Delta Campaign the convoy put to sea.",
        source="family-unique",
        lexical_score=8.0,
        vector_rank=25,
        navigation_path=None,
    )
    unique_meta = dict(pool[UNIQUE_CHANNEL_ID].metadata)
    unique_meta["heading"] = "ALPHA"
    pool[UNIQUE_CHANNEL_ID] = pool[UNIQUE_CHANNEL_ID].model_copy(update={"metadata": unique_meta})
    pool[COMPLETE_ROUTE_ID] = _evidence(
        COMPLETE_ROUTE_ID,
        "Commander Alpha sailed from Harbor One across the open channel into Region Delta during the Northern Delta Campaign.",
        source="family-complete",
        lexical_score=7.0,
        vector_rank=35,
    )
    decoys = [
        _evidence(
            f"family-decoy-{index:02d}:0:120",
            f"{ROUTE_CORE[:-1]} at waypoint {index}.",
            source=f"family-decoy-{index:02d}",
            lexical_score=30.0 - index * 0.05,
            vector_rank=index,
        )
        for index in range(1, 15)
    ]
    fillers = [
        _evidence(
            f"family-fill-{index:02d}:0:120",
            f"Commander Beta marched from Harbor One through Cape Two toward Region Delta in segment {index}.",
            source=f"family-fill-{index:02d}",
            lexical_score=12.0 - index * 0.1,
            vector_rank=30 + index,
        )
        for index in range(1, 12)
    ]
    pool.update({item.id: item for item in decoys + fillers})
    ranked_list = rerank_evidence(QUERY, list(pool.values()), pool_relative=False)
    ranked = {item.id: item for item in ranked_list}
    complete_rank = int(ranked[COMPLETE_ROUTE_ID].metadata.get("rank") or 999)

    def channel(*item_ids: str) -> list[Evidence]:
        return [ranked[item_id] for item_id in item_ids if item_id in ranked]

    decoy_ids = [item.id for item in decoys]
    filler_ids = [item.id for item in fillers]
    intent_results = [
        (INTENTS[0], channel("family-canonical:0:120", "family-bio:0:120", COMPLETE_ROUTE_ID, *decoy_ids, *filler_ids)),
        (INTENTS[1], channel("family-subject:0:120", "family-loc:0:120", "family-episode-prose:0:120")),
        (INTENTS[2], channel(EPISODE_CONTROL_ID, UNIQUE_CHANNEL_ID)),
        (INTENTS[3], channel("family-movement:0:120")),
        (INTENTS[4], channel("family-endpoint:0:120", EMPTY_FACET_ID)),
        (INTENTS[5], channel("family-feature:0:120")),
    ]
    return intent_results, ranked, complete_rank


def test_global_facet_dedup_contract():
    intent_results, ranked, complete_rank = _build_fixture()
    assert len(ranked) > 20
    assert 16 <= complete_rank <= 17

    merged = merge_coverage_results(QUERY, intent_results, budget=DEFAULT_COVERAGE_BUDGET)
    stats = _merge_stats(merged)
    assert complete_rank > BASELINE_REACHABLE_DEPTH
    assert complete_rank <= CORRECTED_REACHABLE_DEPTH
    assert stats["reserved"] <= 2
    assert stats["global_fill"] >= 18
    assert stats["depth"] >= complete_rank

    merged_ids = {item.id for item in merged}
    by_id = {item.id: item for item in merged}

    movement_reserved = sum(
        1 for item in merged if item.id in REDUNDANT_MOVEMENT_IDS and _reason(item) == "intent_movement_coverage"
    )
    complementary_reserved = sum(
        1 for item in merged if item.id in REDUNDANT_COMPLEMENTARY_IDS and _reason(item) == "intent_coverage_slot"
    )
    general_reserved = sum(1 for item in merged if _reason(item) == "intent_coverage_slot")

    assert movement_reserved <= 1
    assert complementary_reserved <= 1
    assert general_reserved <= 2
    assert EMPTY_FACET_ID not in merged_ids or _reason(by_id[EMPTY_FACET_ID]) != "intent_coverage_slot"

    assert EPISODE_CONTROL_ID in ranked
    assert any(
        item.id in REDUNDANT_MOVEMENT_IDS and _reason(item) == GLOBAL_REASON
        for item in merged
    )

    assert UNIQUE_CHANNEL_ID in merged_ids
    assert _reason(by_id[UNIQUE_CHANNEL_ID]) in RESERVED_REASONS
    assert float((ranked[UNIQUE_CHANNEL_ID].metadata.get("retrieval_ranking") or {}).get("route_fragment_relevance") or 0) > 0

    assert COMPLETE_ROUTE_ID in ranked
    complete = by_id.get(COMPLETE_ROUTE_ID)
    assert complete is not None, "complete route starved from final evidence"
    assert _reason(complete) == GLOBAL_REASON
    assert _reason(complete) not in RESERVED_REASONS

    assert len(merged) == DEFAULT_COVERAGE_BUDGET
