from unittest.mock import patch

from backend.app.models import Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, merge_coverage_results
from backend.app.rag.retrieval_intents import RetrievalIntent


QUERY = "Trace the route from Italy into Epirus."


def _item(identifier, text, *, score, semantic=True, lexical=False, qualify=True, source=None):
    ranking = {
        "final_score": score,
        "navigation_penalty": 0.0,
        "lexical_support": 0.2 if lexical and qualify else 0.0,
        "lexical_rank_relevance": 0.0,
        "lexical_score_relevance": 0.0,
        "semantic_relevance": 0.1 if semantic and qualify else 0.0,
        "parent_semantic_prior": 0.2 if semantic and qualify else 0.0,
        "passage_local_support": 0.1 if semantic and qualify else 0.0,
    }
    metadata = {
        "document_id": "doc",
        "source_chunk_id": source or identifier,
        "semantic_candidate": semantic,
        "lexical_candidate": lexical,
        "rank": 1,
        "retrieval_ranking": ranking,
    }
    return Evidence(
        id=identifier, author="Author", work="Work", locator="1",
        excerpt=text, text=text, score=score, metadata=metadata,
    )


def _rerank(_query, items, pool_relative=False):
    ordered = sorted(items, key=lambda item: (-float((item.metadata.get("retrieval_ranking") or {}).get("final_score") or 0), item.id))
    result = []
    for rank, item in enumerate(ordered, start=1):
        ranking = dict(item.metadata.get("retrieval_ranking") or {})
        ranking["final_score"] = float(ranking.get("final_score") or item.score or 0)
        result.append(item.model_copy(update={"metadata": {**item.metadata, "rank": rank, "retrieval_ranking": ranking}}))
    return result


def _merge(channels, budget=5):
    with patch("backend.app.rag.coverage_retrieval.rerank_evidence", side_effect=_rerank):
        return merge_coverage_results(QUERY, channels, budget)


def _reason(item):
    return str((item.metadata.get("retrieval_provenance") or {}).get("final_merge_reason"))


def test_canonical_only_common_rank_leader_remains_selected():
    leader = _item("canon-1", "Notes about Italy and Epirus.", score=0.99)
    filler = [_item(f"f-{index}", f"Camp note {index}.", score=0.5 - index * 0.01) for index in range(8)]
    merged = _merge([(RetrievalIntent("CANONICAL", QUERY), [leader, *filler])], budget=3)
    assert "canon-1" in {item.id for item in merged}
    assert _reason(next(item for item in merged if item.id == "canon-1")) != "canonical_intent_reservation"


def test_rank1_canonical_admitted_qualified_gets_reservation():
    reserved = _item("shared", "Notes about the winter quarters.", score=0.01, lexical=True)
    canonical = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(5)]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), [*canonical, reserved]),
            (RetrievalIntent("EPISODE", "winter episode"), [reserved, _item("ep-2", "Second episode note.", score=0.02)]),
        ],
        budget=4,
    )
    by_id = {item.id: item for item in merged}
    assert "shared" in by_id
    assert _reason(by_id["shared"]) == "canonical_intent_reservation"
    assert len(merged) == 4


def test_rank1_not_canonical_admitted_gets_no_reservation():
    only_episode = _item("solo", "Notes about the winter quarters.", score=0.01, lexical=True)
    canonical = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(6)]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), canonical),
            (RetrievalIntent("EPISODE", "winter episode"), [only_episode]),
        ],
        budget=3,
    )
    assert "solo" not in {item.id for item in merged}
    assert all(_reason(item) != "canonical_intent_reservation" for item in merged)


def test_canonical_plus_rank2_intent_gets_no_reservation():
    second = _item("rank2", "Notes about the winter quarters.", score=0.01, lexical=True)
    first = _item("rank1", "Other episode note.", score=0.02, lexical=True)
    canonical = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(6)]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), [*canonical, second]),
            (RetrievalIntent("EPISODE", "winter episode"), [first, second]),
        ],
        budget=3,
    )
    assert "rank2" not in {item.id for item in merged}
    assert all(item.id != "rank2" or _reason(item) != "canonical_intent_reservation" for item in merged)


def test_unqualified_rank1_gets_no_reservation():
    weak = _item("weak", "Notes about the winter quarters.", score=0.01, lexical=True, qualify=False, semantic=False)
    canonical = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(6)]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), [*canonical, weak]),
            (RetrievalIntent("EPISODE", "winter episode"), [weak]),
        ],
        budget=3,
    )
    assert "weak" not in {item.id for item in merged}


def test_same_id_selected_once_across_channels():
    shared = _item("shared", "Notes about the winter quarters.", score=0.01, lexical=True)
    canonical = [_item("c-0", "Canon note 0.", score=0.99), shared]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), canonical),
            (RetrievalIntent("EPISODE", "winter episode"), [shared]),
            (RetrievalIntent("SUBJECT", "subject note"), [shared]),
        ],
        budget=5,
    )
    assert [item.id for item in merged].count("shared") == 1


def test_channel_order_permutation_same_final_ids():
    shared = _item("shared", "Notes about the winter quarters.", score=0.01, lexical=True)
    canonical = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(3)] + [shared]
    episode = [shared]
    a = _merge(
        [(RetrievalIntent("CANONICAL", QUERY), canonical), (RetrievalIntent("EPISODE", "winter episode"), episode)],
        budget=4,
    )
    b = _merge(
        [(RetrievalIntent("EPISODE", "winter episode"), episode), (RetrievalIntent("CANONICAL", QUERY), canonical)],
        budget=4,
    )
    assert {item.id for item in a} == {item.id for item in b}


def test_global_fill_order_and_budget_unchanged_apart_from_reservation():
    reserved = _item("shared", "Notes about the winter quarters.", score=0.0, lexical=True)
    fillers = [_item(f"c-{index}", f"Canon note {index}.", score=0.9 - index * 0.01) for index in range(8)]
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), [*fillers, reserved]),
            (RetrievalIntent("EPISODE", "winter episode"), [reserved]),
        ],
        budget=4,
    )
    assert len(merged) == 4
    assert merged[0].id == "shared"
    assert _reason(merged[0]) == "canonical_intent_reservation"
    assert [item.id for item in merged[1:]] == ["c-0", "c-1", "c-2"]
    assert all(_reason(item) == "global_rank_fill" for item in merged[1:])
    assert DEFAULT_COVERAGE_BUDGET == 20


def test_existing_facet_coverage_still_selects_movement_slot():
    movement = _item(
        "move",
        "Caesar marched from Italy into Epirus across the Adriatic.",
        score=0.4,
        lexical=True,
    )
    high = _item(
        "high",
        "Julius Caesar crossed from Italy across the Adriatic into Epirus.",
        score=0.95,
        lexical=True,
    )
    merged = _merge(
        [
            (RetrievalIntent("CANONICAL", QUERY), [movement]),
            (RetrievalIntent("MOVEMENT", "Italy Epirus march"), [high]),
        ],
        budget=4,
    )
    reasons = {_reason(item) for item in merged}
    assert "intent_movement_coverage" in reasons or "intent_coverage_slot" in reasons
    assert "high" in {item.id for item in merged}
