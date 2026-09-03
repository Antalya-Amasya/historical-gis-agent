"""Bounded coverage-oriented merge for multi-intent historical retrieval (G5M)."""
from __future__ import annotations

from collections import defaultdict

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import _MOVEMENT_PAIR_STATEMENT, _MOVEMENT_STATEMENT, rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent

DEFAULT_COVERAGE_BUDGET = 20
DEFAULT_PER_INTENT_K = 6


def movement_bearing_text(text: str) -> bool:
    normalized = text or ""
    return bool(_MOVEMENT_PAIR_STATEMENT.search(normalized) or _MOVEMENT_STATEMENT.search(normalized))


def movement_bearing_evidence(evidence: list[Evidence]) -> list[Evidence]:
    return [item for item in evidence if movement_bearing_text(item.text or item.excerpt or "")]


def _annotate(
    item: Evidence,
    intent: RetrievalIntent,
    *,
    rank_within_query: int,
    merge_reason: str,
    matched_intents: list[str],
) -> Evidence:
    metadata = dict(item.metadata)
    provenance = dict(metadata.get("retrieval_provenance") or {})
    provenance.update({
        "retrieval_query": intent.query,
        "retrieval_intent": intent.kind,
        "rank_within_query": rank_within_query,
        "final_merge_reason": merge_reason,
        "matched_intents": list(dict.fromkeys(matched_intents)),
    })
    metadata["retrieval_provenance"] = provenance
    return item.model_copy(update={"metadata": metadata})


def _global_score(item: Evidence) -> float:
    ranking = item.metadata.get("retrieval_ranking") or {}
    return float(ranking.get("final_score", item.score or 0.0))


def merge_coverage_results(
    user_query: str,
    intent_results: list[tuple[RetrievalIntent, list[Evidence]]],
    budget: int,
) -> list[Evidence]:
    """Merge per-intent retrieval with coverage slots then global rank fill."""
    if budget < 1:
        return []

    by_id: dict[str, Evidence] = {}
    intents_by_id: dict[str, list[str]] = defaultdict(list)
    best_rank: dict[str, int] = {}

    for intent, items in intent_results:
        ranked = rerank_evidence(intent.query, items)
        for rank, item in enumerate(ranked, start=1):
            intents_by_id[item.id].append(intent.kind)
            if item.id not in by_id or rank < best_rank[item.id]:
                by_id[item.id] = item
                best_rank[item.id] = rank

    selected_ids: set[str] = set()
    selected: list[Evidence] = []

    def add(item: Evidence, intent: RetrievalIntent, *, rank: int, reason: str) -> None:
        if item.id in selected_ids or len(selected) >= budget:
            return
        selected_ids.add(item.id)
        selected.append(
            _annotate(
                by_id[item.id],
                intent,
                rank_within_query=rank,
                merge_reason=reason,
                matched_intents=intents_by_id[item.id],
            )
        )

    # Coverage slot: best movement-bearing item per intent when available.
    for intent, items in intent_results:
        ranked = rerank_evidence(intent.query, items)
        for rank, item in enumerate(ranked, start=1):
            if movement_bearing_text(item.text or ""):
                add(item, intent, rank=rank, reason="intent_movement_coverage")
                break

    # Coverage slot: at least one representative per intent.
    for intent, items in intent_results:
        ranked = rerank_evidence(intent.query, items)
        for rank, item in enumerate(ranked, start=1):
            add(item, intent, rank=rank, reason="intent_coverage_slot")
            break

    # Fill remaining budget by global relevance across all intents.
    global_ranked = sorted(
        by_id.values(),
        key=lambda item: (-_global_score(item), best_rank.get(item.id, 999), item.id),
    )
    for item in global_ranked:
        if len(selected) >= budget:
            break
        if item.id in selected_ids:
            continue
        intent_kind = intents_by_id[item.id][0]
        intent = next((entry for entry, _ in intent_results if entry.kind == intent_kind), intent_results[0][0])
        add(item, intent, rank=best_rank.get(item.id, 999), reason="global_rank_fill")

    return selected[:budget]


def top_source_share(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for item in evidence:
        family = str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])
        counts[family] += 1
    return max(counts.values()) / len(evidence)
