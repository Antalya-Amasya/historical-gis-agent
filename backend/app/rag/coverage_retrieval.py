"""Bounded coverage-oriented merge for multi-intent historical retrieval (G5M)."""
from __future__ import annotations

from collections import defaultdict

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import _MOVEMENT_PAIR_STATEMENT, _MOVEMENT_STATEMENT, rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent

DEFAULT_COVERAGE_BUDGET = 20
DEFAULT_PER_INTENT_K = 6
DEFAULT_RAW_OBSERVATION_K = 60


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
    matched_channels: list[dict[str, object]] | None = None,
) -> Evidence:
    metadata = dict(item.metadata)
    provenance = dict(metadata.get("retrieval_provenance") or {})
    provenance.update({
        "retrieval_query": intent.query,
        "retrieval_intent": intent.kind,
        "rank_within_query": rank_within_query,
        "final_merge_reason": merge_reason,
        "matched_intents": list(dict.fromkeys(matched_intents)),
        "matched_channels": list(matched_channels or []),
    })
    metadata["retrieval_provenance"] = provenance
    return item.model_copy(update={"metadata": metadata})


def passages_overlap(left: Evidence, right: Evidence) -> bool:
    source = left.metadata.get("source_chunk_id")
    if source is None or source != right.metadata.get("source_chunk_id"):
        return False
    start, end = left.metadata.get("passage_start"), left.metadata.get("passage_end")
    other_start, other_end = right.metadata.get("passage_start"), right.metadata.get("passage_end")
    if None in (start, end, other_start, other_end):
        return False
    return max(start, other_start) < min(end, other_end)


def _merge_channel_candidate(current: Evidence, incoming: Evidence) -> Evidence:
    metadata = dict(current.metadata)
    incoming_meta = incoming.metadata
    if incoming_meta.get("semantic_candidate"):
        metadata["semantic_candidate"] = True
        current_rank = metadata.get("vector_rank")
        incoming_rank = incoming_meta.get("vector_rank")
        if incoming_rank is not None and (current_rank is None or incoming_rank < current_rank):
            metadata["vector_rank"] = incoming_rank
    if incoming_meta.get("lexical_candidate"):
        metadata["lexical_candidate"] = True
        current_lex = float(metadata.get("lexical_score") or 0.0)
        incoming_lex = float(incoming_meta.get("lexical_score") or 0.0)
        if incoming_lex >= current_lex:
            metadata["lexical_score"] = incoming_lex
    return current.model_copy(update={"metadata": metadata})


def merge_coverage_results(
    user_query: str,
    intent_results: list[tuple[RetrievalIntent, list[Evidence]]],
    budget: int,
) -> list[Evidence]:
    """Union channel proposals, score once against the user query, then select."""
    if budget < 1 or not intent_results:
        return []

    by_id: dict[str, Evidence] = {}
    channels_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    stable_channels = sorted(
        enumerate(intent_results),
        key=lambda pair: (pair[1][0].kind, pair[1][0].query, pair[0]),
    )
    intent_by_key = {(intent.kind, intent.query): intent for intent, _items in intent_results}

    for channel_index, (intent, items) in stable_channels:
        seen_local: set[str] = set()
        local_rank = 0
        for item in items:
            if item.id in seen_local:
                continue
            seen_local.add(item.id)
            local_rank += 1
            channels_by_id[item.id].append({
                "kind": intent.kind,
                "query": intent.query,
                "channel_index": channel_index,
                "local_rank": local_rank,
            })
            if item.id not in by_id:
                by_id[item.id] = item
            else:
                by_id[item.id] = _merge_channel_candidate(by_id[item.id], item)

    ranked = rerank_evidence(user_query, list(by_id.values()), pool_relative=False)
    by_id = {item.id: item for item in ranked}
    global_rank = {item.id: int(item.metadata.get("rank") or 999) for item in ranked}

    selected_ids: set[str] = set()
    selected: list[Evidence] = []

    def overlaps_selected(item: Evidence) -> bool:
        return any(passages_overlap(item, prior) for prior in selected)

    def add(item: Evidence, intent: RetrievalIntent, *, rank: int, reason: str) -> bool:
        resolved = by_id[item.id]
        if resolved.id in selected_ids or len(selected) >= budget:
            return False
        if overlaps_selected(resolved):
            return False
        selected_ids.add(resolved.id)
        matched = channels_by_id[resolved.id]
        selected.append(
            _annotate(
                resolved,
                intent,
                rank_within_query=rank,
                merge_reason=reason,
                matched_intents=[str(entry["kind"]) for entry in matched],
                matched_channels=matched,
            )
        )
        return True

    channel_ranked: list[tuple[RetrievalIntent, list[Evidence]]] = []
    for _index, (intent, items) in stable_channels:
        unique_ids = list(dict.fromkeys(item.id for item in items if item.id in by_id))
        ordered = sorted(unique_ids, key=lambda ident: (global_rank.get(ident, 999), ident))
        channel_ranked.append((intent, [by_id[ident] for ident in ordered]))

    for intent, items in channel_ranked:
        for rank, item in enumerate(items, start=1):
            if not movement_bearing_text(item.text or item.excerpt or ""):
                continue
            if add(item, intent, rank=rank, reason="intent_movement_coverage"):
                break

    for intent, items in channel_ranked:
        for rank, item in enumerate(items, start=1):
            if add(item, intent, rank=rank, reason="intent_coverage_slot"):
                break

    for item in ranked:
        if len(selected) >= budget:
            break
        matched = channels_by_id[item.id]
        preferred = next((entry for entry in matched if entry["kind"] == "CANONICAL"), matched[0])
        intent = intent_by_key[(str(preferred["kind"]), str(preferred["query"]))]
        add(item, intent, rank=global_rank.get(item.id, 999), reason="global_rank_fill")

    return selected[:budget]


def top_source_share(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for item in evidence:
        family = str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])
        counts[family] += 1
    return max(counts.values()) / len(evidence)
