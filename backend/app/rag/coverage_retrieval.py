"""Bounded coverage-oriented merge for multi-intent historical retrieval (G5M)."""
from __future__ import annotations

import re
from collections import defaultdict

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import _MOVEMENT_PAIR_STATEMENT, _MOVEMENT_STATEMENT, _explicit_fragment_actor_conflict, rerank_evidence
from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import RetrievalIntent

_SENTENCE_SPAN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_NON_ASSERTED_MOVEMENT = re.compile(r"\b(?:would|could|might|should|may|planned\s+to|intended\s+to)\b", re.I)
_CONTINUATION_MOVEMENT = re.compile(r"\b(?:marched|advanced|proceeded|moved|travel(?:led|ed|ing)|departed|left|went|crossed|returned|sailed|embarked|passed|landed|entered)\b", re.I)
_STEER_PHYSICAL = re.compile(
    r"\b(?:(?:was|had)\s+)?steer(?:ed|ing|s)?\b(?:\s+(?:his|her|their|the)\s+(?:course|voyage|ships?|fleet))?(?:\s+\w+){0,6}?\s+(?:that\s+way|toward(?:s)?|to|into)\b",
    re.I,
)
_NON_PHYSICAL_STEER = re.compile(
    r"\b(?:discussion|conversation|debate|policy|politics|talk)\s+steer(?:ed|ing|s)?\b|"
    r"\bsteer(?:ed|ing|s)?\s+(?:the\s+)?(?:conversation|discussion|debate|policy|talk)\b|"
    r"\b(?:policy|policies)\s+steer(?:ed|ing|s)?\b|"
    r"\bsteer(?:ed|ing|s)?\s+the\s+army\s+toward\s+reform\b",
    re.I,
)
_STEER_REJECT = re.compile(
    r"\b(?:did\s+not|never|not)\s+steer\b|"
    r"\b(?:discussed|debated)\s+steer(?:ing|ed|s)?\b|"
    r"\b(?:prevented|stopped|blocked|forbidden|barred|hindered)\b(?:\s+\w+){0,8}\s+from\s+steer",
    re.I,
)

DEFAULT_COVERAGE_BUDGET = 20
DEFAULT_PER_INTENT_K = 6
DEFAULT_RAW_OBSERVATION_K = 60


def _proposal_source_key(item: Evidence) -> str:
    return str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])


def _coverage_family_key(item: Evidence) -> str:
    metadata = item.metadata or {}
    if metadata.get("parent_id"):
        return str(metadata["parent_id"])
    return _proposal_source_key(item)


def _qualifies_lexical_proposal(item: Evidence) -> bool:
    if not item.metadata.get("lexical_candidate"):
        return False
    ranking = item.metadata.get("retrieval_ranking") or {}
    if float(ranking.get("navigation_penalty", 0.0)) >= 0.32:
        return False
    return (
        float(ranking.get("lexical_support", 0.0)) >= 0.15
        or float(ranking.get("lexical_rank_relevance", 0.0)) >= 0.65
        or float(ranking.get("lexical_score_relevance", 0.0)) >= 0.45
    )


def _qualifies_semantic_proposal(item: Evidence) -> bool:
    if not item.metadata.get("semantic_candidate"):
        return False
    ranking = item.metadata.get("retrieval_ranking") or {}
    if float(ranking.get("navigation_penalty", 0.0)) >= 0.32:
        return False
    parent_prior = float(ranking.get("parent_semantic_prior", 0.0))
    local_support = float(ranking.get("passage_local_support", 0.0))
    semantic_relevance = float(ranking.get("semantic_relevance", 0.0))
    if semantic_relevance >= 0.08 or local_support >= 0.05:
        return parent_prior >= 0.15
    return False


def _semantic_proposal_key(item: Evidence) -> tuple[float, float, int, str]:
    ranking = item.metadata.get("retrieval_ranking") or {}
    return (
        -float(ranking.get("semantic_relevance", 0.0)),
        -float(ranking.get("parent_semantic_prior", 0.0)),
        int(item.metadata.get("rank") or 999),
        item.id,
    )


def _lexical_proposal_key(item: Evidence) -> tuple[float, float, float, str]:
    ranking = item.metadata.get("retrieval_ranking") or {}
    return (
        -float(ranking.get("lexical_rank_relevance", 0.0)),
        -float(ranking.get("lexical_score_relevance", 0.0)),
        -float(item.metadata.get("lexical_score") or 0.0),
        item.id,
    )


def select_qualified_local_proposals(ranked: list[Evidence], observation_k: int) -> list[Evidence]:
    """Preserve bounded lexical-direct and semantic-derived diversity before proposal cutoff."""
    if observation_k < 1 or not ranked:
        return []
    if len(ranked) <= observation_k:
        return list(ranked)

    lexical_budget = max(6, min(20, observation_k // 3))
    semantic_family_budget = max(8, min(30, observation_k // 2))
    selected: list[Evidence] = []
    selected_ids: set[str] = set()

    def add(item: Evidence) -> None:
        if item.id in selected_ids:
            return
        selected_ids.add(item.id)
        selected.append(item)

    for item in sorted(
        (candidate for candidate in ranked if _qualifies_lexical_proposal(candidate)),
        key=_lexical_proposal_key,
    )[:lexical_budget]:
        add(item)

    semantic_by_source: dict[str, list[Evidence]] = defaultdict(list)
    for item in ranked:
        if _qualifies_semantic_proposal(item):
            semantic_by_source[_proposal_source_key(item)].append(item)

    family_representatives = sorted(
        (sorted(items, key=_semantic_proposal_key)[0] for items in semantic_by_source.values()),
        key=_semantic_proposal_key,
    )
    for item in family_representatives[:semantic_family_budget]:
        add(item)

    for item in ranked:
        if len(selected) >= observation_k:
            break
        add(item)

    return selected[:observation_k]


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


def _continuation_asserted_movement(text: str) -> bool:
    if _NON_ASSERTED_MOVEMENT.search(text):
        return False
    if _STEER_PHYSICAL.search(text):
        return not (_NON_PHYSICAL_STEER.search(text) or _STEER_REJECT.search(text))
    return bool(_MOVEMENT_PAIR_STATEMENT.search(text) or _MOVEMENT_STATEMENT.search(text) or _CONTINUATION_MOVEMENT.search(text))


def _movement_continuation_suffix(user_query: str, selected: Evidence, candidate: Evidence) -> Evidence | None:
    ls, le = selected.metadata.get("passage_start"), selected.metadata.get("passage_end")
    rs, rend = candidate.metadata.get("passage_start"), candidate.metadata.get("passage_end")
    source = selected.metadata.get("source_chunk_id")
    if None in (ls, le, rs, rend, source) or source != candidate.metadata.get("source_chunk_id"):
        return None
    if rs < ls or rs >= le or rend <= le:
        return None
    sel_i, cand_i = selected.metadata.get("passage_index"), candidate.metadata.get("passage_index")
    if sel_i is not None and cand_i is not None and int(cand_i) != int(sel_i) + 1:
        return None
    suffix_start, suffix_end = int(le), int(rend)
    if suffix_end - suffix_start > 1200:
        return None
    suffix_text = (candidate.text or "")[suffix_start - int(rs):]
    if not suffix_text.strip() or suffix_text.strip().casefold() in (selected.text or "").casefold():
        return None
    if len([m for m in _SENTENCE_SPAN.finditer(suffix_text) if m.group().strip()]) != 1:
        return None
    if not _continuation_asserted_movement(suffix_text):
        return None
    if _explicit_fragment_actor_conflict(analyze_query(user_query), suffix_text):
        return None
    source = str(source)
    return candidate.model_copy(update={"id": f"{source}:{suffix_start}:{suffix_end}", "text": suffix_text, "excerpt": suffix_text[:500], "metadata": {**candidate.metadata, "source_chunk_id": source, "passage_start": suffix_start, "passage_end": suffix_end, "continuation_of": selected.id, "continuation_suffix": True}})


def _discover_proactive_continuation(
    user_query: str,
    anchor: Evidence,
    candidates: list[Evidence],
    global_rank: dict[str, int],
) -> Evidence | None:
    if anchor.metadata.get("continuation_suffix"):
        return None
    source = anchor.metadata.get("source_chunk_id")
    if source is None:
        return None
    best: Evidence | None = None
    best_key: tuple[int, str, str] | None = None
    for candidate in candidates:
        if candidate.id == anchor.id or candidate.metadata.get("source_chunk_id") != source:
            continue
        suffix = _movement_continuation_suffix(user_query, anchor, candidate)
        if suffix is None:
            continue
        key = (global_rank.get(candidate.id, 999), candidate.id, suffix.id)
        if best_key is None or key < best_key:
            best_key = key
            best = suffix
    return best


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
        resolved = by_id.get(item.id, item)
        if resolved.id in selected_ids or len(selected) >= budget:
            return False
        if overlaps_selected(resolved):
            for prior in selected:
                if not passages_overlap(resolved, prior):
                    continue
                continuation = _movement_continuation_suffix(user_query, prior, resolved)
                if continuation is None or continuation.id in selected_ids:
                    return False
                if overlaps_selected(continuation):
                    return False
                return add(continuation, intent, rank=rank, reason="same_parent_movement_continuation")
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
        if not resolved.metadata.get("continuation_suffix"):
            continuation = _discover_proactive_continuation(user_query, resolved, ranked, global_rank)
            if (
                continuation is not None
                and continuation.id not in selected_ids
                and len(selected) < budget
                and not overlaps_selected(continuation)
            ):
                add(continuation, intent, rank=rank, reason="same_parent_movement_continuation")
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

    global_fill_families: set[str] = set()

    def _try_global_fill(item: Evidence) -> None:
        if len(selected) >= budget or item.id in selected_ids:
            return
        matched = channels_by_id[item.id]
        preferred = next((entry for entry in matched if entry["kind"] == "CANONICAL"), matched[0])
        intent = intent_by_key[(str(preferred["kind"]), str(preferred["query"]))]
        if add(item, intent, rank=global_rank.get(item.id, 999), reason="global_rank_fill"):
            global_fill_families.add(_coverage_family_key(item))

    for item in ranked:
        if len(selected) >= budget:
            break
        if item.id in selected_ids:
            continue
        if _coverage_family_key(item) in global_fill_families:
            continue
        _try_global_fill(item)

    for item in ranked:
        if len(selected) >= budget:
            break
        if item.id in selected_ids:
            continue
        _try_global_fill(item)

    return selected[:budget]


def top_source_share(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for item in evidence:
        family = str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])
        counts[family] += 1
    return max(counts.values()) / len(evidence)
