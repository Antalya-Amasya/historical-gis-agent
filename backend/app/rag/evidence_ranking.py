"""Small, explainable post-retrieval ordering for primary-source evidence.

The vector store remains the recall mechanism.  This module only orders a
bounded candidate set, so it cannot manufacture evidence or turn retrieval
into keyword search.
"""
from __future__ import annotations

import re
import statistics
import unicodedata

from backend.app.models import Evidence


_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset({"a", "an", "and", "at", "battle", "by", "for", "in", "of", "on", "the", "to", "with"})
_NAVIGATION_STARTS = (
    "the following is contained",
    "table of contents",
    "contents",
    "this ebook is for the use",
)
_CONTENTS_LEAD = re.compile(r"^\s*(?:\d+\s+)?the following is contained\b", re.IGNORECASE)
_NUMBERED_BOOK_LIST = re.compile(r"^\s*book\s+[ivxlcdm0-9]+\.?\s+(?:\d+\s+){8,}", re.IGNORECASE)
_ACTION_GROUPS = (
    frozenset({"assassination", "assassinate", "assassinated", "murder", "murdered", "slain", "killed", "stabbed"}),
    frozenset({"battle", "battled", "fought", "fight", "defeated", "defeat", "vanquished", "victory", "victorious"}),
)


def normalized_tokens(text: str) -> frozenset[str]:
    """Case-fold and remove classical diacritics (Cæsar -> caesar)."""
    normalized = unicodedata.normalize("NFKD", text).replace("æ", "ae").replace("Æ", "AE")
    return frozenset(_WORD.findall(normalized.casefold()))


def is_navigation_or_heading(evidence: Evidence) -> bool:
    """Recognize only explicit navigation artefacts, never ordinary prose headings."""
    metadata = evidence.metadata
    source = str(metadata.get("navigation_source", "")).casefold()
    if source in {"toc", "contents", "navigation", "index"}:
        return True
    text = (evidence.text or "").lstrip().casefold()
    return text.startswith(_NAVIGATION_STARTS) or bool(_CONTENTS_LEAD.match(text)) or bool(_NUMBERED_BOOK_LIST.match(text))


def _action_support(query_tokens: frozenset[str], text_tokens: frozenset[str]) -> bool:
    return any(query_tokens & group and text_tokens & group for group in _ACTION_GROUPS)


def rerank_evidence(query: str, evidence: list[Evidence]) -> list[Evidence]:
    """Return the same evidence with transparent, deterministic ordering data.

    Score components intentionally remain modest: vector similarity is still
    the primary signal; explicit TOC/navigation material is demoted only when
    it competes with a statement-bearing primary-source passage.
    """
    query_tokens = normalized_tokens(query)
    entity_tokens = query_tokens - _STOP_WORDS - frozenset().union(*_ACTION_GROUPS)
    semantic_items = [item for item in evidence if item.metadata.get("semantic_candidate")]
    lexical_items = [item for item in evidence if item.metadata.get("lexical_candidate")]
    semantic_order = {item.id: rank for rank, item in enumerate(sorted(semantic_items, key=lambda item: item.metadata.get("vector_rank", 0)), 1)}
    lexical_order = {item.id: rank for rank, item in enumerate(sorted(lexical_items, key=lambda item: (-float(item.metadata.get("lexical_score", 0.0)), item.id)), 1)}

    def percentile(rank: int | None, population: int) -> float:
        return 0.0 if rank is None or not population else (population - rank + 1) / population
    lexical_median = statistics.median(float(item.metadata.get("lexical_score", 0.0)) for item in lexical_items) if lexical_items else 0.0
    ranked: list[Evidence] = []
    for item in evidence:
        text_tokens = normalized_tokens(item.text or "")
        parent_semantic_prior = percentile(semantic_order.get(item.id), len(semantic_items))
        lexical_score = float(item.metadata.get("lexical_score", 0.0))
        entity_support = min(0.08, 0.04 * len(entity_tokens & text_tokens))
        # Action vocabulary is only a supporting signal when the candidate
        # also carries a substantive query entity.  This prevents generic
        # words such as "battle" from promoting an unrelated battle passage.
        action_support = 0.12 if entity_support and _action_support(query_tokens, text_tokens) else 0.0
        statement_bonus = 0.04 if action_support and len(text_tokens) >= 20 else 0.0
        local_support = min(1.0, (entity_support / 0.08) * 0.45 + (action_support / 0.12) * 0.45 + (statement_bonus / 0.04) * 0.10)
        # Parent ANN rank is a prior only.  A weak child cannot inherit a
        # near-perfect relevance merely because its source chunk ranked first.
        semantic_relevance = parent_semantic_prior * (0.20 + 0.80 * local_support) if item.metadata.get("semantic_candidate") else 0.0
        lexical_rank_relevance = percentile(lexical_order.get(item.id), len(lexical_items))
        lexical_score = float(item.metadata.get("lexical_score", 0.0))
        lexical_score_relevance = lexical_score / (lexical_score + lexical_median) if lexical_score > 0 and lexical_median > 0 else 0.0
        lexical_support = 0.60 * lexical_score_relevance + 0.40 * lexical_rank_relevance if item.metadata.get("lexical_candidate") else 0.0
        passage_relevance = max(semantic_relevance, lexical_support)
        channel_confidence = 0.02 if item.metadata.get("semantic_candidate") and item.metadata.get("lexical_candidate") else 0.0
        navigation_penalty = 0.32 if is_navigation_or_heading(item) else 0.0
        final_score = passage_relevance + channel_confidence + entity_support + action_support + statement_bonus - navigation_penalty
        metadata = dict(item.metadata)
        metadata["retrieval_ranking"] = {
            "base_vector_score": round(parent_semantic_prior, 6),
            "semantic_relevance": round(semantic_relevance, 6),
            "parent_semantic_prior": round(parent_semantic_prior, 6),
            "passage_local_support": round(local_support, 6),
            "lexical_rank_relevance": round(lexical_rank_relevance, 6),
            "lexical_score_relevance": round(lexical_score_relevance, 6),
            "lexical_support": round(lexical_support, 6),
            "passage_relevance": round(passage_relevance, 6),
            "channel_confidence": round(channel_confidence, 6),
            "entity_support": round(entity_support, 6),
            "action_support": round(action_support, 6),
            "statement_bonus": round(statement_bonus, 6),
            "navigation_penalty": round(navigation_penalty, 6),
            "final_score": round(final_score, 6),
        }
        ranked.append(item.model_copy(update={"score": round(final_score, 4), "metadata": metadata}))
    ranked.sort(key=lambda item: (-item.metadata["retrieval_ranking"]["final_score"], item.metadata.get("vector_rank", item.metadata.get("rank", 0))))
    return [item.model_copy(update={"metadata": {**item.metadata, "rank": rank}}) for rank, item in enumerate(ranked, start=1)]
