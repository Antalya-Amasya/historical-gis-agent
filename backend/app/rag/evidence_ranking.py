"""Small, explainable post-retrieval ordering for primary-source evidence.

The vector store remains the recall mechanism.  This module only orders a
bounded candidate set, so it cannot manufacture evidence or turn retrieval
into keyword search.
"""
from __future__ import annotations

import re
import statistics

from backend.app.models import Evidence
from backend.app.rag.query_roles import (
    action_support as role_action_support,
    analyze_query,
    generic_support as role_generic_support,
    location_support as role_location_support,
    normalized_tokens,
    person_support as role_person_support,
)

_STRONG_NAV_SOURCES = frozenset({"toc", "contents", "navigation", "index"})
_NAV_SOURCE_HINTS = frozenset({"epub3_nav", "epub_nav", "ncx", "nav"})
_NAVIGATION_STARTS = (
    "the following is contained",
    "table of contents",
    "contents",
    "this ebook is for the use",
)
_CONTENTS_LEAD = re.compile(r"^\s*(?:\d+\s+)?the following is contained\b", re.IGNORECASE)
_NUMBERED_BOOK_LIST = re.compile(r"^\s*book\s+[ivxlcdm0-9]+\.?\s+(?:\d+\s+){8,}", re.IGNORECASE)
_CHAPTER_TOC = re.compile(
    r"^\s*(?:how|about)\b.{0,160}\(chapters?\s+[ivxlcdm0-9]+(?:\s*[-–]\s*[ivxlcdm0-9]+)?\)",
    re.IGNORECASE | re.DOTALL,
)
_FRONT_MATTER = re.compile(
    r"(?is)(?:this ebook is for the use|^\s*title:\s|\*\*\*\s*start of (?:the )?project gutenberg)",
)
_STRUCTURAL_HEADING = re.compile(r"^(?:index|contents|table of contents)\.?\s*$", re.IGNORECASE)
_ROUTE_OR_MOVEMENT_QUERY = re.compile(
    r"\b(?:route|march(?:ed|ing)?|movement|moved|advance(?:d|ment)?|cross(?:ed|ing)?|journey|expedition)\b|路线|行军|行进|进军",
    re.IGNORECASE,
)
_MOVEMENT_STATEMENT = re.compile(
    r"\b(?:march(?:ed|ing)?|moved|advance(?:d|ment)?|cross(?:ed|ing)?|arriv(?:ed|ing)|depart(?:ed|ing)?|left|entered|passed|proceeded|travel(?:led|ed|ing)?)\b",
    re.IGNORECASE,
)


def _is_structural_heading(value: str) -> bool:
    return bool(_STRUCTURAL_HEADING.match((value or "").strip()))


def _text_is_navigation(text: str) -> bool:
    stripped = (text or "").lstrip()
    folded = stripped.casefold()
    return (
        folded.startswith(_NAVIGATION_STARTS)
        or bool(_CONTENTS_LEAD.match(stripped))
        or bool(_NUMBERED_BOOK_LIST.match(stripped))
        or bool(_CHAPTER_TOC.match(stripped))
        or bool(_FRONT_MATTER.search(stripped[:500]))
    )


def is_navigation_or_heading(evidence: Evidence) -> bool:
    """TOC/index metadata or navigational text; structural EPUB hints are not enough."""
    source = str(evidence.metadata.get("navigation_source", "")).casefold()
    if source in _STRONG_NAV_SOURCES:
        return True
    heading = str(evidence.metadata.get("heading") or "")
    if _is_structural_heading(heading):
        return True
    return _text_is_navigation(evidence.text or "")


def is_route_or_movement_query(query: str) -> bool:
    """Identify the narrow retrieval mode whose objective includes episode coverage."""
    return bool(_ROUTE_OR_MOVEMENT_QUERY.search(query or ""))


def diversify_route_evidence(query: str, ranked: list[Evidence]) -> list[Evidence]:
    """Order a route candidate pool for bounded source-family coverage.

    This is selection diversity, not historical ordering: it uses only the
    existing rank, source-chunk identity, structural noise detection, and an
    evidence-local movement statement signal.  It never creates evidence or
    assigns chronology.
    """
    if not is_route_or_movement_query(query):
        return ranked

    def family(item: Evidence) -> str:
        return str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])

    def useful(item: Evidence) -> bool:
        return not is_navigation_or_heading(item)

    selected: list[Evidence] = []
    used: set[str] = set()

    # Take one movement-bearing body passage per source family first.  This
    # prevents many non-overlapping windows from one retrieved source chunk
    # from crowding out independent movement-episode evidence.
    for item in ranked:
        key = family(item)
        if key not in used and useful(item) and _MOVEMENT_STATEMENT.search(item.text or ""):
            selected.append(item)
            used.add(key)
    # Preserve source diversity even when a relevant passage has no explicit
    # verb (for example a compact battle or arrival statement).
    for item in ranked:
        key = family(item)
        if key not in used and useful(item):
            selected.append(item)
            used.add(key)
    # Structural items and additional siblings are only fallback material;
    # their original deterministic rank remains their order within fallback.
    selected_ids = {item.id for item in selected}
    selected.extend(item for item in ranked if item.id not in selected_ids)
    return selected


def rerank_evidence(query: str, evidence: list[Evidence]) -> list[Evidence]:
    """Return the same evidence with transparent, deterministic ordering data.

    Score components remain modest: vector similarity is still the primary
    semantic signal; role-aware supports only order a bounded candidate set.
    """
    roles = analyze_query(query)
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
        person = role_person_support(roles, text_tokens)
        location = role_location_support(roles, text_tokens)
        action = role_action_support(roles, text_tokens, person=person, location=location)
        generic = role_generic_support(roles, text_tokens)
        statement_bonus = 0.04 if action >= 0.12 and len(text_tokens) >= 20 else 0.0
        joint = 0.04 if person > 0 and location > 0 else 0.0
        entity_support = person
        local_support = min(1.0, (person / 0.08) * 0.40 + (action / 0.12) * 0.40 + (location / 0.06) * 0.10 + (statement_bonus / 0.04) * 0.10) if (person or action or location or statement_bonus) else 0.0
        semantic_relevance = parent_semantic_prior * (0.20 + 0.80 * local_support) if item.metadata.get("semantic_candidate") else 0.0
        lexical_rank_relevance = percentile(lexical_order.get(item.id), len(lexical_items))
        lexical_score = float(item.metadata.get("lexical_score", 0.0))
        lexical_score_relevance = lexical_score / (lexical_score + lexical_median) if lexical_score > 0 and lexical_median > 0 else 0.0
        role_parts = []
        if roles.person_terms:
            role_parts.append(person / 0.08)
        if roles.location_terms:
            role_parts.append(location / 0.06)
        coverage = sum(role_parts) / len(role_parts) if role_parts else 1.0
        lexical_support = (0.60 * lexical_score_relevance + 0.40 * lexical_rank_relevance) * (0.35 + 0.65 * coverage) if item.metadata.get("lexical_candidate") else 0.0
        passage_relevance = max(semantic_relevance, lexical_support)
        channel_confidence = 0.02 if item.metadata.get("semantic_candidate") and item.metadata.get("lexical_candidate") else 0.0
        navigation_penalty = 0.32 if is_navigation_or_heading(item) else 0.0
        final_score = passage_relevance + channel_confidence + entity_support + location + action + generic + joint + statement_bonus - navigation_penalty
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
            "person_support": round(person, 6),
            "location_support": round(location, 6),
            "generic_support": round(generic, 6),
            "joint_support": round(joint, 6),
            "action_support": round(action, 6),
            "statement_bonus": round(statement_bonus, 6),
            "navigation_penalty": round(navigation_penalty, 6),
            "final_score": round(final_score, 6),
        }
        ranked.append(item.model_copy(update={"score": round(final_score, 4), "metadata": metadata}))
    ranked.sort(key=lambda item: (-item.metadata["retrieval_ranking"]["final_score"], item.metadata.get("vector_rank", item.metadata.get("rank", 0))))
    return [item.model_copy(update={"metadata": {**item.metadata, "rank": rank}}) for rank, item in enumerate(ranked, start=1)]
