"""Small, explainable post-retrieval ordering for primary-source evidence.

The vector store remains the recall mechanism.  This module only orders a
bounded candidate set, so it cannot manufacture evidence or turn retrieval
into keyword search.
"""
from __future__ import annotations

import re
import statistics
import math
from dataclasses import replace

from backend.app.models import Evidence
from backend.app.rag.retrieval_intents import primary_route_subject
from backend.app.rag.query_roles import (
    action_support as role_action_support,
    analyze_query,
    episode_context_terms,
    extract_subject_context_terms,
    generic_support as role_generic_support,
    location_support as role_location_support,
    movement_scoring_terms,
    normalized_tokens,
    person_support as role_person_support,
    route_movement_query,
    _MOVEMENT_TERM_TO_FAMILY,
    _ROUTE_LOCATION_PREP,
    _GEOGRAPHIC_COMPOUND_HEADS,
    _GEOGRAPHIC_COMPOUND_TAILS,
    _PRAENOMINA,
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
    r"\b(?:route|march(?:ed|ing)?|movements?|moved|advance(?:d|ment)?|cross(?:ed|ing)?|journey|expedition)\b|路线|行军|行进|进军",
    re.IGNORECASE,
)
_MOVEMENT_STATEMENT = re.compile(
    r"\b(?:march(?:ed|ing)?|moved|advance(?:d|ment)?|cross(?:ed|ing)?|arriv(?:ed|ing)|depart(?:ed|ing)?|left|entered|passed|proceeded|travel(?:led|ed|ing)?)\b",
    re.IGNORECASE,
)
_MOVEMENT_PAIR_STATEMENT = re.compile(
    r"\b(?:march(?:ed|es|ing)?|moved|advance(?:d|ment)?|cross(?:ed|ing)?|"
    r"arriv(?:ed|ing)|depart(?:ed|ing)?|left|entered|passed|proceeded|"
    r"travel(?:led|ed|ing)?)\b(?!\s+(?:was|is|were|has|had)\b)"
    r"[^,;.!?]{0,50}\bfrom\b[^,;.!?]{1,80}"
    r"\b(?:to|into|toward(?:s)?)\b",
    re.IGNORECASE,
)
_ROUTE_FRAGMENT_MAX = 0.06
_FRAGMENT_PRONOUNS = frozenset({"he", "she", "they", "it", "his", "her", "their"})
_EXPLICIT_ACTOR_MOVEMENT = re.compile(
    r"(?<![A-Za-z])(?P<actor>[A-Z][a-z'']+(?:\s+[A-Z][a-z'']+)?)\s+"
    r"(?:crossed|marched|marches|marching|travel(?:led|ed|ing)|landed|sailed|sailing|"
    r"advanced|advancing|proceeded|proceeding|passed|passing|enter(?:ed|ing)|depart(?:ed|ing)|"
    r"left|moved|moving|put)\b",
)
_FRAGMENT_MOVEMENT_PHRASE = re.compile(r"\b(?:put to sea|made over|quitted|quit)\b", re.IGNORECASE)


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


_ORDINARY_POLAR_QA = re.compile(
    r"^\s*(?:did|does|do|was|were|is|are|has|have|had)\b",
    re.IGNORECASE,
)
_ROUTE_STYLE_DIVERSITY_CUE = re.compile(
    r"\b(?:trace|reconstruct|map|follow|show|route|routes|journey|expedition)\b|路线|军队",
    re.IGNORECASE,
)


def _applies_route_source_diversity(query: str) -> bool:
    """Flat-pool source-family diversity is for route-style retrieval, not polar QA.

    Coverage branch selection still uses `is_route_or_movement_query` unchanged.
    A yes/no question may contain a movement noun without asking for route
    reconstruction, and in that case the rerank order must be preserved.
    """
    if not is_route_or_movement_query(query):
        return False
    if _ORDINARY_POLAR_QA.search(query or "") and not _ROUTE_STYLE_DIVERSITY_CUE.search(query or ""):
        return False
    return True


def _explicit_fragment_actor_conflict(roles, text: str) -> bool:
    for match in _EXPLICIT_ACTOR_MOVEMENT.finditer(text or ""):
        actor_tokens = normalized_tokens(match.group("actor"))
        if not actor_tokens or actor_tokens <= _FRAGMENT_PRONOUNS:
            continue
        if not (actor_tokens & roles.person_terms):
            return True
    return False


_PERSON_SCAFFOLD = frozenset({"commander", "general", "king", "emperor", "vi"})


def _passage_person_signals(query: str, roles, text: str, text_tokens: frozenset[str]) -> tuple[float, float] | None:
    ptokens = normalized_tokens(primary_route_subject(query, roles) or "")
    if not ptokens:
        return None
    core = (ptokens - _PERSON_SCAFFOLD) or ptokens
    present = core & text_tokens
    primary, secondary = 0.0, 0.0
    opponents = roles.person_terms - ptokens
    for match in _EXPLICIT_ACTOR_MOVEMENT.finditer(text or ""):
        actor = normalized_tokens(match.group("actor"))
        if not actor or actor <= _FRAGMENT_PRONOUNS:
            continue
        if actor & core and (core <= actor or actor <= ptokens):
            primary = 0.08
            break
        if not (actor & core) and actor & opponents:
            secondary = 0.02
    if primary == 0.0 and present and secondary == 0.0 and not _explicit_fragment_actor_conflict(roles, text):
        primary = 0.04
    return primary, secondary


def _local_person_support(roles, text: str, tokens: frozenset[str]) -> float:
    """Ranking-only literal support; unrelated names cannot veto a local surname.

    Caius/Gaius is an orthographic variant, not a registry membership test.
    A conflicting given name must occur in the same surface name phrase.
    This neither resolves same-full-name homonyms nor assigns actor authority.
    """
    canonical = lambda term: "gaius" if term == "caius" else term
    sequence = tuple(canonical(term) for term in roles.person_sequence)
    terms = frozenset(canonical(term) for term in roles.person_terms)
    tokens = frozenset(canonical(term) for term in tokens)
    local_roles = replace(roles, person_terms=terms, person_sequence=sequence)
    support = role_person_support(local_roles, tokens)
    if len(sequence) < 2 or not terms & tokens or terms <= tokens:
        return support
    surname = sequence[-1]
    if surname not in tokens:
        return support if roles.multiple_person_phrases_detected else 0.0
    # Only an adjacent capitalized name span supplies a competing praenomen.
    for match in re.finditer(r"\b(?:[A-Z][a-z]+\s+){0,2}[A-Z][a-z]+\b", text or ""):
        name = frozenset(canonical(term) for term in normalized_tokens(match.group()))
        if surname in name and (name & _PRAENOMINA) - terms:
            return 0.0
    return max(support, 0.04)


def route_fragment_relevance(
    query: str,
    roles,
    item: Evidence,
    text: str,
    text_tokens: frozenset[str],
    *,
    person: float,
    location: float,
    action: float,
) -> float:
    if not roles.person_terms or not route_movement_query(normalized_tokens(query), roles):
        return 0.0
    if person >= 0.08 or (person >= 0.04 and location > 0 and action >= 0.12):
        return 0.0
    if _explicit_fragment_actor_conflict(roles, text):
        return 0.0
    if not (
        action > 0
        or _MOVEMENT_STATEMENT.search(text or "")
        or _FRAGMENT_MOVEMENT_PHRASE.search(text or "")
        or movement_scoring_terms(roles) & text_tokens
    ):
        return 0.0
    if not (extract_subject_context_terms(item.metadata) & roles.person_terms):
        return 0.0
    if not (
        location > 0
        or _MOVEMENT_PAIR_STATEMENT.search(text or "")
        or episode_context_terms(roles) & text_tokens
        or text_tokens & (_GEOGRAPHIC_COMPOUND_TAILS | _GEOGRAPHIC_COMPOUND_HEADS)
    ):
        return 0.0
    transition = _FRAGMENT_MOVEMENT_PHRASE.search(text or "") or _MOVEMENT_PAIR_STATEMENT.search(text or "")
    if location == 0 and action == 0:
        return _ROUTE_FRAGMENT_MAX if transition else 0.0
    if person == 0 and location > 0 and action > 0 and re.search(r"\b(?:he|she|they)\b", text or "", re.I) and episode_context_terms(roles) & text_tokens:
        return _ROUTE_FRAGMENT_MAX / 2
    return 0.0


def diversify_route_evidence(query: str, ranked: list[Evidence]) -> list[Evidence]:
    """Order a route candidate pool for bounded source-family coverage.

    This is selection diversity, not historical ordering: it uses only the
    existing rank, source-chunk identity, structural noise detection, and an
    evidence-local movement statement signal.  It never creates evidence or
    assigns chronology.
    """
    if not _applies_route_source_diversity(query):
        return ranked

    def family(item: Evidence) -> str:
        return str(item.metadata.get("source_chunk_id") or item.id.split(":", 1)[0])

    def useful(item: Evidence) -> bool:
        return not is_navigation_or_heading(item)

    roles = _ranking_roles(query)

    def in_query_scope(item: Evidence) -> bool:
        """Require a pair candidate to retain an available actor/region scope.

        This is only a bounded retrieval-selection gate.  It does not assert
        that either textual endpoint is historically valid or ordered.
        """
        ranking = item.metadata.get("retrieval_ranking") or {}
        scoped_support = []
        if roles.person_terms:
            scoped_support.append(float(ranking.get("person_support", 0.0)))
        if roles.location_terms:
            scoped_support.append(float(ranking.get("location_support", 0.0)))
        if roles.person_terms and not roles.multiple_person_phrases_detected:
            return float(ranking.get("person_support", 0.0)) > 0
        return not scoped_support or any(value > 0 for value in scoped_support)

    selected: list[Evidence] = []
    used: set[str] = set()
    pair_used: set[str] = set()

    # Preserve one locally scoped directional-movement passage per source
    # family before a broader movement sibling consumes that family's slot.
    # The lexical shape affects retrieval coverage only; extraction remains
    # the authority for endpoints and movement direction.
    for item in ranked:
        key = family(item)
        if (
            key not in pair_used
            and useful(item)
            and in_query_scope(item)
            and _MOVEMENT_PAIR_STATEMENT.search(item.text or "")
        ):
            selected.append(item)
            pair_used.add(key)

    # Take one movement-bearing body passage per source family first.  This
    # prevents many non-overlapping windows from one retrieved source chunk
    # from crowding out independent movement-episode evidence.
    for item in ranked:
        key = family(item)
        if key not in used and useful(item) and in_query_scope(item) and _MOVEMENT_STATEMENT.search(item.text or ""):
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


# Calibrated on the v2 E5 candidate distribution: median ~.75, P90 ~.772.
# The ceiling equals the existing navigation penalty and stays below the
# .40 passage-local person component. These signals select evidence only.
_VECTOR_BONUS_MAX = 0.32
_VECTOR_SIMILARITY_FLOOR = 0.75
_VECTOR_SIMILARITY_CEILING = 0.78
_NOMINAL_MOVEMENT_FORMS = {"entry": "enter", "arrival": "arrive", "departure": "depart"}


def _ranking_roles(query: str):
    """Correct grammatical/nominal roles locally, without expanding retrieval queries."""
    roles = analyze_query(query)
    prepositions = _ROUTE_LOCATION_PREP | {"against"}
    nominal = normalized_tokens(query) & _NOMINAL_MOVEMENT_FORMS.keys()
    person = roles.person_terms - prepositions - nominal
    forms = frozenset().union(*(
        _MOVEMENT_TERM_TO_FAMILY[_NOMINAL_MOVEMENT_FORMS[term]] for term in nominal
    ))
    return replace(
        roles, person_terms=frozenset(person),
        person_sequence=tuple(term for term in roles.person_sequence if term in person),
        context_terms=roles.context_terms - prepositions,
        action_terms=roles.action_terms | nominal,
        movement_inflection_terms=roles.movement_inflection_terms | forms,
    )


def _query_scope_weights(roles, tokens, *, pool_relative: bool):
    groups = [frozenset({term}) for term in sorted(
        roles.person_terms | roles.location_terms | roles.context_terms
    )]
    if roles.action_terms:
        groups.append(roles.action_terms | roles.expanded_action_terms | movement_scoring_terms(roles))
    weighted = []
    for group in groups:
        count = sum(bool(group & item) for item in tokens)
        if count:
            weight = math.log1p(len(tokens) / (1 + count)) if pool_relative else 1.0
            weighted.append((group, weight))
    return weighted


def _vector_signal(item: Evidence) -> tuple[float | None, float]:
    # Evidence.score is overwritten by reranking; only the original distance
    # is a stable vector signal. It belongs to the source chunk, not its child.
    distance = item.metadata.get("distance")
    if not item.metadata.get("semantic_candidate") or distance is None:
        return None, 0.0
    distance = float(distance)
    if not math.isfinite(distance) or distance < 0:
        return None, 0.0
    similarity = 1 / (1 + distance)
    strength = max(0.0, min(1.0, (similarity - _VECTOR_SIMILARITY_FLOOR) /
                               (_VECTOR_SIMILARITY_CEILING - _VECTOR_SIMILARITY_FLOOR)))
    return similarity, _VECTOR_BONUS_MAX * strength * strength * (3 - 2 * strength)


def rerank_evidence(query: str, evidence: list[Evidence], *, pool_relative: bool = True) -> list[Evidence]:
    """Return the same evidence with transparent, deterministic ordering data.

    Source-vector similarity contributes a bounded retrieval prior alongside
    passage-local role and lexical signals; it is not child-level relevance.
    Scope coverage reduces unsupported keyword matches without adding authority.
    Coverage comparison must pass pool_relative=False so channel-local
    percentile/median calibration is not treated as a global score.
    """
    roles = _ranking_roles(query)
    tokens_by_id = {item.id: normalized_tokens(item.text or "") for item in evidence}
    scope_weights = _query_scope_weights(roles, list(tokens_by_id.values()), pool_relative=pool_relative)
    scope_total = sum(weight for _, weight in scope_weights)
    semantic_items = [item for item in evidence if item.metadata.get("semantic_candidate")]
    lexical_items = [item for item in evidence if item.metadata.get("lexical_candidate")]
    semantic_order = {item.id: rank for rank, item in enumerate(sorted(semantic_items, key=lambda item: item.metadata.get("vector_rank", 0)), 1)}
    lexical_sorted = sorted(lexical_items, key=lambda item: (-float(item.metadata.get("lexical_score", 0.0)), item.id))
    # Equal lexical evidence must not acquire different relevance from source IDs.
    # Keep the best competition rank for ties; recall and score magnitude stay intact.
    lexical_ranks = {}
    lexical_order = {}
    for rank, item in enumerate(lexical_sorted, 1):
        lexical_score = float(item.metadata.get("lexical_score", 0.0))
        lexical_order[item.id] = lexical_ranks.setdefault(lexical_score, rank)

    def percentile(rank: int | None, population: int) -> float:
        return 0.0 if rank is None or not population else (population - rank + 1) / population
    lexical_median = statistics.median(float(item.metadata.get("lexical_score", 0.0)) for item in lexical_items) if lexical_items else 0.0
    ranked: list[Evidence] = []
    for item in evidence:
        text_tokens = tokens_by_id[item.id]
        parent_semantic_prior = percentile(semantic_order.get(item.id), len(semantic_items))
        passage_text = item.text or ""
        person = _local_person_support(roles, passage_text, text_tokens)
        signals = _passage_person_signals(query, roles, passage_text, text_tokens)
        primary_subject = signals[0] if signals else 0.0
        secondary_context = signals[1] if signals else 0.0
        location = role_location_support(roles, text_tokens)
        if signals:
            action = role_action_support(
                roles, text_tokens, person=primary_subject if primary_subject >= 0.08 else 0.0, location=location
            )
            if primary_subject >= 0.08:
                person_local = (primary_subject / 0.08) * 0.40
            elif primary_subject == 0.04:
                person_local = (primary_subject / 0.08) * 0.40
            else:
                person_local = 0.0
            entity_support = 0.0 if primary_subject >= 0.08 else (
                0.04 if primary_subject >= 0.04 else 0.0
            )
            joint = 0.0
        else:
            action = role_action_support(roles, text_tokens, person=person, location=location)
            statement_bonus = 0.04 if action >= 0.12 and len(text_tokens) >= 20 else 0.0
            joint = 0.04 if person > 0 and location > 0 else 0.0
            route_local_evidence = action > 0 or statement_bonus > 0
            entity_support = person
            person_local = (person / 0.08) * 0.40
            if person >= 0.08 and roles.multiple_person_phrases_detected:
                person_local = (0.04 / 0.08) * 0.40
                entity_support = 0.04
            elif person >= 0.08 and not route_local_evidence:
                entity_support = 0.04
                person_local = (0.04 / 0.08) * 0.40
        generic = role_generic_support(roles, text_tokens)
        statement_bonus = 0.04 if action >= 0.12 and len(text_tokens) >= 20 else 0.0
        route_local_evidence = action > 0 or statement_bonus > 0
        frag_person = primary_subject if signals and primary_subject >= 0.08 else (0.0 if signals else person)
        frag_action = (
            role_action_support(roles, text_tokens, person=person, location=location)
            if signals and primary_subject < 0.08
            else action
        )
        fragment = route_fragment_relevance(
            query, roles, item, passage_text, text_tokens,
            person=frag_person, location=location, action=frag_action,
        )
        if signals:
            joint = 0.04 if location > 0 and primary_subject >= 0.08 else 0.0
        fragment_local = (fragment / _ROUTE_FRAGMENT_MAX) * 0.20 if fragment else 0.0
        local_support = min(1.0, person_local + (action / 0.12) * 0.40 + (location / 0.06) * 0.10 + (statement_bonus / 0.04) * 0.10 + fragment_local) if (primary_subject or person or action or location or statement_bonus or fragment) else 0.0
        semantic_relevance = local_support if item.metadata.get("semantic_candidate") else 0.0
        lexical_score = float(item.metadata.get("lexical_score", 0.0))
        if pool_relative:
            lexical_rank_relevance = percentile(lexical_order.get(item.id), len(lexical_items))
            lexical_score_relevance = lexical_score / (lexical_score + lexical_median) if lexical_score > 0 and lexical_median > 0 else 0.0
        else:
            lexical_rank_relevance = 0.0
            lexical_score_relevance = lexical_score / (lexical_score + 1.0) if lexical_score > 0 else 0.0
        role_parts = []
        if roles.person_terms:
            role_parts.append(((primary_subject + secondary_context) if signals else person) / 0.08)
        if roles.location_terms:
            role_parts.append(location / 0.06)
        coverage = sum(role_parts) / len(role_parts) if role_parts else 1.0
        lexical_support = (0.60 * lexical_score_relevance + 0.40 * lexical_rank_relevance) * (0.35 + 0.65 * coverage) if item.metadata.get("lexical_candidate") else 0.0
        passage_relevance = min(1.0, max(semantic_relevance, lexical_support) + (fragment if fragment and (semantic_relevance == 0 and lexical_support > 0 or fragment < _ROUTE_FRAGMENT_MAX) else 0.0))
        channel_confidence = 0.02 if item.metadata.get("semantic_candidate") and item.metadata.get("lexical_candidate") else 0.0
        navigation_penalty = 0.32 if is_navigation_or_heading(item) else 0.0
        query_coverage = (sum(weight for group, weight in scope_weights if group & text_tokens) /
                          scope_total) if scope_total else 1.0
        # Preserve already supported actor/movement and surname/place evidence.
        # This is lexical ranking support, never actor identity or admission.
        if (person > 0 and action >= 0.12) or (
            roles.person_sequence and roles.person_sequence[-1] in text_tokens and location > 0
        ):
            query_coverage = 1.0
        vector_similarity, semantic_vector_bonus = _vector_signal(item)
        source_vector_bonus = semantic_vector_bonus
        # An incomplete multi-token name is a reference, not full local identity.
        # Apply the existing .04/.08 support ratio to its inherited vector prior.
        person_vector_scale = (person / 0.08 if len(roles.person_sequence) > 1
                               and not roles.multiple_person_phrases_detected
                               and 0 < person < 0.08 else 1.0)
        semantic_vector_bonus *= person_vector_scale
        if navigation_penalty:
            semantic_vector_bonus = 0.0
        heuristic_score = passage_relevance + channel_confidence + location + action + generic + statement_bonus
        # A source vector can be strong while this exact child names another actor.
        # Bound the correction by the existing maximum inherited vector prior;
        # candidates remain eligible and no historical identity is asserted.
        competing_person = bool(roles.person_terms and person == 0 and
                                _explicit_fragment_actor_conflict(roles, passage_text))
        # Restore the existing .40 full / .20 reference distinction even for
        # lexical-only evidence, whose local component is otherwise not used.
        person_identity_bonus = (0.20 if len(roles.person_sequence) > 1
                                 and not roles.multiple_person_phrases_detected
                                 and person >= 0.08 and not navigation_penalty else 0.0)
        supported_score = heuristic_score * query_coverage + semantic_vector_bonus + person_identity_bonus
        person_mismatch_penalty = min(_VECTOR_BONUS_MAX, supported_score / 2) if competing_person else 0.0
        final_score = supported_score - navigation_penalty - person_mismatch_penalty
        metadata = dict(item.metadata)
        metadata["retrieval_ranking"] = {
            "semantic_rank": item.metadata.get("vector_rank"),
            "lexical_rank": lexical_order.get(item.id),
            "merged_rank": item.metadata.get("merged_rank"),
            "vector_similarity": round(vector_similarity, 6) if vector_similarity is not None else None,
            "semantic_vector_bonus": round(semantic_vector_bonus, 6),
            "source_vector_bonus": round(source_vector_bonus, 6),
            "person_vector_scale": round(person_vector_scale, 6),
            "query_scope_coverage": round(query_coverage, 6),
            "heuristic_score": round(heuristic_score, 6),
            "pre_rerank_score": item.metadata.get("pre_rerank_score", item.score),
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
            "person_local_support": round(person_local, 6),
            "person_identity_bonus": round(person_identity_bonus, 6),
            "person_mismatch_penalty": round(person_mismatch_penalty, 6),
            "explicit_competing_person": competing_person,
            "primary_subject_support": round(primary_subject, 6),
            "secondary_person_context_support": round(secondary_context, 6),
            "location_support": round(location, 6),
            "generic_support": round(generic, 6),
            "joint_support": round(joint, 6),
            "action_support": round(action, 6),
            "statement_bonus": round(statement_bonus, 6),
            "route_fragment_relevance": round(fragment, 6),
            "navigation_penalty": round(navigation_penalty, 6),
            "final_score": round(final_score, 6),
        }
        ranked.append(item.model_copy(update={"score": round(final_score, 4), "metadata": metadata}))
    ranked.sort(key=lambda item: (-item.metadata["retrieval_ranking"]["final_score"], item.metadata.get("vector_rank", 999999), item.id))
    return [item.model_copy(update={"metadata": {**item.metadata, "rank": rank}}) for rank, item in enumerate(ranked, start=1)]
