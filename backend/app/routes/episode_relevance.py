"""Episode relevance gate for legacy movement-claim route admission (G5L)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from backend.app.models import Evidence, HistoricalClaim, EventPlaceRole
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    bounded_window_text,
    classify_evidence_relevance,
    event_relevance,
    has_normalized_subject_overlap,
    narrative_subject_proper_nouns,
    normalize_subject_name,
    normalized_query_subject_terms,
    normalized_terms,
    query_proper_nouns,
    query_terms,
    relation_supporting_statements,
    statement_evidence_window,
    _named_proper_nouns,
    _spatial_role_proper_nouns,
)
from backend.app.routes.temporal import EvidenceTemporalResolver

if TYPE_CHECKING:
    from backend.app.models import HistoricalEvent

_GENERIC_EPISODE_SUBJECTS = frozenset({
    "commander", "general", "emperor", "king", "consul", "trace",
})
_NON_PERSON_SUBJECTS = frozenset({
    "synthetic", "test", "evidence", "according", "however", "meanwhile", "source",
}) | _GENERIC_EPISODE_SUBJECTS
_ANAPHORIC_SUBJECT = re.compile(
    r"\b(?:he|she|they|his|her|their|him|them)\b",
    re.IGNORECASE,
)
_EPISODE_ADMISSIBLE = frozenset({
    EvidenceRelevance.DIRECT_SUBJECT,
    EvidenceRelevance.DIRECT_CAMPAIGN,
    EvidenceRelevance.DIRECT_EVENT,
    EvidenceRelevance.SAME_CONFLICT_RELEVANT,
})
_CAMPAIGN_OBJECTIVE = re.compile(
    r"\b(?:march|route|campaign|crossing|flight|escape|retreat|advance|movement|"
    r"arrival|defeat|battle|war|siege|expedition)\b",
    re.IGNORECASE,
)


class EpisodeRelevance(str, Enum):
    DIRECT_QUERY_EPISODE = "DIRECT_QUERY_EPISODE"
    SAME_CAMPAIGN_RELEVANT = "SAME_CAMPAIGN_RELEVANT"
    SAME_SUBJECT_OTHER_EPISODE = "SAME_SUBJECT_OTHER_EPISODE"
    OTHER_CAMPAIGN = "OTHER_CAMPAIGN"
    UNKNOWN = "UNKNOWN"


_EPISODE_ROUTE_ADMISSIBLE = frozenset({
    EpisodeRelevance.DIRECT_QUERY_EPISODE,
    EpisodeRelevance.SAME_CAMPAIGN_RELEVANT,
    EpisodeRelevance.UNKNOWN,
})


def episode_route_admission_allowed(episode: EpisodeRelevance) -> bool:
    return episode in _EPISODE_ROUTE_ADMISSIBLE


_TEMPORAL_RESOLVER = EvidenceTemporalResolver()


def _explicit_temporal_intervals(text: str) -> list[tuple[int, int]]:
    readings, codes = _TEMPORAL_RESOLVER.resolve(text, "probe")
    if not readings or "TEMPORAL_CONFLICT" in codes:
        return []
    intervals: list[tuple[int, int]] = []
    for item in readings:
        if item.normalized_start is None:
            continue
        start = int(item.normalized_start)
        end = int(item.normalized_end) if item.normalized_end is not None else start
        if start > end:
            start, end = end, start
        intervals.append((start, end))
    return intervals


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def _explicit_temporal_contradiction(
    statement: str,
    contexts: tuple[str, ...] | None,
    window: str | None = None,
) -> bool:
    if not contexts:
        return False
    query_intervals: list[tuple[int, int]] = []
    for context in contexts:
        query_intervals.extend(_explicit_temporal_intervals(context or ""))
    if not query_intervals:
        return False
    statement_intervals = _explicit_temporal_intervals(statement.strip())
    if not statement_intervals and window:
        statement_intervals = _explicit_temporal_intervals(window.strip())
    if not statement_intervals:
        return False
    return not any(
        _intervals_overlap(query, stmt)
        for query in query_intervals
        for stmt in statement_intervals
    )


@dataclass(frozen=True)
class _MovementEpisodeProbe:
    source_place: str | None
    destination_place: str | None
    statement: str
    supporting_evidence_ids: tuple[str, ...]


def _normalize_subject_name(value: str) -> str:
    return normalize_subject_name(value)


def _query_campaign_phrase_terms(contexts: tuple[str, ...] | None) -> set[str]:
    if not contexts:
        return set()
    terms: set[str] = set()
    for context in contexts:
        for match in _CAMPAIGN_EPISODE_PHRASE.finditer(context or ""):
            terms |= normalized_terms(match.group(1))
    return terms


def _campaign_phrase_modifier_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in _EVIDENCE_CAMPAIGN_PHRASE.finditer(text):
        fragment = match.group(1) or match.group(2)
        if fragment:
            terms |= normalized_terms(fragment)
    return terms


def _query_has_campaign_episode_phrase(contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return False
    return any(_CAMPAIGN_EPISODE_PHRASE.search(context or "") for context in contexts)


def _narrative_subjects(value: str) -> set[str]:
    spatial = {_normalize_subject_name(name) for name in _spatial_role_proper_nouns(value)}
    return {
        _normalize_subject_name(name)
        for name in narrative_subject_proper_nouns(value)
        if _normalize_subject_name(name) not in spatial
        and _normalize_subject_name(name) not in _NON_PERSON_SUBJECTS
    }


def _query_subjects(contexts: tuple[str, ...] | None) -> set[str]:
    if not contexts:
        return set()
    campaign_terms = _query_campaign_phrase_terms(contexts)
    subjects = {_normalize_subject_name(name) for name in query_proper_nouns(contexts)}
    subjects |= {
        _normalize_subject_name(name)
        for name in narrative_subject_proper_nouns(" ".join(contexts))
    }
    return {subject for subject in subjects if subject not in campaign_terms}


def _other_campaign_subject_conflict(text: str, contexts: tuple[str, ...] | None) -> bool:
    """True only when narrative subjects are wholly disjoint from query subjects."""
    if not contexts:
        return False
    query_subjects = _query_subjects(contexts)
    evidence_subjects = _narrative_subjects(text)
    if not query_subjects or not evidence_subjects:
        return False
    if query_subjects & evidence_subjects:
        return False
    if query_subjects & normalized_terms(text):
        return False
    if _ANAPHORIC_SUBJECT.search(text):
        return False
    return True


def _evidence_text(item: Evidence) -> str:
    return " ".join(dict.fromkeys(value for value in (item.text, item.excerpt) if value))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]


def _claim_statement_window(claim: HistoricalClaim, evidence_by_id: dict[str, Evidence]) -> str | None:
    statement = (claim.textual_basis or claim.text or "").strip()
    if not statement:
        return None
    for ref in claim.supporting_evidence_ids:
        item = evidence_by_id.get(ref)
        if item is None:
            continue
        sentences = _split_sentences(_evidence_text(item))
        for index, sentence in enumerate(sentences):
            if statement in sentence or sentence in statement:
                return bounded_window_text(sentences, index)
    return None


def _claim_place_tokens(claim: HistoricalClaim) -> set[str]:
    return _movement_place_tokens(claim.source_place, claim.destination_place, claim.traversed_place)


def _movement_place_tokens(
    source_place: str | None,
    destination_place: str | None,
    traversed_place: str | None = None,
) -> set[str]:
    tokens: set[str] = set()
    for value in (source_place, destination_place, traversed_place):
        if not value:
            continue
        tokens |= normalized_terms(value)
    return tokens


_GENERIC_PLACE_TOKENS = frozenset({
    "region", "province", "city", "port", "gulf", "bay", "world", "event",
})
_EPISODE_FRAMING = frozenset({
    "trace", "route", "reconstruct", "major", "movements", "movement", "from", "through",
    "leading", "until", "after", "during", "across", "into", "toward", "towards", "first",
    "world", "back", "defeat", "arrival", "great", "bactria", "hindu", "ending", "near",
    "between", "eastern", "mediterranean", "minor", "toward", "anatolia",
})


def _query_episode_anchor_terms(contexts: tuple[str, ...] | None) -> set[str]:
    if not contexts:
        return set()
    terms = query_terms(contexts)
    return {
        term for term in terms
        if len(term) >= 4 and term not in _EPISODE_FRAMING and term not in _GENERIC_PLACE_TOKENS
    }


def _episode_anchor_overlap_places(
    statement: str,
    place_tokens: set[str],
    contexts: tuple[str, ...] | None,
) -> bool:
    return bool(_episode_place_anchor_terms(statement, place_tokens, contexts))


def _episode_anchor_overlap(claim: HistoricalClaim, statement: str, contexts: tuple[str, ...] | None) -> bool:
    return _episode_anchor_overlap_places(statement, _claim_place_tokens(claim), contexts)


def _normalized_subject_overlap(statement: str, contexts: tuple[str, ...] | None) -> bool:
    return has_normalized_subject_overlap(statement, contexts)


def _episode_subject_overlap(statement: str, contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return False
    if has_normalized_subject_overlap(statement, contexts):
        return True
    query_subjects = _query_subjects(contexts) - _GENERIC_EPISODE_SUBJECTS
    narrative_subjects = _narrative_subjects(statement) - _GENERIC_EPISODE_SUBJECTS
    return bool(query_subjects & narrative_subjects)


_QUERY_ORIGIN_DESTINATION = re.compile(
    r"\bfrom\s+(?:the\s+)?([A-Z][A-Za-z'’\u2019-]+(?:\s+[A-Z][A-Za-z'’\u2019-]+)?)\s+to\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'’\u2019-]+(?:\s+[A-Z][A-Za-z'’\u2019-]+)?)",
    re.IGNORECASE,
)
_SUBJECT_CONSTRAINT_NOISE = frozenset({
    "bce", "ce", "trace", "route", "during", "campaign", "war", "expedition",
})


def _query_explicit_origin_destination(contexts: tuple[str, ...] | None) -> tuple[str, str] | None:
    if not contexts:
        return None
    for context in contexts:
        match = _QUERY_ORIGIN_DESTINATION.search(context or "")
        if match:
            return match.group(1), match.group(2)
    return None


def _movement_contradicts_query_endpoints(
    source_place: str | None,
    destination_place: str | None,
    contexts: tuple[str, ...] | None,
) -> bool:
    pair = _query_explicit_origin_destination(contexts)
    if not pair or not (source_place and destination_place):
        return False
    query_origin, query_dest = pair
    query_o = normalized_terms(query_origin)
    query_d = normalized_terms(query_dest)
    move_o = normalized_terms(source_place)
    move_d = normalized_terms(destination_place)
    if (move_o & query_o) or (move_d & query_d) or (move_o & query_d) or (move_d & query_o):
        return False
    return True


def _explicit_query_subject_satisfied(combined: str, contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return True
    campaign_terms = _query_campaign_phrase_terms(contexts)
    query_persons = normalized_query_subject_terms(contexts) - campaign_terms - _SUBJECT_CONSTRAINT_NOISE
    query_persons -= _GENERIC_EPISODE_SUBJECTS
    if not query_persons:
        return True
    spatial = {_normalize_subject_name(name) for name in _spatial_role_proper_nouns(combined)}
    evidence_persons = {
        _normalize_subject_name(name) for name in _named_proper_nouns(combined)
    } - spatial - campaign_terms - _campaign_phrase_modifier_terms(combined) - _SUBJECT_CONSTRAINT_NOISE
    evidence_persons -= _GENERIC_EPISODE_SUBJECTS
    if evidence_persons:
        return bool(query_persons & evidence_persons)
    if _other_campaign_subject_conflict(combined, contexts):
        return False
    return has_normalized_subject_overlap(combined, contexts)


_DURING_EPISODE = re.compile(
    r"\bduring\s+([A-Z][A-Za-z'’]+(?:\s+[A-Z][A-Za-z'’]+)?)",
    re.IGNORECASE,
)
_CAMPAIGN_EPISODE_PHRASE = re.compile(
    r"\b(?:during|in|throughout|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'’\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'’\u2019-]+){0,4})\s+campaign\b",
    re.IGNORECASE,
)
_EVIDENCE_CAMPAIGN_PHRASE = re.compile(
    r"(?:\b(?:during|in|throughout|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z'’\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'’\u2019-]+){0,4})\s+campaign\b"
    r"|\b(?:the\s+)?([A-Z][A-Za-z'’\u2019-]+(?:\s+(?:the\s+)?[A-Z][A-Za-z'’\u2019-]+){0,4})\s+campaign\b(?!\s+[A-Z]))",
    re.IGNORECASE,
)
_GEO_PREP = re.compile(
    r"\b(?:in|into|from|to|toward|towards|near|across|through|via|around)\s+"
    r"([A-Z][A-Za-z'’]+(?:\s+[A-Z][A-Za-z'’]+)?)",
    re.IGNORECASE,
)


def _statement_local_place_terms(statement: str, place_tokens: set[str]) -> set[str]:
    terms: set[str] = set()
    for pattern in (_GEO_PREP, _DURING_EPISODE):
        for match in pattern.finditer(statement):
            terms |= normalized_terms(match.group(1))
    terms |= place_tokens & normalized_terms(statement)
    return terms


def _episode_place_anchor_terms(
    statement: str,
    place_tokens: set[str],
    contexts: tuple[str, ...] | None,
) -> set[str]:
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return set()
    place_local = {
        token for token in place_tokens & anchors if token not in _GENERIC_PLACE_TOKENS
    }
    local_places = _statement_local_place_terms(statement, place_tokens)
    statement_place = (
        (normalized_terms(statement) & anchors & local_places)
        - _GENERIC_EPISODE_SUBJECTS
        - _GENERIC_PLACE_TOKENS
        - (_narrative_subjects(statement) & _query_subjects(contexts))
    )
    return place_local | statement_place


def _explicit_origin_destination_claim(claim: HistoricalClaim) -> bool:
    return bool(claim.source_place and claim.destination_place)


def _explicit_od_episode_compatible_places(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    contexts: tuple[str, ...] | None,
) -> bool:
    """True when explicit O→D aligns with the requested episode beyond a single weak overlap."""
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return True
    source_tokens = normalized_terms(source_place or "")
    dest_tokens = normalized_terms(destination_place or "")
    source_overlap = {token for token in source_tokens & anchors if token not in _GENERIC_PLACE_TOKENS}
    dest_overlap = {token for token in dest_tokens & anchors if token not in _GENERIC_PLACE_TOKENS}
    place_anchor_terms = _episode_place_anchor_terms(statement, source_tokens | dest_tokens, contexts)
    if source_overlap and dest_overlap:
        return True
    if len(place_anchor_terms) >= 2:
        return True
    return False


def _explicit_od_episode_compatible(
    claim: HistoricalClaim,
    statement: str,
    contexts: tuple[str, ...] | None,
) -> bool:
    return _explicit_od_episode_compatible_places(
        claim.source_place, claim.destination_place, statement, contexts,
    )


def _positive_same_subject_other_episode_evidence(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    contexts: tuple[str, ...] | None,
    *,
    require_anchor_overlap: bool,
) -> bool:
    """True only when explicit evidence supports a different episode, not mere non-overlap."""
    if not contexts:
        return False
    if not _normalized_subject_overlap(statement, contexts):
        return False
    if not _CAMPAIGN_OBJECTIVE.search(" ".join(contexts)):
        return False
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return False
    if not (source_place and destination_place):
        return False
    place_tokens = _movement_place_tokens(source_place, destination_place)
    anchor_hits = {
        token
        for token in place_tokens & anchors
        if token not in _GENERIC_PLACE_TOKENS
    }
    has_overlap = bool(anchor_hits) or _episode_anchor_overlap_places(
        statement, place_tokens, contexts,
    )
    if require_anchor_overlap:
        if not has_overlap:
            return False
    elif has_overlap:
        return False
    if _explicit_od_episode_compatible_places(
        source_place, destination_place, statement, contexts,
    ):
        return False
    return True


def _same_subject_other_episode_places(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    contexts: tuple[str, ...] | None,
    *,
    relevance: EvidenceRelevance,
) -> bool:
    if relevance is EvidenceRelevance.OTHER_CAMPAIGN:
        return False
    if relevance in _EPISODE_ADMISSIBLE:
        return False
    return _positive_same_subject_other_episode_evidence(
        source_place,
        destination_place,
        statement,
        contexts,
        require_anchor_overlap=False,
    )


def _same_subject_other_episode(
    claim: HistoricalClaim,
    statement: str,
    contexts: tuple[str, ...] | None,
    *,
    relevance: EvidenceRelevance,
) -> bool:
    return _same_subject_other_episode_places(
        claim.source_place,
        claim.destination_place,
        statement,
        contexts,
        relevance=relevance,
    )


_PRIOR_EPISODE_MARKERS = re.compile(
    r"\b(?:earlier|previous|prior|former)\s+(?:war|campaign|conflict|march|expedition)\b",
    re.IGNORECASE,
)
_CURRENT_EPISODE_MARKERS = re.compile(
    r"\b(?:current|this|present)\s+(?:war|campaign|conflict|march|expedition)\b",
    re.IGNORECASE,
)


def _episode_framing_conflict(statement: str, contexts: tuple[str, ...] | None) -> bool:
    if not contexts or not statement.strip():
        return False
    if not _PRIOR_EPISODE_MARKERS.search(statement):
        return False
    return any(_CURRENT_EPISODE_MARKERS.search(context or "") for context in contexts)


def _episode_local_text(statement: str, window: str | None) -> str:
    local = (window or statement).strip()
    return local


def _episode_rescue_after_explicit_od_mismatch(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    window: str | None,
    contexts: tuple[str, ...] | None,
) -> bool:
    local = statement.strip()
    context_text = (window or statement).strip()
    if not local or not context_text or not _episode_subject_overlap(context_text, contexts):
        return False
    query_intervals: list[tuple[int, int]] = []
    for context in contexts or ():
        query_intervals.extend(_explicit_temporal_intervals(context or ""))
    statement_intervals = _explicit_temporal_intervals(context_text)
    if query_intervals and statement_intervals and not _explicit_temporal_contradiction(
        context_text, contexts, window,
    ):
        return True
    anchors = _query_episode_anchor_terms(contexts)
    context_episode_hits = (
        (normalized_terms(context_text) & anchors)
        - _GENERIC_EPISODE_SUBJECTS
        - _GENERIC_PLACE_TOKENS
        - _narrative_subjects(context_text)
    )
    if context_episode_hits:
        return True
    source_tokens = normalized_terms(source_place or "")
    if {token for token in source_tokens & anchors if token not in _GENERIC_PLACE_TOKENS}:
        return True
    place_tokens = _movement_place_tokens(source_place, destination_place)
    place_stmt_anchors = _episode_place_anchor_terms(local, place_tokens, contexts)
    if len(place_stmt_anchors) >= 2:
        return True
    return False


def _direct_subject_local_episode_signal(
    probe: _MovementEpisodeProbe,
    statement: str,
    contexts: tuple[str, ...] | None,
) -> bool:
    if not (probe.source_place and probe.destination_place):
        return False
    local = statement.strip()
    if not local:
        return False
    source_tokens = normalized_terms(probe.source_place)
    dest_tokens = normalized_terms(probe.destination_place)
    statement_tokens = normalized_terms(local)
    if not (source_tokens & statement_tokens and dest_tokens & statement_tokens):
        return False
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return _episode_subject_overlap(local, contexts) or bool(_query_subjects(contexts) & statement_tokens)
    source_hits = {token for token in source_tokens & anchors if token not in _GENERIC_PLACE_TOKENS}
    dest_hits = {token for token in dest_tokens & anchors if token not in _GENERIC_PLACE_TOKENS}
    place_anchor_terms = _episode_place_anchor_terms(local, source_tokens | dest_tokens, contexts)
    if dest_hits and not source_hits and len(place_anchor_terms) < 2:
        return False
    if len(place_anchor_terms) >= 2 or source_hits:
        return True
    return bool((_query_subjects(contexts) - _GENERIC_EPISODE_SUBJECTS) & statement_tokens & source_tokens)


def _statement_supports_query_campaign(statement: str, contexts: tuple[str, ...] | None) -> bool:
    query_terms = _query_campaign_phrase_terms(contexts)
    if not query_terms:
        return False
    evidence_terms = _campaign_phrase_modifier_terms(statement)
    if not evidence_terms:
        return False
    return bool(query_terms & evidence_terms)


def _explicit_query_constraints_satisfied(
    statement: str,
    window: str | None,
    contexts: tuple[str, ...] | None,
) -> bool:
    if not contexts:
        return True
    combined = " ".join(part for part in (statement.strip(), (window or "").strip()) if part)
    if not _explicit_query_subject_satisfied(combined, contexts):
        return False
    if _query_has_campaign_episode_phrase(contexts):
        if not _statement_supports_query_campaign(combined, contexts):
            return False
    query_intervals: list[tuple[int, int]] = []
    for context in contexts:
        query_intervals.extend(_explicit_temporal_intervals(context or ""))
    if query_intervals:
        statement_intervals = _explicit_temporal_intervals(statement.strip())
        if not statement_intervals and window:
            statement_intervals = _explicit_temporal_intervals(window.strip())
        if not statement_intervals:
            return False
        if _explicit_temporal_contradiction(statement, contexts, window):
            return False
    return True


def _explicit_endpoint_constraint_satisfied(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    contexts: tuple[str, ...] | None,
) -> bool:
    if not _query_endpoint_scope_terms(contexts):
        return True
    if not (source_place and destination_place):
        return True
    return _explicit_od_episode_compatible_places(
        source_place, destination_place, statement, contexts,
    )


def _strong_direct_episode_signal(
    statement: str,
    window: str | None,
    probe: _MovementEpisodeProbe,
    contexts: tuple[str, ...] | None,
    *,
    subject_relevance: EvidenceRelevance | None = None,
) -> bool:
    if _explicit_temporal_contradiction(statement, contexts, window):
        return False
    combined = " ".join(part for part in (statement.strip(), (window or "").strip()) if part)
    if (
        _statement_supports_query_campaign(combined, contexts)
        and has_normalized_subject_overlap(statement, contexts)
        and _explicit_query_constraints_satisfied(statement, window, contexts)
        and _explicit_endpoint_constraint_satisfied(
            probe.source_place, probe.destination_place, statement, contexts,
        )
    ):
        return True
    explicit_od = bool(probe.source_place and probe.destination_place)
    if explicit_od and _explicit_od_episode_compatible_places(
        probe.source_place, probe.destination_place, statement, contexts,
    ):
        return True
    if subject_relevance is EvidenceRelevance.DIRECT_SUBJECT and _direct_subject_local_episode_signal(
        probe, statement, contexts,
    ):
        return True
    if explicit_od and _episode_rescue_after_explicit_od_mismatch(
        probe.source_place,
        probe.destination_place,
        statement,
        window,
        contexts,
    ):
        return True
    if not _query_episode_anchor_terms(contexts):
        return _episode_subject_overlap(statement.strip(), contexts)
    place_tokens = _movement_place_tokens(probe.source_place, probe.destination_place)
    return len(_episode_place_anchor_terms(statement.strip(), place_tokens, contexts)) >= 2


def _probe_statement_window(
    probe: _MovementEpisodeProbe,
    evidence_by_id: dict[str, Evidence],
) -> str | None:
    statement = probe.statement.strip()
    if not statement:
        return None
    for ref in probe.supporting_evidence_ids:
        item = evidence_by_id.get(ref)
        if item is None:
            continue
        sentences = _split_sentences(_evidence_text(item))
        for index, sentence in enumerate(sentences):
            if statement in sentence or sentence in statement:
                return bounded_window_text(sentences, index)
    return None


def _query_endpoint_scope_terms(contexts: tuple[str, ...] | None) -> set[str]:
    if not contexts:
        return set()
    terms: set[str] = set()
    for context in contexts:
        for name in _spatial_role_proper_nouns(context or ""):
            terms |= normalized_terms(name)
    return {term for term in terms if term not in _GENERIC_PLACE_TOKENS}


def _query_has_episode_constraints(contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return False
    for context in contexts:
        if _explicit_temporal_intervals(context or ""):
            return True
    if _query_has_campaign_episode_phrase(contexts):
        return True
    query_subjects = _query_subjects(contexts) - _GENERIC_EPISODE_SUBJECTS
    episode_anchors = {
        _normalize_subject_name(term)
        for term in _query_episode_anchor_terms(contexts)
    } - query_subjects - _GENERIC_EPISODE_SUBJECTS
    if episode_anchors:
        return True
    return bool(_query_endpoint_scope_terms(contexts))


def _classify_movement_episode(
    probe: _MovementEpisodeProbe,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
    *,
    subject_relevance: EvidenceRelevance | None = None,
) -> tuple[EpisodeRelevance, dict[str, object]]:
    statement = probe.statement.strip()
    window = _probe_statement_window(probe, evidence_by_id)
    local = _episode_local_text(statement, window)
    chunk_parts = [statement]
    for ref in probe.supporting_evidence_ids:
        item = evidence_by_id.get(ref)
        if item is not None:
            chunk_parts.append(_evidence_text(item))
    chunk = " ".join(chunk_parts)
    if _other_campaign_subject_conflict(local or chunk, contexts):
        tag = EvidenceRelevance.OTHER_CAMPAIGN
    elif subject_relevance is not None:
        tag = subject_relevance
    else:
        tag = classify_evidence_relevance(statement or chunk, contexts, window_text=window)
        if tag is EvidenceRelevance.OTHER_CAMPAIGN:
            tag = EvidenceRelevance.UNKNOWN
    explicit_od = bool(probe.source_place and probe.destination_place)
    place_tokens = _movement_place_tokens(probe.source_place, probe.destination_place)
    temporal_conflict = _explicit_temporal_contradiction(statement, contexts, window)
    if tag is EvidenceRelevance.OTHER_CAMPAIGN:
        episode = EpisodeRelevance.OTHER_CAMPAIGN
    elif temporal_conflict:
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    elif _episode_framing_conflict(statement, contexts):
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    elif tag in {
        EvidenceRelevance.DIRECT_SUBJECT,
        EvidenceRelevance.DIRECT_CAMPAIGN,
        EvidenceRelevance.DIRECT_EVENT,
    }:
        if _strong_direct_episode_signal(
            statement, window, probe, contexts, subject_relevance=tag,
        ):
            episode = EpisodeRelevance.DIRECT_QUERY_EPISODE
        elif tag is EvidenceRelevance.DIRECT_SUBJECT:
            if _positive_same_subject_other_episode_evidence(
                probe.source_place,
                probe.destination_place,
                local,
                contexts,
                require_anchor_overlap=True,
            ):
                episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
            else:
                episode = EpisodeRelevance.UNKNOWN
        else:
            episode = EpisodeRelevance.DIRECT_QUERY_EPISODE
    elif tag is EvidenceRelevance.SAME_CONFLICT_RELEVANT:
        episode = EpisodeRelevance.SAME_CAMPAIGN_RELEVANT
    elif _strong_direct_episode_signal(
        statement, window, probe, contexts, subject_relevance=tag,
    ):
        episode = EpisodeRelevance.DIRECT_QUERY_EPISODE
    elif _same_subject_other_episode_places(
        probe.source_place, probe.destination_place, local, contexts, relevance=tag,
    ):
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    else:
        episode = EpisodeRelevance.UNKNOWN
    admitted = episode_route_admission_allowed(episode)
    if episode is EpisodeRelevance.DIRECT_QUERY_EPISODE:
        combined = " ".join(part for part in (statement.strip(), (window or "").strip()) if part)
        if not _explicit_query_constraints_satisfied(statement, window, contexts):
            if not _explicit_query_subject_satisfied(combined, contexts):
                episode = EpisodeRelevance.OTHER_CAMPAIGN
            else:
                episode = EpisodeRelevance.UNKNOWN
            admitted = False
        elif _movement_contradicts_query_endpoints(
            probe.source_place, probe.destination_place, contexts,
        ):
            episode = EpisodeRelevance.UNKNOWN
            admitted = False
    elif episode is EpisodeRelevance.UNKNOWN:
        if not explicit_od:
            admitted = False
        elif tag is EvidenceRelevance.DIRECT_SUBJECT:
            if _query_has_episode_constraints(contexts):
                admitted = False
            elif not _episode_subject_overlap(local, contexts):
                admitted = False
    return episode, {
        "origin": probe.source_place,
        "destination": probe.destination_place,
        "source_statement": statement[:240] if statement else None,
        "subject_relevance": tag.value,
        "episode_classification": episode.value,
        "admitted": admitted,
        "admission_reason": "EPISODE_RELEVANT" if admitted else f"REJECT_{episode.value}",
    }


def _movement_probe_for_event(event: HistoricalEvent) -> _MovementEpisodeProbe | None:
    origin = next((item for item in event.place_bindings if item.role is EventPlaceRole.ORIGIN), None)
    destination = next((item for item in event.place_bindings if item.role is EventPlaceRole.DESTINATION), None)
    if origin is None or destination is None:
        return None
    statement = next((item for item in relation_supporting_statements(event) if item.strip()), "")
    if not statement:
        statement = (event.summary or "").strip()
    if not statement:
        return None
    return _MovementEpisodeProbe(
        source_place=origin.place.canonical_name,
        destination_place=destination.place.canonical_name,
        statement=statement,
        supporting_evidence_ids=tuple(dict.fromkeys(event.evidence_refs)),
    )


def _participant_episode_admissions(
    relation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> list[dict[str, object]]:
    admissions: list[dict[str, object]] = []
    for event_id in dict.fromkeys(relation.event_ids):
        event = events_by_id.get(event_id)
        if event is None:
            admissions.append({
                "event_id": event_id,
                "admitted": False,
                "admission_reason": "REJECT_MISSING_EVENT",
                "episode_classification": EpisodeRelevance.UNKNOWN.value,
            })
            continue
        probe = _movement_probe_for_event(event)
        if probe is None:
            admissions.append({
                "event_id": event_id,
                "admitted": False,
                "admission_reason": "REJECT_MISSING_MOVEMENT",
                "episode_classification": EpisodeRelevance.UNKNOWN.value,
            })
            continue
        tag = event_relevance(event, evidence_by_id, contexts)
        _, detail = _classify_movement_episode(
            probe, evidence_by_id, contexts, subject_relevance=tag,
        )
        admissions.append({"event_id": event_id, **detail})
    return admissions


def classify_event_anchor_episode(
    relation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
    *,
    subject_relevance: EvidenceRelevance,
) -> tuple[EpisodeRelevance, dict[str, object]]:
    statements: list[str] = []
    for event_id in relation.event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue
        statements.extend(relation_supporting_statements(event))
    statement = next((item for item in statements if item.strip()), "")
    if not statement:
        for event_id in relation.event_ids:
            event = events_by_id.get(event_id)
            if event is not None and event.summary:
                statement = event.summary
                break
    refs: list[str] = list(relation.evidence_refs)
    for event_id in relation.event_ids:
        event = events_by_id.get(event_id)
        if event is not None:
            refs.extend(event.evidence_refs)
    probe = _MovementEpisodeProbe(
        source_place=relation.earlier,
        destination_place=relation.later,
        statement=statement,
        supporting_evidence_ids=tuple(dict.fromkeys(refs)),
    )
    episode, detail = _classify_movement_episode(
        probe, evidence_by_id, contexts, subject_relevance=subject_relevance,
    )
    participant_admissions = _participant_episode_admissions(
        relation, events_by_id, evidence_by_id, contexts,
    )
    if len(dict.fromkeys(relation.event_ids)) > 1:
        detail = dict(detail)
        detail["participant_episode_admission"] = participant_admissions
        rejected = next((item for item in participant_admissions if not item["admitted"]), None)
        if rejected is not None:
            detail["admitted"] = False
            detail["episode_classification"] = rejected["episode_classification"]
            detail["admission_reason"] = rejected["admission_reason"]
            episode = EpisodeRelevance(rejected["episode_classification"])
        else:
            detail["admitted"] = all(item["admitted"] for item in participant_admissions)
    detail["earlier"] = relation.earlier
    detail["later"] = relation.later
    detail["event_ids"] = list(relation.event_ids)
    return episode, detail


def classify_legacy_claim_episode(
    claim: HistoricalClaim,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> tuple[EpisodeRelevance, dict[str, object]]:
    probe = _MovementEpisodeProbe(
        source_place=claim.source_place,
        destination_place=claim.destination_place,
        statement=(claim.textual_basis or claim.text or "").strip(),
        supporting_evidence_ids=tuple(claim.supporting_evidence_ids),
    )
    return _classify_movement_episode(probe, evidence_by_id, contexts)


def filter_legacy_movement_claims(
    claims: list[HistoricalClaim],
    evidence: list[Evidence],
    contexts: tuple[str, ...] | None,
) -> tuple[list[HistoricalClaim], list[dict[str, object]]]:
    if not contexts:
        return claims, []
    evidence_by_id = {item.id: item for item in evidence}
    admitted: list[HistoricalClaim] = []
    diagnostics: list[dict[str, object]] = []
    for claim in claims:
        episode, detail = classify_legacy_claim_episode(claim, evidence_by_id, contexts)
        diagnostics.append(detail)
        if detail["admitted"]:
            admitted.append(claim)
    return admitted, diagnostics


def legacy_claim_admission_allowed(
    claim: HistoricalClaim,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> bool:
    if not contexts:
        return True
    _, detail = classify_legacy_claim_episode(claim, evidence_by_id, contexts)
    return bool(detail["admitted"])
