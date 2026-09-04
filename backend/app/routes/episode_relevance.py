"""Episode relevance gate for legacy movement-claim route admission (G5L)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from backend.app.models import Evidence, HistoricalClaim
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    bounded_window_text,
    classify_evidence_relevance,
    narrative_subject_proper_nouns,
    normalized_terms,
    query_proper_nouns,
    query_terms,
    relation_supporting_statements,
    statement_evidence_window,
    _spatial_role_proper_nouns,
)

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
_BCE_YEAR = re.compile(r"\b(\d{1,4})\s*(?:bce|bc)\b", re.IGNORECASE)
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


@dataclass(frozen=True)
class _MovementEpisodeProbe:
    source_place: str | None
    destination_place: str | None
    statement: str
    supporting_evidence_ids: tuple[str, ...]


def _normalize_subject_name(value: str) -> str:
    return re.sub(r"(?:'s|’s)$", "", value.casefold())


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
    return {_normalize_subject_name(name) for name in query_proper_nouns(contexts)}


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


def _query_years(contexts: tuple[str, ...] | None) -> set[int]:
    years: set[int] = set()
    if not contexts:
        return years
    for context in contexts:
        for match in _BCE_YEAR.finditer(context or ""):
            years.add(int(match.group(1)))
    return years


def _statement_years(text: str) -> set[int]:
    return {int(match.group(1)) for match in _BCE_YEAR.finditer(text)}


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
    if not contexts:
        return False
    return bool(_query_subjects(contexts) & _narrative_subjects(statement))


def _episode_subject_overlap(statement: str, contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return False
    query_subjects = _query_subjects(contexts) - _GENERIC_EPISODE_SUBJECTS
    narrative_subjects = _narrative_subjects(statement) - _GENERIC_EPISODE_SUBJECTS
    return bool(query_subjects & narrative_subjects)


_DURING_EPISODE = re.compile(
    r"\bduring\s+([A-Z][A-Za-z'’]+(?:\s+[A-Z][A-Za-z'’]+)?)",
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


def _same_subject_other_episode_places(
    source_place: str | None,
    destination_place: str | None,
    statement: str,
    contexts: tuple[str, ...] | None,
    *,
    relevance: EvidenceRelevance,
) -> bool:
    if not contexts or relevance in _EPISODE_ADMISSIBLE:
        return False
    if relevance is EvidenceRelevance.OTHER_CAMPAIGN:
        return False
    if not _normalized_subject_overlap(statement, contexts):
        return False
    if not _CAMPAIGN_OBJECTIVE.search(" ".join(contexts)):
        return False
    place_tokens = _movement_place_tokens(source_place, destination_place)
    if _episode_anchor_overlap_places(statement, place_tokens, contexts):
        return False
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return False
    return True


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
    if _query_years(contexts) and _statement_years(context_text) and not _time_incompatible(context_text, contexts):
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


def _strong_direct_episode_signal(
    statement: str,
    window: str | None,
    probe: _MovementEpisodeProbe,
    contexts: tuple[str, ...] | None,
    *,
    subject_relevance: EvidenceRelevance | None = None,
) -> bool:
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


def _time_incompatible(statement: str, contexts: tuple[str, ...] | None) -> bool:
    query_years = _query_years(contexts)
    if len(query_years) != 1:
        return False
    statement_years = _statement_years(statement)
    if not statement_years:
        return False
    (query_year,) = tuple(query_years)
    return all(abs(year - query_year) > 2 for year in statement_years)


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
    if tag is EvidenceRelevance.OTHER_CAMPAIGN:
        episode = EpisodeRelevance.OTHER_CAMPAIGN
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
            episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
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
    elif _time_incompatible(local, contexts):
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    else:
        episode = EpisodeRelevance.UNKNOWN
    admitted = episode_route_admission_allowed(episode)
    return episode, {
        "origin": probe.source_place,
        "destination": probe.destination_place,
        "source_statement": statement[:240] if statement else None,
        "subject_relevance": tag.value,
        "episode_classification": episode.value,
        "admitted": admitted,
        "admission_reason": "EPISODE_RELEVANT" if admitted else f"REJECT_{episode.value}",
    }


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
