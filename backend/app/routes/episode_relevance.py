"""Episode relevance gate for legacy movement-claim route admission (G5L)."""
from __future__ import annotations

import re
from enum import Enum

from backend.app.models import Evidence, HistoricalClaim
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    bounded_window_text,
    classify_evidence_relevance,
    narrative_subject_proper_nouns,
    normalized_terms,
    query_proper_nouns,
    query_terms,
    _spatial_role_proper_nouns,
)

_NON_PERSON_SUBJECTS = frozenset({
    "synthetic", "test", "evidence", "according", "however", "meanwhile", "source",
})
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
    tokens: set[str] = set()
    for value in (claim.source_place, claim.destination_place, claim.traversed_place):
        if not value:
            continue
        tokens |= normalized_terms(value)
    return tokens


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
    return {term for term in terms if len(term) >= 4 and term not in _EPISODE_FRAMING}


def _episode_anchor_overlap(claim: HistoricalClaim, statement: str, contexts: tuple[str, ...] | None) -> bool:
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return False
    involved = normalized_terms(statement) | _claim_place_tokens(claim)
    return bool(involved & anchors)


def _normalized_subject_overlap(statement: str, contexts: tuple[str, ...] | None) -> bool:
    if not contexts:
        return False
    return bool(_query_subjects(contexts) & _narrative_subjects(statement))


def _explicit_origin_destination_claim(claim: HistoricalClaim) -> bool:
    return bool(claim.source_place and claim.destination_place)


def _explicit_od_episode_compatible(
    claim: HistoricalClaim,
    statement: str,
    contexts: tuple[str, ...] | None,
) -> bool:
    """True when explicit O→D has episode-compatibility beyond subject overlap."""
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return True
    source_tokens = normalized_terms(claim.source_place or "")
    dest_tokens = normalized_terms(claim.destination_place or "")
    source_overlap = source_tokens & anchors
    dest_overlap = dest_tokens & anchors
    stmt_anchors = normalized_terms(statement) & anchors
    if source_overlap or dest_overlap:
        return True
    if source_tokens:
        return len(stmt_anchors) >= 2
    return bool(stmt_anchors)


def _same_subject_other_episode(
    claim: HistoricalClaim,
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
    if _episode_anchor_overlap(claim, statement, contexts):
        return False
    anchors = _query_episode_anchor_terms(contexts)
    if not anchors:
        return False
    return True


def _time_incompatible(statement: str, contexts: tuple[str, ...] | None) -> bool:
    query_years = _query_years(contexts)
    if len(query_years) != 1:
        return False
    statement_years = _statement_years(statement)
    if not statement_years:
        return False
    (query_year,) = tuple(query_years)
    return all(abs(year - query_year) > 2 for year in statement_years)


def classify_legacy_claim_episode(
    claim: HistoricalClaim,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> tuple[EpisodeRelevance, dict[str, object]]:
    statement = (claim.textual_basis or claim.text or "").strip()
    window = _claim_statement_window(claim, evidence_by_id)
    chunk_parts = [statement]
    for ref in claim.supporting_evidence_ids:
        item = evidence_by_id.get(ref)
        if item is not None:
            chunk_parts.append(_evidence_text(item))
    chunk = " ".join(chunk_parts)
    if _other_campaign_subject_conflict(statement or chunk, contexts):
        tag = EvidenceRelevance.OTHER_CAMPAIGN
    else:
        tag = classify_evidence_relevance(statement or chunk, contexts, window_text=window)
        if tag is EvidenceRelevance.OTHER_CAMPAIGN:
            tag = EvidenceRelevance.UNKNOWN
    if tag is EvidenceRelevance.OTHER_CAMPAIGN:
        episode = EpisodeRelevance.OTHER_CAMPAIGN
    elif tag in {
        EvidenceRelevance.DIRECT_SUBJECT,
        EvidenceRelevance.DIRECT_CAMPAIGN,
        EvidenceRelevance.DIRECT_EVENT,
    }:
        explicit_od = _explicit_origin_destination_claim(claim)
        if (
            tag is EvidenceRelevance.DIRECT_SUBJECT
            and explicit_od
            and not _explicit_od_episode_compatible(claim, statement or chunk, contexts)
        ):
            episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
        else:
            episode = EpisodeRelevance.DIRECT_QUERY_EPISODE
    elif tag is EvidenceRelevance.SAME_CONFLICT_RELEVANT:
        episode = EpisodeRelevance.SAME_CAMPAIGN_RELEVANT
    elif _normalized_subject_overlap(statement or chunk, contexts) and _episode_anchor_overlap(
        claim, statement or chunk, contexts,
    ):
        episode = EpisodeRelevance.DIRECT_QUERY_EPISODE
    elif _same_subject_other_episode(claim, statement or chunk, contexts, relevance=tag):
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    elif _time_incompatible(statement or chunk, contexts):
        episode = EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    else:
        episode = EpisodeRelevance.UNKNOWN
    admitted = episode in {
        EpisodeRelevance.DIRECT_QUERY_EPISODE,
        EpisodeRelevance.SAME_CAMPAIGN_RELEVANT,
        EpisodeRelevance.UNKNOWN,
    }
    return episode, {
        "origin": claim.source_place,
        "destination": claim.destination_place,
        "source_statement": statement[:240] if statement else None,
        "subject_relevance": tag.value,
        "episode_classification": episode.value,
        "admitted": admitted,
        "admission_reason": "EPISODE_RELEVANT" if admitted else f"REJECT_{episode.value}",
    }


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
