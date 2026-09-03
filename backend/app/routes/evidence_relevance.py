"""Generalized evidence relevance and relation admission helpers (G5E)."""
from __future__ import annotations

import re
import unicodedata
from enum import Enum

from backend.app.models import Evidence, HistoricalEvent, HistoricalEventType

_QUERY_STOP = frozenset(
    "a an and at by for from how in of on or the to what which who why with "
    "military actions battle history event political province route campaign war".split()
)
_ACTION_EQUIVALENTS = {
    "assassination": "violent_death",
    "assassinated": "violent_death",
    "murder": "violent_death",
    "murdered": "violent_death",
    "slain": "violent_death",
    "killed": "violent_death",
}
_SPATIAL_PREPOSITION = re.compile(
    r"\b(?:from|to|at|in|into|near|through|across|toward(?:s)?|between|of)\s+(?:the\s+)?$",
    re.IGNORECASE,
)
_SUBJECT_VERB_CONTEXT = re.compile(
    r"\b(?P<name>[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,})\b[^.!?;]{0,100}\b(?:"
    r"marched|marches|marching|led|commanded|crossed|fought|besieged|sailed|withdrew|"
    r"advanced|proceeded|departed|returned|said|declared|was|were|had|have|having"
    r")\b",
    re.IGNORECASE,
)
_MOVEMENT_CONTINUATION = re.compile(
    r"\b(?:they|he|she|it|his|her|their|these|those|the\s+army|the\s+fleet|the\s+forces|"
    r"the\s+consul|the\s+general|the\s+people|the\s+party|his\s+forces|her\s+forces|"
    r"the\s+ten\s+thousand|ten\s+thousand)\b",
    re.IGNORECASE,
)
_MOVEMENT_OBJECTIVE = re.compile(
    r"\b(?:march(?:ed|ing|es)?|travel(?:led|ed|ing)?|journey|retreat(?:ed|ing)?|"
    r"route|expedition|crossing|withdrawal|flight|escape(?:d)?)\b",
    re.IGNORECASE,
)
class EvidenceRelevance(str, Enum):
    DIRECT_SUBJECT = "DIRECT_SUBJECT"
    DIRECT_CAMPAIGN = "DIRECT_CAMPAIGN"
    DIRECT_EVENT = "DIRECT_EVENT"
    SAME_CONFLICT_RELEVANT = "SAME_CONFLICT_RELEVANT"
    SAME_PERIOD_BACKGROUND = "SAME_PERIOD_BACKGROUND"
    OTHER_CAMPAIGN = "OTHER_CAMPAIGN"
    UNRELATED = "UNRELATED"
    UNKNOWN = "UNKNOWN"


def normalized_terms(value: str | None) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value or "").casefold().replace("æ", "ae").replace("œ", "oe")
    return {
        _ACTION_EQUIVALENTS.get(term, term)
        for term in re.findall(r"[a-z][a-z']{2,}", normalized)
        if term not in _QUERY_STOP
    }


def query_terms(contexts: tuple[str, ...] | None) -> set[str]:
    terms: set[str] = set()
    if not contexts:
        return terms
    for context in contexts:
        if context and context.strip():
            terms |= normalized_terms(context)
    return terms


def query_proper_nouns(contexts: tuple[str, ...] | None) -> set[str]:
    nouns: set[str] = set()
    if not contexts:
        return nouns
    for context in contexts:
        nouns.update(re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}", context or ""))
    return {noun.casefold() for noun in nouns}


_SENTENCE_INITIAL_NON_NAMES = frozenset({
    "from", "then", "there", "after", "before", "when", "while", "during", "upon",
    "with", "without", "into", "onto", "through", "between", "among", "because",
    "although", "however", "therefore", "meanwhile", "later", "earlier", "thus",
    "now", "here", "still", "also", "but", "and", "or", "the", "this", "that",
    "these", "those", "where", "once", "soon", "next", "finally", "meanwhile",
})


def _named_proper_nouns(value: str) -> set[str]:
    nouns: set[str] = set()
    for match in re.finditer(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,})\b", value):
        noun = match.group(1).casefold()
        if noun in _SENTENCE_INITIAL_NON_NAMES:
            continue
        nouns.add(noun)
    return nouns


def _spatial_role_proper_nouns(value: str) -> set[str]:
    nouns: set[str] = set()
    for match in re.finditer(
        r"\b(?:from|to|at|in|into|near|through|across|toward(?:s)?|between|of)\s+(?:the\s+)?"
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}){0,3})",
        value,
        re.IGNORECASE,
    ):
        nouns.add(match.group(1).casefold())
    return nouns


def narrative_subject_proper_nouns(value: str) -> set[str]:
    spatial = _spatial_role_proper_nouns(value)
    subjects: set[str] = set()
    for match in _SUBJECT_VERB_CONTEXT.finditer(value):
        name = match.group("name").casefold()
        if name in spatial or name in _SENTENCE_INITIAL_NON_NAMES:
            continue
        subjects.add(name)
    for noun in _named_proper_nouns(value):
        if noun in spatial:
            continue
        if re.search(rf"\b{re.escape(noun)}\b[^.!?;]{{0,40}}\b(?:and|with|who|whom|whose)\b", value, re.IGNORECASE):
            subjects.add(noun)
    return subjects


def has_query_term_overlap(text: str, contexts: tuple[str, ...] | None) -> bool:
    terms = query_terms(contexts)
    if not terms:
        return True
    return bool(normalized_terms(text) & terms)


def has_subject_campaign_conflict(text: str, contexts: tuple[str, ...] | None) -> bool:
    """True when evidence names a narrative subject absent from the query subjects."""
    if not contexts:
        return False
    query_subjects = query_proper_nouns(contexts) | narrative_subject_proper_nouns(" ".join(contexts))
    if not query_subjects:
        return False
    evidence_subjects = narrative_subject_proper_nouns(text)
    if not evidence_subjects:
        return False
    if evidence_subjects & query_subjects:
        return False
    query_only = query_subjects - evidence_subjects
    evidence_only = evidence_subjects - query_subjects
    return bool(query_only and evidence_only)


def classify_evidence_relevance(
    text: str,
    contexts: tuple[str, ...] | None,
    *,
    window_text: str | None = None,
) -> EvidenceRelevance:
    if not contexts:
        return EvidenceRelevance.UNKNOWN
    combined = " ".join(part for part in (window_text, text) if part).strip() or text
    if has_subject_campaign_conflict(text, contexts):
        return EvidenceRelevance.OTHER_CAMPAIGN
    if has_query_term_overlap(text, contexts):
        if query_proper_nouns(contexts) & _named_proper_nouns(text):
            return EvidenceRelevance.DIRECT_SUBJECT
        return EvidenceRelevance.DIRECT_CAMPAIGN
    if window_text and has_query_term_overlap(window_text, contexts):
        if _MOVEMENT_CONTINUATION.search(text) or _MOVEMENT_OBJECTIVE.search(text):
            return EvidenceRelevance.DIRECT_SUBJECT
        return EvidenceRelevance.DIRECT_EVENT
    if has_query_term_overlap(combined, contexts):
        return EvidenceRelevance.DIRECT_EVENT
    if _MOVEMENT_OBJECTIVE.search(text) and has_query_term_overlap(combined, contexts):
        return EvidenceRelevance.SAME_CONFLICT_RELEVANT
    if normalized_terms(text) & query_terms(contexts):
        return EvidenceRelevance.SAME_CONFLICT_RELEVANT
    return EvidenceRelevance.UNKNOWN


def bounded_window_text(sentences: list[str], index: int) -> str:
    start = max(0, index - 1)
    end = min(len(sentences), index + 2)
    return " ".join(sentences[start:index] + sentences[index + 1 : end])


def movement_eligibility_with_context(
    sentence: str,
    event_type: HistoricalEventType,
    contexts: tuple[str, ...] | None,
    *,
    sentences: list[str],
    index: int,
    evidence_text: str,
) -> bool:
    if not contexts:
        return True
    if has_query_term_overlap(sentence, contexts):
        return True
    window = bounded_window_text(sentences, index)
    relevance = classify_evidence_relevance(sentence, contexts, window_text=window)
    if relevance is EvidenceRelevance.OTHER_CAMPAIGN:
        return False
    if relevance in {
        EvidenceRelevance.DIRECT_SUBJECT,
        EvidenceRelevance.DIRECT_CAMPAIGN,
        EvidenceRelevance.DIRECT_EVENT,
        EvidenceRelevance.SAME_CONFLICT_RELEVANT,
    }:
        return True
    if event_type is HistoricalEventType.MOVEMENT:
        evidence_tag = classify_evidence_relevance(evidence_text, contexts)
        if evidence_tag in {
            EvidenceRelevance.DIRECT_SUBJECT,
            EvidenceRelevance.DIRECT_CAMPAIGN,
            EvidenceRelevance.DIRECT_EVENT,
            EvidenceRelevance.SAME_CONFLICT_RELEVANT,
        }:
            if _MOVEMENT_CONTINUATION.search(sentence) or re.search(
                r"\b(?:from|travel(?:led|ed|ing)?|journey)\b", sentence, re.IGNORECASE
            ):
                return True
            if re.search(r"\bfrom\b[^.]{0,120}\b(?:to|into)\b", sentence, re.IGNORECASE):
                return True
        if index > 0 and has_query_term_overlap(sentences[index - 1], contexts):
            if _MOVEMENT_CONTINUATION.search(sentence) or _MOVEMENT_OBJECTIVE.search(sentence):
                return True
    return False


def event_relevance(
    event: HistoricalEvent,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> EvidenceRelevance:
    parts = [event.summary or ""]
    for ref in event.evidence_refs:
        item = evidence_by_id.get(ref)
        if item is not None:
            parts.append(" ".join(value for value in (item.text, item.excerpt) if value))
    return classify_evidence_relevance(" ".join(parts), contexts)
