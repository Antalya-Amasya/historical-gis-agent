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
            terms |= {
                normalize_subject_name(term)
                if re.search(r"(?:'s|'s)$", term)
                else term
                for term in normalized_terms(context)
            }
    return terms


def query_proper_nouns(contexts: tuple[str, ...] | None) -> set[str]:
    nouns: set[str] = set()
    if not contexts:
        return nouns
    for context in contexts:
        nouns.update(re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}", context or ""))
    return {noun.casefold() for noun in nouns}


def normalize_subject_name(value: str) -> str:
    return re.sub(r"(?:'s|'s)$", "", value.casefold())


def _person_name_tokens(prefix: str, spatial: set[str]) -> tuple[str, ...] | None:
    match = _PERSON_NAME_TAIL.search(prefix.strip())
    if not match:
        return None
    tokens: list[str] = []
    for part in match.group(1).split():
        token = normalize_subject_name(part)
        if token in _TEMPORAL_ERA_SUBJECTS or token in spatial or token in _SENTENCE_INITIAL_NON_NAMES:
            continue
        if _NUMERIC_YEAR.match(token):
            continue
        tokens.append(token)
    return tuple(tokens) if tokens else None


def explicit_person_identities(value: str) -> list[tuple[str, ...]]:
    spatial = _spatial_role_proper_nouns(value)
    identities: list[tuple[str, ...]] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", value):
        if not sentence.strip():
            continue
        verb = _VERB_HEAD.search(sentence)
        if verb is None:
            continue
        name = _person_name_tokens(sentence[:verb.start()], spatial)
        if name:
            identities.append(name)
    return list(dict.fromkeys(identities))


def _query_person_tokens(raw_name: str) -> tuple[str, ...] | None:
    tokens: list[str] = []
    for part in raw_name.split():
        token = normalize_subject_name(part)
        if token in _QUERY_STOP or token in _TEMPORAL_ERA_SUBJECTS or token in _SENTENCE_INITIAL_NON_NAMES:
            continue
        if _NUMERIC_YEAR.match(token):
            continue
        tokens.append(token)
    return tuple(tokens) if tokens else None


def normalized_query_person_identities(contexts: tuple[str, ...] | None) -> list[tuple[str, ...]]:
    if not contexts:
        return []
    identities: list[tuple[str, ...]] = []
    for context in contexts:
        match = _QUERY_PERSON.search(context or "")
        if not match:
            continue
        tokens = _query_person_tokens(match.group(1))
        if tokens:
            identities.append(tokens)
    return list(dict.fromkeys(identities))


def person_identities_match(text: str, contexts: tuple[str, ...] | None) -> bool:
    query_ids = normalized_query_person_identities(contexts)
    evidence_ids = explicit_person_identities(text)
    if query_ids:
        for query_id in query_ids:
            for evidence_id in evidence_ids:
                if query_id == evidence_id:
                    return True
                if len(query_id) == 1 and len(evidence_id) == 1 and query_id[0] == evidence_id[0]:
                    return True
        if all(len(query_id) == 1 for query_id in query_ids):
            query_tokens = {token for query_id in query_ids for token in query_id}
            evidence_tokens = {token for evidence_id in evidence_ids for token in evidence_id}
            evidence_tokens |= normalized_narrative_subjects(text)
            return bool(query_tokens & evidence_tokens)
        return False
    query_subjects = normalized_query_subject_terms(contexts)
    if not query_subjects:
        return False
    evidence_subjects = normalized_narrative_subjects(text)
    if query_subjects & evidence_subjects:
        return True
    evidence_nouns = {normalize_subject_name(noun) for noun in _named_proper_nouns(text)}
    return bool(query_subjects & evidence_nouns)


def person_identities_conflict(text: str, contexts: tuple[str, ...] | None) -> bool:
    query_ids = normalized_query_person_identities(contexts)
    evidence_ids = explicit_person_identities(text)
    multi_query = [query_id for query_id in query_ids if len(query_id) >= 2]
    multi_evidence = [evidence_id for evidence_id in evidence_ids if len(evidence_id) >= 2]
    if multi_query and multi_evidence:
        return all(evidence_id not in multi_query for evidence_id in multi_evidence)
    return False


def normalized_query_subject_terms(contexts: tuple[str, ...] | None) -> set[str]:
    if not contexts:
        return set()
    terms = {normalize_subject_name(name) for name in query_proper_nouns(contexts)}
    terms |= {normalize_subject_name(name) for name in narrative_subject_proper_nouns(" ".join(contexts))}
    return terms


def normalized_narrative_subjects(text: str) -> set[str]:
    return {normalize_subject_name(name) for name in narrative_subject_proper_nouns(text)}


def has_normalized_subject_overlap(text: str, contexts: tuple[str, ...] | None) -> bool:
    return person_identities_match(text, contexts)


_SENTENCE_INITIAL_NON_NAMES = frozenset({
    "from", "then", "there", "after", "before", "when", "while", "during", "upon",
    "with", "without", "into", "onto", "through", "between", "among", "because",
    "although", "however", "therefore", "meanwhile", "later", "earlier", "thus",
    "now", "here", "still", "also", "but", "and", "or", "the", "this", "that",
    "these", "those", "where", "once", "soon", "next", "finally", "meanwhile",
})
_TEMPORAL_ERA_SUBJECTS = frozenset({"bce", "bc", "ce", "ad"})
_VERB_HEAD = re.compile(
    r"\b(?:marched|marches|marching|led|commanded|crossed|fought|besieged|sailed|withdrew|"
    r"advanced|proceeded|departed|returned|said|declared|was|were|had|have|having)\b",
    re.IGNORECASE,
)
_QUERY_PERSON = re.compile(
    r"\b(?:"
    r"(?:(?i:trace|reconstruct|follow))\s+(?:(?i:the)\s+(?i:route)\s+(?i:of)\s+)?"
    r"|(?:(?i:the)\s+(?i:route)\s+(?i:of)\s+)"
    r")"
    r"((?:[A-Z][A-Za-z'’\u2019-]+(?:\s+[A-Z][A-Za-z'’\u2019-]+){0,3}))"
    r"(?:['\u2019]s)?"
    r"(?=\s+(?:(?i:route|from|in|during|through|across|into|toward(?:s)?)\b)|(?=[.!?;])|$)",
)
_PERSON_NAME_TAIL = re.compile(
    r"((?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ'’\u2019-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ'’\u2019-]+){0,3}))\s*$",
)
_NUMERIC_YEAR = re.compile(r"^\d{1,4}$")


def _proper_nouns_before_verb(fragment: str, spatial: set[str]) -> set[str]:
    verb = _VERB_HEAD.search(fragment)
    if not verb:
        return set()
    prefix = fragment[:verb.start()]
    names: set[str] = set()
    for match in re.finditer(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,})\b", prefix):
        token = match.group(1).casefold()
        if token in _TEMPORAL_ERA_SUBJECTS or token in spatial or token in _SENTENCE_INITIAL_NON_NAMES:
            continue
        names.add(token)
    return names


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
        match_text = match.group(0)
        if name in _TEMPORAL_ERA_SUBJECTS:
            subjects |= _proper_nouns_before_verb(match_text, spatial)
            continue
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
    if person_identities_conflict(text, contexts):
        return True
    if any(len(query_id) >= 2 for query_id in normalized_query_person_identities(contexts)):
        return False
    query_subjects = normalized_query_subject_terms(contexts)
    if not query_subjects:
        return False
    evidence_subjects = normalized_narrative_subjects(text)
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
    if has_normalized_subject_overlap(text, contexts):
        return EvidenceRelevance.DIRECT_SUBJECT
    if has_query_term_overlap(text, contexts):
        if person_identities_match(text, contexts):
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


_STATEMENT_ADMISSIBLE = frozenset({
    EvidenceRelevance.DIRECT_SUBJECT,
    EvidenceRelevance.DIRECT_CAMPAIGN,
    EvidenceRelevance.DIRECT_EVENT,
    EvidenceRelevance.SAME_CONFLICT_RELEVANT,
})


def _evidence_item_text(item: Evidence) -> str:
    return " ".join(dict.fromkeys(value for value in (item.text, item.excerpt) if value))


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]


def relation_supporting_statements(event: HistoricalEvent) -> list[str]:
    """Statements that ground a movement relation, preferring explicit movement assertions."""
    statements = list(dict.fromkeys(event.source_statements or []))
    if event.summary and event.summary not in statements:
        statements.insert(0, event.summary)
    movement_statements = [
        statement
        for statement in statements
        if _MOVEMENT_OBJECTIVE.search(statement)
        or re.search(r"\bfrom\b[^.]{0,160}\b(?:to|into)\b", statement, re.IGNORECASE)
    ]
    return movement_statements or statements[:1] or ([event.summary] if event.summary else [])


def statement_evidence_window(
    statement: str,
    event: HistoricalEvent,
    evidence_by_id: dict[str, Evidence],
) -> str | None:
    """Bounded adjacent-sentence context within the same evidence item as the statement."""
    normalized = statement.strip()
    if not normalized:
        return None
    for ref in event.evidence_refs:
        item = evidence_by_id.get(ref)
        if item is None:
            continue
        sentences = _split_sentences(_evidence_item_text(item))
        for index, sentence in enumerate(sentences):
            if normalized in sentence or sentence in normalized:
                return bounded_window_text(sentences, index)
    return None


def statement_relation_relevance(
    statement: str,
    contexts: tuple[str, ...] | None,
    *,
    event: HistoricalEvent | None = None,
    evidence_by_id: dict[str, Evidence] | None = None,
) -> EvidenceRelevance:
    window = (
        statement_evidence_window(statement, event, evidence_by_id)
        if event is not None and evidence_by_id is not None
        else None
    )
    return classify_evidence_relevance(statement, contexts, window_text=window)


def same_movement_relation_relevance(
    event: HistoricalEvent,
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
) -> EvidenceRelevance:
    """Prefer relation-supporting statements over whole-chunk relevance."""
    chunk_relevance = event_relevance(event, evidence_by_id, contexts)
    best: EvidenceRelevance | None = None
    for statement in relation_supporting_statements(event):
        tag = statement_relation_relevance(
            statement, contexts, event=event, evidence_by_id=evidence_by_id,
        )
        if tag is EvidenceRelevance.OTHER_CAMPAIGN:
            return EvidenceRelevance.OTHER_CAMPAIGN
        if tag in _STATEMENT_ADMISSIBLE:
            return tag
        if best is None or tag is not EvidenceRelevance.UNKNOWN:
            best = tag
    if best in _STATEMENT_ADMISSIBLE:
        return best
    if chunk_relevance is EvidenceRelevance.OTHER_CAMPAIGN and best is EvidenceRelevance.UNKNOWN:
        return EvidenceRelevance.UNKNOWN
    return chunk_relevance


_ROUTE_ADMISSIBLE = _STATEMENT_ADMISSIBLE


def relation_admission_allowed(
    relation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
    *,
    rule,
) -> bool:
    if not contexts:
        return True
    final = classify_relation_relevance(
        relation, events_by_id, evidence_by_id, contexts, rule=rule,
    )
    from backend.app.routes.episode_relevance import classify_event_anchor_episode

    episode, detail = classify_event_anchor_episode(
        relation, events_by_id, evidence_by_id, contexts, subject_relevance=final,
    )
    if detail.get("admitted"):
        return True
    if final is EvidenceRelevance.OTHER_CAMPAIGN:
        return False
    if rule.value == "SAME_MOVEMENT_EVENT":
        evidence_allowed = final in _STATEMENT_ADMISSIBLE
    elif rule.value == "TEMPORAL_ORDER":
        evidence_allowed = final in _ROUTE_ADMISSIBLE or final is EvidenceRelevance.UNKNOWN
    elif rule.value == "SOURCE_STRUCTURAL_ORDER":
        evidence_allowed = final in _ROUTE_ADMISSIBLE
    else:
        evidence_allowed = True
    if not evidence_allowed:
        return False
    return bool(detail["admitted"])


def classify_relation_relevance(
    relation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
    *,
    rule,
) -> EvidenceRelevance:
    from backend.app.routes.event_route_orchestration import OrderingRule

    if not contexts:
        return EvidenceRelevance.UNKNOWN
    if rule is OrderingRule.SAME_MOVEMENT_EVENT:
        tags = [
            same_movement_relation_relevance(events_by_id[event_id], evidence_by_id, contexts)
            for event_id in relation.event_ids
            if event_id in events_by_id
        ]
        if any(tag is EvidenceRelevance.OTHER_CAMPAIGN for tag in tags):
            return EvidenceRelevance.OTHER_CAMPAIGN
        for tag in tags:
            if tag in _STATEMENT_ADMISSIBLE:
                return tag
        return tags[0] if tags else EvidenceRelevance.UNKNOWN
    tags = [
        event_relevance(events_by_id[event_id], evidence_by_id, contexts)
        for event_id in relation.event_ids
        if event_id in events_by_id
    ]
    if any(tag is EvidenceRelevance.OTHER_CAMPAIGN for tag in tags):
        return EvidenceRelevance.OTHER_CAMPAIGN
    for tag in tags:
        if tag in _STATEMENT_ADMISSIBLE:
            return tag
    return tags[0] if tags else EvidenceRelevance.UNKNOWN


def relation_admission_diagnostic(
    relation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    contexts: tuple[str, ...] | None,
    *,
    rule,
) -> dict[str, object]:
    from backend.app.routes.event_route_orchestration import OrderingRule

    statements: list[str] = []
    statement_tags: list[str] = []
    window_tags: list[str] = []
    chunk_tags: list[str] = []
    for event_id in relation.event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue
        chunk_tags.append(event_relevance(event, evidence_by_id, contexts).value)
        for statement in relation_supporting_statements(event):
            statements.append(statement)
            statement_tags.append(
                classify_evidence_relevance(statement, contexts).value,
            )
            window = statement_evidence_window(statement, event, evidence_by_id)
            window_tags.append(
                classify_evidence_relevance(statement, contexts, window_text=window).value
                if window
                else EvidenceRelevance.UNKNOWN.value
            )
    final = classify_relation_relevance(
        relation, events_by_id, evidence_by_id, contexts, rule=rule,
    )
    allowed = relation_admission_allowed(
        relation, events_by_id, evidence_by_id, contexts, rule=rule,
    )
    from backend.app.routes.episode_relevance import classify_event_anchor_episode

    episode, episode_detail = classify_event_anchor_episode(
        relation, events_by_id, evidence_by_id, contexts, subject_relevance=final,
    )
    return {
        "relation_type": rule.value,
        "earlier": relation.earlier,
        "later": relation.later,
        "event_ids": list(relation.event_ids),
        "supporting_statements": statements,
        "statement_relevance": statement_tags,
        "window_relevance": window_tags,
        "chunk_relevance": chunk_tags,
        "final_relation_relevance": final.value,
        "episode_classification": episode.value,
        "episode_admitted": episode_detail.get("admitted"),
        "admission": "ALLOW" if allowed else "REJECT",
        "reason": (
            f"EPISODE_{episode_detail.get('admission_reason')}"
            if allowed is False and not episode_detail.get("admitted")
            else "LOCAL_STATEMENT_RELEVANCE" if allowed and rule is OrderingRule.SAME_MOVEMENT_EVENT
            else "CAMPAIGN_RELEVANCE_REJECTED" if not allowed else "ADMISSIBLE"
        ),
    }
