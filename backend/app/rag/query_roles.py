"""Deterministic query-token roles for hybrid retrieval.

Tokens are not treated as equal keywords.  PERSON is an identity constraint,
LOCATION is geographic support, ACTION is event semantics, and GENERIC is a
weak topical remainder.  No NER, LLM, or query-specific branches.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP_WORDS = frozenset({"a", "an", "and", "at", "battle", "by", "for", "in", "of", "on", "the", "to", "with"})
_QUERY_SCAFFOLD = frozenset({
    "what", "who", "where", "when", "why", "how",
    "did", "does", "do", "was", "were", "is", "are", "am", "be", "been", "being",
    "has", "have", "had",
    "tell", "show", "find", "describe", "explain",
    "happen", "happened", "happens", "happening",
    "me", "about", "please",
    "can", "could", "would", "should", "may", "might",
    "which", "whose", "whom",
})
_DETERMINERS = frozenset({"a", "an", "the"})
_GENERIC_TERMS = frozenset({
    "action", "actions", "affairs", "campaign", "campaigns", "event", "events",
    "history", "historical", "military", "operation", "operations",
})
_LOCATION_PREP = frozenset({"in", "at", "near", "from", "into", "through"})
_LOCATION_OF_HEADS = frozenset({"battle", "siege", "war"})
_LOCATION_ALIASES = {
    "spain": frozenset({"spain", "spanish", "hispania"}),
    "spanish": frozenset({"spain", "spanish", "hispania"}),
    "hispania": frozenset({"spain", "spanish", "hispania"}),
}
_ACTION_GROUPS = (
    frozenset({"assassination", "assassinate", "assassinated", "murder", "murdered", "slain", "killed", "stabbed"}),
    frozenset({"battle", "battled", "fought", "fight", "defeated", "defeat", "vanquished", "victory", "victorious"}),
)
_MOVEMENT_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"move", "moved", "moving"}),
    frozenset({"march", "marched", "marching"}),
    frozenset({"advance", "advanced", "advancing"}),
    frozenset({"proceed", "proceeded", "proceeding"}),
    frozenset({"pass", "passed", "passing"}),
    frozenset({"cross", "crossed", "crossing"}),
    frozenset({"sail", "sailed", "sailing"}),
    frozenset({"depart", "departed", "departing"}),
    frozenset({"leave", "left", "leaving"}),
    frozenset({"arrive", "arrived", "arriving"}),
    frozenset({"reach", "reached", "reaching"}),
    frozenset({"enter", "entered", "entering"}),
    frozenset({"land", "landed", "landing"}),
    frozenset({"flee", "fled", "fleeing", "fly"}),
    frozenset({"come", "came", "coming"}),
    frozenset({"go", "went", "going"}),
    frozenset({"travel", "traveled", "travelled", "traveling", "travelling"}),
    frozenset({"journey", "journeyed"}),
    frozenset({"route"}),
)
_MOVEMENT_TERMS = frozenset().union(*_MOVEMENT_FAMILIES)
_MOVEMENT_TERM_TO_FAMILY = {term: family for family in _MOVEMENT_FAMILIES for term in family}
_GENERIC_MOVEMENT_TRIGGERS = frozenset({
    "route", "routes", "movement", "movements", "travel", "journey", "trace", "reconstruct",
})
_GENERIC_EXPANSION_FAMILIES = tuple(
    family for family in _MOVEMENT_FAMILIES if family != frozenset({"route"})
)
_ACTION_UNION = frozenset().union(*_ACTION_GROUPS) | _MOVEMENT_TERMS
# Closed praenomen set used only to reject a different named person who shares a surname.
_PRAENOMINA = frozenset({
    "lucius", "marcus", "gaius", "caius", "quintus", "publius", "gnaeus",
    "aulus", "sextus", "servius", "tiberius", "decimus", "spurius",
})


def normalized_tokens(text: str) -> frozenset[str]:
    """Case-fold and remove classical diacritics (Cæsar -> caesar)."""
    normalized = unicodedata.normalize("NFKD", text).replace("æ", "ae").replace("Æ", "AE")
    return frozenset(_WORD.findall(normalized.casefold()))


@dataclass(frozen=True)
class QueryRoleAnalysis:
    person_terms: frozenset[str]
    person_sequence: tuple[str, ...]
    location_terms: frozenset[str]
    action_terms: frozenset[str]
    generic_terms: frozenset[str]
    context_terms: frozenset[str]
    expanded_action_terms: frozenset[str]
    movement_inflection_terms: frozenset[str]
    stop_terms: frozenset[str]

    @property
    def location_match_terms(self) -> frozenset[str]:
        aliases = set(self.location_terms)
        for term in self.location_terms:
            aliases.update(_LOCATION_ALIASES.get(term, ()))
        return frozenset(aliases)


def _movement_family_expansions(action_terms: frozenset[str]) -> frozenset[str]:
    """Sibling inflections for explicit movement terms (person-gated at score time)."""
    expanded: set[str] = set()
    for term in action_terms & _MOVEMENT_TERMS:
        expanded.update(_MOVEMENT_TERM_TO_FAMILY[term] - action_terms)
    return frozenset(expanded)


def _generic_movement_expansions(norms: list[str], person_sequence: tuple[str, ...], action_terms: frozenset[str]) -> frozenset[str]:
    """Route/movement intent expands verb families on subject-gated passages only."""
    if not person_sequence or not (set(norms) & _GENERIC_MOVEMENT_TRIGGERS):
        return frozenset()
    expanded: set[str] = set()
    for family in _GENERIC_EXPANSION_FAMILIES:
        expanded.update(family)
    return frozenset(expanded - action_terms)


_EPISODE_CONTEXT_SCAFFOLD = _STOP_WORDS | _QUERY_SCAFFOLD | frozenset({
    "s", "his", "her", "their", "from", "into", "through", "across", "until", "after",
    "during", "leading", "against", "over", "near", "between", "around", "upon",
})


def route_movement_query(norms: frozenset[str] | list[str], roles: QueryRoleAnalysis) -> bool:
    norm_set = set(norms)
    return bool(norm_set & _GENERIC_MOVEMENT_TRIGGERS) or bool(roles.action_terms & _MOVEMENT_TERMS)


def episode_context_terms(roles: QueryRoleAnalysis) -> frozenset[str]:
    """Episode/location anchors for conjunctive route scoring."""
    context = frozenset(
        term for term in roles.context_terms
        if term not in _EPISODE_CONTEXT_SCAFFOLD and term not in _MOVEMENT_TERMS
    )
    return roles.location_match_terms | context


def movement_scoring_terms(roles: QueryRoleAnalysis) -> frozenset[str]:
    return (roles.action_terms & _MOVEMENT_TERMS) | roles.movement_inflection_terms | roles.expanded_action_terms


_SUBJECT_CONTEXT_SKIP = frozenset({"book", "chapter", "contents", "part", "preserved", "section", "volume"})
_KNOWN_NARRATIVE_PERSONS = frozenset({"alexander", "brutus", "caesar", "cato", "cicero", "hercules", "lucullus", "mithridates", "pompey", "theseus", "xenophon"})

def _biography_subject_label(label: str) -> bool:
    label = label.strip()
    head = label.split()[0].casefold().rstrip("'s") if label else ""
    return len(label) >= 3 and label == label.upper() and head not in _SUBJECT_CONTEXT_SKIP and not re.fullmatch(r"[ivxlc]+", label, re.I)

def extract_subject_context_terms(metadata: dict) -> frozenset[str]:
    """Local biography-subject tokens from navigation_path / heading only."""
    terms: set[str] = set()
    nav_raw = metadata.get("navigation_path_json") or metadata.get("navigation_path")
    if isinstance(nav_raw, str):
        try: nav_raw = json.loads(nav_raw)
        except json.JSONDecodeError: nav_raw = []
    for item in nav_raw or []:
        label = str(item).strip()
        if len(label.split()) == 1 and _biography_subject_label(label): terms.update(normalized_tokens(label))
    heading = str(metadata.get("heading") or "").strip()
    if heading and _biography_subject_label(heading): terms.update(normalized_tokens(heading))
    return frozenset(terms)

def body_conflicting_person(query_person: frozenset[str], body_tokens: frozenset[str]) -> bool:
    return bool((body_tokens & _KNOWN_NARRATIVE_PERSONS) - query_person)


def analyze_query(query: str) -> QueryRoleAnalysis:
    """Assign PERSON / LOCATION / ACTION / GENERIC roles from the raw query."""
    normalized = unicodedata.normalize("NFKD", query or "").replace("æ", "ae").replace("Æ", "AE")
    pairs = tuple((match.group(), match.group().casefold()) for match in _WORD.finditer(normalized))
    norms = [norm for _, norm in pairs]
    action_terms = frozenset(token for token in norms if token in _ACTION_UNION)
    movement_inflection = _movement_family_expansions(action_terms)
    expanded = (
        frozenset().union(*(group for group in _ACTION_GROUPS if action_terms & group))
    ) - action_terms
    stop_in_query = frozenset(token for token in norms if token in _STOP_WORDS)
    location: set[str] = set()
    for index, norm in enumerate(norms):
        nxt = norms[index + 1] if index + 1 < len(norms) else None
        if nxt is None:
            continue
        if norm in _LOCATION_PREP and nxt not in _STOP_WORDS and nxt not in _ACTION_UNION:
            location.add(nxt)
        if norm in _LOCATION_OF_HEADS and nxt == "of" and index + 2 < len(norms):
            tail = norms[index + 2]
            if tail not in _STOP_WORDS:
                location.add(tail)
    generic = frozenset(token for token in norms if token in _GENERIC_TERMS)
    reserved = _STOP_WORDS | _ACTION_UNION | _GENERIC_TERMS | location | _QUERY_SCAFFOLD
    after_determiner = {index + 1 for index, norm in enumerate(norms) if norm in _DETERMINERS}
    leftover = [(index, raw, norm) for index, (raw, norm) in enumerate(pairs) if norm not in reserved]
    titled = [(index, raw, norm) for index, raw, norm in leftover if raw[:1].isupper() and index not in after_determiner]
    if titled:
        person_sequence = [norm for _, _, norm in titled]
        context = [norm for index, raw, norm in leftover if (index, raw, norm) not in titled]
    else:
        person_sequence = [norm for _, _, norm in leftover]
        context = []
    expanded |= _generic_movement_expansions(norms, tuple(person_sequence), action_terms)
    return QueryRoleAnalysis(
        person_terms=frozenset(person_sequence),
        person_sequence=tuple(person_sequence),
        location_terms=frozenset(location),
        action_terms=action_terms,
        generic_terms=generic,
        context_terms=frozenset(context),
        expanded_action_terms=expanded,
        movement_inflection_terms=movement_inflection,
        stop_terms=stop_in_query,
    )


def person_support(roles: QueryRoleAnalysis, text_tokens: frozenset[str]) -> float:
    """Full-name match outranks surname/reference; a conflicting praenomen is not identity."""
    if not roles.person_terms:
        return 0.0
    if roles.person_terms <= text_tokens:
        return 0.08
    overlap = roles.person_terms & text_tokens
    if not overlap:
        return 0.0
    if len(roles.person_sequence) == 1:
        return 0.04
    surname = roles.person_sequence[-1]
    given = frozenset(roles.person_sequence[:-1])
    if surname not in text_tokens:
        return 0.04 if overlap else 0.0
    if given and not (given & text_tokens) and ((text_tokens & _PRAENOMINA) - roles.person_terms):
        return 0.0
    return 0.04


def location_support(roles: QueryRoleAnalysis, text_tokens: frozenset[str]) -> float:
    return 0.06 if roles.location_match_terms & text_tokens else 0.0


def action_support(roles: QueryRoleAnalysis, text_tokens: frozenset[str], *, person: float, location: float) -> float:
    native = bool(roles.action_terms & text_tokens)
    expanded = bool((roles.expanded_action_terms | roles.movement_inflection_terms) & text_tokens)
    if person > 0 or location > 0:
        matched = native or expanded
    else:
        matched = native
    if not matched:
        return 0.0
    if person > 0 or location > 0:
        return 0.12
    return 0.02


def generic_support(roles: QueryRoleAnalysis, text_tokens: frozenset[str]) -> float:
    return min(0.02, 0.01 * len(roles.generic_terms & text_tokens))
