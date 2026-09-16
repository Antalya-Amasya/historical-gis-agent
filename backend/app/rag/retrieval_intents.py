"""Deterministic retrieval-intent decomposition for movement route queries (G5M)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.app.rag.query_roles import QueryRoleAnalysis, analyze_query, normalized_tokens

_RETRIEVAL_INTENTS = frozenset({"CANONICAL", "SUBJECT", "EPISODE", "MOVEMENT", "ENDPOINT", "FEATURE", "REGION"})
_YEAR_RE = re.compile(r"\b(\d{1,4})\s*(?:bce|bc)\b", re.IGNORECASE)
_SUBJECT_SCAFFOLD = frozenset({
    "trace", "reconstruct", "show", "map", "follow", "route", "routes", "movements", "movement",
    "major", "during", "after", "through", "into", "ending", "near", "back", "bce", "bc",
    "defeat", "arrival", "campaign", "campaigns", "world", "greek", "ten", "first", "until",
})
_ROUTE_QUERY_HINTS = frozenset({"route", "trace", "reconstruct", "journey", "march", "movements", "movement"})
_MOVEMENT_HINTS = frozenset({
    "march", "marched", "marching", "retreat", "retreated", "cross", "crossed", "crossing",
    "route", "travel", "travelled", "traveled", "journey", "advance", "advanced", "withdraw",
    "withdrew", "sail", "sailed", "land", "landed", "reach", "reached", "depart", "departed",
    "arrive", "arrived", "escape", "escaped", "flight", "expedition", "movement", "move",
    "moved", "proceed", "proceeded", "embark", "embarked",
})
_FEATURE_HINTS = frozenset({
    "river", "rivers", "mountain", "mountains", "sea", "strait", "straits", "pass", "port",
    "ports", "alps", "adriatic", "mediterranean", "isthmus", "gulf", "channel", "crossing",
})
_EPISODE_HINTS = frozenset({"battle", "war", "siege", "campaign", "defeat", "victory", "revolt"})
_ENDPOINT_FROM = re.compile(
    r"\bfrom\s+(?:the\s+)?([A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+(?:\s+(?:of\s+the\s+)?[A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+){0,3})",
)
_ENDPOINT_TO = re.compile(
    r"\b(?:to|into|toward(?:s)?)\s+(?:the\s+)?([A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+(?:\s+(?:of\s+the\s+)?[A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+){0,3})",
)
_ENDPOINT_BACK_TO = re.compile(
    r"\bback\s+to\s+(?:the\s+)?([A-Z][A-Za-zÀ-ÖØ-öø-ÿ']+(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ']+){0,2})\b",
)
_EPISODE_LOCATION_EXCLUDE = frozenset({
    "pharsalus", "cunaxa", "pontus", "anatolia", "epirus", "egypt", "greece", "italy",
    "bactria", "hindu", "kush", "hydaspes", "armenia", "hispania", "africa", "syria",
    "asia", "minor", "mediterranean", "adriatic", "brundisium", "carthage", "war",
    "mithridatic", "campaign", "indian",
})
_ROMAN_NUMERAL = re.compile(r"^v+i{0,3}$", re.IGNORECASE)
_MULTIWORD_SUBJECT = re.compile(
    r"\b(?:the\s+)?((?:Ten|Three|Five|Six|Seven|Eight|Nine)\s+Thousand)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalIntent:
    kind: str
    query: str

    def __post_init__(self) -> None:
        if self.kind not in _RETRIEVAL_INTENTS:
            raise ValueError(f"unsupported retrieval intent: {self.kind}")


def _possessive_subject(query: str) -> str | None:
    for match in re.finditer(r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,2})'s\b", query):
        tokens = [token for token in match.group(1).split() if token.casefold() not in _SUBJECT_SCAFFOLD]
        if tokens:
            return " ".join(tokens)
    return None


_PRIMARY_ROUTE_GRAMMAR = frozenset({"route", "routes", "movement", "movements", "march", "campaign", "crossing"})


def primary_route_subject(query: str, roles: QueryRoleAnalysis | None = None) -> str | None:
    """Person/entity whose movement the user requested; not broad person_terms."""
    text = (query or "").strip()
    if not text or not (normalized_tokens(text) & (_ROUTE_QUERY_HINTS | _PRIMARY_ROUTE_GRAMMAR)):
        return None
    possessive = _possessive_subject(text)
    if possessive:
        return possessive
    multi = _MULTIWORD_SUBJECT.search(text)
    if multi:
        return multi.group(1)
    return None


def _subject_phrase(query: str, roles: QueryRoleAnalysis) -> str:
    parts: list[str] = []
    match = _MULTIWORD_SUBJECT.search(query)
    if match:
        parts.append(match.group(1))
    possessive = _possessive_subject(query)
    if possessive and possessive.casefold() not in {part.casefold() for part in parts}:
        parts.append(possessive)
    possessive_tokens = {token.casefold() for token in possessive.split()} if possessive else set()
    location_folded = {term.casefold() for term in roles.location_terms}
    excluded = _SUBJECT_SCAFFOLD | location_folded | _EPISODE_LOCATION_EXCLUDE | set(roles.action_terms)
    for name in roles.person_sequence:
        token = name.strip()
        folded = token.casefold()
        if not token or folded in excluded or folded in possessive_tokens or _ROMAN_NUMERAL.match(folded):
            continue
        if folded == "thousand" and any("thousand" in part.casefold() for part in parts):
            continue
        if folded in {part.casefold() for part in parts}:
            continue
        if any(folded in part.casefold().split() or part.casefold() in folded for part in parts):
            continue
        parts.append(token.title() if folded == token else token)
    if not parts:
        for name in roles.person_sequence:
            if name.casefold() not in excluded:
                parts.append(name)
                break
    return " ".join(parts[:4])


def _query_year(query: str) -> str | None:
    match = _YEAR_RE.search(query)
    return f"{match.group(1)} BCE" if match else None


def _endpoint_terms(query: str, roles: QueryRoleAnalysis) -> tuple[str, ...]:
    terms: list[str] = []
    for pattern in (_ENDPOINT_FROM, _ENDPOINT_TO, _ENDPOINT_BACK_TO):
        for match in pattern.finditer(query):
            phrase = match.group(1).strip()
            if phrase and phrase.casefold() not in {term.casefold() for term in terms}:
                terms.append(phrase)
    for term in sorted(roles.location_terms):
        if term.casefold() not in {"bce", "bc"} and term not in {item.casefold() for item in terms}:
            terms.append(term.title() if term.islower() else term)
    return tuple(terms[:6])


def _region_terms(query: str, roles: QueryRoleAnalysis) -> tuple[str, ...]:
    regions: list[str] = []
    for match in re.finditer(
        r"\b(?:the\s+)?([A-Z][A-Za-z]+)\s+(world|minor|mediterranean|peninsula|empire|coast|plains?)\b",
        query,
    ):
        phrase = f"{match.group(1)} {match.group(2)}"
        if phrase.casefold() not in {item.casefold() for item in regions}:
            regions.append(phrase)
    for term in sorted(roles.context_terms):
        if term in {"greek", "world", "asia", "europe", "africa", "italy", "greece", "egypt", "anatolia"}:
            if term not in {item.casefold() for item in regions}:
                regions.append(term.title() if term.islower() else term)
    return tuple(list(dict.fromkeys(regions))[:4])


def _feature_terms(query: str) -> tuple[str, ...]:
    tokens = normalized_tokens(query)
    return tuple(sorted(token for token in tokens if token in _FEATURE_HINTS))


def _episode_terms(query: str, roles: QueryRoleAnalysis) -> tuple[str, ...]:
    tokens = normalized_tokens(query)
    found = [token for token in tokens if token in _EPISODE_HINTS]
    found.extend(token for token in roles.action_terms if token in _EPISODE_HINTS)
    return tuple(dict.fromkeys(found))


def _movement_terms(query: str) -> tuple[str, ...]:
    tokens = normalized_tokens(query)
    return tuple(sorted(token for token in tokens if token in _MOVEMENT_HINTS))


def _compose(*parts: str) -> str:
    seen: set[str] = set()
    tokens: list[str] = []
    for part in parts:
        for token in re.findall(r"[A-Za-z0-9]+", part or ""):
            folded = token.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            tokens.append(token)
    return " ".join(tokens)


def _movement_language_query(query: str, *, movement: tuple[str, ...]) -> str:
    tokens = normalized_tokens(query)
    parts: list[str] = list(movement[:4])
    if tokens & _ROUTE_QUERY_HINTS or movement:
        parts.extend(["journey", "travel", "by", "land", "march", "route"])
    return _compose(*parts)


def decompose_movement_query(query: str) -> tuple[RetrievalIntent, ...]:
    """Return 3–5 complementary deterministic retrieval intents for a route query."""
    text = (query or "").strip()
    if not text:
        return ()
    roles = analyze_query(text)
    subject = primary_route_subject(text, roles)
    subject_prefix = subject or ""
    year = _query_year(text)
    endpoints = _endpoint_terms(text, roles)
    regions = _region_terms(text, roles)
    features = _feature_terms(text)
    movement = _movement_terms(text)
    episode_terms = _episode_terms(text, roles)
    intents: list[RetrievalIntent] = []

    if subject:
        intents.append(RetrievalIntent("SUBJECT", subject))

    episode_query = _compose(subject_prefix, year, *episode_terms, *endpoints[:2], *regions[:1])
    if episode_query and episode_query.casefold() != (subject or "").casefold():
        intents.append(RetrievalIntent("EPISODE", episode_query))

    movement_language = _movement_language_query(text, movement=movement)
    if movement_language:
        intents.append(RetrievalIntent("MOVEMENT", movement_language))

    if subject:
        subject_movement = _compose(subject, *movement[:3], "march", "route", "travel")
        if subject_movement and subject_movement.casefold() != movement_language.casefold():
            intents.append(RetrievalIntent("MOVEMENT", subject_movement))

    if endpoints or regions:
        intents.append(
            RetrievalIntent(
                "ENDPOINT",
                _compose(subject_prefix, *endpoints[:3], *regions[:2], "travel", "march", "journey"),
            )
        )

    if regions and not endpoints:
        intents.append(RetrievalIntent("REGION", _compose(subject_prefix, *regions, "march", "travel", "journey")))

    if features:
        intents.append(RetrievalIntent("FEATURE", _compose(subject_prefix, *features, *movement[:2], "cross", "march")))

    deduped: list[RetrievalIntent] = []
    seen: set[tuple[str, str]] = set()
    for intent in intents:
        key = (intent.kind, intent.query.casefold())
        if not intent.query.strip() or key in seen:
            continue
        seen.add(key)
        deduped.append(intent)
    if not deduped:
        fallback = _movement_language_query(text, movement=movement) or text
        deduped = [RetrievalIntent("MOVEMENT", fallback)]
    return tuple(deduped[:5])
