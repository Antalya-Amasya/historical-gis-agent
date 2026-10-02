"""Canonical query-relative route relation admission for observation components."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.routes.soft_phase_membership import SoftPhaseMembershipIndex
from dataclasses import dataclass
from enum import Enum

from backend.app.models import EventActorStatus, EventPlaceRole, Evidence, HistoricalEvent, HistoricalEventType, TemporalGroundingStatus, TemporalPrecision
from backend.app.routes.episode_relevance import (
    _endpoint_place_tokens,
    _edge_on_directed_path,
    _movement_pairs_from_text,
    _place_token_set,
    _campaign_phrase_modifier_terms,
)
from backend.app.routes.query_scope_parser import build_query_route_scope
from backend.app.routes.evidence_relevance import (
    normalize_subject_name,
    relation_supporting_statements,
)
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
)

_TYPED_MOVEMENT_AUTHORITIES = frozenset({
    ObservationOrderingAuthority.BEFORE_SUBORDINATE,
    ObservationOrderingAuthority.AFTER_SUBORDINATE,
    ObservationOrderingAuthority.BEFORE_POSTPOSED,
    ObservationOrderingAuthority.AFTER_POSTPOSED,
    ObservationOrderingAuthority.FIRST_THEN,
})
_EXPLICIT_OTHER_EPISODE_STATEMENT = re.compile(
    r"\b(?:exile\b|with\s+antiochus\b)",
    re.IGNORECASE,
)
_EVENT_SUBJECT_TOKENS = frozenset({
    "battle", "battles", "war", "wars", "campaign", "campaigns",
    "siege", "sieges", "expedition", "expeditions",
})
_LEADING_COMMANDS = frozenset({"explain", "show", "trace", "display", "reconstruct", "follow"})
_TRAILING_ROUTE_NOUNS = frozenset({
    "route", "routes", "movement", "movements", "march", "marches",
    "journey", "journeys", "advance", "advances", "return", "returns",
    "travel", "travels", "voyage", "voyages",
})
# Collective, office, and event words are not unseen person identities.
_NON_PERSON_SUBJECT_TOKENS = _LEADING_COMMANDS | _TRAILING_ROUTE_NOUNS | _EVENT_SUBJECT_TOKENS | frozenset({
    "army", "armies", "consul", "consuls", "commander", "commanders",
    "troops", "troop", "forces", "force", "soldiers", "soldier", "fleet", "fleets",
    "compare",
})
_LITERAL_NAME = r"[A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){0,3}"
_LITERAL_MULTI_NAME = r"[A-Z][A-Za-z'\u2019-]+(?:\s+[A-Z][A-Za-z'\u2019-]+){1,3}"
_LITERAL_NAME_LIST = rf"{_LITERAL_MULTI_NAME}(?:\s+(?:and|or)\s+{_LITERAL_NAME})*"
_LITERAL_MOVEMENT_OBJECT = (
    r"(?:historical\s+)?(?:routes?|movements?|marches?|journeys?|advances?|returns?|travels?|voyages?)"
)
_SOURCE_ROLE_PREFIX = re.compile(
    r"\b(?:(?:described|reported|recorded|written|documented|narrated)\s+by|according\s+to)\s*$",
    re.IGNORECASE,
)
# Subject position only. Multi-token literals keep single-token places off this path;
# known one-token people still resolve through the alias fallback.
_LITERAL_SUBJECT_PATTERNS = (
    re.compile(
        rf"(?i:\b(?:show|display|trace|reconstruct|follow)\s+)(?P<names>{_LITERAL_NAME_LIST})\s+"
        rf"(?i:{_LITERAL_MOVEMENT_OBJECT}\b)"
    ),
    re.compile(
        rf"(?i:\b(?:movements?|marches?|journeys?|advances?|returns?|travels?|voyages?)\s+of\s+)"
        rf"(?P<names>{_LITERAL_NAME_LIST})\b"
    ),
    re.compile(rf"(?i:\broutes?\s+for\s+)(?P<names>{_LITERAL_NAME_LIST})\b"),
    re.compile(rf"(?:^|\A\s*|(?<=[.!?]\s))(?P<names>{_LITERAL_MULTI_NAME})\s+(?i:routes?\s*:)"),
    re.compile(
        rf"(?i:\bcompare\s+)(?P<left>{_LITERAL_MULTI_NAME})\s+(?i:routes?\s+with\s+)(?P<right>{_LITERAL_NAME})\b"
    ),
)


class AuthorityState(str, Enum):
    MATCH = "MATCH"
    WRONG = "WRONG"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class QueryRouteScope:
    subject: str | None
    origin: str | None
    destination: str | None
    episode: str | None
    has_episode_constraint: bool = False
    endpoint_strict: bool = True
    has_endpoint_constraint: bool = False
    temporal_start: int | None = None
    temporal_end: int | None = None
    temporal_precision: TemporalPrecision | None = None
    has_temporal_constraint: bool = False
    subject_ambiguous: bool = False


@dataclass(frozen=True)
class QueryRouteRelationAdmission:
    subject_match: AuthorityState
    episode_match: AuthorityState
    relation_episode_identity: AuthorityState
    event_episode_compatibility: AuthorityState
    route_phase_match: AuthorityState
    movement_assertion: AuthorityState
    temporal_match: AuthorityState
    admitted: bool
    reason_codes: tuple[str, ...]


def _fallback_person_subjects(query: str, parsed) -> set[str]:
    # Reuse identity aliases, excluding geographic entries in that shared data.
    # Presence supplies query compatibility only, never an evidence actor.
    from backend.app.agent.evidence_support import _subject_alias_registry
    from backend.app.geography.place_registry import records, physical_records

    place_names = {
        str(name).casefold() for item in [*records(), *physical_records()]
        for name in [item["canonical_name"], *item["aliases"]]
    }
    subjects: set[str] = set()
    for canonical, aliases in _subject_alias_registry().items():
        if place_names.intersection(form.casefold() for form in (canonical, *aliases)):
            continue
        for alias in (canonical, *aliases):
            pattern = re.escape(alias)
            if not re.search(r"[\u4e00-\u9fff]", alias):
                pattern = rf"(?<!\w){pattern}(?!\w)"
            for match in re.finditer(pattern, query, re.I):
                if any(span.role.value != "SUBJECT"
                       and span.start <= match.start() and match.end() <= span.end
                       for span in parsed.spans):
                    continue
                # A documentary source mention is not the requested mover.
                if _SOURCE_ROLE_PREFIX.search(query[:match.start()]):
                    continue
                subjects.add(canonical)
    return subjects


def _sanitize_requested_subject(subject: str | None) -> str | None:
    """Drop command verbs and route nouns that the narrow parser folded into a name."""
    if not subject or not subject.strip():
        return None
    parts = subject.split()
    while parts and parts[0].casefold().strip(".,:;\"'") in _LEADING_COMMANDS:
        parts.pop(0)
    while parts and parts[-1].casefold().strip(".,:;\"'") in _TRAILING_ROUTE_NOUNS:
        parts.pop()
    if not parts or parts[0].casefold() in {"the", "a", "an"}:
        return None
    # "Trace Second Punic War route" occupies the trace-subject slot, but an
    # event name is not a requested person.
    if any(part.casefold().strip(".,:;\"'") in _EVENT_SUBJECT_TOKENS for part in parts):
        return None
    return " ".join(parts)


def _known_place_names() -> set[str]:
    from backend.app.geography.place_registry import physical_records, records
    from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES

    names = {
        str(name).casefold()
        for item in [*records(), *physical_records()]
        for name in [item["canonical_name"], *item["aliases"]]
    }
    for alias in HISTORICAL_PLACE_ALIASES:
        names.add(alias.canonical_name.casefold())
        names.update(item.casefold() for item in alias.aliases)
    return names


def _inside_non_subject_span(parsed, start: int, end: int) -> bool:
    return any(
        span.role.value != "SUBJECT" and span.start <= start and end <= span.end
        for span in parsed.spans
    )


def _literal_subject_label(name: str) -> str:
    identity = _normalized_person_identity(name)
    from backend.app.agent.evidence_support import _subject_alias_registry

    if identity in _subject_alias_registry():
        return identity
    return " ".join(name.split())


def _accept_literal_person(name: str, places: set[str]) -> str | None:
    cleaned = " ".join(name.split())
    tokens = [token.casefold() for token in cleaned.split()]
    if not tokens or tokens[0] in {"the", "a", "an"}:
        return None
    if any(token in _NON_PERSON_SUBJECT_TOKENS for token in tokens):
        return None
    if cleaned.casefold() in places or any(token in places for token in tokens):
        return None
    return _literal_subject_label(cleaned)


def _literal_route_subjects(query: str, parsed) -> dict[str, str]:
    """Literal requested-subject phrases. Compatibility only; never an evidence actor."""
    places = _known_place_names()
    found: dict[str, str] = {}
    for pattern in _LITERAL_SUBJECT_PATTERNS:
        for match in pattern.finditer(query):
            groups = []
            if "names" in pattern.groupindex:
                groups.append((match.start("names"), match.group("names")))
            else:
                groups.append((match.start("left"), match.group("left")))
                groups.append((match.start("right"), match.group("right")))
            for start, raw in groups:
                if not raw:
                    continue
                for piece in re.finditer(_LITERAL_NAME, raw):
                    abs_start = start + piece.start()
                    abs_end = start + piece.end()
                    if _SOURCE_ROLE_PREFIX.search(query[:abs_start]):
                        continue
                    if _inside_non_subject_span(parsed, abs_start, abs_end):
                        continue
                    label = _accept_literal_person(piece.group(0), places)
                    if label:
                        found.setdefault(_normalized_person_identity(label), label)
    return found


def parse_query_route_scope(contexts: tuple[str, ...] | None) -> QueryRouteScope:
    parsed = build_query_route_scope(contexts)
    subject = _sanitize_requested_subject(parsed.subject)
    ambiguous = False
    if subject is None:
        from backend.app.agent.loop import infer_requested_output

        query = next((item for item in (contexts or ()) if (item or "").strip()), "")
        if infer_requested_output(query) == "historical_route":
            candidates = {
                _normalized_person_identity(label): label
                for label in _fallback_person_subjects(query, parsed)
            }
            for key, label in _literal_route_subjects(query, parsed).items():
                candidates.setdefault(key, label)
            ambiguous = len(candidates) > 1
            if len(candidates) == 1:
                subject = next(iter(candidates.values()))
    return QueryRouteScope(
        subject=subject,
        origin=parsed.origin,
        destination=parsed.destination,
        episode=parsed.episode,
        has_episode_constraint=parsed.has_episode_constraint,
        endpoint_strict=parsed.endpoint_strict,
        has_endpoint_constraint=parsed.has_endpoint_constraint,
        temporal_start=parsed.temporal_start,
        temporal_end=parsed.temporal_end,
        temporal_precision=parsed.temporal_precision,
        has_temporal_constraint=parsed.has_temporal_constraint,
        subject_ambiguous=ambiguous,
    )


def classify_observation_relation_admission(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...],
    *,
    soft_phase_membership_index: SoftPhaseMembershipIndex | None = None,
) -> QueryRouteRelationAdmission:
    scope = parse_query_route_scope(query_contexts)
    adapter = _observation_relation_adapter(relation, observations_by_id)
    statement = _observation_relation_statement(relation, events_by_id)
    subject_match = _classify_subject_match(relation, observations_by_id, scope)
    relation_episode_identity = _classify_relation_episode_identity(
        relation, events_by_id, evidence_by_id,
    )
    episode_match = _classify_episode_match(
        relation,
        adapter,
        events_by_id,
        evidence_by_id,
        query_contexts,
        statement,
        scope,
        relation_episode_identity=relation_episode_identity,
    )
    route_phase_match = _classify_route_phase_match(
        adapter.earlier,
        adapter.later,
        scope,
        statement,
        relation=relation,
        observations_by_id=observations_by_id,
        events_by_id=events_by_id,
        soft_phase_membership_index=soft_phase_membership_index,
    )
    movement_assertion = _classify_movement_assertion(
        relation, observations_by_id, events_by_id, evidence_by_id,
    )
    event_episode_compatibility = _classify_event_episode_compatibility(
        adapter, events_by_id, evidence_by_id,
    )
    temporal_match = _classify_temporal_match(relation, events_by_id, scope)
    if relation_episode_identity is AuthorityState.WRONG:
        episode_match = AuthorityState.WRONG
    reason_codes = _reason_codes_for_admission(
        subject_match,
        episode_match,
        relation_episode_identity,
        event_episode_compatibility,
        route_phase_match,
        movement_assertion,
        temporal_match,
        scope,
        query_active=bool(query_contexts),
    )
    admitted = _compose_admitted(
        subject_match,
        episode_match,
        relation_episode_identity,
        event_episode_compatibility,
        route_phase_match,
        movement_assertion,
        temporal_match,
        scope,
        query_active=bool(query_contexts),
    )
    return QueryRouteRelationAdmission(
        subject_match=subject_match,
        episode_match=episode_match,
        relation_episode_identity=relation_episode_identity,
        event_episode_compatibility=event_episode_compatibility,
        route_phase_match=route_phase_match,
        movement_assertion=movement_assertion,
        temporal_match=temporal_match,
        admitted=admitted,
        reason_codes=reason_codes,
    )


def primary_rejection_reason(admission: QueryRouteRelationAdmission) -> str:
    priority = (
        "QUERY_SUBJECT_REJECTED",
        "RELATION_EPISODE_CONFLICT",
        "QUERY_EPISODE_REJECTED",
        "INTER_EVENT_EPISODE_INCOMPATIBLE",
        "INTER_EVENT_EPISODE_UNKNOWN",
        "QUERY_ROUTE_PHASE_REJECTED",
        "QUERY_ROUTE_PHASE_UNKNOWN",
        "QUERY_TEMPORAL_REJECTED",
        "QUERY_TEMPORAL_UNKNOWN",
        "MOVEMENT_ASSERTION_REJECTED",
    )
    for code in priority:
        if code in admission.reason_codes:
            return code
    if admission.reason_codes:
        return admission.reason_codes[0]
    return "QUERY_ROUTE_ADMISSION_REJECTED"


def _observation_relation_adapter(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
):
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    earlier_label = earlier.label if earlier is not None else relation.earlier_observation_id
    later_label = later.label if later is not None else relation.later_observation_id
    return type("RelationAdapter", (), {
        "rule": type("Rule", (), {"value": relation.ordering_rule.value})(),
        "event_ids": relation.event_ids,
        "earlier": earlier_label,
        "later": later_label,
        "evidence_refs": relation.evidence_refs,
    })()


def _observation_relation_statement(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
) -> str:
    for event_id in relation.event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue
        for item in relation_supporting_statements(event):
            if item.strip():
                return item.strip()
        if event.summary:
            return event.summary.strip()
    return ""


def _normalized_person_identity(value: str) -> str:
    from backend.app.agent.evidence_support import _subject_alias_registry

    normalized = " ".join(
        token for part in value.split() if (token := normalize_subject_name(part))
    )
    for canonical, aliases in _subject_alias_registry().items():
        if normalized in {normalize_subject_name(alias) for alias in (canonical, *aliases)}:
            return canonical
    return normalized


def _classify_subject_match(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    scope: QueryRouteScope,
) -> AuthorityState:
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    if earlier is None or later is None:
        return AuthorityState.UNKNOWN
    if earlier.actor_status is not EventActorStatus.EXPLICIT or later.actor_status is not EventActorStatus.EXPLICIT:
        return AuthorityState.UNKNOWN
    if not earlier.actor_text or not later.actor_text:
        return AuthorityState.UNKNOWN
    if earlier.actor_text.casefold() != later.actor_text.casefold():
        return AuthorityState.UNKNOWN
    if scope.subject_ambiguous:
        return AuthorityState.UNKNOWN
    if scope.subject is None:
        return AuthorityState.MATCH
    actor_identity = _normalized_person_identity(earlier.actor_text)
    subject_identity = _normalized_person_identity(scope.subject)
    if actor_identity == subject_identity:
        return AuthorityState.MATCH
    return AuthorityState.WRONG


def _statement_supports_scope_subject(statement: str, scope: QueryRouteScope) -> bool:
    if scope.subject_ambiguous:
        return False
    if scope.subject is None:
        return True
    subject = scope.subject.casefold().strip()
    if not subject:
        return True
    statement_cf = statement.casefold()
    return subject in statement_cf


def _statement_supports_scope_episode(statement: str, scope: QueryRouteScope) -> bool:
    if not scope.has_episode_constraint or not scope.episode:
        return True
    query_terms = _scope_episode_query_terms(scope)
    evidence_terms = _campaign_phrase_modifier_terms(statement)
    if not query_terms:
        return True
    if not evidence_terms:
        return False
    return bool(query_terms & evidence_terms)


def _scope_proven_chain_member(
    source_place: str,
    destination_place: str,
    statement: str,
    scope: QueryRouteScope,
) -> bool:
    if not scope.origin or not scope.destination:
        return False
    query_origin = _place_token_set(scope.origin)
    query_destination = _place_token_set(scope.destination)
    current = (_place_token_set(source_place), _place_token_set(destination_place))
    if current[0] & query_origin and current[1] & query_destination:
        return True
    edges: list[tuple[frozenset[str], frozenset[str]]] = []
    for occurrence in re.split(r"(?<=[.!?])\s+", statement.strip()):
        local = occurrence.strip()
        if not local:
            continue
        if re.search(
            r"\b(?:did\s+not|never|not)\s+(?:march|travel|move|go|proceed)",
            local,
            re.IGNORECASE,
        ):
            continue
        if not _statement_supports_scope_subject(local, scope):
            continue
        if not _statement_supports_scope_episode(local, scope):
            continue
        for edge in _movement_pairs_from_text(local):
            if edge not in edges:
                edges.append(edge)
    if current not in edges and _statement_supports_scope_subject(statement.strip(), scope):
        edges.append(current)
    return _edge_on_directed_path(current, query_origin, query_destination, edges)


def _statement_signals_explicit_other_episode(
    statement: str,
    query_contexts: tuple[str, ...],
) -> bool:
    if not statement.strip() or not query_contexts:
        return False
    if _EXPLICIT_OTHER_EPISODE_STATEMENT.search(statement):
        return True
    return False


def _event_movement_statement(
    event: HistoricalEvent,
    evidence_by_id: dict[str, Evidence],
) -> str:
    if event.source_statements:
        for item in event.source_statements:
            if item.strip():
                return item.strip()
    for ref in event.evidence_refs:
        item = evidence_by_id.get(ref)
        if item is not None and item.text:
            return item.text.strip()
    return (event.summary or "").strip()


def _explicit_campaign_phrase_term_sets(text: str) -> list[frozenset[str]]:
    if not text.strip():
        return []
    from backend.app.routes.episode_relevance import _EVIDENCE_EPISODE_PATTERNS

    term_sets: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for pattern in _EVIDENCE_EPISODE_PATTERNS:
        for match in pattern.finditer(text):
            terms = frozenset(_campaign_phrase_modifier_terms(match.group(0)))
            if terms and terms not in seen:
                term_sets.append(terms)
                seen.add(terms)
    return term_sets


def _phrase_term_sets_conflict(term_sets: list[frozenset[str]]) -> bool:
    if len(term_sets) < 2:
        return False
    for left in term_sets:
        for right in term_sets:
            if left is not right and left.isdisjoint(right):
                return True
    return False


def _statement_has_conflicting_episode_phrases(statement: str) -> bool:
    return _phrase_term_sets_conflict(_explicit_campaign_phrase_term_sets(statement))


def _relation_local_statements(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> list[str]:
    statements: list[str] = []
    seen: set[str] = set()
    for event_id in dict.fromkeys(relation.event_ids):
        event = events_by_id.get(event_id)
        if event is None:
            continue
        local_added = False
        for item in relation_supporting_statements(event):
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                statements.append(cleaned)
                seen.add(cleaned)
                local_added = True
        if not local_added:
            for item in event.source_statements:
                cleaned = item.strip()
                if cleaned and cleaned not in seen:
                    statements.append(cleaned)
                    seen.add(cleaned)
                    local_added = True
        if not local_added and (event.summary or "").strip():
            cleaned = event.summary.strip()
            if cleaned not in seen:
                statements.append(cleaned)
                seen.add(cleaned)
    return statements


def _classify_relation_episode_identity(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> AuthorityState:
    events = [events_by_id[event_id] for event_id in relation.event_ids if event_id in events_by_id]
    if not events:
        return AuthorityState.UNKNOWN
    any_positive = False
    for event in events:
        statements = _event_local_statements(event)
        phrase_sets: list[frozenset[str]] = []
        for statement in statements:
            if _statement_has_conflicting_episode_phrases(statement):
                return AuthorityState.WRONG
            phrase_sets.extend(_explicit_campaign_phrase_term_sets(statement))
        if _phrase_term_sets_conflict(phrase_sets):
            return AuthorityState.WRONG
        if phrase_sets:
            any_positive = True
    if any_positive:
        return AuthorityState.MATCH
    return AuthorityState.UNKNOWN


def _relation_local_campaign_terms(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> set[str]:
    terms: set[str] = set()
    for statement in _relation_local_statements(relation, events_by_id, evidence_by_id):
        terms |= _campaign_phrase_modifier_terms(statement)
    return terms


def _scope_episode_query_terms(scope: QueryRouteScope) -> set[str]:
    if not scope.episode:
        return set()
    return _campaign_phrase_modifier_terms(f"during {scope.episode}")


def _classify_episode_match(
    relation: ObservationOrderingRelation,
    adapter,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...],
    statement: str,
    scope: QueryRouteScope,
    *,
    relation_episode_identity: AuthorityState,
) -> AuthorityState:
    if relation_episode_identity is AuthorityState.WRONG:
        return AuthorityState.WRONG
    if not scope.has_episode_constraint:
        return AuthorityState.UNKNOWN
    if not scope.episode:
        return AuthorityState.UNKNOWN
    local_statements = _relation_local_statements(relation, events_by_id, evidence_by_id)
    if any(_statement_has_conflicting_episode_phrases(item) for item in local_statements):
        return AuthorityState.WRONG
    if _statement_signals_explicit_other_episode(statement, query_contexts):
        return AuthorityState.WRONG
    query_terms = _scope_episode_query_terms(scope)
    evidence_terms = _relation_local_campaign_terms(relation, events_by_id, evidence_by_id)
    if query_terms and evidence_terms:
        if query_terms & evidence_terms:
            return AuthorityState.MATCH
        return AuthorityState.WRONG
    if query_terms and not evidence_terms:
        return AuthorityState.UNKNOWN
    return AuthorityState.UNKNOWN


def _event_local_statements(event: HistoricalEvent) -> list[str]:
    statements: list[str] = []
    seen: set[str] = set()
    for item in relation_supporting_statements(event):
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            statements.append(cleaned)
            seen.add(cleaned)
    for item in event.source_statements:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            statements.append(cleaned)
            seen.add(cleaned)
    if (event.summary or "").strip() and event.summary.strip() not in seen:
        statements.append(event.summary.strip())
    return statements


def _classify_event_episode_compatibility(
    adapter,
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> AuthorityState:
    if len(adapter.event_ids) < 2:
        return AuthorityState.MATCH
    from backend.app.routes.evidence_relevance import EvidenceRelevance, event_relevance

    events = [events_by_id[event_id] for event_id in adapter.event_ids if event_id in events_by_id]
    if len(events) < 2:
        return AuthorityState.UNKNOWN
    for event in events:
        if event_relevance(event, evidence_by_id, ()) is EvidenceRelevance.OTHER_CAMPAIGN:
            return AuthorityState.WRONG
    event_terms: list[set[str]] = []
    for event in events:
        local_statements = _event_local_statements(event)
        if any(_statement_has_conflicting_episode_phrases(item) for item in local_statements):
            return AuthorityState.WRONG
        terms: set[str] = set()
        for item in local_statements:
            terms |= _campaign_phrase_modifier_terms(item)
        event_terms.append(terms)
    non_empty = [terms for terms in event_terms if terms]
    if not non_empty:
        return AuthorityState.MATCH
    if len(non_empty) == 1:
        return AuthorityState.UNKNOWN
    for left in non_empty:
        for right in non_empty:
            if left is not right and left.isdisjoint(right):
                return AuthorityState.WRONG
    return AuthorityState.MATCH


def _directed_place_surface_tokens(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(part.casefold() for part in value.split() if part.strip())


def _directed_place_identity(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if left.casefold().strip() == right.casefold().strip():
        return True
    left_surface = _directed_place_surface_tokens(left)
    right_surface = _directed_place_surface_tokens(right)
    if left_surface and left_surface == right_surface:
        return True
    if left_surface and right_surface:
        return False
    left_alias = _endpoint_place_tokens(left)
    right_alias = _endpoint_place_tokens(right)
    return bool(left_alias) and left_alias == right_alias


def _place_identity_matches(left: str | None, right: str | None) -> bool:
    return _directed_place_identity(left, right)


def _observation_place_identities(
    observation: RouteObservation | None,
    events_by_id: dict[str, HistoricalEvent] | None,
    fallback_label: str,
) -> tuple[str, ...]:
    names: list[str] = []
    if fallback_label:
        names.append(fallback_label)
    if observation is None:
        return tuple(dict.fromkeys(names))
    if observation.label:
        names.append(observation.label)
    event = None if events_by_id is None else events_by_id.get(observation.event_id)
    if event is not None and observation.place_role is not None:
        for binding in event.place_bindings:
            if binding.role is not observation.place_role:
                continue
            if binding.mention.raw_text:
                names.append(binding.mention.raw_text)
            if binding.place is not None and binding.place.canonical_name:
                names.append(binding.place.canonical_name)
    return tuple(dict.fromkeys(item for item in names if item))


def _identities_match_scope(identities: tuple[str, ...], scope_value: str) -> bool:
    return any(_place_identity_matches(item, scope_value) for item in identities)


def _same_event_origin_destination_leg(
    relation: ObservationOrderingRelation | None,
    observations_by_id: dict[str, RouteObservation] | None,
    events_by_id: dict[str, HistoricalEvent] | None,
) -> bool:
    if relation is None or observations_by_id is None or events_by_id is None:
        return False
    if len(relation.event_ids) != 1:
        return False
    event = events_by_id.get(relation.event_ids[0])
    if event is None or event.event_type is not HistoricalEventType.MOVEMENT:
        return False
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    if earlier is None or later is None:
        return False
    return (
        earlier.place_role is EventPlaceRole.ORIGIN
        and later.place_role is EventPlaceRole.DESTINATION
        and relation.authority == "SAME_MOVEMENT_EVENT"
    )


def _trusted_soft_phase_membership_proof(
    index: object | None,
    relation: ObservationOrderingRelation,
    scope: QueryRouteScope,
):
    if index is None:
        return None
    from backend.app.routes.soft_phase_membership import SoftPhaseMembershipIndex

    if type(index) is not SoftPhaseMembershipIndex:
        return None
    return index.proof_for(relation, scope)


def _classify_route_phase_match(
    earlier_label: str,
    later_label: str,
    scope: QueryRouteScope,
    statement: str,
    *,
    relation: ObservationOrderingRelation | None = None,
    observations_by_id: dict[str, RouteObservation] | None = None,
    events_by_id: dict[str, HistoricalEvent] | None = None,
    soft_phase_membership_index: SoftPhaseMembershipIndex | None = None,
) -> AuthorityState:
    if not scope.has_endpoint_constraint:
        return AuthorityState.UNKNOWN
    origin = scope.origin or ""
    destination = scope.destination or ""
    earlier_obs = None
    later_obs = None
    if relation is not None and observations_by_id is not None:
        earlier_obs = observations_by_id.get(relation.earlier_observation_id)
        later_obs = observations_by_id.get(relation.later_observation_id)
    earlier_identities = _observation_place_identities(earlier_obs, events_by_id, earlier_label)
    later_identities = _observation_place_identities(later_obs, events_by_id, later_label)
    earlier_matches_origin = _identities_match_scope(earlier_identities, origin)
    later_matches_destination = _identities_match_scope(later_identities, destination)
    earlier_matches_destination = _identities_match_scope(earlier_identities, destination)
    later_matches_origin = _identities_match_scope(later_identities, origin)
    if scope.endpoint_strict:
        if earlier_matches_origin and later_matches_destination:
            return AuthorityState.MATCH
        if earlier_matches_destination and later_matches_origin:
            return AuthorityState.WRONG
        if earlier_matches_destination and not later_matches_destination and not later_matches_origin:
            if _scope_proven_chain_member(
                earlier_label, later_label, statement, scope,
            ):
                return AuthorityState.MATCH
            if _same_event_origin_destination_leg(relation, observations_by_id, events_by_id):
                return AuthorityState.MATCH
            return AuthorityState.WRONG
        if later_matches_origin and not earlier_matches_origin:
            if _same_event_origin_destination_leg(relation, observations_by_id, events_by_id):
                return AuthorityState.MATCH
            return AuthorityState.WRONG
        if earlier_matches_origin and not later_matches_destination:
            if _scope_proven_chain_member(
                earlier_label, later_label, statement, scope,
            ):
                return AuthorityState.MATCH
            return AuthorityState.UNKNOWN
        if later_matches_destination and not earlier_matches_origin:
            if _scope_proven_chain_member(
                earlier_label, later_label, statement, scope,
            ):
                return AuthorityState.MATCH
            return AuthorityState.UNKNOWN
        if not (
            earlier_matches_origin
            or earlier_matches_destination
            or later_matches_origin
            or later_matches_destination
        ):
            return AuthorityState.UNKNOWN
        return AuthorityState.UNKNOWN
    if earlier_matches_destination and later_matches_origin:
        return AuthorityState.WRONG
    if (
        earlier_matches_origin
        or later_matches_destination
        or _scope_proven_chain_member(earlier_label, later_label, statement, scope)
    ):
        return AuthorityState.MATCH
    if relation is not None and _trusted_soft_phase_membership_proof(
        soft_phase_membership_index,
        relation,
        scope,
    ) is not None:
        return AuthorityState.MATCH
    return AuthorityState.UNKNOWN


def relation_non_phase_eligible(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
    query_contexts: tuple[str, ...],
    scope: QueryRouteScope,
) -> bool:
    adapter = _observation_relation_adapter(relation, observations_by_id)
    statement = _observation_relation_statement(relation, events_by_id)
    subject_match = _classify_subject_match(relation, observations_by_id, scope)
    relation_episode_identity = _classify_relation_episode_identity(
        relation, events_by_id, evidence_by_id,
    )
    episode_match = _classify_episode_match(
        relation,
        adapter,
        events_by_id,
        evidence_by_id,
        query_contexts,
        statement,
        scope,
        relation_episode_identity=relation_episode_identity,
    )
    movement_assertion = _classify_movement_assertion(
        relation, observations_by_id, events_by_id, evidence_by_id,
    )
    event_episode_compatibility = _classify_event_episode_compatibility(
        adapter, events_by_id, evidence_by_id,
    )
    temporal_match = _classify_temporal_match(relation, events_by_id, scope)
    if relation_episode_identity is AuthorityState.WRONG:
        return False
    if not query_contexts:
        return movement_assertion is AuthorityState.MATCH and event_episode_compatibility is AuthorityState.MATCH
    required: list[AuthorityState] = [event_episode_compatibility, movement_assertion]
    if scope.subject is not None or scope.subject_ambiguous:
        required.append(subject_match)
    if scope.has_episode_constraint:
        required.append(episode_match)
    if scope.has_temporal_constraint:
        required.append(temporal_match)
    if any(state is AuthorityState.WRONG for state in required):
        return False
    return all(state is AuthorityState.MATCH for state in required)


def _movement_statement(event: HistoricalEvent, evidence_by_id: dict[str, Evidence]) -> str:
    for ref in event.evidence_refs:
        item = evidence_by_id.get(ref)
        if item is not None and item.text:
            return item.text
    return next((item for item in event.source_statements if item.strip()), event.summary or "")


def _relation_has_typed_movement_provenance(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    event: HistoricalEvent,
) -> bool:
    if event.route_orderings:
        earlier = observations_by_id.get(relation.earlier_observation_id)
        later = observations_by_id.get(relation.later_observation_id)
        if earlier is None or later is None:
            return False
        for ordering in event.route_orderings:
            if (
                _place_identity_matches(ordering.earlier.canonical, earlier.label)
                and _place_identity_matches(ordering.later.canonical, later.label)
            ):
                return True
    origin = next(
        (
            binding
            for binding in event.place_bindings
            if binding.role is EventPlaceRole.ORIGIN
            and binding.place is not None
        ),
        None,
    )
    destination = next(
        (
            binding
            for binding in event.place_bindings
            if binding.role is EventPlaceRole.DESTINATION
            and binding.place is not None
        ),
        None,
    )
    if origin is None or destination is None:
        return False
    earlier = observations_by_id.get(relation.earlier_observation_id)
    later = observations_by_id.get(relation.later_observation_id)
    if earlier is None or later is None:
        return False
    return (
        _place_identity_matches(origin.place.canonical_name, earlier.label)
        and _place_identity_matches(destination.place.canonical_name, later.label)
    )


def _classify_movement_assertion(
    relation: ObservationOrderingRelation,
    observations_by_id: dict[str, RouteObservation],
    events_by_id: dict[str, HistoricalEvent],
    evidence_by_id: dict[str, Evidence],
) -> AuthorityState:
    if relation.ordering_rule not in _TYPED_MOVEMENT_AUTHORITIES:
        if relation.ordering_rule in {
            ObservationOrderingAuthority.TEMPORAL_ORDER,
            ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER,
        }:
            if len(relation.event_ids) < 2:
                return AuthorityState.UNKNOWN
            from backend.app.routes.event_route_orchestration import (
                _positive_asserted_structural_order_pair,
            )
            earlier_event = events_by_id.get(relation.event_ids[0])
            later_event = events_by_id.get(relation.event_ids[1])
            if earlier_event is None or later_event is None:
                return AuthorityState.UNKNOWN
            if _positive_asserted_structural_order_pair(earlier_event, later_event):
                return AuthorityState.MATCH
            return AuthorityState.WRONG
        return AuthorityState.UNKNOWN
    for event_id in relation.event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            return AuthorityState.UNKNOWN
        if event.event_type is not HistoricalEventType.MOVEMENT:
            return AuthorityState.WRONG
        statement = _movement_statement(event, evidence_by_id)
        if not EvidenceGroundedHistoricalEventExtractor._has_positive_movement_assertion(statement):
            return AuthorityState.WRONG
        if not _relation_has_typed_movement_provenance(relation, observations_by_id, event):
            return AuthorityState.WRONG
    return AuthorityState.MATCH


def _event_temporal_match(event: HistoricalEvent | None, scope: QueryRouteScope) -> AuthorityState:
    if event is None:
        return AuthorityState.UNKNOWN
    if event.temporal_grounding.status is TemporalGroundingStatus.CONFLICT:
        return AuthorityState.WRONG
    from backend.app.routes.event_route_orchestration import _temporal_interval

    interval = _temporal_interval(event)
    if interval is None:
        interval = _event_statement_interval(event)
    if interval == "CONFLICT":
        return AuthorityState.WRONG
    if interval is None:
        return AuthorityState.UNKNOWN
    query_start = scope.temporal_start
    query_end = scope.temporal_end if scope.temporal_end is not None else scope.temporal_start
    if query_start is None or query_end is None:
        return AuthorityState.UNKNOWN
    if interval[1] < query_start or query_end < interval[0]:
        return AuthorityState.WRONG
    return AuthorityState.MATCH


def _event_statement_interval(event: HistoricalEvent) -> tuple[int, int] | None | str:
    from backend.app.routes.temporal import EvidenceTemporalResolver

    text = " ".join(item for item in event.source_statements if item.strip()) or (event.summary or "")
    readings, codes = EvidenceTemporalResolver().resolve(text, event.id)
    if "TEMPORAL_CONFLICT" in codes:
        return "CONFLICT"
    usable = [
        item for item in readings
        if item.normalized_start is not None and item.status is TemporalGroundingStatus.EVIDENCE_GROUNDED
    ]
    if not usable:
        return None
    start = int(usable[0].normalized_start)
    end = int(usable[0].normalized_end) if usable[0].normalized_end is not None else start
    return (start, end) if start <= end else (end, start)


def _classify_temporal_match(
    relation: ObservationOrderingRelation,
    events_by_id: dict[str, HistoricalEvent],
    scope: QueryRouteScope,
) -> AuthorityState:
    if not scope.has_temporal_constraint:
        return AuthorityState.UNKNOWN
    states = [_event_temporal_match(events_by_id.get(event_id), scope) for event_id in relation.event_ids]
    if not states:
        return AuthorityState.UNKNOWN
    if any(state is AuthorityState.WRONG for state in states):
        return AuthorityState.WRONG
    if any(state is AuthorityState.UNKNOWN for state in states):
        return AuthorityState.UNKNOWN
    return AuthorityState.MATCH


def _compose_admitted(
    subject_match: AuthorityState,
    episode_match: AuthorityState,
    relation_episode_identity: AuthorityState,
    event_episode_compatibility: AuthorityState,
    route_phase_match: AuthorityState,
    movement_assertion: AuthorityState,
    temporal_match: AuthorityState,
    scope: QueryRouteScope,
    *,
    query_active: bool,
) -> bool:
    required: list[AuthorityState] = [event_episode_compatibility]
    if relation_episode_identity is AuthorityState.WRONG:
        return False
    if query_active:
        required.append(movement_assertion)
        if scope.subject is not None or scope.subject_ambiguous:
            required.append(subject_match)
        if scope.has_episode_constraint:
            required.append(episode_match)
        if scope.has_endpoint_constraint:
            required.append(route_phase_match)
        if scope.has_temporal_constraint:
            required.append(temporal_match)
    if any(state is AuthorityState.WRONG for state in required):
        return False
    return all(state is AuthorityState.MATCH for state in required)


def _reason_codes_for_admission(
    subject_match: AuthorityState,
    episode_match: AuthorityState,
    relation_episode_identity: AuthorityState,
    event_episode_compatibility: AuthorityState,
    route_phase_match: AuthorityState,
    movement_assertion: AuthorityState,
    temporal_match: AuthorityState,
    scope: QueryRouteScope,
    *,
    query_active: bool,
) -> tuple[str, ...]:
    codes: list[str] = []
    if query_active and (scope.subject is not None or scope.subject_ambiguous):
        if subject_match is AuthorityState.WRONG:
            codes.append("QUERY_SUBJECT_REJECTED")
        elif subject_match is AuthorityState.UNKNOWN:
            codes.append("QUERY_SUBJECT_UNKNOWN")
    if relation_episode_identity is AuthorityState.WRONG:
        codes.append("RELATION_EPISODE_CONFLICT")
    if query_active and scope.has_episode_constraint:
        if episode_match is AuthorityState.WRONG:
            codes.append("QUERY_EPISODE_REJECTED")
        elif episode_match is AuthorityState.UNKNOWN:
            codes.append("QUERY_EPISODE_UNKNOWN")
    if event_episode_compatibility is AuthorityState.WRONG:
        codes.append("INTER_EVENT_EPISODE_INCOMPATIBLE")
    elif event_episode_compatibility is AuthorityState.UNKNOWN:
        codes.append("INTER_EVENT_EPISODE_UNKNOWN")
    if query_active and scope.has_endpoint_constraint:
        if route_phase_match is AuthorityState.WRONG:
            codes.append("QUERY_ROUTE_PHASE_REJECTED")
        elif route_phase_match is AuthorityState.UNKNOWN:
            codes.append("QUERY_ROUTE_PHASE_UNKNOWN")
    if query_active and scope.has_temporal_constraint:
        if temporal_match is AuthorityState.WRONG:
            codes.append("QUERY_TEMPORAL_REJECTED")
        elif temporal_match is AuthorityState.UNKNOWN:
            codes.append("QUERY_TEMPORAL_UNKNOWN")
    if query_active and movement_assertion is AuthorityState.WRONG:
        codes.append("MOVEMENT_ASSERTION_REJECTED")
    elif query_active and movement_assertion is AuthorityState.UNKNOWN:
        codes.append("MOVEMENT_ASSERTION_UNKNOWN")
    return tuple(codes)
