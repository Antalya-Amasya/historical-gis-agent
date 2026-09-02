"""Evidence-local movement semantic extraction.

Candidate generation favors recall; downstream gates keep edge admission precise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from backend.app.models import PlaceMentionValidationClass
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.place_mention_validation import validate_broad_place_mention

EndpointRole = Literal["origin", "destination", "traversal"]

_MOVEMENT_CUE = re.compile(
    r"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|"
    r"departed|arrived|entered|crossed|withdrew|retreated|fled|left|leaving|reached|came|"
    r"returned|passed|set\s+out|hastened|sailed|sailing|traversed|traversing|conducted|led|went|descended)\b",
    re.IGNORECASE,
)
_NON_MOVEMENT = re.compile(
    r"\b(?:fought|battle|born|controlled|province|political\s+movement|moved\s+the\s+senate|"
    r"speech\s+about|according\s+to|made\s+equal\s+in\s+command|brought|buried|joined|"
    r"march\s+on\b|advanced\s+to\s+the\s+(?:city|town|camp))\b",
    re.IGNORECASE,
)
_PLACE_SPAN = re.compile(
    r"(?:the\s+)?([A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*(?:\s+(?:the\s+)?[A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*){0,3})"
)
_DISCOURSE_FROM_THERE = re.compile(
    r"^\s*(?:and\s+)?(?:from\s+there|thence)\b",
    re.IGNORECASE,
)
_DISCOURSE_THEN = re.compile(
    r"^\s*(?:and\s+)?(?:then|thereupon|next|after\s+this)\b",
    re.IGNORECASE,
)
_MOVEMENT_AFTER_DISCOURSE = re.compile(
    r"\b(?:marched|advanced|proceeded|passed|went|moved|travelled|traveled|reached|arrived|entered)\b",
    re.IGNORECASE,
)
_MOVEMENT_GOVERNED_FROM = re.compile(
    r"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|"
    r"departed|left|leaving|withdrew|retreated|fled|came|went|crossed|crossing|returned|"
    r"hastened|set\s+out|descended)\b(?:\s+\w+){0,12}?\bfrom\b",
    re.IGNORECASE,
)
_MEDIATED_ORIGIN_PREFIX = re.compile(
    r"^(?:the\s+)?(?:passage|valley|crossing|banks?|mouth|shores?|foot)\s+of\s+(?:the\s+)?",
    re.IGNORECASE,
)
_VALLEY_OF_PREFIX = re.compile(
    r"^(?:the\s+)?([A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*(?:\s+(?:the\s+)?[A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*){0,2})\s+valley\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MovementEndpoint:
    surface: str
    canonical: str | None
    role: EndpointRole
    position: int

    @property
    def place_name(self) -> str:
        return self.canonical or self.surface


@dataclass(frozen=True)
class MovementEdgeCandidate:
    origin: MovementEndpoint | None
    destination: MovementEndpoint | None
    traversal: MovementEndpoint | None
    movement_relation: str
    cross_sentence_link: bool = False


@dataclass(frozen=True)
class SentenceMovementSemantics:
    is_movement: bool
    edges: tuple[MovementEdgeCandidate, ...]
    endpoints: tuple[MovementEndpoint, ...]
    should_abstain: bool = False
    abstain_reason: str | None = None


def _alias_map(aliases: list[tuple[int, HistoricalPlaceAlias, str]]) -> dict[int, tuple[HistoricalPlaceAlias, str]]:
    return {position: (place, alias) for position, place, alias in aliases}


def _validated_span(sentence: str, start: int, *, before: int | None = None) -> MovementEndpoint | None:
    window = sentence[start:before]
    match = _PLACE_SPAN.search(window)
    if not match:
        return None
    surface = match.group(1).strip()
    if not surface or surface.lower() in {"he", "she", "they", "it", "there", "thence"}:
        return None
    absolute_start = start + match.start(1)
    fake = re.compile(re.escape(surface))
    fake_match = fake.search(sentence, absolute_start)
    if fake_match is None:
        return None
    validation = validate_broad_place_mention(surface, sentence, fake_match)
    if validation.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE:
        return None
    return surface, absolute_start


def _endpoint_after(
    sentence: str,
    start: int,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
    *,
    before: int | None = None,
    role: EndpointRole,
) -> MovementEndpoint | None:
    alias_hits = [
        (position, place, alias)
        for position, place, alias in aliases
        if position >= start and (before is None or position < before)
    ]
    if alias_hits:
        position, place, alias = alias_hits[0]
        return MovementEndpoint(
            surface=sentence[position:position + len(alias)],
            canonical=place.canonical_name,
            role=role,
            position=position,
        )
    validated = _validated_span(sentence, start, before=before)
    if validated is None:
        return None
    surface, position = validated
    return MovementEndpoint(surface=surface, canonical=None, role=role, position=position)


def _origin_after_from(
    sentence: str,
    from_end: int,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
    *,
    before: int | None = None,
) -> MovementEndpoint | None:
    """Resolve a movement source after ``from``, including mediated phrases."""
    remainder = sentence[from_end:]
    stripped = remainder.lstrip()
    offset = from_end + (len(remainder) - len(stripped))
    mediated = _MEDIATED_ORIGIN_PREFIX.match(stripped)
    if mediated:
        return _endpoint_after(sentence, offset + mediated.end(), aliases, before=before, role="origin")
    valley = _VALLEY_OF_PREFIX.match(stripped)
    if valley:
        surface = valley.group(1).strip()
        alias_hits = [
            (position, place, alias)
            for position, place, alias in aliases
            if position >= offset and (before is None or position < before)
        ]
        for position, place, alias in alias_hits:
            if alias.casefold() in surface.casefold() or surface.casefold() in alias.casefold():
                return MovementEndpoint(
                    surface=sentence[position:position + len(alias)],
                    canonical=place.canonical_name,
                    role="origin",
                    position=position,
                )
        fake = re.search(re.escape(surface), sentence[offset:before] if before else sentence[offset:], re.IGNORECASE)
        if fake:
            validation = validate_broad_place_mention(surface, sentence, fake)
            if validation.validation_class is not PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE:
                return MovementEndpoint(
                    surface=surface,
                    canonical=None,
                    role="origin",
                    position=offset + valley.start(1),
                )
    return _endpoint_after(sentence, from_end, aliases, before=before, role="origin")


def _has_movement_cue(sentence: str) -> bool:
    return bool(_MOVEMENT_CUE.search(sentence)) and not _NON_MOVEMENT.search(sentence)


def _edge(
    origin: MovementEndpoint | None,
    destination: MovementEndpoint | None,
    *,
    traversal: MovementEndpoint | None = None,
    movement_relation: str,
    cross_sentence_link: bool = False,
) -> MovementEdgeCandidate | None:
    if origin and destination and origin.place_name.casefold() == destination.place_name.casefold():
        return None
    if not origin and not destination and not traversal:
        return None
    if origin or destination:
        return MovementEdgeCandidate(
            origin=origin,
            destination=destination,
            traversal=traversal,
            movement_relation=movement_relation,
            cross_sentence_link=cross_sentence_link,
        )
    if traversal:
        return MovementEdgeCandidate(
            origin=None,
            destination=None,
            traversal=traversal,
            movement_relation="traversal",
        )
    return None


def analyze_sentence(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
    *,
    prior_endpoints: tuple[MovementEndpoint, ...] = (),
) -> SentenceMovementSemantics:
    lower = sentence.lower()
    endpoints: list[MovementEndpoint] = []
    edges: list[MovementEdgeCandidate] = []

    if not _has_movement_cue(sentence):
        return SentenceMovementSemantics(is_movement=False, edges=(), endpoints=())

    # A. marched/advanced ... from X to/into/toward Y
    movement_from = re.search(
        r"\b(?:marched|advanced|proceeded|moved|travelled|traveled|returned|withdrew|retreated|"
        r"hastened|led(?:\s+(?:his|the)\s+army)?|set\s+out|went|descended|fled|march(?:ed|ing)?)\b.{0,180}?\bfrom\s+",
        lower,
    )
    if movement_from:
        from_match = re.search(r"\bfrom\s+", lower[movement_from.start():])
        from_start = movement_from.start() + from_match.start() if from_match else None
        from_end = movement_from.start() + from_match.end() if from_match else None
        to_match = re.search(r"\b(?:to|into|toward|towards)\s+", lower[from_end:] if from_end is not None else "")
        if from_end is not None and to_match:
            to_start = from_end + to_match.start()
            to_end = from_end + to_match.end()
            origin = _origin_after_from(sentence, from_end, aliases, before=to_start)
            destination = _endpoint_after(sentence, to_end, aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="from_to")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # A-cross: crossed/moved ... from X into/to/toward Y
    if not edges:
        cross_from = re.search(
            r"\b(?:crossed|crossing|moved|travelled|traveled|went|came|advanced|proceeded)\b.{0,160}?\bfrom\s+",
            lower,
        )
        if cross_from:
            from_match = re.search(r"\bfrom\s+", lower[cross_from.start():])
            from_end = cross_from.start() + from_match.end() if from_match else None
            to_match = re.search(r"\b(?:into|to|toward|towards)\s+", lower[from_end:] if from_end is not None else "")
            if from_end is not None and to_match:
                to_start = from_end + to_match.start()
                to_end = from_end + to_match.end()
                origin = _origin_after_from(sentence, from_end, aliases, before=to_start)
                destination = _endpoint_after(sentence, to_end, aliases, role="destination")
                candidate = _edge(origin, destination, movement_relation="crossed_from_into")
                if candidate:
                    edges.append(candidate)
                    endpoints.extend(item for item in (origin, destination) if item)

    # B. departed from X for/toward Y
    if not edges:
        depart = re.search(r"\b(?:departed(?:\s+from)?|set\s+out\s+from)\s+", lower)
        dest = re.search(r"\b(?:for|toward|towards|to|into)\s+", lower[depart.end():] if depart else "")
        if depart and dest:
            origin = _origin_after_from(sentence, depart.end(), aliases, before=depart.end() + dest.start())
            destination = _endpoint_after(sentence, depart.end() + dest.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="departed_for")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # C. left X ... reached/arrived/entered Y
    if not edges:
        leave = re.search(r"\b(?:left|leaving|departed(?:\s+from)?)\s+", lower)
        arrive = re.search(r"\b(?:reached|arriv(?:ed|ing)\s+(?:at|in)|came\s+to|entered|passed\s+into)\s+", lower)
        if leave and arrive and leave.start() < arrive.start():
            origin = _endpoint_after(sentence, leave.end(), aliases, before=arrive.start(), role="origin")
            destination = _endpoint_after(sentence, arrive.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="departure_arrival")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # D. crossed X (into Y)?
    if not edges:
        cross = re.search(r"\b(?:crossed|crossing|traversed|traversing)\s+(?:the\s+)?", lower)
        into = re.search(r"\binto\s+", lower[cross.end():] if cross else "")
        arrive = re.search(r"\b(?:came\s+to|arriv(?:ed|ing)\s+(?:at|in)|entered|passed\s+into|reached)\s+", lower)
        if cross:
            traversal = _endpoint_after(
                sentence, cross.end(), aliases,
                before=(cross.end() + into.start()) if into else (arrive.start() if arrive else None),
                role="traversal",
            )
            destination = None
            if into:
                destination = _endpoint_after(sentence, cross.end() + into.end(), aliases, role="destination")
                candidate = _edge(None, destination, traversal=traversal, movement_relation="crossing_into")
            elif arrive and cross.start() < arrive.start():
                destination = _endpoint_after(sentence, arrive.end(), aliases, role="destination")
                candidate = _edge(traversal, destination, movement_relation="crossing_arrival") if traversal else None
            else:
                candidate = _edge(None, None, traversal=traversal, movement_relation="traversal") if traversal else None
            if candidate:
                edges.append(candidate)
                if traversal:
                    endpoints.append(traversal)
                if destination:
                    endpoints.append(destination)

    # E. reached/arrived then led to
    if not edges:
        reached = re.search(r"\b(?:reached|arrived\s+(?:at|in)|came\s+to)\s+", lower)
        lead = re.search(r"\b(?:led|conducted)\b.{0,180}?\b(?:to|into)\s+", lower)
        if reached and lead and reached.start() < lead.start():
            origin = _endpoint_after(sentence, reached.end(), aliases, before=lead.start(), role="origin")
            destination = _endpoint_after(sentence, lead.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="arrival_then_lead")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # F. sailed for X ... thence passed on to Y
    if not edges:
        sailed = re.search(r"\b(?:sail|sailed|sailing)\s+for\s+", lower)
        thence = re.search(r"\bthence\s+passed\s+on\s+to\s+", lower)
        if sailed and thence and sailed.start() < thence.start():
            sailed_destination = _endpoint_after(sentence, sailed.end(), aliases, before=thence.start(), role="origin")
            thence_destination = _endpoint_after(sentence, thence.end(), aliases, role="destination")
            candidate = _edge(sailed_destination, thence_destination, movement_relation="thence_passed_on_to")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (sailed_destination, thence_destination) if item)

    # G. reached/arrived/entered Y (destination only)
    if not edges:
        arrive_only = re.search(
            r"\b(?:reached|arriv(?:ed|ing)\s+(?:at|in)|came\s+to|entered|passed\s+into)\s+",
            lower,
        )
        if arrive_only:
            destination = _endpoint_after(sentence, arrive_only.end(), aliases, role="destination")
            if destination:
                endpoints.append(destination)

    # H. marched/reached to Y only (destination with preposition)
    if not edges and not any(item.role == "destination" for item in endpoints):
        dest_only = re.search(
            r"\b(?:marched|marches|marching|march|advanced|proceeded|reached|arrived\s+(?:at|in)|"
            r"came\s+to|entered|passed\s+into|set\s+out)\s+(?:\w+\s+){0,4}(?:to|into)\s+",
            lower,
        )
        if dest_only:
            destination = _endpoint_after(sentence, dest_only.end(), aliases, role="destination")
            if destination:
                endpoints.append(destination)

    # I. left X (origin only)
    if not edges:
        left_only = re.search(r"\b(?:left|leaving)\s+", lower)
        if left_only:
            origin = _endpoint_after(sentence, left_only.end(), aliases, role="origin")
            if origin:
                endpoints.append(origin)

    # J. came from X
    if not edges:
        came_from = re.search(r"\bcame\s+from\s+", lower)
        if came_from:
            origin = _endpoint_after(sentence, came_from.end(), aliases, role="origin")
            if origin:
                endpoints.append(origin)

    # K. passed/went through X to Y
    if not edges:
        through = re.search(r"\b(?:passed|went)\s+through\s+", lower)
        toward = re.search(r"\b(?:to|toward|towards|into)\s+", lower[through.end():] if through else "")
        if through and toward:
            origin = _endpoint_after(sentence, through.end(), aliases, before=through.end() + toward.start(), role="origin")
            destination = _endpoint_after(sentence, through.end() + toward.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="through_to")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # L. led through X toward Y
    if not edges:
        led_through = re.search(r"\bled\b.{0,40}?\bthrough\s+", lower)
        toward = re.search(r"\b(?:toward|towards|to|into)\s+", lower[led_through.end():] if led_through else "")
        if led_through and toward:
            origin = _endpoint_after(sentence, led_through.end(), aliases, before=led_through.end() + toward.start(), role="origin")
            destination = _endpoint_after(sentence, led_through.end() + toward.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="led_through_toward")
            if candidate:
                edges.append(candidate)
                endpoints.extend(item for item in (origin, destination) if item)

    # M. discourse: from there / thence ... to Y
    if not edges and prior_endpoints and (_DISCOURSE_FROM_THERE.search(sentence) or _DISCOURSE_THEN.search(sentence)):
        if not _MOVEMENT_AFTER_DISCOURSE.search(sentence):
            return SentenceMovementSemantics(
                is_movement=True, edges=(), endpoints=tuple(endpoints), should_abstain=True,
                abstain_reason="discourse_without_movement_predicate",
            )
        antecedents = [item for item in prior_endpoints if item.role in {"origin", "destination"}]
        if len(antecedents) != 1:
            return SentenceMovementSemantics(
                is_movement=True, edges=(), endpoints=tuple(endpoints),
                should_abstain=True, abstain_reason="ambiguous_discourse_antecedent",
            )
        to_match = re.search(r"\b(?:to|into|toward|towards)\s+", lower)
        dest_match = re.search(
            r"\b(?:reached|arrived\s+(?:at|in)|came\s+to|entered|passed\s+into|marched\s+to|advanced\s+to|proceeded\s+to)\s+",
            lower,
        )
        destination = None
        if to_match:
            destination = _endpoint_after(sentence, to_match.end(), aliases, role="destination")
        elif dest_match:
            destination = _endpoint_after(sentence, dest_match.end(), aliases, role="destination")
        antecedent = antecedents[0]
        origin = MovementEndpoint(
            surface=antecedent.surface,
            canonical=antecedent.canonical,
            role="origin",
            position=antecedent.position,
        )
        candidate = _edge(origin, destination, movement_relation="discourse_continuation", cross_sentence_link=True)
        if candidate:
            edges.append(candidate)
            endpoints.extend([origin, destination] if destination else [origin])

    return SentenceMovementSemantics(
        is_movement=bool(edges or endpoints),
        edges=tuple(edges),
        endpoints=tuple(endpoints),
    )
