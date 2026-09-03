"""Evidence-local movement semantic extraction.

Candidate generation favors recall; downstream gates keep edge admission precise.
Clause-level role assignment uses movement predicates plus directional markers,
not campaign-specific phrasing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from backend.app.models import PlaceMentionValidationClass
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.place_mention_validation import validate_broad_place_mention

EndpointRole = Literal["origin", "destination", "traversal"]

_MOVEMENT_PREDICATE = re.compile(
    r"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|"
    r"departed|arrived|entered|crossed|crossing|withdrew|retreated|fled|left|leaving|reached|came|"
    r"returned|passed|passing|set\s+out|hastened|sailed|sailing|embarked|embark|landed|landing|"
    r"traversed|traversing|conducted|led|went|descended|repassed|travel|travelling|traveling|"
    r"escaped|withdrawing|retreating|fell\s+back|made\s+(?:his|her|their)\s+way|put\s+(?:in|out)|"
    r"journey(?:ed)?|route(?:d)?)\b",
    re.IGNORECASE,
)
_MOVEMENT_CUE = _MOVEMENT_PREDICATE
_NON_MOVEMENT = re.compile(
    r"\b(?:fought|battle|born|controlled|province|political\s+movement|moved\s+the\s+senate|"
    r"speech\s+about|according\s+to|made\s+equal\s+in\s+command|brought|buried|joined|"
    r"march\s+on\b|advanced\s+to\s+the\s+(?:city|town|camp)|crossed\s+the\s+(?:theater|theatre|stage|room))\b",
    re.IGNORECASE,
)
_NON_SPATIAL_PROCEEDED = re.compile(
    r"\bproceeded\s+to\s+(?:alter|discuss|debate|vote|appoint|elect|consider|pass|enact|execute|"
    r"carry|take|hold|conduct|complete|finish|begin|start|make|do|allay|divide|separate|lay|draw|"
    r"bring|raise|form|establish|settle|arrange|organize|organise|inquire|investigate)\b",
    re.IGNORECASE,
)
_NON_SPATIAL_FROM = re.compile(
    r"\b(?:suffered|learned|benefited|benefitted|died|derived|known|heard|distinguished|"
    r"removed|apart|different|acquired)\s+from\b",
    re.IGNORECASE,
)
_DISTANCE_FROM = re.compile(
    r"\b\d+\s+(?:miles?|leagues?|stadia|furlongs?|days?'?\s+march|kilometers?|km)\s+from\b",
    re.IGNORECASE,
)
_REFERENCE_FROM = re.compile(
    r"\b(?:news|intelligence|report|account|word|tidings|letter|message|story|version|tradition)\s+from\b",
    re.IGNORECASE,
)
_GENERIC_FROM_OBJECT = re.compile(
    r"\bfrom\s+(?:the\s+)?(?:harbor|harbour|port|sea|shore|coast|camp|city|town|forum|field|battle|war)\b",
    re.IGNORECASE,
)
_TERRITORIAL_EXTENT = re.compile(
    r"\b(?:extended|spread|stretched|influence|power|possessions|dominion|authority|fear|"
    r"report|news|fame|renown|reputation)\b[^.]{0,40}?\b(?:as\s+far\s+as|to|into|through|from)\b",
    re.IGNORECASE,
)
_SOURCE_MARKER = re.compile(r"\b(?:from|out\s+of|away\s+from)\b", re.IGNORECASE)
_TARGET_MARKER = re.compile(r"\b(?:to|into|toward|towards|for)\b", re.IGNORECASE)
_TRAVERSAL_MARKER = re.compile(r"\b(?:through|across|by\s+way\s+of|over)\s+(?:the\s+)?", re.IGNORECASE)
_ARRIVAL_PREDICATE = re.compile(
    r"\b(?:reached|arriv(?:ed|ing)|landed|landing|entered|came)\b", re.IGNORECASE,
)
_DEPARTURE_PREDICATE = re.compile(
    r"\b(?:departed|left|leaving|withdrew|withdrawing|retreated|retreating|fled|embarked|"
    r"embark|sailed|sailing|set\s+out|put\s+out)\b",
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
    r"hastened|set\s+out|descended|sailed|embarked|escaped|travel(?:led|ed|ing)?|made\s+(?:his|her|their)\s+way)\b"
    r"(?:\s+\w+){0,16}?\bfrom\b",
    re.IGNORECASE,
)
_MEDIATED_ORIGIN_PREFIX = re.compile(
    r"^(?:the\s+)?(?:passage|valley|crossing|banks?|mouth|shores?|foot|straits?)\s+of\s+(?:the\s+)?",
    re.IGNORECASE,
)
_VALLEY_OF_PREFIX = re.compile(
    r"^(?:the\s+)?([A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*(?:\s+(?:the\s+)?[A-Z][A-Za-z'À-ÖØ-öø-ÿÆæŒœ]*){0,2})\s+valley\b",
    re.IGNORECASE,
)
_GOVERNED_WINDOW = 100


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


def _clause_start(sentence: str, position: int) -> int:
    return max(sentence.rfind(",", 0, position), sentence.rfind(";", 0, position)) + 1


def _has_movement_predicate(text: str) -> bool:
    return bool(_MOVEMENT_PREDICATE.search(text))


def _has_movement_cue(sentence: str) -> bool:
    if _NON_MOVEMENT.search(sentence):
        return False
    if _NON_SPATIAL_PROCEEDED.search(sentence):
        return False
    if _TERRITORIAL_EXTENT.search(sentence) and not _MOVEMENT_GOVERNED_FROM.search(sentence):
        if not re.search(r"\b(?:marched|advanced|proceeded|travelled|traveled|sailed|went|crossed|passed)\b", sentence, re.I):
            return False
    return bool(_MOVEMENT_CUE.search(sentence))


def _predicate_before(clause: str, marker_start: int) -> bool:
    prefix = clause[:marker_start]
    if not _has_movement_predicate(prefix):
        return False
    last_pred = None
    for match in _MOVEMENT_PREDICATE.finditer(prefix):
        last_pred = match
    if last_pred is None:
        return False
    return marker_start - last_pred.end() <= _GOVERNED_WINDOW


def _valid_source_marker(clause: str, marker_start: int) -> bool:
    window = clause[max(0, marker_start - 60):marker_start + 4]
    if _NON_SPATIAL_FROM.search(window):
        return False
    if _DISTANCE_FROM.search(window):
        return False
    if _REFERENCE_FROM.search(window):
        return False
    if _GENERIC_FROM_OBJECT.search(clause[max(0, marker_start - 5):marker_start + 40]):
        return False
    if not _predicate_before(clause, marker_start) and not _MOVEMENT_GOVERNED_FROM.search(clause):
        if re.search(r"\b(?:travel|journey|route)\b", clause[:marker_start], re.I):
            if _TARGET_MARKER.search(clause, marker_start):
                return True
        return False
    return True


def _valid_target_marker(clause: str, marker_start: int, *, role_token: str) -> bool:
    after_to = clause[marker_start + len(role_token):]
    if role_token == "to" and re.match(
        r"\s+(?:travel|alter|discuss|vote|see|fight|make|do|take|go|march|advance|proceed|move|carry|hold|be)\b",
        after_to,
        re.I,
    ):
        return False
    if role_token == "for" and not re.search(
        r"\b(?:departed|sail|sailed|sailing|set\s+out|embark|embarked|marched|advanced|went)\b",
        clause[:marker_start],
        re.I,
    ):
        return False
    if re.search(r"\b(?:equal\s+in\s+command|made\s+equal|according|gave\s+the\s+name)\b", clause[:marker_start], re.I):
        return False
    if not _predicate_before(clause, marker_start):
        if role_token in {"to", "into"} and re.search(
            r"\b(?:marched|advanced|proceeded|travelled|traveled|went|hastened|returned|fled|withdrew|retreated|"
            r"departed|left|sailed|crossed|passed|came|reached|travel(?:led|ed|ing)?|made\s+(?:his|her|their)\s+way)\s+"
            r"(?:\w+\s+){0,6}(?:to|into)\b",
            clause[:marker_start + 6],
            re.I,
        ):
            return True
        if role_token in {"to", "into"} and re.search(r"\bbegan\s+to\s+march\s+to\b", clause, re.I):
            return True
        if role_token in {"to", "into"} and re.search(r"\bon\s+(?:their|his|her|its)\s+march\s+to\b", clause, re.I):
            return True
        return False
    return True


def _arrival_place_start(clause: str, predicate_end: int) -> int | None:
    tail = clause[predicate_end:]
    came_to = re.match(r"\s*to\s+(?:the\s+)?", tail, re.I)
    if came_to:
        return predicate_end + came_to.end()
    direct = re.match(r"\s+(?:at|in|on|upon)\s+(?:the\s+)?", tail, re.I)
    if direct:
        return predicate_end + direct.end()
    gap = re.match(r"(?:\s+\w+){0,8}\s+(?:at|in|on|upon)\s+(?:the\s+)?", tail, re.I)
    if gap:
        return predicate_end + gap.end()
    place = re.match(r"\s*(?:the\s+)?([A-Z])", tail)
    if place:
        return predicate_end + place.start(1)
    return None


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


def _parse_clause(
    sentence: str,
    clause_start: int,
    clause_end: int,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
) -> tuple[list[MovementEdgeCandidate], list[MovementEndpoint], bool]:
    clause = sentence[clause_start:clause_end]
    lower = clause.lower()
    if not _has_movement_predicate(lower) and not re.search(r"\btravel\b", lower, re.I):
        return [], [], False

    origins: list[MovementEndpoint] = []
    destinations: list[MovementEndpoint] = []
    traversals: list[MovementEndpoint] = []

    for match in _SOURCE_MARKER.finditer(lower):
        if not _valid_source_marker(clause, match.start()):
            continue
        abs_from_end = clause_start + match.end()
        next_target = _TARGET_MARKER.search(lower, match.end())
        before = clause_start + next_target.start() if next_target else clause_end
        origin = _origin_after_from(sentence, abs_from_end, aliases, before=before)
        if origin is not None:
            origins.append(origin)

    for match in _TARGET_MARKER.finditer(lower):
        role_token = match.group(0).lower()
        if not _valid_target_marker(clause, match.start(), role_token=role_token):
            continue
        abs_to_end = clause_start + match.end()
        destination = _endpoint_after(sentence, abs_to_end, aliases, before=clause_end, role="destination")
        if destination is not None:
            destinations.append(destination)

    for match in _TRAVERSAL_MARKER.finditer(lower):
        if not _predicate_before(clause, match.start()):
            continue
        abs_end = clause_start + match.end()
        next_target = _TARGET_MARKER.search(lower, match.end())
        before = clause_start + next_target.start() if next_target else clause_end
        traversal = _endpoint_after(sentence, abs_end, aliases, before=before, role="traversal")
        if traversal is not None:
            traversals.append(traversal)

    for match in _ARRIVAL_PREDICATE.finditer(lower):
        place_start = _arrival_place_start(clause, match.end())
        if place_start is None:
            continue
        token = match.group(0).lower()
        if token not in {"reached", "arrived", "arriving", "landed", "landing", "entered", "came"}:
            if not _predicate_before(clause, match.start()):
                continue
        abs_start = clause_start + place_start
        destination = _endpoint_after(sentence, abs_start, aliases, before=clause_end, role="destination")
        if destination is not None:
            destinations.append(destination)

    # Crossed X before arriving at Y
    crossed_only = re.search(r"\b(?:crossed|crossing)\s+(?:the\s+)?", lower)
    arrive_after = re.search(r"\b(?:before\s+)?arriv(?:ed|ing)\s+(?:at|in)\s+", lower)
    if crossed_only and arrive_after and crossed_only.start() < arrive_after.start():
        traversal = _endpoint_after(
            sentence, clause_start + crossed_only.end(), aliases,
            before=clause_start + arrive_after.start(), role="traversal",
        )
        place_start = _arrival_place_start(clause, arrive_after.end())
        if traversal and place_start is not None:
            destination = _endpoint_after(sentence, clause_start + place_start, aliases, before=clause_end, role="destination")
            candidate = _edge(traversal, destination, movement_relation="crossing_arrival")
            if candidate:
                return [candidate], [traversal, destination], False

    # Crossed/moved from X into/to Y
    cross_from = re.search(
        r"\b(?:crossed|crossing|moved|travelled|traveled|went|came|advanced|proceeded)\b.{0,160}?\bfrom\s+",
        lower,
    )
    if cross_from:
        from_token = re.search(r"\bfrom\s+", lower[cross_from.start():])
        if from_token:
            abs_from_end = clause_start + cross_from.start() + from_token.end()
            to_match = _TARGET_MARKER.search(lower, cross_from.start() + from_token.end())
            if to_match:
                origin = _origin_after_from(sentence, abs_from_end, aliases, before=clause_start + to_match.start())
                destination = _endpoint_after(sentence, clause_start + to_match.end(), aliases, before=clause_end, role="destination")
                candidate = _edge(origin, destination, movement_relation="crossed_from_into")
                if candidate:
                    return [candidate], [item for item in (origin, destination) if item], False

    # Maritime compound: sailed/embarked from X ... landed/arrived at Y
    sailed_from = re.search(
        r"\b(?:sail|sailed|sailing|embark|embarked|put\s+out)\b.{0,140}?\bfrom\s+",
        lower,
    )
    landed = re.search(
        r"\b(?:landed|landing|arriv(?:ed|ing))\b",
        lower[sailed_from.end():] if sailed_from else "",
    )
    if sailed_from and landed:
        from_match = re.search(r"\bfrom\s+", lower[sailed_from.start():])
        if from_match:
            abs_from_end = clause_start + sailed_from.start() + from_match.end()
            land_abs = clause_start + sailed_from.end() + landed.end()
            land_start = _arrival_place_start(clause[sailed_from.end():], landed.end())
            if land_start is not None:
                land_abs = clause_start + sailed_from.end() + land_start
            origin = _origin_after_from(sentence, abs_from_end, aliases, before=land_abs)
            destination = _endpoint_after(sentence, land_abs, aliases, before=clause_end, role="destination")
            if origin and destination:
                return (
                    [_edge(origin, destination, movement_relation="maritime_from_landed")],
                    [origin, destination],
                    False,
                )

    cross_from_land = re.search(
        r"\b(?:crossed|crossing|sailed|sailing)\b.{0,120}?\bfrom\s+",
        lower,
    )
    landed_after = re.search(r"\b(?:landed|landing|arriv(?:ed|ing))\b", lower)
    if cross_from_land and landed_after and cross_from_land.start() < landed_after.start():
        from_match = re.search(r"\bfrom\s+", lower[cross_from_land.start():])
        if from_match:
            abs_from_end = clause_start + cross_from_land.start() + from_match.end()
            place_start = _arrival_place_start(sentence, landed_after.end())
            if place_start is None:
                after_land = sentence[landed_after.end():]
                place_matches = list(_PLACE_SPAN.finditer(after_land))
                if place_matches:
                    place_start = landed_after.end() + place_matches[-1].start(1)
            if place_start is not None:
                origin = _origin_after_from(sentence, abs_from_end, aliases, before=place_start)
                destination = _endpoint_after(sentence, place_start, aliases, before=clause_end, role="destination")
                candidate = _edge(origin, destination, movement_relation="maritime_from_landed")
                if candidate:
                    return [candidate], [item for item in (origin, destination) if item], False

    # Crossed/repassed ... into Y
    crossed = re.search(r"\b(?:crossed|crossing|repassed|passing\s+over)\b", lower)
    into = re.search(r"\binto\s+", lower[crossed.end():] if crossed else "")
    if crossed and into:
        abs_cross_end = clause_start + crossed.end()
        abs_into_end = clause_start + crossed.end() + into.end()
        traversal = _endpoint_after(sentence, abs_cross_end, aliases, before=clause_start + crossed.end() + into.start(), role="traversal")
        destination = _endpoint_after(sentence, abs_into_end, aliases, before=clause_end, role="destination")
        if traversal and destination:
            candidate = _edge(None, destination, traversal=traversal, movement_relation="crossing_into")
            if candidate:
                return [candidate], [traversal, destination], False

    # Advanced/marched through X into/toward Y
    through_move = re.search(
        r"\b(?:advanced|marched|proceeded|moved|went|passed|passing)\b.{0,80}?\bthrough\s+",
        lower,
    )
    if through_move:
        abs_through_end = clause_start + through_move.end()
        target = _TARGET_MARKER.search(lower, through_move.end())
        traversal = _endpoint_after(
            sentence, abs_through_end, aliases,
            before=clause_start + target.start() if target else clause_end,
            role="traversal",
        )
        if traversal:
            traversals.append(traversal)
        if target and _valid_target_marker(clause, target.start(), role_token=target.group(0).lower()):
            destination = _endpoint_after(sentence, clause_start + target.end(), aliases, before=clause_end, role="destination")
            if destination:
                destinations.append(destination)

    # Through X to Y — traversal + destination edge (legacy claim compatibility uses traversal as edge origin)
    through_to = re.search(r"\b(?:passed|passing|went|marched|advanced)\s+through\s+", lower)
    toward = _TARGET_MARKER.search(lower[through_to.end():] if through_to else "")
    if through_to and toward:
        abs_through_end = clause_start + through_to.end()
        abs_target_start = clause_start + through_to.end() + toward.start()
        abs_target_end = clause_start + through_to.end() + toward.end()
        traversal = _endpoint_after(sentence, abs_through_end, aliases, before=abs_target_start, role="traversal")
        destination = _endpoint_after(sentence, abs_target_end, aliases, before=clause_end, role="destination")
        if traversal and destination:
            candidate = _edge(traversal, destination, movement_relation="through_to")
            if candidate:
                return [candidate], [traversal, destination], False

    # Led through X toward/to Y
    led_through = re.search(r"\bled\b.{0,40}?\bthrough\s+", lower)
    toward = _TARGET_MARKER.search(lower[led_through.end():] if led_through else "")
    if led_through and toward:
        abs_through_end = clause_start + led_through.end()
        abs_target_start = clause_start + led_through.end() + toward.start()
        abs_target_end = clause_start + led_through.end() + toward.end()
        traversal = _endpoint_after(sentence, abs_through_end, aliases, before=abs_target_start, role="traversal")
        destination = _endpoint_after(sentence, abs_target_end, aliases, before=clause_end, role="destination")
        if traversal and destination:
            candidate = _edge(traversal, destination, movement_relation="led_through_toward")
            if candidate:
                return [candidate], [traversal, destination], False

    # Crossed X then entered/arrived Y in the same clause (including and-coordination).
    if crossed_only:
        arrive_coord = re.search(
            r"\b(?:came\s+to|arriv(?:ed|ing)\s+(?:at|in)|entered|passed\s+into|reached)\s+",
            lower,
        )
        if arrive_coord and crossed_only.start() < arrive_coord.start():
            traversal = _endpoint_after(
                sentence, clause_start + crossed_only.end(), aliases,
                before=clause_start + arrive_coord.start(), role="traversal",
            )
            place_start = _arrival_place_start(clause, arrive_coord.end())
            if traversal and place_start is not None:
                destination = _endpoint_after(sentence, clause_start + place_start, aliases, before=clause_end, role="destination")
                candidate = _edge(traversal, destination, movement_relation="crossing_arrival")
                if candidate:
                    return [candidate], [traversal, destination], False

    if crossed_only and not arrive_after:
        tail = lower[crossed_only.end():]
        if not re.search(r"\b(?:into|from)\s+", tail):
            traversal = _endpoint_after(
                sentence, clause_start + crossed_only.end(), aliases, before=clause_end, role="traversal",
            )
            if traversal:
                candidate = _edge(None, None, traversal=traversal, movement_relation="traversal")
                if candidate:
                    return [candidate], [traversal], False

    # Left X ... reached/arrived Y (same clause)
    leave = re.search(r"\b(?:left|leaving)\s+", lower)
    arrive = _ARRIVAL_PREDICATE.search(lower)
    if leave and arrive and leave.start() < arrive.start():
        origin = _endpoint_after(sentence, clause_start + leave.end(), aliases, before=clause_start + arrive.start(), role="origin")
        place_start = _arrival_place_start(clause, arrive.end())
        if origin and place_start is not None:
            destination = _endpoint_after(sentence, clause_start + place_start, aliases, before=clause_end, role="destination")
            if destination:
                candidate = _edge(origin, destination, movement_relation="departure_arrival")
                if candidate:
                    return [candidate], [origin, destination], False

    unique_origins = {item.place_name.casefold(): item for item in origins}
    unique_dests = {item.place_name.casefold(): item for item in destinations}
    unique_travs = {item.place_name.casefold(): item for item in traversals}

    if len(unique_origins) > 1 or len(unique_dests) > 1:
        return [], [], True

    origin = next(iter(unique_origins.values()), None)
    destination = next(iter(unique_dests.values()), None)
    traversal = next(iter(unique_travs.values()), None)

    edges: list[MovementEdgeCandidate] = []
    endpoints: list[MovementEndpoint] = []

    if origin and destination:
        candidate = _edge(origin, destination, traversal=traversal, movement_relation="from_to")
        if candidate:
            edges.append(candidate)
            endpoints.extend([origin, destination])
            if traversal:
                endpoints.append(traversal)
    elif origin:
        endpoints.append(origin)
    elif destination:
        endpoints.append(destination)
    elif traversal:
        candidate = _edge(None, None, traversal=traversal, movement_relation="traversal")
        if candidate:
            edges.append(candidate)
            endpoints.append(traversal)

    return edges, endpoints, False


def _parse_sentence_compound(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
) -> tuple[list[MovementEdgeCandidate], list[MovementEndpoint]]:
    lower = sentence.lower()
    leave = re.search(r"\b(?:left|leaving)\s+", lower)
    arrive = _ARRIVAL_PREDICATE.search(lower)
    if leave and arrive and leave.start() < arrive.start():
        origin = _endpoint_after(sentence, leave.end(), aliases, before=arrive.start(), role="origin")
        place_start = _arrival_place_start(sentence, arrive.end())
        if origin and place_start is not None:
            destination = _endpoint_after(sentence, place_start, aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="departure_arrival")
            if candidate:
                return [candidate], [origin, destination]

    went_through = re.search(r"\bwent\s+through\s+", lower)
    landed = re.search(r"\b(?:landed|landing)\b", lower)
    if went_through and landed and went_through.start() < landed.start():
        remainder = sentence[went_through.end():]
        stripped = remainder.lstrip()
        offset = went_through.end() + (len(remainder) - len(stripped))
        mediated = _MEDIATED_ORIGIN_PREFIX.match(stripped)
        if mediated:
            traversal = _endpoint_after(
                sentence, offset + mediated.end(), aliases, before=landed.start(), role="traversal",
            )
        else:
            traversal = _endpoint_after(
                sentence, went_through.end(), aliases, before=landed.start(), role="traversal",
            )
        place_start = _arrival_place_start(sentence, landed.end())
        if place_start is None:
            after_land = sentence[landed.end():]
            place_matches = list(_PLACE_SPAN.finditer(after_land))
            if place_matches:
                place_start = landed.end() + place_matches[-1].start(1)
        if traversal and place_start is not None:
            destination = _endpoint_after(sentence, place_start, aliases, role="destination")
            candidate = _edge(traversal, destination, movement_relation="through_landed")
            if candidate:
                return [candidate], [traversal, destination]

    initial_from = re.match(
        r"^\s*from\s+(?:the\s+)?",
        sentence,
        re.IGNORECASE,
    )
    marched_to = re.search(
        r"\b(?:marched|advanced|proceeded|moved|travelled|traveled|went|hastened|returned|withdrew|retreated)\b"
        r"(?:\s+\w+){0,12}\s+(?:to|into)\s+",
        lower,
    )
    if (
        initial_from
        and marched_to
        and not _DISCOURSE_FROM_THERE.search(sentence)
        and not re.match(r"^\s*from\s+(?:there|thence)\b", sentence, re.IGNORECASE)
    ):
        from_end = initial_from.end()
        to_match = _TARGET_MARKER.search(lower, from_end)
        if to_match and marched_to.start() < to_match.start():
            origin = _origin_after_from(sentence, from_end, aliases, before=to_match.start())
            destination = _endpoint_after(sentence, to_match.end(), aliases, role="destination")
            candidate = _edge(origin, destination, movement_relation="initial_from_marched_to")
            if candidate:
                return [candidate], [item for item in (origin, destination) if item]

    reached = re.search(r"\b(?:reached|arriv(?:ed|ing)\s+(?:at|in))\s+", lower)
    lead = re.search(r"\b(?:led|conducted)\b.{0,180}?\b(?:to|into)\s+", lower)
    if reached and lead and reached.start() < lead.start():
        origin = _endpoint_after(sentence, reached.end(), aliases, before=lead.start(), role="origin")
        destination = _endpoint_after(sentence, lead.end(), aliases, role="destination")
        candidate = _edge(origin, destination, movement_relation="arrival_then_lead")
        if candidate:
            return [candidate], [item for item in (origin, destination) if item]

    crossed = re.search(r"\b(?:having\s+)?(?:crossed|crossing)\s+(?:the\s+)?", lower)
    arrive = _ARRIVAL_PREDICATE.search(lower)
    if crossed and arrive and crossed.start() < arrive.start():
        between = lower[crossed.end():arrive.start()]
        if not (re.search(r"\band\b", between) and not re.search(r"\bbefore\b", between)):
            traversal = _endpoint_after(
                sentence, crossed.end(), aliases, before=arrive.start(), role="traversal",
            )
            place_start = _arrival_place_start(sentence, arrive.end())
            if traversal and place_start is not None:
                destination = _endpoint_after(sentence, place_start, aliases, role="destination")
                candidate = _edge(traversal, destination, movement_relation="crossing_arrival")
                if candidate:
                    return [candidate], [traversal, destination]

    return [], []


def _generalized_parse(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
) -> tuple[list[MovementEdgeCandidate], list[MovementEndpoint], bool, str | None]:
    compound_edges, compound_endpoints = _parse_sentence_compound(sentence, aliases)
    if compound_edges:
        return compound_edges, compound_endpoints, False, None

    edges: list[MovementEdgeCandidate] = []
    endpoints: list[MovementEndpoint] = []
    breaks: list[int] = [0]
    for match in re.finditer(r"[,;]", sentence):
        breaks.append(match.end())
    breaks.append(len(sentence))

    for index in range(len(breaks) - 1):
        clause_start = breaks[index]
        clause_end = breaks[index + 1]
        clause_edges, clause_endpoints, ambiguous = _parse_clause(sentence, clause_start, clause_end, aliases)
        if ambiguous:
            return [], [], True, "ambiguous_clause_endpoints"
        if clause_edges:
            edges.extend(clause_edges)
        endpoints.extend(clause_endpoints)

    if edges:
        return edges, endpoints, False, None

    # Whole-sentence from-to when movement predicate governs a single span
    lower = sentence.lower()
    if _MOVEMENT_GOVERNED_FROM.search(lower):
        from_match = _MOVEMENT_GOVERNED_FROM.search(lower)
        assert from_match is not None
        from_token = re.search(r"\bfrom\s+", lower[from_match.start():])
        if from_token:
            from_end = from_match.start() + from_token.end()
            to_match = _TARGET_MARKER.search(lower, from_end)
            if to_match and _valid_target_marker(sentence, to_match.start(), role_token=to_match.group(0).lower()):
                origin = _origin_after_from(sentence, from_end, aliases, before=to_match.start())
                destination = _endpoint_after(sentence, to_match.end(), aliases, role="destination")
                candidate = _edge(origin, destination, movement_relation="from_to")
                if candidate:
                    return [candidate], [item for item in (origin, destination) if item], False, None

    return [], endpoints, False, None


def _discourse_continuation(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
    prior_endpoints: tuple[MovementEndpoint, ...],
    endpoints: list[MovementEndpoint],
) -> tuple[list[MovementEdgeCandidate], list[MovementEndpoint], bool, str | None]:
    if not prior_endpoints or not (_DISCOURSE_FROM_THERE.search(sentence) or _DISCOURSE_THEN.search(sentence)):
        return [], endpoints, False, None
    if not _MOVEMENT_AFTER_DISCOURSE.search(sentence):
        return [], endpoints, True, "discourse_without_movement_predicate"
    antecedents = [item for item in prior_endpoints if item.role in {"origin", "destination"}]
    if len(antecedents) != 1:
        return [], endpoints, True, "ambiguous_discourse_antecedent"
    lower = sentence.lower()
    to_match = re.search(r"\b(?:to|into|toward|towards)\s+", lower)
    dest_match = re.search(
        r"\b(?:reached|arrived\s+(?:at|in)|came\s+to|entered|passed\s+into|marched\s+to|advanced\s+to|proceeded\s+to)\s+",
        lower,
    )
    destination = None
    if to_match:
        destination = _endpoint_after(sentence, to_match.end(), aliases, role="destination")
    elif dest_match:
        place_start = _arrival_place_start(sentence, dest_match.end())
        if place_start is not None:
            destination = _endpoint_after(sentence, place_start, aliases, role="destination")
    antecedent = antecedents[0]
    origin = MovementEndpoint(
        surface=antecedent.surface,
        canonical=antecedent.canonical,
        role="origin",
        position=antecedent.position,
    )
    candidate = _edge(origin, destination, movement_relation="discourse_continuation", cross_sentence_link=True)
    edges: list[MovementEdgeCandidate] = []
    if candidate:
        edges.append(candidate)
        endpoints = endpoints + ([origin, destination] if destination else [origin])
    return edges, endpoints, False, None


def _thence_maritime(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
) -> tuple[MovementEdgeCandidate | None, list[MovementEndpoint]]:
    lower = sentence.lower()
    sailed = re.search(r"\b(?:sail|sailed|sailing)\s+for\s+", lower)
    thence = re.search(r"\b(?:thence|from\s+there)\s+passed\s+on\s+to\s+", lower)
    if sailed and thence and sailed.start() < thence.start():
        sailed_destination = _endpoint_after(sentence, sailed.end(), aliases, before=thence.start(), role="origin")
        thence_destination = _endpoint_after(sentence, thence.end(), aliases, role="destination")
        candidate = _edge(sailed_destination, thence_destination, movement_relation="thence_passed_on_to")
        if candidate:
            return candidate, [item for item in (sailed_destination, thence_destination) if item]
    return None, []


def analyze_sentence(
    sentence: str,
    aliases: list[tuple[int, HistoricalPlaceAlias, str]],
    *,
    prior_endpoints: tuple[MovementEndpoint, ...] = (),
) -> SentenceMovementSemantics:
    endpoints: list[MovementEndpoint] = []
    edges: list[MovementEdgeCandidate] = []

    if not _has_movement_cue(sentence):
        return SentenceMovementSemantics(is_movement=False, edges=(), endpoints=())

    gen_edges, gen_endpoints, should_abstain, abstain_reason = _generalized_parse(sentence, aliases)
    if should_abstain:
        return SentenceMovementSemantics(
            is_movement=True, edges=(), endpoints=(), should_abstain=True, abstain_reason=abstain_reason,
        )
    if gen_edges:
        edges.extend(gen_edges)
        endpoints.extend(gen_endpoints)
    elif gen_endpoints:
        endpoints.extend(gen_endpoints)

    if not edges:
        thence_edge, thence_endpoints = _thence_maritime(sentence, aliases)
        if thence_edge:
            edges.append(thence_edge)
            endpoints.extend(thence_endpoints)

    if not edges:
        disc_edges, disc_endpoints, disc_abstain, disc_reason = _discourse_continuation(
            sentence, aliases, prior_endpoints, endpoints,
        )
        if disc_abstain:
            return SentenceMovementSemantics(
                is_movement=True, edges=(), endpoints=tuple(endpoints),
                should_abstain=True, abstain_reason=disc_reason,
            )
        if disc_edges:
            edges.extend(disc_edges)
            endpoints = disc_endpoints

    return SentenceMovementSemantics(
        is_movement=bool(edges or endpoints),
        edges=tuple(edges),
        endpoints=tuple(endpoints),
    )
