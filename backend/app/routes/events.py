"""Deterministic, evidence-only extraction of general historical events."""
from __future__ import annotations

import hashlib
import re
import unicodedata

from backend.app.models import (
    Evidence,
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    PlaceMentionValidationClass,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.evidence_relevance import movement_eligibility_with_context, narrative_subject_proper_nouns
from backend.app.routes.movement_semantics import MovementEndpoint, _SET_SAIL, analyze_sentence, _has_movement_cue
from backend.app.routes.place_mention_validation import validate_broad_place_mention
from backend.app.routes.temporal import EvidenceTemporalResolver, TemporalResolutionContext


class EvidenceGroundedHistoricalEventExtractor:
    """Extract separate event statements without place resolution or route inference."""

    _TYPE_PATTERNS = (
        (HistoricalEventType.ASSASSINATION, r"\b(?:assassinated|assassination|murdered)\b"),
        (HistoricalEventType.REFORM, r"\b(?:reform(?:ed)?|reformers?|land law|legislation|proposed a law)\b"),
        (HistoricalEventType.BATTLE, r"\b(?:battle|fought at|defeated .* at)\b"),
        (HistoricalEventType.SIEGE, r"\b(?:siege|besieged)\b"),
        (HistoricalEventType.TREATY, r"\b(?:treaty|peace agreement|concluded peace)\b"),
        (HistoricalEventType.ELECTION, r"\b(?:elected|election|chosen as)\b"),
        (HistoricalEventType.REBELLION, r"\b(?:rebellion|revolt(?:ed)?|uprising|insurrection)\b"),
        (HistoricalEventType.MOVEMENT, rf"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|arrived|entered|crossed|withdrew|retreated|fled|left|sailed|embarked|landed|went|returned|passed|repassed|travel|escaped|descended|made\s+(?:his|her|their)\s+way|{_SET_SAIL})\b"),
        (HistoricalEventType.MILITARY, r"\b(?:campaign|army|war|invaded|conquered|captured)\b"),
        (HistoricalEventType.POLITICAL, r"\b(?:senate .*\bdecree|tribune .*\b(?:proposed|elected|opposed)|consul .*\b(?:appointed|elected|sent)|assembly .*\b(?:elected|passed)|issued a decree)\b"),
    )
    _PLACE_PATTERN = re.compile(r"\b(?P<role>(?i:at|in|near|from|to|into|through))\s+(?:(?i:the)\s+)?(?P<place>[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})")
    _MOVEMENT_VERBS = re.compile(
        rf"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|arrived|entered|crossed|withdrew|retreated|fled|left|leaving|reached|came|passed|sailed|embarked|landed|went|returned|repassed|travel|escaped|descended|made\s+(?:his|her|their)\s+way|{_SET_SAIL})\b",
        re.IGNORECASE,
    )
    _NON_MOVEMENT_TO_CONTEXT = re.compile(
        r"\b(?:according|equal\s+in\s+command|made\s+equal\s+in\s+command|brought|intelligence\s+was\s+brought|buried|joined|announced)\b",
        re.IGNORECASE,
    )
    _ATTRIBUTIVE_AFTER_PLACE = re.compile(
        r"\s+(?:custom|war|manner|style|fashion|tradition|practice|people|triumph|riches)\b",
        re.IGNORECASE,
    )
    _TROOP_PROVENANCE_FROM = re.compile(
        r"\b(?:archers|horse|cavalry|infantry|soldiers|men|troops|forces|convoys|people|legions?)\s+from\b",
        re.IGNORECASE,
    )
    _ANAPHORIC_MOVEMENT_FROM = re.compile(
        r"^\s*(?:and\s+)?(?:after\s+)?(?:marching|moving|advancing|proceeding|"
        r"departing|retreating|withdrawing|travelling|traveling)\s+from\s+"
        r"(?:it|there|that\s+place)\s+(?:to|into|toward(?:s)?)\s+",
        re.IGNORECASE,
    )
    _PRECEDING_REGION = re.compile(
        r"\b(?:led|leads|entered|moved|marched|advanced|proceeded|returned|"
        r"retreated|withdrew|came|passed)\b[^;.!?]{0,100}\b(?:in|into|to)\s+"
        r"(?:the\s+)?(?:country|territor(?:y|ies)|lands?)\s+of\s+(?:the\s+)?"
        r"(?P<place>[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})\s*;\s*$",
        re.IGNORECASE,
    )
    _MOVEMENT_TO_PREFIX = re.compile(
        r"(?:\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|arrived|entered|crossed|withdrew|retreated|fled|left|leaving|reached|came|passed)\s+(?:\w+\s+){0,6}(?:to|into)\b"
        r"|\b(?:march|marches|marching)\s+to\b"
        r"|\bbegan\s+to\s+march\s+to\b"
        r"|\bon\s+(?:their|his|her|its)\s+march\s+to\b"
        r"|\bfrom\s+(?:the\s+)?[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}\s+to\b"
        r"|\bcame\s+to\b)",
        re.IGNORECASE,
    )
    _MOVEMENT_FROM_PREFIX = re.compile(
        rf"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|left|leaving|withdrew|retreated|fled|came|went|crossed|crossing|returned|hastened|set\s+out|descended|sailed|embarked|escaped|travel(?:led|ed|ing)?|made\s+(?:his|her|their)\s+way|repassed|{_SET_SAIL})\s+(?:\w+\s+){{0,12}}from\b",
        re.IGNORECASE,
    )
    _MOVEMENT_GOVERNED_FROM = re.compile(
        rf"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|left|leaving|withdrew|retreated|fled|came|went|crossed|crossing|returned|hastened|set\s+out|descended|sailed|embarked|escaped|travel(?:led|ed|ing)?|made\s+(?:his|her|their)\s+way|{_SET_SAIL})\b(?:\s+\w+){{0,12}}?\bfrom\b",
        re.IGNORECASE,
    )
    _MEDIATED_FROM_PREFIX = re.compile(
        r"\bfrom\s+(?:the\s+)?(?:passage|valley|crossing|banks?|mouth|shores?|foot|straits?)\s+of\s+(?:the\s+)?",
        re.IGNORECASE,
    )
    _NON_SPATIAL_FROM = re.compile(
        r"\b(?:suffered|learned|escaped|benefited|benefitted|died|derived|known|heard|distinguished|removed|apart|different)\s+from\b",
        re.IGNORECASE,
    )
    _DISTANCE_FROM = re.compile(
        r"\b\d+\s+(?:miles?|leagues?|stadia|kilometers?|km)\s+from\b",
        re.IGNORECASE,
    )
    _REFERENCE_FROM = re.compile(
        r"\b(?:news|intelligence|report|account|word|tidings|letter|message|story|version|tradition)\s+from\b",
        re.IGNORECASE,
    )
    _ACCOUNT_FROM = re.compile(
        r"\bfrom\s+this\s+account\b|\bfrom\s+the\s+account\b",
        re.IGNORECASE,
    )
    _RETROSPECTIVE = re.compile(
        r"\b(?:after|following|because of|since)\s+(?:the\s+)?(?:battle|defeat|death|murder|assassination)\b[^,;:.]*[,;:]?\s*",
        re.IGNORECASE,
    )
    _NON_COMPLETED = re.compile(
        r"\b(?:would|could|might|should|may|planned\s+to|intended\s+to|wanted\s+to|hoped\s+to|feared\s+(?:that|lest)|if)\b",
        re.IGNORECASE,
    )
    _NON_ASSERTIVE_GOVERNOR = re.compile(
        r"\b(?:discussed|debated|considered|contemplated)\b",
        re.IGNORECASE,
    )
    _PREVENTED_FROM = re.compile(
        r"\b(?:prevented|stopped|blocked|forbidden|barred|hindered)\b(?:\s+\w+){0,8}\sfrom\b",
        re.IGNORECASE,
    )
    _REPORTED_SPEECH = re.compile(r"[\"“”]|\b(?:said|declared|claimed|reported|urged)\s+(?:that|:)", re.IGNORECASE)
    _NAVIGATION_HEADING = re.compile(
        r"^\s*(?:how\b.*\bchapters?\b|(?:chapter|book)\s+[ivxlcdm0-9]+\b)", re.IGNORECASE
    )
    _ACTOR = re.compile(
        r"\b(?:the\s+)?(?:army|armies|senate|assembly|people|romans|carthaginians|rebels|consul|tribune|leader|reformer|commander|king|queen)\b",
        re.IGNORECASE,
    )
    _MOVEMENT_CLAUSE_SPLIT = re.compile(r"[,;]|\bbut\b|\band\b", re.IGNORECASE)
    _POLARITY_CLAUSE_SPLIT = re.compile(r"[,;]|\bbut\b", re.IGNORECASE)
    _NEGATED_AUXILIARIES = frozenset(
        {"did", "does", "do", "had", "has", "have", "was", "were", "is", "are", "could", "would", "should", "might", "may"}
    )
    _NEGATION_WINDOW = 3
    _QUERY_STOP = frozenset("a an and at by for from how in of on or the to what which who why with military actions battle history event political province".split())
    _ACTION_EQUIVALENTS = {
        "assassination": "violent_death", "assassinated": "violent_death", "murder": "violent_death",
        "murdered": "violent_death", "slain": "violent_death", "killed": "violent_death",
    }
    _NON_PLACE_PROPER_NAMES = frozenset({"caesar"})
    _PERSON_NAME_CONTEXT = re.compile(
        r"\b(?:service|army|armies|forces|son|daughter|wife|brother|sister|friend|legate|general|"
        r"command|companions?|followers?|troops|people|party|faction|faction's)\s+of\s+(?:the\s+)?$",
        re.IGNORECASE,
    )
    _UNDER_WITH_PERSON = re.compile(
        r"\b(?:under|with|by)\s+(?:the\s+)?$",
        re.IGNORECASE,
    )

    def __init__(self, mention_extractor: HistoricalPlaceMentionExtractor | None = None,
                 temporal_resolver: EvidenceTemporalResolver | None = None) -> None:
        self.mention_extractor = mention_extractor or HistoricalPlaceMentionExtractor()
        self.temporal_resolver = temporal_resolver or EvidenceTemporalResolver()

    @staticmethod
    def _text(item: Evidence) -> str:
        # ``topic`` is retrieval/navigation metadata, not a primary-source
        # assertion.  Treating it as sentence text made chapter headings such
        # as "Actium" manufacture unrelated events from their paragraphs.
        return " ".join(dict.fromkeys(value for value in (item.text, item.excerpt) if value))

    @classmethod
    def _normalized_terms(cls, value: str | None) -> set[str]:
        normalized = unicodedata.normalize("NFKD", value or "").casefold().replace("æ", "ae").replace("œ", "oe")
        return {
            cls._ACTION_EQUIVALENTS.get(term, term)
            for term in re.findall(r"[a-z][a-z']{2,}", normalized)
            if term not in cls._QUERY_STOP
        }

    @classmethod
    def _is_query_relevant(cls, sentence: str, query_terms: set[str] | None) -> bool:
        if not query_terms:
            return True
        return bool(cls._normalized_terms(sentence) & query_terms)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]

    @classmethod
    def _movement_clauses(cls, sentence: str) -> list[str]:
        return [part.strip() for part in cls._MOVEMENT_CLAUSE_SPLIT.split(sentence) if part.strip()]

    @classmethod
    def _negation_governs_movement_predicate(cls, clause: str, predicate_match: re.Match[str]) -> bool:
        prefix = clause[:predicate_match.start()]
        last_boundary = None
        for match in re.finditer(r"\b(?:and|but)\b", prefix, re.IGNORECASE):
            last_boundary = match
        if last_boundary is not None:
            prefix = prefix[last_boundary.end():]
        tokens = [match.group(0).casefold() for match in re.finditer(r"\b[\w'\u2019]+\b", prefix)]
        if not tokens:
            return False
        for index in range(len(tokens) - 1, max(-1, len(tokens) - 7), -1):
            token = tokens[index]
            gap = len(tokens) - index - 1
            if gap > cls._NEGATION_WINDOW:
                continue
            if token == "never":
                return True
            if token == "not":
                if gap == 0 or (index > 0 and tokens[index - 1] in cls._NEGATED_AUXILIARIES):
                    return True
            if token == "longer" and index > 0 and tokens[index - 1] == "no":
                return True
            if token.endswith("n't"):
                return True
        return False

    @classmethod
    def _clause_has_positive_movement(cls, clause: str) -> bool:
        if not _has_movement_cue(clause) and not cls._MOVEMENT_VERBS.search(clause):
            return False
        matches = list(cls._MOVEMENT_VERBS.finditer(clause))
        if matches:
            return any(not cls._negation_governs_movement_predicate(clause, match) for match in matches)
        return _has_movement_cue(clause)

    @classmethod
    def _has_positive_movement_assertion(cls, sentence: str) -> bool:
        return any(cls._clause_has_positive_movement(clause) for clause in cls._movement_clauses(sentence))

    @classmethod
    def _non_completed_governs_clause(cls, clause: str) -> bool:
        if not cls._NON_COMPLETED.search(clause):
            return False
        if not (_has_movement_cue(clause) or cls._MOVEMENT_VERBS.search(clause)):
            return False
        for match in cls._MOVEMENT_VERBS.finditer(clause):
            if cls._NON_COMPLETED.search(clause[: match.start()]):
                return True
        return bool(_has_movement_cue(clause))

    @classmethod
    def _non_assertive_governs_clause(cls, clause: str) -> bool:
        if not (_has_movement_cue(clause) or cls._MOVEMENT_VERBS.search(clause)):
            return False
        for match in cls._MOVEMENT_VERBS.finditer(clause):
            prefix = clause[: match.start()]
            if cls._NON_ASSERTIVE_GOVERNOR.search(prefix) or cls._PREVENTED_FROM.search(prefix):
                return True
        return bool(
            _has_movement_cue(clause)
            and (cls._NON_ASSERTIVE_GOVERNOR.search(clause) or cls._PREVENTED_FROM.search(clause))
        )

    @classmethod
    def _has_completed_movement_assertion(cls, sentence: str) -> bool:
        return any(
            cls._clause_has_positive_movement(clause)
            and not cls._non_completed_governs_clause(clause)
            and not cls._non_assertive_governs_clause(clause)
            for clause in cls._movement_clauses(sentence)
        )

    def _event_type(self, sentence: str) -> HistoricalEventType:
        # A retrospective reference can name a battle or death while the main
        # assertion describes another event.  Classify the asserted clause.
        lower = self._RETROSPECTIVE.sub("", sentence).lower()
        for event_type, pattern in self._TYPE_PATTERNS:
            if re.search(pattern, lower):
                if event_type is HistoricalEventType.MOVEMENT and not self._has_positive_movement_assertion(sentence):
                    continue
                return event_type
        if _has_movement_cue(sentence) and self._has_positive_movement_assertion(sentence):
            return HistoricalEventType.MOVEMENT
        return HistoricalEventType.UNKNOWN

    @staticmethod
    def _role(token: str) -> EventPlaceRole:
        return {"at": EventPlaceRole.EVENT_SITE, "from": EventPlaceRole.ORIGIN, "to": EventPlaceRole.DESTINATION, "into": EventPlaceRole.DESTINATION}.get(token, EventPlaceRole.RELATED_PLACE)

    @classmethod
    def _clause_start(cls, sentence: str, position: int) -> int:
        return max(sentence.rfind(",", 0, position), sentence.rfind(";", 0, position)) + 1

    @classmethod
    def _movement_clause_boundaries(cls, sentence: str, *, split: re.Pattern[str]) -> list[tuple[int, int]]:
        boundaries: list[tuple[int, int]] = []
        start = 0
        for match in split.finditer(sentence):
            end = match.start()
            if sentence[start:end].strip():
                boundaries.append((start, end))
            start = match.end()
        if sentence[start:].strip():
            boundaries.append((start, len(sentence)))
        return boundaries or [(0, len(sentence))]

    @classmethod
    def _local_clause(cls, sentence: str, position: int) -> tuple[str, int]:
        for start, end in cls._movement_clause_boundaries(sentence, split=cls._POLARITY_CLAUSE_SPLIT):
            if start <= position < end:
                return sentence[start:end], start
        return sentence, 0

    @classmethod
    def _negated_destination_marker(cls, clause: str, marker_start: int, role_token: str) -> bool:
        if role_token not in {"to", "into"}:
            return False
        return bool(re.search(r"\bnot\s+(?:to|into)\s*$", clause[:marker_start], re.IGNORECASE))

    @classmethod
    def _governing_movement_match(cls, clause: str, marker_start: int) -> re.Match[str] | None:
        prefix = clause[:marker_start]
        matches = list(cls._MOVEMENT_VERBS.finditer(prefix))
        if matches:
            return matches[-1]
        governed = [match for match in cls._MOVEMENT_GOVERNED_FROM.finditer(clause) if match.start() < marker_start]
        return governed[-1] if governed else None

    @classmethod
    def _governs_movement_endpoint_clause_local(
        cls, clause: str, marker_start: int, place_end: int, role_token: str,
    ) -> bool:
        prefix = clause[:marker_start]
        local = clause[:marker_start]
        governed = clause[:marker_start + len(role_token)]
        if role_token in {"to", "into"}:
            if re.search(r"\baccording\s+$", prefix, re.IGNORECASE):
                return False
            if cls._NON_MOVEMENT_TO_CONTEXT.search(clause[max(0, marker_start - 60):marker_start]):
                return False
            if cls._ATTRIBUTIVE_AFTER_PLACE.match(clause[place_end:]):
                return False
            return bool(cls._MOVEMENT_TO_PREFIX.search(governed) or cls._MOVEMENT_TO_PREFIX.search(prefix[-80:]))
        if role_token == "from":
            from_window = clause[max(0, marker_start - 60):marker_start + len(role_token)]
            if cls._NON_SPATIAL_FROM.search(from_window):
                return False
            if cls._DISTANCE_FROM.search(from_window):
                return False
            if cls._REFERENCE_FROM.search(from_window):
                return False
            if cls._TROOP_PROVENANCE_FROM.search(from_window):
                return False
            if cls._MOVEMENT_FROM_PREFIX.search(governed) or cls._MOVEMENT_FROM_PREFIX.search(local):
                return True
            mediated_clause = clause[:place_end]
            if cls._MEDIATED_FROM_PREFIX.search(mediated_clause) and cls._MOVEMENT_GOVERNED_FROM.search(mediated_clause):
                return True
            return not local.strip() and bool(cls._MOVEMENT_VERBS.search(clause[place_end:]))
        return True

    @classmethod
    def _positive_movement_governs_endpoint(
        cls, sentence: str, marker_start: int, place_end: int, role_token: str,
    ) -> bool:
        clause, clause_start = cls._local_clause(sentence, marker_start)
        rel_marker = marker_start - clause_start
        rel_place_end = place_end - clause_start
        if cls._negated_destination_marker(clause, rel_marker, role_token):
            return False
        governed = (
            cls._governs_movement_endpoint_clause_local(clause, rel_marker, rel_place_end, role_token)
            or cls._governs_movement_endpoint(sentence, marker_start, place_end, role_token)
        )
        if not governed:
            return False
        governing = cls._governing_movement_match(clause, rel_marker)
        if governing is None:
            prefix = sentence[clause_start:marker_start]
            matches = list(cls._MOVEMENT_VERBS.finditer(prefix))
            governing = matches[-1] if matches else None
        if governing is None:
            tail = cls._MOVEMENT_VERBS.search(sentence[place_end:])
            if tail is not None:
                gov_abs = place_end + tail.start()
                gov_clause, gov_clause_start = cls._local_clause(sentence, gov_abs)
                gov_match = next(
                    (
                        match for match in cls._MOVEMENT_VERBS.finditer(gov_clause)
                        if gov_clause_start + match.start() == gov_abs
                    ),
                    None,
                )
                if gov_match is not None:
                    return not cls._negation_governs_movement_predicate(gov_clause, gov_match)
            return False
        gov_abs = clause_start + governing.start()
        gov_clause, gov_clause_start = cls._local_clause(sentence, gov_abs)
        gov_match = next(
            (
                match for match in cls._MOVEMENT_VERBS.finditer(gov_clause)
                if gov_clause_start + match.start() == gov_abs
            ),
            None,
        )
        if gov_match is None:
            return False
        return not cls._negation_governs_movement_predicate(gov_clause, gov_match)

    @staticmethod
    def _movement_role_marker(
        sentence: str, mention: HistoricalEventPlaceMention,
    ) -> tuple[int, str, int] | None:
        for match in re.finditer(re.escape(mention.raw_text), sentence, re.IGNORECASE):
            place_end = match.end()
            if mention.role is EventPlaceRole.ORIGIN:
                markers = list(re.finditer(r"\bfrom\b", sentence[:match.start()], re.IGNORECASE))
                if markers:
                    marker = markers[-1]
                    return marker.start(), place_end, "from"
            elif mention.role is EventPlaceRole.DESTINATION:
                for pattern in (r"\binto\b", r"\bto\b"):
                    markers = list(re.finditer(pattern, sentence[:match.start()], re.IGNORECASE))
                    if markers:
                        marker = markers[-1]
                        return marker.start(), place_end, marker.group(0).lower()
        return None

    @classmethod
    def _endpoint_has_positive_contradiction(
        cls, sentence: str, marker_start: int, place_end: int, role_token: str,
    ) -> bool:
        clause, clause_start = cls._local_clause(sentence, marker_start)
        rel_marker = marker_start - clause_start
        if cls._negated_destination_marker(clause, rel_marker, role_token) or cls._non_completed_governs_clause(clause):
            return True
        governing = cls._governing_movement_match(clause, rel_marker)
        return governing is not None and cls._negation_governs_movement_predicate(clause, governing)

    def _enforce_movement_endpoint_polarity(
        self, sentence: str, places: list[HistoricalEventPlaceMention],
    ) -> list[HistoricalEventPlaceMention]:
        semantics = analyze_sentence(sentence, self.mention_extractor.aliases_in(sentence))
        role_map = {"origin": EventPlaceRole.ORIGIN, "destination": EventPlaceRole.DESTINATION}
        semantic_directional = {
            (endpoint.surface.casefold(), role_map[endpoint.role])
            for endpoint in semantics.endpoints
            if endpoint.role in role_map
        }
        for edge in semantics.edges:
            for endpoint in (edge.origin, edge.destination):
                if endpoint is not None and endpoint.role in role_map:
                    semantic_directional.add((endpoint.surface.casefold(), role_map[endpoint.role]))
        for mention in places:
            if mention.role not in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}:
                continue
            marker = self._movement_role_marker(sentence, mention)
            if marker is None:
                continue
            if self._positive_movement_governs_endpoint(sentence, *marker):
                continue
            if (
                (mention.raw_text.casefold(), mention.role) in semantic_directional
                and not self._endpoint_has_positive_contradiction(sentence, *marker)
            ):
                continue
            mention.role = EventPlaceRole.RELATED_PLACE
        return places

    @classmethod
    def _governs_movement_endpoint(cls, sentence: str, endpoint_start: int, place_end: int, role_token: str) -> bool:
        prefix = sentence[:endpoint_start]
        local = sentence[cls._clause_start(sentence, endpoint_start):endpoint_start]
        governed = sentence[cls._clause_start(sentence, endpoint_start):endpoint_start + len(role_token)]
        if role_token in {"to", "into"}:
            if re.search(r"\baccording\s+$", prefix, re.IGNORECASE):
                return False
            if cls._NON_MOVEMENT_TO_CONTEXT.search(sentence[max(0, endpoint_start - 60):endpoint_start]):
                return False
            if cls._ATTRIBUTIVE_AFTER_PLACE.match(sentence[place_end:]):
                return False
            return bool(cls._MOVEMENT_TO_PREFIX.search(governed) or cls._MOVEMENT_TO_PREFIX.search(prefix[-80:]))
        if role_token == "from":
            from_window = sentence[max(0, endpoint_start - 60):endpoint_start + len(role_token)]
            if cls._NON_SPATIAL_FROM.search(from_window):
                return False
            if cls._DISTANCE_FROM.search(from_window):
                return False
            if cls._REFERENCE_FROM.search(from_window):
                return False
            if cls._ACCOUNT_FROM.search(sentence):
                return False
            if cls._TROOP_PROVENANCE_FROM.search(from_window):
                return False
            if cls._MOVEMENT_FROM_PREFIX.search(governed) or cls._MOVEMENT_FROM_PREFIX.search(local):
                return True
            mediated_clause = sentence[cls._clause_start(sentence, endpoint_start):place_end]
            if cls._MEDIATED_FROM_PREFIX.search(mediated_clause) and cls._MOVEMENT_GOVERNED_FROM.search(mediated_clause):
                return True
            return not local.strip() and bool(cls._MOVEMENT_VERBS.search(sentence[place_end:]))
        return True

    @classmethod
    def _movement_endpoint_role(cls, sentence: str, match: re.Match[str], role_token: str) -> EventPlaceRole:
        """Keep ORIGIN/DESTINATION only when a movement predicate locally governs the preposition."""
        base = cls._role(role_token)
        if base not in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}:
            return base
        if cls._positive_movement_governs_endpoint(sentence, match.start(), match.end("place"), role_token):
            return base
        return EventPlaceRole.RELATED_PLACE

    @classmethod
    def _is_person_name_context(cls, sentence: str, position: int) -> bool:
        prefix = sentence[:position]
        if cls._PERSON_NAME_CONTEXT.search(prefix):
            return True
        if cls._UNDER_WITH_PERSON.search(prefix):
            return not bool(cls._MOVEMENT_VERBS.search(prefix[-80:]))
        return False

    def _places(self, sentence: str, evidence_id: str) -> list[HistoricalEventPlaceMention]:
        values: list[HistoricalEventPlaceMention] = []
        aliases = self.mention_extractor.aliases_in(sentence)
        alias_by_span = {(alias.lower(), position): place for position, place, alias in aliases}
        for match in self._PLACE_PATTERN.finditer(sentence):
            raw = match.group("place")
            if raw.casefold() in self._NON_PLACE_PROPER_NAMES:
                continue
            if self._is_person_name_context(sentence, match.start("place")):
                continue
            place = alias_by_span.get((raw.lower(), match.start("place")))
            role_token = match.group("role").lower()
            validation = validate_broad_place_mention(
                raw, sentence, match, canonical_hint=place.canonical_name if place else None,
            )
            values.append(HistoricalEventPlaceMention(
                raw_text=raw,
                canonical_hint=place.canonical_name if place else None,
                role=self._movement_endpoint_role(sentence, match, role_token),
                evidence_refs=[evidence_id],
                resolution_status=EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY if place else EventPlaceResolutionStatus.TEXT_ONLY,
                alias_provenance=place.provenance if place else None,
                validation_class=validation.validation_class,
                validation_reason=validation.reason,
            ))
        for position, place, alias in aliases:
            if self._is_person_name_context(sentence, position):
                continue
            if any(item.canonical_hint == place.canonical_name for item in values):
                continue
            values.append(HistoricalEventPlaceMention(
                raw_text=sentence[position:position + len(alias)], canonical_hint=place.canonical_name,
                role=EventPlaceRole.RELATED_PLACE, evidence_refs=[evidence_id],
                resolution_status=EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY,
                alias_provenance=place.provenance,
                validation_class=PlaceMentionValidationClass.GEOGRAPHIC_PLACE_CANDIDATE,
            ))
        # Directional verbs attach roles to their immediately following known
        # place.  This is evidence-local syntax, never list order or geography.
        for position, place, _alias in aliases:
            prefix = sentence[:position]
            role = None
            if re.search(
                r"\b(?:left|leaving|departed(?:\s+from)?|from)\s+(?:the\s+)?(?:"
                r"(?:passage|valley|crossing|banks?|mouth|shores?|foot)\s+of\s+(?:the\s+)?)?$",
                prefix,
                re.IGNORECASE,
            ):
                role = EventPlaceRole.ORIGIN
            elif re.search(r"\b(?:reached|arrived\s+(?:at|in)|came\s+to|entered|passed\s+into|marched?\s+to|marches\s+to|marching\s+to|march\s+to|to|into)\s+(?:the\s+)?$", prefix, re.IGNORECASE):
                role = EventPlaceRole.DESTINATION
            if role is None:
                continue
            if role is EventPlaceRole.ORIGIN:
                from_prep = re.search(
                    r"\bfrom\s+(?:the\s+)?(?:(?:passage|valley|crossing|banks?|mouth|shores?|foot)\s+of\s+(?:the\s+)?)?$",
                    prefix,
                    re.IGNORECASE,
                )
                if from_prep:
                    from_window = sentence[max(0, from_prep.start() - 40):from_prep.end()]
                    if self._TROOP_PROVENANCE_FROM.search(from_window) or not self._positive_movement_governs_endpoint(
                        sentence, from_prep.start(), position, "from"
                    ):
                        continue
            elif role is EventPlaceRole.DESTINATION:
                to_prep = re.search(r"\b(to|into)\s+(?:the\s+)?$", prefix, re.IGNORECASE)
                if to_prep and not self._positive_movement_governs_endpoint(
                    sentence, to_prep.start(), position, to_prep.group(1).lower()
                ):
                    continue
            for item in values:
                if item.canonical_hint == place.canonical_name:
                    item.role = role
        return values

    @staticmethod
    def _place_identity(mention: HistoricalEventPlaceMention) -> str:
        return (mention.canonical_hint or mention.raw_text).casefold()

    def _assign_movement_endpoint_role(
        self,
        places: list[HistoricalEventPlaceMention],
        endpoint: MovementEndpoint,
        role: EventPlaceRole,
        sentence: str,
        evidence_id: str,
    ) -> None:
        if role not in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION, EventPlaceRole.RELATED_PLACE}:
            return
        existing = next(
            (
                item for item in places
                if item.raw_text.casefold() == endpoint.surface.casefold()
                or (endpoint.canonical and item.canonical_hint == endpoint.canonical)
            ),
            None,
        )
        if existing is not None:
            if role in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}:
                existing.role = role
            return
        validation = validate_broad_place_mention(
            endpoint.surface,
            sentence,
            re.search(re.escape(endpoint.surface), sentence, re.IGNORECASE),
        )
        if validation.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE:
            return
        if self._is_person_name_context(sentence, endpoint.position):
            return
        places.append(HistoricalEventPlaceMention(
            raw_text=endpoint.surface,
            canonical_hint=endpoint.canonical,
            role=role,
            evidence_refs=[evidence_id],
            resolution_status=(
                EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY
                if endpoint.canonical
                else EventPlaceResolutionStatus.TEXT_ONLY
            ),
            alias_provenance=None,
            validation_class=validation.validation_class,
            validation_reason=validation.reason,
        ))

    @classmethod
    def _collapse_same_place_origin_destination(cls, places: list[HistoricalEventPlaceMention]) -> list[HistoricalEventPlaceMention]:
        """Fail closed on impossible same-place O/D pairs produced by overlapping heuristics."""
        origin_identities = {cls._place_identity(item) for item in places if item.role is EventPlaceRole.ORIGIN}
        for mention in places:
            if mention.role is EventPlaceRole.DESTINATION and cls._place_identity(mention) in origin_identities:
                mention.role = EventPlaceRole.RELATED_PLACE
        return places

    def _apply_movement_semantics(
        self,
        sentence: str,
        places: list[HistoricalEventPlaceMention],
        evidence_id: str,
        *,
        prior_endpoints: tuple = (),
    ) -> list[HistoricalEventPlaceMention]:
        semantics = analyze_sentence(
            sentence,
            self.mention_extractor.aliases_in(sentence),
            prior_endpoints=prior_endpoints,
        )
        if semantics.should_abstain:
            return places
        role_map = {
            "origin": EventPlaceRole.ORIGIN,
            "destination": EventPlaceRole.DESTINATION,
            "traversal": EventPlaceRole.RELATED_PLACE,
        }
        for endpoint in semantics.endpoints:
            self._assign_movement_endpoint_role(
                places, endpoint, role_map[endpoint.role], sentence, evidence_id,
            )
        for edge in semantics.edges:
            traversal_as_origin = edge.movement_relation in {
                "through_to", "maritime_from_landed", "through_landed",
                "led_through_toward",
            }
            if edge.origin is not None:
                role = role_map[edge.origin.role]
                if traversal_as_origin and edge.origin.role == "traversal" and edge.destination is not None:
                    role = EventPlaceRole.ORIGIN
                self._assign_movement_endpoint_role(
                    places, edge.origin, role, sentence, evidence_id,
                )
            if edge.destination is not None:
                self._assign_movement_endpoint_role(
                    places, edge.destination, role_map[edge.destination.role], sentence, evidence_id,
                )
        return self._collapse_same_place_origin_destination(places)

    def _anaphoric_origin(
        self, previous: str | None, sentence: str, evidence_id: str,
    ) -> HistoricalEventPlaceMention | None:
        """Resolve only an explicit same-sentence regional antecedent.

        The semicolon and directional ``from it/there/that place`` syntax make
        the antecedent evidence-local.  No retrieval order, geography, or
        world knowledge participates in this textual role assignment.
        """
        if not previous or not self._ANAPHORIC_MOVEMENT_FROM.search(sentence):
            return None
        match = self._PRECEDING_REGION.search(previous)
        if not match:
            return None
        raw = match.group("place")
        alias = next(
            (place for _position, place, value in self.mention_extractor.aliases_in(raw) if value.casefold() == raw.casefold()),
            None,
        )
        return HistoricalEventPlaceMention(
            raw_text=raw,
            canonical_hint=alias.canonical_name if alias else None,
            role=EventPlaceRole.ORIGIN,
            evidence_refs=[evidence_id],
            resolution_status=(
                EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY if alias else EventPlaceResolutionStatus.TEXT_ONLY
            ),
            alias_provenance=alias.provenance if alias else None,
        )

    @classmethod
    def _is_relevant_to_query_contexts(cls, sentence: str, contexts: tuple[str, ...] | None) -> bool:
        if not contexts:
            return True
        return any(
            cls._is_query_relevant(sentence, cls._normalized_terms(context) if context and context.strip() else None)
            for context in contexts
        )

    def _eligible(
        self,
        sentence: str,
        event_type: HistoricalEventType,
        query_contexts: tuple[str, ...] | None,
        *,
        sentences: list[str] | None = None,
        index: int = 0,
        evidence_text: str = "",
    ) -> bool:
        if event_type is HistoricalEventType.UNKNOWN:
            return False
        if event_type is HistoricalEventType.MOVEMENT:
            if not self._has_completed_movement_assertion(sentence):
                return False
        elif self._NON_COMPLETED.search(sentence):
            return False
        if self._REPORTED_SPEECH.search(sentence) or self._NAVIGATION_HEADING.search(sentence):
            return False
        # Proper names or a concrete collective/office keep this conservative
        # without requiring a place or a normalized date.
        if not self._proper_tokens(sentence) and not self._ACTOR.search(sentence):
            return False
        if self._is_relevant_to_query_contexts(sentence, query_contexts):
            return True
        if not query_contexts:
            return True
        return movement_eligibility_with_context(
            sentence,
            event_type,
            query_contexts,
            sentences=sentences or [sentence],
            index=index,
            evidence_text=evidence_text,
        )

    @staticmethod
    def _proper_tokens(value: str) -> set[str]:
        return set(re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿÆæŒœ']{2,}", value))

    def extract(
        self,
        evidence: list[Evidence],
        *,
        query: str | None = None,
        query_contexts: tuple[str, ...] | None = None,
    ) -> tuple[list[HistoricalEvent], dict[str, object]]:
        events: list[HistoricalEvent] = []
        temporal_codes: set[str] = set()
        contexts = query_contexts
        if contexts is None and query and query.strip():
            contexts = (query.strip(),)
        for item in evidence:
            item_text = self._text(item)
            sentences = self._sentences(item_text)
            prior_endpoints: tuple = ()
            temporal_context = TemporalResolutionContext()
            for index, sentence in enumerate(sentences):
                event_type = self._event_type(sentence)
                if not self._eligible(
                    sentence,
                    event_type,
                    contexts,
                    sentences=sentences,
                    index=index,
                    evidence_text=item_text,
                ):
                    continue
                places = self._places(sentence, item.id)
                if event_type is HistoricalEventType.MOVEMENT:
                    places = self._apply_movement_semantics(
                        sentence, places, item.id, prior_endpoints=prior_endpoints,
                    )
                origin = (
                    self._anaphoric_origin(sentences[index - 1] if index else None, sentence, item.id)
                    if event_type is HistoricalEventType.MOVEMENT else None
                )
                if origin is not None:
                    places.insert(0, origin)
                places = self._enforce_movement_endpoint_polarity(sentence, places)
                if event_type is HistoricalEventType.MOVEMENT:
                    prior_endpoints = analyze_sentence(
                        sentence,
                        self.mention_extractor.aliases_in(sentence),
                        prior_endpoints=prior_endpoints,
                    ).endpoints
                statement = f"{sentences[index - 1]} {sentence}" if origin is not None else sentence
                digest = hashlib.sha256(f"{item.id}:{index}:{statement}".encode("utf-8")).hexdigest()[:12]
                temporal_readings, codes = self.temporal_resolver.resolve(
                    statement, item.id, context=temporal_context,
                )
                temporal_codes.update(codes)
                temporal = self.temporal_resolver.primary(temporal_readings, item.id)
                events.append(HistoricalEvent(
                    id=f"event-{digest}", name=f"{event_type.value.title()} event", summary=statement,
                    period=item.period, event_type=event_type, temporal_grounding=temporal,
                    place_mentions=places, evidence_refs=[item.id], grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
                    limitations=["Extracted from one explicit evidence statement; no coordinates, chronology merge, or route inference was performed."],
                    candidate_ids=[f"event-{digest}"], source_statements=[statement], temporal_groundings=temporal_readings or [temporal],
                ))
        reason_codes: list[str] = ["EVENT_EXTRACTED"] if events else ["NO_EVENT_EVIDENCE", "INSUFFICIENT_GROUNDING"]
        if events and any(event.temporal_grounding.status is TemporalGroundingStatus.UNRESOLVED for event in events):
            reason_codes.append("TEMPORAL_UNRESOLVED")
        if "TEMPORAL_CONFLICT" in temporal_codes:
            reason_codes.append("TEMPORAL_CONFLICT")
        if events and any(not event.place_mentions for event in events):
            reason_codes.append("PLACE_UNRESOLVED")
        return events, {"evidence_count": len(evidence), "event_count": len(events), "reason_codes": reason_codes}


class HistoricalEventConsolidator:
    """Conservative deterministic grouping of statement candidates, never semantic clustering."""

    @staticmethod
    def _family(event_type: HistoricalEventType) -> str:
        return "MILITARY" if event_type in {HistoricalEventType.BATTLE, HistoricalEventType.MILITARY} else event_type.value

    @staticmethod
    def _place_key(event: HistoricalEvent) -> tuple[str, ...]:
        return tuple(sorted({(item.canonical_hint or item.raw_text).casefold() for item in event.place_mentions}))

    @staticmethod
    def _temporal_key(event: HistoricalEvent) -> str:
        if event.temporal_grounding.status is TemporalGroundingStatus.EVIDENCE_GROUNDED and event.temporal_grounding.normalized_start:
            return f"normalized:{event.temporal_grounding.normalized_start}:{event.temporal_grounding.normalized_end}"
        return (event.temporal_grounding.raw_expression or "<unresolved>").casefold()

    @staticmethod
    def _movement_route_key(event: HistoricalEvent) -> tuple[str, str] | None:
        origin = next((mention for mention in event.place_mentions if mention.role is EventPlaceRole.ORIGIN), None)
        destination = next((mention for mention in event.place_mentions if mention.role is EventPlaceRole.DESTINATION), None)
        if origin is None or destination is None:
            return None
        return (
            (origin.canonical_hint or origin.raw_text).casefold(),
            (destination.canonical_hint or destination.raw_text).casefold(),
        )

    @staticmethod
    def _movement_actor_key(event: HistoricalEvent) -> str:
        statement = (event.source_statements or [event.summary])[0]
        subjects = narrative_subject_proper_nouns(statement)
        if not subjects:
            return "<unknown>"
        if len(subjects) == 1:
            return next(iter(subjects))
        return ",".join(sorted(subjects))

    @staticmethod
    def _occurrence_key(event: HistoricalEvent) -> str:
        statement = (event.source_statements or [event.summary])[0].casefold().strip()
        return hashlib.sha256(statement.encode("utf-8")).hexdigest()[:12]

    def _movement_key(self, event: HistoricalEvent) -> str | None:
        route = self._movement_route_key(event)
        if route is None:
            return None
        origin, destination = route
        return "|".join(
            (
                "MOVEMENT",
                self._movement_actor_key(event),
                f"{origin}>{destination}",
                self._temporal_key(event),
                self._occurrence_key(event),
            )
        )

    def _key(self, event: HistoricalEvent) -> str | None:
        if event.event_type is HistoricalEventType.MOVEMENT:
            return self._movement_key(event)
        places = self._place_key(event)
        if not places:
            return None
        return "|".join((self._family(event.event_type), ",".join(places), self._temporal_key(event)))

    @staticmethod
    def _merged_type(events: list[HistoricalEvent]) -> HistoricalEventType:
        types = {item.event_type for item in events}
        return HistoricalEventType.BATTLE if HistoricalEventType.BATTLE in types else events[0].event_type

    @staticmethod
    def _unique(values):
        return list(dict.fromkeys(values))

    def consolidate(self, candidates: list[HistoricalEvent]) -> tuple[list[HistoricalEvent], dict[str, object]]:
        buckets: dict[str, list[HistoricalEvent]] = {}
        separate: list[HistoricalEvent] = []
        ambiguous = 0
        for candidate in candidates:
            key = self._key(candidate)
            if key is None:
                separate.append(candidate)
                ambiguous += 1
            else:
                buckets.setdefault(key, []).append(candidate)
        consolidated: list[HistoricalEvent] = []
        merged_count = 0
        for key, members in buckets.items():
            if len(members) == 1:
                consolidated.append(members[0].model_copy(update={"identity_key": key}))
                continue
            primary = members[0]
            refs = self._unique(ref for item in members for ref in item.evidence_refs)
            statements = self._unique(statement for item in members for statement in (item.source_statements or [item.summary]))
            places = []
            seen_places = set()
            for item in members:
                for mention in item.place_mentions:
                    marker = (mention.raw_text, mention.canonical_hint, mention.role.value)
                    if marker not in seen_places:
                        places.append(mention)
                        seen_places.add(marker)
            temporal = []
            seen_temporal = set()
            for item in members:
                for grounding in item.temporal_groundings or [item.temporal_grounding]:
                    marker = grounding.model_dump_json()
                    if marker not in seen_temporal:
                        temporal.append(grounding)
                        seen_temporal.add(marker)
            consolidated.append(primary.model_copy(update={
                "id": f"consolidated-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}",
                "event_type": self._merged_type(members), "identity_key": key,
                "candidate_ids": self._unique(identifier for item in members for identifier in (item.candidate_ids or [item.id])),
                "evidence_refs": refs, "source_statements": statements, "place_mentions": places,
                "temporal_groundings": temporal,
                "limitations": self._unique([*primary.limitations, "Consolidated only from candidates with an identical deterministic identity key."]),
            }))
            merged_count += len(members) - 1
        conflicts = 0
        type_sets: dict[tuple[tuple[str, ...], str], set[str]] = {}
        for candidate in candidates:
            place_key = self._place_key(candidate)
            if place_key:
                type_sets.setdefault((place_key, self._temporal_key(candidate)), set()).add(self._family(candidate.event_type))
        conflicts = sum(len(types) > 1 for types in type_sets.values())
        result = [*consolidated, *separate]
        reason_codes = ["EVENTS_CONSOLIDATED"] if merged_count else ["NO_SAFE_EVENT_MERGE"]
        if ambiguous:
            reason_codes.append("EVENT_IDENTITY_AMBIGUOUS")
        if conflicts:
            reason_codes.append("EVENT_TYPE_CONFLICT")
        temporal_conflicts = sum(len({self._temporal_key(item) for item in candidates if self._place_key(item) == place_key and self._family(item.event_type) == family}) > 1 for place_key, family in {(self._place_key(item), self._family(item.event_type)) for item in candidates if self._place_key(item)})
        if temporal_conflicts:
            reason_codes.append("TEMPORAL_CONFLICT")
        return result, {
            "candidate_event_count": len(candidates), "consolidated_event_count": len(result),
            "merged_candidate_count": merged_count, "ambiguous_candidate_count": ambiguous,
            "conflicting_candidate_count": conflicts + temporal_conflicts, "reason_codes": reason_codes,
        }
