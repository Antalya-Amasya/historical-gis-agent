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
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.temporal import EvidenceTemporalResolver


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
        (HistoricalEventType.MOVEMENT, r"\b(?:marched|marches|marching|march|advanced|proceeded|moved|travelled|traveled|departed|arrived|entered|crossed|withdrew|retreated)\b"),
        (HistoricalEventType.MILITARY, r"\b(?:campaign|army|war|invaded|conquered|captured)\b"),
        (HistoricalEventType.POLITICAL, r"\b(?:senate .*\bdecree|tribune .*\b(?:proposed|elected|opposed)|consul .*\b(?:appointed|elected|sent)|assembly .*\b(?:elected|passed)|issued a decree)\b"),
    )
    _PLACE_PATTERN = re.compile(r"\b(?P<role>(?i:at|in|near|from|to|into|through))\s+(?:(?i:the)\s+)?(?P<place>[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})")
    _RETROSPECTIVE = re.compile(
        r"\b(?:after|following|because of|since)\s+(?:the\s+)?(?:battle|defeat|death|murder|assassination)\b[^,;:.]*[,;:]?\s*",
        re.IGNORECASE,
    )
    _NON_COMPLETED = re.compile(
        r"\b(?:would|could|might|should|may|planned\s+to|intended\s+to|wanted\s+to|hoped\s+to|feared\s+(?:that|lest)|if)\b",
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
    _QUERY_STOP = frozenset("a an and at by for from how in of on or the to what which who why with military actions battle history event political province".split())
    _ACTION_EQUIVALENTS = {
        "assassination": "violent_death", "assassinated": "violent_death", "murder": "violent_death",
        "murdered": "violent_death", "slain": "violent_death", "killed": "violent_death",
    }
    # A narrow corpus-observed false positive: the prepositional place pattern
    # once interpreted the author/actor Caesar as an EVENT_SITE.  This is an
    # exact exclusion, not a general person/entity inference layer.
    _NON_PLACE_PROPER_NAMES = frozenset({"caesar"})

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

    def _event_type(self, sentence: str) -> HistoricalEventType:
        # A retrospective reference can name a battle or death while the main
        # assertion describes another event.  Classify the asserted clause.
        lower = self._RETROSPECTIVE.sub("", sentence).lower()
        for event_type, pattern in self._TYPE_PATTERNS:
            if re.search(pattern, lower):
                return event_type
        return HistoricalEventType.UNKNOWN

    @staticmethod
    def _role(token: str) -> EventPlaceRole:
        return {"at": EventPlaceRole.EVENT_SITE, "from": EventPlaceRole.ORIGIN, "to": EventPlaceRole.DESTINATION, "into": EventPlaceRole.DESTINATION}.get(token, EventPlaceRole.RELATED_PLACE)

    def _places(self, sentence: str, evidence_id: str) -> list[HistoricalEventPlaceMention]:
        values: list[HistoricalEventPlaceMention] = []
        aliases = self.mention_extractor.aliases_in(sentence)
        alias_by_span = {(alias.lower(), position): place for position, place, alias in aliases}
        for match in self._PLACE_PATTERN.finditer(sentence):
            raw = match.group("place")
            if raw.casefold() in self._NON_PLACE_PROPER_NAMES:
                continue
            place = alias_by_span.get((raw.lower(), match.start("place")))
            values.append(HistoricalEventPlaceMention(
                raw_text=raw,
                canonical_hint=place.canonical_name if place else None,
                role=self._role(match.group("role").lower()),
                evidence_refs=[evidence_id],
                resolution_status=EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY if place else EventPlaceResolutionStatus.TEXT_ONLY,
                alias_provenance=place.provenance if place else None,
            ))
        for position, place, alias in aliases:
            if any(item.canonical_hint == place.canonical_name for item in values):
                continue
            values.append(HistoricalEventPlaceMention(
                raw_text=sentence[position:position + len(alias)], canonical_hint=place.canonical_name,
                role=EventPlaceRole.RELATED_PLACE, evidence_refs=[evidence_id],
                resolution_status=EventPlaceResolutionStatus.NORMALIZED_TEXT_ONLY,
                alias_provenance=place.provenance,
            ))
        # Directional verbs attach roles to their immediately following known
        # place.  This is evidence-local syntax, never list order or geography.
        for position, place, _alias in aliases:
            prefix = sentence[:position]
            role = None
            if re.search(r"\b(?:left|leaving|departed(?:\s+from)?|from)\s+(?:the\s+)?$", prefix, re.IGNORECASE):
                role = EventPlaceRole.ORIGIN
            elif re.search(r"\b(?:reached|arrived\s+(?:at|in)|came\s+to|entered|passed\s+into|marched?\s+to|marches\s+to|marching\s+to|march\s+to|to|into)\s+(?:the\s+)?$", prefix, re.IGNORECASE):
                role = EventPlaceRole.DESTINATION
            if role is None:
                continue
            for item in values:
                if item.canonical_hint == place.canonical_name:
                    item.role = role
        return values

    @classmethod
    def _is_relevant_to_query_contexts(cls, sentence: str, contexts: tuple[str, ...] | None) -> bool:
        if not contexts:
            return True
        return any(
            cls._is_query_relevant(sentence, cls._normalized_terms(context) if context and context.strip() else None)
            for context in contexts
        )

    def _eligible(self, sentence: str, event_type: HistoricalEventType, query_contexts: tuple[str, ...] | None) -> bool:
        if event_type is HistoricalEventType.UNKNOWN:
            return False
        if self._NON_COMPLETED.search(sentence) or self._REPORTED_SPEECH.search(sentence) or self._NAVIGATION_HEADING.search(sentence):
            return False
        # Proper names or a concrete collective/office keep this conservative
        # without requiring a place or a normalized date.
        if not self._proper_tokens(sentence) and not self._ACTOR.search(sentence):
            return False
        return self._is_relevant_to_query_contexts(sentence, query_contexts)

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
            for index, sentence in enumerate(self._sentences(self._text(item))):
                event_type = self._event_type(sentence)
                if not self._eligible(sentence, event_type, contexts):
                    continue
                digest = hashlib.sha256(f"{item.id}:{index}:{sentence}".encode("utf-8")).hexdigest()[:12]
                places = self._places(sentence, item.id)
                temporal_readings, codes = self.temporal_resolver.resolve(sentence, item.id)
                temporal_codes.update(codes)
                temporal = self.temporal_resolver.primary(temporal_readings, item.id)
                events.append(HistoricalEvent(
                    id=f"event-{digest}", name=f"{event_type.value.title()} event", summary=sentence,
                    period=item.period, event_type=event_type, temporal_grounding=temporal,
                    place_mentions=places, evidence_refs=[item.id], grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
                    limitations=["Extracted from one explicit evidence statement; no coordinates, chronology merge, or route inference was performed."],
                    candidate_ids=[f"event-{digest}"], source_statements=[sentence], temporal_groundings=temporal_readings or [temporal],
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

    def _key(self, event: HistoricalEvent) -> str | None:
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
