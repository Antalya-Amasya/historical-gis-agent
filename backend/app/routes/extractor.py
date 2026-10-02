"""Evidence -> explicit movement claims -> MCP-resolved schematic routes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from backend.app.geography.feature_semantics import simulation_anchor_eligible
from backend.app.models import ExtractedHistoricalPlaceMention, GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint
from backend.app.routes.episode_relevance import filter_legacy_movement_claims
from backend.app.routes.evidence_relevance import (
    _named_proper_nouns,
    _spatial_role_proper_nouns,
    narrative_subject_proper_nouns,
    normalize_subject_name,
    normalized_terms,
)
from backend.app.routes.movement_semantics import MovementEdgeCandidate, _DISCOURSE_ANAPHORA, analyze_sentence
from backend.app.routes.place_aliases import HISTORICAL_PLACE_ALIASES, HistoricalPlaceAlias


class GeographyResolver(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


@dataclass(frozen=True)
class RouteBuildOutcome:
    route: HistoricalRoute | None
    diagnostics: dict[str, object]


def evidence_structural_key(item: Evidence) -> tuple[str, int, int] | None:
    """Only structure-derived source positions may order a connected chain."""
    document_id = str(item.metadata.get("document_id") or item.source_file or "")
    spine_index, start_offset = item.metadata.get("spine_index"), item.metadata.get("start_offset")
    if not document_id or not isinstance(spine_index, int) or not isinstance(start_offset, int):
        return None
    return document_id, spine_index, start_offset


def _legacy_od_has_positive_authority(
    sentence: str,
    source_place: str,
    destination_place: str,
    *,
    mention_extractor: HistoricalPlaceMentionExtractor | None = None,
    semantic_edge: MovementEdgeCandidate | None = None,
    source_surface: str | None = None,
    destination_surface: str | None = None,
) -> bool:
    from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor

    if not EvidenceGroundedHistoricalEventExtractor._has_positive_movement_assertion(sentence):
        return False

    mention_extractor = mention_extractor or HistoricalPlaceMentionExtractor()

    def place_names(*names: str | None) -> set[str]:
        tokens: set[str] = set()
        for name in names:
            if name:
                tokens.add(" ".join(name.casefold().split()))
        return tokens

    def endpoint_tokens(endpoint) -> set[str]:
        if endpoint is None:
            return set()
        return place_names(endpoint.canonical, endpoint.surface)

    source_tokens = place_names(source_place, source_surface)
    dest_tokens = place_names(destination_place, destination_surface)

    def completed_clause(clause: str) -> bool:
        return (
            EvidenceGroundedHistoricalEventExtractor._clause_has_positive_movement(clause)
            and not EvidenceGroundedHistoricalEventExtractor._non_completed_governs_clause(clause)
            and not EvidenceGroundedHistoricalEventExtractor._non_assertive_governs_clause(clause)
        )

    positive_clauses = [
        clause
        for clause in EvidenceGroundedHistoricalEventExtractor._movement_clauses(sentence)
        if completed_clause(clause)
    ]
    if not positive_clauses:
        return False

    all_clauses = EvidenceGroundedHistoricalEventExtractor._movement_clauses(sentence)
    positive_indexes = [
        index
        for index, clause in enumerate(all_clauses)
        if completed_clause(clause)
    ]

    def directed_edge_in_clause(clause: str) -> bool:
        for edge in analyze_sentence(clause, mention_extractor.aliases_in(clause)).edges:
            origin = edge.origin or edge.traversal
            destination = edge.destination
            if origin is None or destination is None:
                continue
            if source_tokens & endpoint_tokens(origin) and dest_tokens & endpoint_tokens(destination):
                return True
        return False

    occurrence_boundary = re.compile(
        r"\b(?:"
        r"years?\s+later|much\s+later|long\s+after|decades?\s+later|centuries?\s+later|"
        r"in\s+(?:a\s+)?(?:different|later|separate|subsequent)\s+(?:campaign|war|expedition|episode)|"
        r"in\s+another\s+(?:campaign|war|expedition|episode)|"
        r"(?:different|another|later|separate|subsequent)\s+(?:campaign|war|expedition|episode)"
        r")\b",
        re.IGNORECASE,
    )

    for clause in positive_clauses:
        if occurrence_boundary.search(clause):
            continue
        if directed_edge_in_clause(clause):
            return True

    def clause_actor_tokens(clause: str) -> set[str]:
        spatial = _spatial_role_proper_nouns(clause)
        place_tokens = set(normalized_terms(" ".join(spatial)))
        for _position, place, alias in mention_extractor.aliases_in(clause):
            place_tokens |= normalized_terms(place.canonical_name)
            place_tokens |= normalized_terms(alias)
        actors = {
            normalize_subject_name(name)
            for name in _named_proper_nouns(clause)
            if name not in spatial and normalize_subject_name(name) not in place_tokens
        }
        actors |= {
            actor
            for actor in narrative_subject_proper_nouns(clause)
            if actor not in place_tokens
        }
        return {token for token in actors if token not in {"bce", "bc", "ce", "ad"}}

    def subjects_allow_cross_clause_continuation(left: str, right: str) -> bool:
        left_actors = clause_actor_tokens(left)
        right_actors = clause_actor_tokens(right)
        if left_actors and right_actors:
            return bool(left_actors & right_actors)
        if right_actors and not left_actors:
            return False
        return True

    # Validate the established bounded edge; do not reconstruct it with a
    # different alias environment or lose its cross-clause relation.
    edges = (semantic_edge,) if semantic_edge else analyze_sentence(
        sentence, mention_extractor.aliases_in(sentence),
    ).edges
    for edge in edges:
        if edge.movement_relation not in {"crossing_arrival", "crossing_into", "thence_passed_on_to", "departure_arrival"}:
            continue
        origin = edge.traversal or edge.origin
        destination = edge.destination
        if origin is None or destination is None:
            continue
        origin_tokens = endpoint_tokens(origin)
        destination_tokens = endpoint_tokens(destination)
        if not (source_tokens & origin_tokens and dest_tokens & destination_tokens):
            continue
        for left in positive_indexes:
            for right in positive_indexes:
                if right <= left:
                    continue
                window = all_clauses[left:right + 1]
                if not all(completed_clause(clause) for clause in window):
                    continue
                if any(occurrence_boundary.search(clause) for clause in window[1:]):
                    continue
                if not (
                    re.search(rf"(?<!\w){re.escape(origin.surface)}(?!\w)", window[0], re.I)
                    and re.search(rf"(?<!\w){re.escape(destination.surface)}(?!\w)", window[-1], re.I)
                ):
                    continue
                if not all(subjects_allow_cross_clause_continuation(a, b)
                           for a, b in zip(window, window[1:])):
                    continue
                return True
    return False


class HistoricalPlaceMentionExtractor:
    """Literal place mentions for display/audit; mentions alone never create routes."""

    def __init__(self, aliases: tuple[HistoricalPlaceAlias, ...] = HISTORICAL_PLACE_ALIASES):
        self.aliases = aliases

    @staticmethod
    def _text(item: Evidence) -> str:
        return " ".join(dict.fromkeys(value for value in (item.text, item.excerpt, item.topic) if value))

    def aliases_in(self, text: str) -> list[tuple[int, HistoricalPlaceAlias, str]]:
        lower = text.lower()
        candidates: list[tuple[int, int, HistoricalPlaceAlias, str]] = []
        for place in self.aliases:
            for alias in place.aliases:
                for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", lower):
                    candidates.append((match.start(), match.end(), place, alias))
        if not candidates:
            return []
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[3], item[2].canonical_name))
        accepted: list[tuple[int, int, HistoricalPlaceAlias, str]] = []
        for candidate in candidates:
            start, end, place, alias = candidate
            if any(start < other_end and end > other_start for other_start, other_end, _, _ in accepted):
                continue
            accepted.append(candidate)
        accepted.sort(key=lambda item: item[0])
        return [(start, place, alias) for start, _, place, alias in accepted]

    def extract(self, evidence: list[Evidence]) -> list[ExtractedHistoricalPlaceMention]:
        grouped: dict[str, ExtractedHistoricalPlaceMention] = {}
        for evidence_index, item in enumerate(evidence):
            text = self._text(item)
            for position, place, alias in self.aliases_in(text):
                existing = grouped.get(place.canonical_name)
                if existing is None:
                    start, end = max(0, position - 80), min(len(text), position + len(alias) + 160)
                    grouped[place.canonical_name] = ExtractedHistoricalPlaceMention(raw_name=text[position:position + len(alias)], normalized_name=place.canonical_name, sequence_hint=evidence_index * 100000 + position, date_or_period=item.period, evidence_refs=[item.id], context_excerpt=text[start:end], confidence=0.8, alias_provenance=place.provenance)
                elif item.id not in existing.evidence_refs:
                    existing.evidence_refs.append(item.id)
        return sorted(grouped.values(), key=lambda mention: mention.sequence_hint)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n+", text) if part.strip()]

    @staticmethod
    def _place_after(start: int, aliases: list[tuple[int, HistoricalPlaceAlias, str]], *, before: int | None = None) -> HistoricalPlaceAlias | None:
        candidates = [place for position, place, _ in aliases if position >= start and (before is None or position < before)]
        return candidates[0] if candidates else None

    def continuation_endpoints(self, previous: str | None, sentence: str, semantics, *, evidence_id: str) -> tuple:
        """Resume only a completed destination in the adjacent evidence statement.

        This supplies location bindings, never a pronoun's actor authority.
        """
        from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
        from backend.app.models import EventActorStatus

        if not previous or semantics is None or semantics.should_abstain or not _DISCOURSE_ANAPHORA.search(sentence):
            return ()
        extractor = EvidenceGroundedHistoricalEventExtractor(self)
        for text in (previous, sentence):
            if (not extractor._has_completed_movement_assertion(text)
                    or extractor._incomplete_movement_fact(text, evidence_id) is not None
                    or extractor._REPORTED_SPEECH.search(text)
                    or re.search(r"\b(?:years?\s+later|decades?\s+later|long\s+after|"
                                 r"(?:new|different|another|later|separate|subsequent)\s+(?:campaign|war|episode|expedition))\b", text, re.I)):
                return ()
        # A named place omitted by the semantic parse is a competing endpoint,
        # not permission to pick the parser's first destination in a list.
        current_semantics = analyze_sentence(sentence, self.aliases_in(sentence))
        for text, parsed in ((previous, semantics), (sentence, current_semantics)):
            if parsed.should_abstain:
                return ()
            identities = {endpoint.place_name.casefold() for endpoint in parsed.endpoints}
            if any(place.canonical_name.casefold() not in identities
                   for _start, place, _surface in self.aliases_in(text)):
                return ()
        previous_actor = extractor._ground_movement_actor(previous)
        current_actor = extractor._ground_movement_actor(sentence)
        if current_actor.actor_status is EventActorStatus.EXPLICIT:
            if (previous_actor.actor_status is not EventActorStatus.EXPLICIT
                    or normalize_subject_name(current_actor.actor_text or "") != normalize_subject_name(previous_actor.actor_text or "")):
                return ()
        # Other named people are competing referents, not a reason to guess.
        for text, actor in ((previous, previous_actor), (sentence, current_actor)):
            masked = text
            if actor.actor_span:
                start, end = actor.actor_span
                masked = masked[:start] + " " * (end - start) + masked[end:]
            for start, _place, surface in reversed(self.aliases_in(masked)):
                masked = masked[:start] + " " * len(surface) + masked[start + len(surface):]
            masked = _DISCOURSE_ANAPHORA.sub("", masked)
            if _named_proper_nouns(masked) - _spatial_role_proper_nouns(masked):
                return ()
        destinations = {endpoint.place_name.casefold(): endpoint
                        for endpoint in semantics.endpoints if endpoint.role == "destination"}
        return tuple(destinations.values()) if len(destinations) == 1 else ()

    def movement_claims(self, evidence: list[Evidence], *, event_id: str) -> list[HistoricalClaim]:
        """Extract explicit movement claims from evidence-local movement semantics.

        Traversal-only statements retain evidence provenance but cannot form a route edge.
        """
        from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor

        claims: list[HistoricalClaim] = []
        event_extractor = EvidenceGroundedHistoricalEventExtractor(self)
        claim_number = 0
        for item in evidence:
            document = str(item.metadata.get("document_id") or item.source_file or item.author)
            text = self._text(item)
            previous, previous_semantics, previous_end = None, None, 0
            for sentence in self._sentences(text):
                start = text.find(sentence, previous_end)
                if "\n" in text[previous_end:start]:
                    previous = None
                prior_endpoints = self.continuation_endpoints(previous, sentence, previous_semantics, evidence_id=item.id)
                semantics = analyze_sentence(sentence, self.aliases_in(sentence), prior_endpoints=prior_endpoints)
                basis = f"{previous} {sentence}" if any(edge.cross_sentence_link for edge in semantics.edges) else sentence
                previous, previous_semantics, previous_end = sentence, semantics, start + len(sentence)
                incomplete = event_extractor._incomplete_movement_fact(sentence, item.id)
                for edge in semantics.edges:
                    if incomplete is not None:
                        bounded_end = len(incomplete.source_statement)
                        edge_endpoints = (edge.origin or edge.traversal, edge.destination)
                        if any(endpoint and endpoint.position < bounded_end for endpoint in edge_endpoints):
                            continue
                    claim_number += 1
                    if edge.origin and edge.destination:
                        if not edge.cross_sentence_link and not _legacy_od_has_positive_authority(
                            sentence,
                            edge.origin.place_name,
                            edge.destination.place_name,
                            mention_extractor=self,
                            semantic_edge=edge,
                            source_surface=edge.origin.surface,
                            destination_surface=edge.destination.surface,
                        ):
                            continue
                        claims.append(HistoricalClaim(
                            id=f"{event_id}-movement-{claim_number}",
                            claim_type="MOVEMENT",
                            text=basis,
                            textual_basis=basis,
                            source_place=edge.origin.place_name,
                            destination_place=edge.destination.place_name,
                            movement_relation=edge.movement_relation,
                            sequence_status="explicit",
                            supporting_evidence_ids=[item.id],
                            source_documents=[document],
                            confidence=0.9,
                        ))
                        continue
                    if edge.traversal and edge.destination and edge.movement_relation in {
                        "crossing_into", "crossing_arrival",
                    }:
                        if not _legacy_od_has_positive_authority(
                            sentence,
                            edge.traversal.place_name,
                            edge.destination.place_name,
                            mention_extractor=self,
                            semantic_edge=edge,
                            source_surface=edge.traversal.surface,
                            destination_surface=edge.destination.surface,
                        ):
                            continue
                        claims.append(HistoricalClaim(
                            id=f"{event_id}-movement-{claim_number}",
                            claim_type="MOVEMENT",
                            text=sentence,
                            textual_basis=sentence,
                            source_place=edge.traversal.place_name,
                            destination_place=edge.destination.place_name,
                            movement_relation=edge.movement_relation,
                            sequence_status="explicit",
                            supporting_evidence_ids=[item.id],
                            source_documents=[document],
                            confidence=0.9,
                        ))
                        continue
                    if edge.traversal and edge.movement_relation == "traversal":
                        claims.append(HistoricalClaim(
                            id=f"{event_id}-movement-{claim_number}",
                            claim_type="MOVEMENT",
                            text=sentence,
                            textual_basis=sentence,
                            traversed_place=edge.traversal.place_name,
                            movement_relation="traversal",
                            sequence_status="unordered",
                            supporting_evidence_ids=[item.id],
                            source_documents=[document],
                            confidence=0.8,
                        ))
        return claims


class HistoricalRouteExtractor:
    """Build a schematic route only from explicit movement-claim edges."""

    def __init__(self, geography_client: GeographyResolver, mention_extractor: HistoricalPlaceMentionExtractor | None = None):
        self.geography_client = geography_client
        self.mention_extractor = mention_extractor or HistoricalPlaceMentionExtractor()

    _evidence_order_key = staticmethod(evidence_structural_key)

    def build(self, evidence: list[Evidence], *, event_id: str, name: str, period: str) -> HistoricalRoute | None:
        return self.build_with_diagnostics(evidence, event_id=event_id, name=name, period=period).route

    def build_with_diagnostics(
        self,
        evidence: list[Evidence],
        *,
        event_id: str,
        name: str,
        period: str,
        query_contexts: tuple[str, ...] | None = None,
    ) -> RouteBuildOutcome:
        mentions = self.mention_extractor.extract(evidence)
        claims = self.mention_extractor.movement_claims(evidence, event_id=event_id)
        episode_diagnostics: list[dict[str, object]] = []
        if query_contexts:
            claims, episode_diagnostics = filter_legacy_movement_claims(claims, evidence, query_contexts)
        ordered = [claim for claim in claims if claim.sequence_status == "explicit" and claim.source_place and claim.destination_place]
        diagnostics: dict[str, object] = {
            "evidence_count": len(evidence),
            "recognized_place_mentions": len(mentions),
            "movement_claim_count": len(claims),
            "legacy_episode_admission": episode_diagnostics,
            "explicit_edge_count": len(ordered),
            "connected_edge_count": 0,
            "unresolved_anchor_count": 0,
            "route_point_count": 0,
            "reason_codes": [],
        }
        if not ordered:
            diagnostics["reason_codes"] = (
                ["NO_EPISODE_RELEVANT_LEGACY_CLAIMS"]
                if query_contexts and episode_diagnostics and not any(item.get("admitted") for item in episode_diagnostics)
                else ["NO_MOVEMENT_CLAIMS"]
            )
            return RouteBuildOutcome(None, diagnostics)
        evidence_by_id = {item.id: item for item in evidence}
        claim_keys = {
            claim.id: self._evidence_order_key(evidence_by_id[claim.supporting_evidence_ids[0]])
            for claim in ordered
            if claim.supporting_evidence_ids and claim.supporting_evidence_ids[0] in evidence_by_id
        }
        can_chain = (
            len(claim_keys) == len(ordered)
            and len({key[0] for key in claim_keys.values() if key}) == 1
            and len(set(claim_keys.values())) == len(claim_keys)
            and all(claim_keys.values())
        )
        if can_chain:
            ordered = sorted(ordered, key=lambda claim: claim_keys[claim.id])
        elif len(ordered) > 1:
            diagnostics["reason_codes"] = ["DISCONNECTED_EVIDENCE"]
        anchor_names: list[str] = []
        anchor_claims: dict[str, list[HistoricalClaim]] = {}
        for claim in ordered:
            if not anchor_names:
                anchor_names.extend([claim.source_place, claim.destination_place])
                diagnostics["connected_edge_count"] = 1
            elif can_chain and anchor_names[-1] == claim.source_place:
                anchor_names.append(claim.destination_place)
                diagnostics["connected_edge_count"] = int(diagnostics["connected_edge_count"]) + 1
            else:
                continue  # disconnected evidence cannot be silently fused into a route
            for place_name in (claim.source_place, claim.destination_place):
                anchor_claims.setdefault(place_name, []).append(claim)
        if len(anchor_names) < 2:
            diagnostics["reason_codes"] = ["NO_EXPLICIT_EDGES"]
            return RouteBuildOutcome(None, diagnostics)
        points: list[HistoricalRoutePoint] = []
        for place_name in anchor_names:
            supporting_claims = anchor_claims[place_name]
            resolved = self.geography_client.call("resolve_ancient_place", {"name": place_name, "period": period})
            if not resolved.get("found"):
                diagnostics["unresolved_anchor_count"] = int(diagnostics["unresolved_anchor_count"]) + 1
                diagnostics["reason_codes"] = ["UNRESOLVED_ANCHOR"]
                return RouteBuildOutcome(None, diagnostics)  # required evidence-grounded anchor cannot receive an invented coordinate
            place = HistoricalPlace.model_validate({key: value for key, value in resolved.items() if key != "found"})
            if not simulation_anchor_eligible(place):
                diagnostics["reason_codes"] = ["NON_EXACT_ROUTE_POINT"]
                return RouteBuildOutcome(None, diagnostics)
            refs = list(dict.fromkeys(ref for claim in supporting_claims for ref in claim.supporting_evidence_ids))
            sources = [item.author for item in evidence if item.id in refs]
            points.append(HistoricalRoutePoint(sequence=len(points) + 1, historical_place=place, event_summary=supporting_claims[0].textual_basis or supporting_claims[0].text, date_or_period=period, evidence_refs=refs, confidence=min(min(claim.confidence for claim in supporting_claims), place.confidence), coordinate_role=place.coordinate_role, source_support=list(dict.fromkeys(sources)), claim_ids=[claim.id for claim in supporting_claims]))
        coordinates = [(point.historical_place.longitude, point.historical_place.latitude) for point in points]
        diagnostics["route_point_count"] = len(points)
        route_claims = [claim for claim in claims if claim.sequence_status == "explicit" and claim.source_place and claim.destination_place]
        return RouteBuildOutcome(HistoricalRoute(id=f"{event_id}-evidence-route", event_id=event_id, name=name, period=period, ordered_points=points, geometry=GeoJsonLineString(coordinates=coordinates), evidence_refs=list(dict.fromkeys(ref for claim in ordered for ref in claim.supporting_evidence_ids)), assumptions=["Evidence-grounded regional anchors are connected with schematic straight segments."], limitations=["Historical reconstruction only; not an exact march track or road route.", "Geometry is a model-derived connection between evidence-grounded anchors, not historical track evidence.", "Representative river, mountain, or regional coordinates are not exact passage locations."], historical_confidence=round(sum(point.confidence for point in points) / len(points), 2), claims=route_claims), diagnostics)
