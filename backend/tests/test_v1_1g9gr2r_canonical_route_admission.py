"""V1.1G9G-R2R: canonical query-route relation admission."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    Evidence,
    GeographicFeatureKind,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_constraints import project_transition_constraints
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.query_route_admission import (
    AuthorityState,
    QueryRouteRelationAdmission,
    classify_observation_relation_admission,
    parse_query_route_scope,
)
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
    RouteObservationKind,
    project_observation_ordering,
)

GENERIC_QUERY = "Trace Commander Alpha's route from Port A to City B during Campaign One."
BROAD_QUERY = "Show Commander Alpha's historical route."
HANNIBAL_QUERY = "Trace Hannibal's route from New Carthage toward Italy during the Second Punic War."

GENERIC_ALIASES = (
    HistoricalPlaceAlias("Port A", ("port a",), "audited"),
    HistoricalPlaceAlias("City B", ("city b",), "audited"),
    HistoricalPlaceAlias("City Z", ("city z",), "audited"),
    HistoricalPlaceAlias("River X", ("river x",), "audited"),
)

HANNIBAL_ALIASES = GENERIC_ALIASES + (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
    HistoricalPlaceAlias("Ephesus", ("ephesus",), "audited"),
    HistoricalPlaceAlias("Graecia", ("greece", "graecia"), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(HistoricalPlaceMentionExtractor(GENERIC_ALIASES))
_HANNIBAL_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES))


def _evidence(text: str, *, eid: str = "ev", offset: int = 100) -> Evidence:
    return Evidence(
        id=eid,
        author="Fixture",
        work="Test",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": offset},
    )


class _GenericGeography:
    _PLACES = {
        "port a": ("Port A", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 40.0, 10.0),
        "city b": ("City B", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 41.0, 11.0),
        "city z": ("City Z", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 42.0, 12.0),
        "river x": ("River X", PlaceSpatialSemantics.RIVER, "representative_point", 40.5, 10.5),
    }

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        key = arguments["name"].casefold()
        for alias, (name, semantics, role, lat, lon) in self._PLACES.items():
            if key in {alias, name.casefold()}:
                place = HistoricalPlace(
                    id=alias.replace(" ", "-"),
                    canonical_name=name,
                    latitude=lat,
                    longitude=lon,
                    source="fixture",
                    confidence=0.8,
                    spatial_semantics=semantics,
                    coordinate_role=role,
                )
                payload = place.model_dump()
                payload["found"] = True
                return payload
        return {"found": False}


class _HannibalGeography(_GenericGeography):
    _PLACES = {
        **_GenericGeography._PLACES,
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "island": ("Island", PlaceSpatialSemantics.ISLAND, "feature_centroid", 39.0, 9.0),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
        "ephesus": ("Ephesus", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.94, 27.34),
        "graecia": ("Graecia", PlaceSpatialSemantics.REGION, "regional_centroid", 39.0, 22.0),
    }


def _observation(
    observation_id: str,
    *,
    label: str,
    actor_text: str | None,
    actor_status: EventActorStatus,
    event_id: str,
    evidence_refs: tuple[str, ...],
    kind: RouteObservationKind = RouteObservationKind.PLACE,
    place_role: EventPlaceRole | None = None,
) -> RouteObservation:
    return RouteObservation(
        observation_id=observation_id,
        kind=kind,
        event_id=event_id,
        label=label,
        actor_text=actor_text,
        actor_status=actor_status,
        evidence_refs=evidence_refs,
        place_role=place_role,
        feature_kind=GeographicFeatureKind.RIVER if kind is RouteObservationKind.TRANSITION else None,
    )


def _relation(
    earlier: str,
    later: str,
    *,
    event_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=event_ids,
        evidence_refs=evidence_refs,
        authority=authority.value,
    )


def _pipeline(text: str, *, query: str = GENERIC_QUERY, extractor=_EXTRACTOR, geography=_GenericGeography()):
    evidence = _evidence(text)
    events, _ = extractor.extract([evidence], query_contexts=(query,))
    resolved, _ = HistoricalEventPlaceResolver(geography).resolve(events)
    constraints, _ = project_transition_constraints(resolved, [evidence])
    observations, relations, _ = project_observation_ordering(resolved, [], constraints, [evidence])
    return observations, relations, resolved, [evidence], (query,)


def _admission_for_labels(
    text: str,
    earlier_label: str,
    later_label: str,
    *,
    query: str = GENERIC_QUERY,
    actor_text: str = "Commander Alpha",
    actor_status: EventActorStatus = EventActorStatus.EXPLICIT,
    extractor=_EXTRACTOR,
    geography=_GenericGeography,
) -> QueryRouteRelationAdmission | None:
    observations, relations, resolved, evidence, contexts = _pipeline(
        text, query=query, extractor=extractor, geography=geography(),
    )
    if relations:
        obs_by = {item.observation_id: item for item in observations}
        for relation in relations:
            if (
                obs_by[relation.earlier_observation_id].label == earlier_label
                and obs_by[relation.later_observation_id].label == later_label
            ):
                return classify_observation_relation_admission(
                    relation, obs_by, {event.id: event for event in resolved}, {evidence[0].id: evidence[0]}, contexts,
                )
    event_id = resolved[0].id if resolved else "e1"
    observations = [
        _observation("a", label=earlier_label, actor_text=actor_text, actor_status=actor_status, event_id=event_id, evidence_refs=(evidence[0].id,), place_role=EventPlaceRole.ORIGIN),
        _observation("b", label=later_label, actor_text=actor_text, actor_status=actor_status, event_id=event_id, evidence_refs=(evidence[0].id,), place_role=EventPlaceRole.DESTINATION),
    ]
    relation = _relation("a", "b", event_ids=(event_id,), evidence_refs=(evidence[0].id,))
    return classify_observation_relation_admission(
        relation,
        {item.observation_id: item for item in observations},
        {event.id: event for event in resolved},
        {evidence[0].id: evidence[0]},
        contexts,
    )


def _assemble(text: str, *, query: str = GENERIC_QUERY):
    observations, relations, resolved, evidence, contexts = _pipeline(text, query=query)
    return assemble_observation_components(
        observations, relations, resolved, evidence, query_contexts=contexts,
    )


def test_query_route_scope_parses_generic_endpoints_and_subject():
    scope = parse_query_route_scope((GENERIC_QUERY,))
    assert scope.subject == "Commander Alpha"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert scope.has_episode_constraint is True


def test_matrix_forward_admission():
    admission = _admission_for_labels(
        "During Campaign One, Commander Alpha marched from Port A to City B.",
        "Port A",
        "City B",
    )
    assert admission is not None
    assert admission.admitted is True
    assert admission.subject_match is AuthorityState.MATCH
    assert admission.episode_match is AuthorityState.MATCH
    assert admission.route_phase_match is AuthorityState.MATCH
    assert admission.movement_assertion is AuthorityState.MATCH


@pytest.mark.parametrize(
    ("text", "earlier", "later", "actor", "actor_status", "expected_admitted", "phase", "episode", "movement"),
    [
        (
            "During Campaign One, Commander Alpha marched from City B to Port A.",
            "City B",
            "Port A",
            "Commander Alpha",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.WRONG,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
        ),
        (
            "During Campaign One, Commander Alpha marched from City B to City Z.",
            "City B",
            "City Z",
            "Commander Alpha",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.WRONG,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
        ),
        (
            "During Campaign One, Commander Alpha marched from Port A to River X.",
            "Port A",
            "River X",
            "Commander Alpha",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.UNKNOWN,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
        ),
        (
            "During Campaign Two, Commander Alpha marched from Port A to City B.",
            "Port A",
            "City B",
            "Commander Alpha",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.MATCH,
            AuthorityState.WRONG,
            AuthorityState.MATCH,
        ),
        (
            "Commander Alpha marched from Port A to City B.",
            "Port A",
            "City B",
            "Commander Alpha",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.MATCH,
            AuthorityState.UNKNOWN,
            AuthorityState.MATCH,
        ),
        (
            "During Campaign One, Commander Beta marched from Port A to City B.",
            "Port A",
            "City B",
            "Commander Beta",
            EventActorStatus.EXPLICIT,
            False,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
        ),
        (
            "During Campaign One, Commander Alpha marched from Port A to City B.",
            "Port A",
            "City B",
            None,
            EventActorStatus.UNKNOWN,
            False,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
            AuthorityState.MATCH,
        ),
    ],
)
def test_generic_admission_matrix_rows(
    text,
    earlier,
    later,
    actor,
    actor_status,
    expected_admitted,
    phase,
    episode,
    movement,
):
    admission = _admission_for_labels(
        text,
        earlier,
        later,
        actor_text=actor,
        actor_status=actor_status,
    )
    assert admission is not None
    assert admission.admitted is expected_admitted
    assert admission.route_phase_match is phase
    assert admission.episode_match is episode
    assert admission.movement_assertion is movement
    if actor_status is EventActorStatus.UNKNOWN:
        assert admission.subject_match is AuthorityState.UNKNOWN
    elif actor == "Commander Beta":
        assert admission.subject_match is AuthorityState.WRONG


def test_background_statements_do_not_admit_components():
    for text in (
        "Campaign One involved fighting near River X.",
        "Commander Alpha was the principal general of Campaign One.",
        "There was a battle near City B.",
    ):
        assembly = _assemble(text)
        assert not assembly.components
        if assembly.rejected_edges:
            assert assembly.rejected_edges[0]["reason"] in {
                "MOVEMENT_ASSERTION_REJECTED",
                "QUERY_ROUTE_ADMISSION_REJECTED",
                "QUERY_EPISODE_REJECTED",
                "QUERY_ROUTE_PHASE_UNKNOWN",
                "ACTOR_AUTHORITY_REJECTED",
            }


def test_mixed_window_evaluates_relations_independently():
    campaign_one = _admission_for_labels(
        "During Campaign One, Commander Alpha marched from Port A to City B.",
        "Port A",
        "City B",
    )
    campaign_two = _admission_for_labels(
        "During Campaign Two, Commander Alpha marched from Port A to City B.",
        "Port A",
        "City B",
    )
    assert campaign_one is not None and campaign_two is not None
    assert campaign_one.admitted is True
    assert campaign_two.admitted is False


def test_broad_query_does_not_require_endpoint_phase():
    text = "Commander Alpha marched from Port A to City B."
    admission = _admission_for_labels(text, "Port A", "City B", query=BROAD_QUERY)
    assert admission is not None
    assert admission.route_phase_match is AuthorityState.UNKNOWN
    assert admission.admitted is True


def test_reverse_retreat_rejected():
    admission = _admission_for_labels(
        "During Campaign One, Commander Alpha retreated from City B to Port A.",
        "City B",
        "Port A",
    )
    assert admission is not None
    assert not admission.admitted
    assert admission.route_phase_match is AuthorityState.WRONG
    assert "QUERY_ROUTE_PHASE_REJECTED" in admission.reason_codes


def test_forward_chain_member_can_admit_intermediate():
    text = (
        "During Campaign One, Commander Alpha marched from Port A to River X, "
        "then from River X to City B."
    )
    observations, relations, resolved, evidence, contexts = _pipeline(text)
    obs_by = {item.observation_id: item for item in observations}
    labels = {
        (obs_by[r.earlier_observation_id].label, obs_by[r.later_observation_id].label): classify_observation_relation_admission(
            r, obs_by, {event.id: event for event in resolved}, {evidence[0].id: evidence[0]}, contexts,
        )
        for r in relations
    }
    if ("Port A", "River X") in labels:
        assert labels[("Port A", "River X")].route_phase_match is AuthorityState.MATCH
    if ("River X", "City B") in labels:
        assert labels[("River X", "City B")].route_phase_match is AuthorityState.MATCH


def test_historical_island_passage_isolated_evidence_rejected():
    """STALE_TEST_CONTRACT: isolated Rhodanus->Island lacks endpoint-anchored graph authority."""
    text = (
        "After four days' march from the passage of the Rhone, "
        "Hannibal arrived at the place called the Island."
    )
    observations, relations, resolved, evidence, contexts = _pipeline(
        text,
        query=HANNIBAL_QUERY,
        extractor=_HANNIBAL_EXTRACTOR,
        geography=_HannibalGeography(),
    )
    assembly = assemble_observation_components(
        observations, relations, resolved, evidence, query_contexts=contexts,
    )
    assert relations
    assert not assembly.components


def test_historical_wrong_episode_rejected():
    admission = _admission_for_labels(
        "Later Hannibal left Italy for exile and went to Antiochus.",
        "Italia",
        "Ephesus",
        query=HANNIBAL_QUERY,
        actor_text="Hannibal",
        extractor=_HANNIBAL_EXTRACTOR,
        geography=_HannibalGeography,
    )
    assert admission is not None
    assert not admission.admitted


def test_historical_new_carthage_passage_still_admits():
    text = "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    observations, relations, resolved, evidence, contexts = _pipeline(
        text,
        query=HANNIBAL_QUERY,
        extractor=_HANNIBAL_EXTRACTOR,
        geography=_HannibalGeography(),
    )
    assembly = assemble_observation_components(
        observations, relations, resolved, evidence, query_contexts=contexts,
    )
    assert relations
    assert assembly.components


def test_live_hannibal_replay_does_not_admit_isolated_rhodanus_island():
    """STALE_TEST_CONTRACT: live replay must not admit Rhodanus->Island without anchored graph."""
    texts = _live_hannibal_evidence()
    if texts is None:
        pytest.skip("Chroma corpus unavailable for live Hannibal replay")
    items = [_evidence(text, eid=f"ev-{index}", offset=100 + index * 100) for index, text in enumerate(texts)]
    candidates, _ = _HANNIBAL_EXTRACTOR.extract(items, query=HANNIBAL_QUERY, query_contexts=(HANNIBAL_QUERY,))
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        items,
        event_id="hannibal-route",
        name="Hannibal",
        period="Second Punic War",
        query_contexts=(HANNIBAL_QUERY,),
    )
    obs_by = {item.observation_id: item for item in outcome.observations}
    labels = {
        obs_by[oid].label
        for component in outcome.observation_components
        for oid in component.observation_ids
    }
    if "Rhodanus" in labels and "Island" in labels:
        pytest.fail("isolated Rhodanus->Island must not assemble without endpoint-anchored graph")


def _live_hannibal_evidence() -> list[str] | None:
    for path in (
        Path(r"C:\D\python\historical-gis-cursor\data\chroma_server_roman_republic_v2"),
        Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
    ):
        if not path.exists():
            continue
        try:
            import chromadb
            from backend.app.core.config import settings
            from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
            from backend.app.rag.http_store import ChromaHttpEvidenceStore
            from backend.app.rag.query_bridge import HistoricalQueryBridge
            from backend.app.rag.retriever import ChromaHistoricalRetriever

            client = chromadb.PersistentClient(path=str(path))
            if settings.rag_collection not in [collection.name for collection in client.list_collections()]:
                continue
            collection = client.get_collection(settings.rag_collection)
            if collection.count() < 1000:
                continue
            provider = SentenceTransformerEmbeddingProvider(
                settings.rag_embedding_model,
                settings.rag_embedding_device,
                settings.rag_embedding_batch_size,
            )
            retriever = ChromaHistoricalRetriever(
                ChromaHttpEvidenceStore(collection, provider),
                HistoricalQueryBridge(settings.rag_query_bridge_enabled),
            )
            evidence = retriever.retrieve_with_coverage(HANNIBAL_QUERY)
        except Exception:
            continue
        if evidence:
            return [item.text for item in evidence[:20]]
    return None
