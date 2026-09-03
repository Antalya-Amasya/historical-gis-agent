"""G5L: live/direct movement parity and legacy episode relevance safety."""
from __future__ import annotations

from backend.app.agent.tools import (
    AgentToolRegistry,
    _cumulative_event_query_contexts,
)
from backend.app.models import (
    AgentState,
    AgentToolHistoryEntry,
    Evidence,
    HistoricalClaim,
    HistoricalEventType,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_legacy_claim_episode,
    filter_legacy_movement_claims,
)
from backend.app.routes.events import (
    EvidenceGroundedHistoricalEventExtractor,
    HistoricalEventConsolidator,
)
from backend.app.routes.extractor import HistoricalRouteExtractor
from backend.app.rag.retriever import HistoricalRetriever


def ev(identifier: str, text: str, *, author: str = "Author", work: str = "Work") -> Evidence:
    return Evidence(
        id=identifier,
        author=author,
        work=work,
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


class Retriever(HistoricalRetriever):
    def __init__(self, items: list[Evidence]):
        self.items = items

    def retrieve(self, *_args, **_kwargs):
        return self.items


class Geography:
    places = {
        "Rhodanus": (43.3, 4.8),
        "Italia": (41.9, 12.5),
        "Macedonia": (40.6, 22.9),
        "Apollonia": (40.72, 19.45),
        "Athamania": (39.5, 21.2),
        "Thessalia": (39.6, 22.4),
        "Corfinium": (42.0, 14.0),
        "Sicily": (37.5, 14.0),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        if name not in self.places:
            return {"found": False}
        lat, lon = self.places[name]
        return {
            "found": True,
            "id": f"fixture-{name}",
            "canonical_name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "fixture",
            "source_id": name,
            "confidence": 0.8,
            "coordinate_role": "representative_point",
        }


CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


def _movement_count(events) -> int:
    return sum(1 for event in events if event.event_type is HistoricalEventType.MOVEMENT)


def _direct_extract(evidence, contexts):
    extractor = EvidenceGroundedHistoricalEventExtractor()
    consolidator = HistoricalEventConsolidator()
    candidates, _ = extractor.extract(evidence, query_contexts=contexts)
    consolidated, _ = consolidator.consolidate(candidates)
    return consolidated


def test_live_and_direct_extraction_share_movement_semantics():
    text = "Caesar marched from Macedonia to Apollonia before the campaign in Epirus."
    items = [ev("c1", text)]
    state = AgentState(session_id="parity", user_query=CAESAR_QUERY, requested_output="historical_route")
    tools = AgentToolRegistry(Retriever(items), Geography())
    tools.execute("search_historical_evidence", {"query": CAESAR_QUERY, "top_k": 5}, state)
    contexts = _cumulative_event_query_contexts(
        state, current_query=CAESAR_QUERY, current_evidence_count=len(items),
    )
    direct = _direct_extract(state.historical_evidence, contexts)
    assert _movement_count(state.historical_events) == _movement_count(direct)
    assert _movement_count(direct) >= 1


def test_movement_rich_evidence_not_filtered_by_context_ordering():
    items = [
        ev("m1", "After Pharsalus, Pompey sailed from Greece toward Cyprus."),
        ev("m2", "He later arrived in Egypt near Pelusium."),
    ]
    contexts = (
        POMPEY_QUERY,
        "Pompey eastern Mediterranean escape",
        "Pelusium Egypt arrival",
    )
    extractor = EvidenceGroundedHistoricalEventExtractor()
    for ordering in (contexts, tuple(reversed(contexts))):
        events, _ = extractor.extract(items, query_contexts=ordering)
        assert _movement_count(events) >= 1


def test_irrelevant_movement_still_filtered_by_subject_conflict():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Hannibal marched from Rhodanus into Gaul.",
        textual_basis="Hannibal marched from Rhodanus into Gaul.",
        source_place="Rhodanus",
        destination_place="Gaul",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["h1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("h1", claim.text, author="Polybius", work="Histories")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False


def test_consolidation_preserves_movement_class():
    items = [ev("m1", "Caesar marched from Macedonia to Apollonia in 48 BCE.")]
    extractor = EvidenceGroundedHistoricalEventExtractor()
    consolidator = HistoricalEventConsolidator()
    candidates, _ = extractor.extract(items, query_contexts=(CAESAR_QUERY,))
    consolidated, _ = consolidator.consolidate(candidates)
    assert _movement_count(candidates) == _movement_count(consolidated) >= 1


def test_retrieval_gap_is_not_treated_as_parser_failure():
    items = [ev("bg", "The Senate debated grain supplies in Rome.")]
    extractor = EvidenceGroundedHistoricalEventExtractor()
    events, diagnostics = extractor.extract(items, query_contexts=(CAESAR_QUERY,))
    assert _movement_count(events) == 0
    assert "NO_EVENT_EVIDENCE" in diagnostics["reason_codes"]


def test_same_subject_other_episode_legacy_edge_rejected():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Caesar marched from Corfinium to Sicily.",
        textual_basis="Caesar marched from Corfinium to Sicily.",
        source_place="Corfinium",
        destination_place="Sicily",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Julius Caesar", work="Civil War")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_other_campaign_legacy_edge_rejected():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Hannibal crossed from Rhodanus into Italia.",
        textual_basis="Hannibal crossed from Rhodanus into Italia.",
        source_place="Rhodanus",
        destination_place="Italia",
        movement_relation="crossed_from_into",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Polybius", work="Histories")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False


def test_direct_query_episode_legacy_edge_admitted():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Caesar crossed the Adriatic and landed in Epirus near Apollonia.",
        textual_basis="Caesar crossed the Adriatic and landed in Epirus near Apollonia.",
        source_place="Macedonia",
        destination_place="Apollonia",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Caesar", work="Civil War")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode in {EpisodeRelevance.DIRECT_QUERY_EPISODE, EpisodeRelevance.SAME_CAMPAIGN_RELEVANT}
    assert detail["admitted"] is True


def test_same_campaign_relevant_intermediate_edge_admitted():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Caesar marched from Macedonia to Apollonia in Epirus during the campaign.",
        textual_basis="Caesar marched from Macedonia to Apollonia in Epirus during the campaign.",
        source_place="Macedonia",
        destination_place="Apollonia",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text)]
    admitted, _ = filter_legacy_movement_claims([claim], evidence, (CAESAR_QUERY,))
    assert admitted == [claim]


def test_place_overlap_alone_is_insufficient_for_other_episode_edge():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Hannibal crossed from Rhodanus into Italia.",
        textual_basis="Hannibal crossed from Rhodanus into Italia.",
        source_place="Rhodanus",
        destination_place="Italia",
        movement_relation="crossed_from_into",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Polybius", work="Histories")]
    episode, _ = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN


def test_missing_year_remains_unknown_not_auto_reject():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Pompey sailed eastward from Greece with his fleet.",
        textual_basis="Pompey sailed eastward from Greece with his fleet.",
        source_place="Greece",
        destination_place="Egypt",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Plutarch", work="Lives")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (POMPEY_QUERY,))
    assert episode in {EpisodeRelevance.UNKNOWN, EpisodeRelevance.DIRECT_QUERY_EPISODE}
    assert detail["admitted"] is True


def test_no_relevant_legacy_edge_yields_clean_insufficient_route():
    items = [
        ev(
            "poly",
            "Hannibal crossed from Rhodanus into Italia.",
            author="Polybius",
            work="Histories",
        ),
        ev(
            "sicily",
            "Caesar marched from Corfinium to Sicily.",
            author="Caesar",
            work="Civil War",
        ),
    ]
    state = AgentState(session_id="legacy", user_query=CAESAR_QUERY, requested_output="historical_route")
    state.historical_evidence = items
    tools = AgentToolRegistry(Retriever(items), Geography())
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "caesar-48", "name": "Caesar 48", "period": "48 BCE"},
        state,
    )
    assert result["result"]["route"] is None
    assert "NO_EPISODE_RELEVANT_LEGACY_CLAIMS" in result["result"]["diagnostics"]["reason_codes"]


def test_pompey_escape_query_rejects_athamania_thessalia_primary_legacy_route():
    items = [
        ev(
            "plut",
            "Pompey withdrew through Athamania into Thessalia while reorganizing his forces.",
            author="Plutarch",
            work="Parallel Lives",
        ),
    ]
    state = AgentState(session_id="pompey", user_query=POMPEY_QUERY, requested_output="historical_route")
    state.tool_history.append(
        AgentToolHistoryEntry(
            tool_name="search_historical_evidence",
            arguments={"query": POMPEY_QUERY},
            result_summary="search_historical_evidence evidence_count=1 accumulated_evidence_count=1",
            success=True,
            outcome="success",
            duration_ms=1,
        )
    )
    state.historical_evidence = items
    tools = AgentToolRegistry(Retriever(items), Geography())
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "pompey-48", "name": "Pompey flight", "period": "48 BCE"},
        state,
    )
    route = result["result"]["route"]
    # Either no route, or not the wrong-episode Athamania->Thessalia chain alone.
    if route is not None:
        names = [p.historical_place.canonical_name for p in state.historical_route.ordered_points]
        assert names != ["Athamania", "Thessalia"]
