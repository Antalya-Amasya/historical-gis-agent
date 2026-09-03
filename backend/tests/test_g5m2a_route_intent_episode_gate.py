"""G5M-2a: movement plural route intent and legacy episode anchor gate."""
from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence, HistoricalClaim
from backend.app.rag.evidence_ranking import is_route_or_movement_query
from backend.app.routes.episode_relevance import EpisodeRelevance, classify_legacy_claim_episode

CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


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


class _CoverageSpyRetriever:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def retrieve(self, query, top_k=5, filters=None):
        self.calls.append(("retrieve", top_k))
        return []

    def retrieve_with_coverage(self, query, budget=20, *, per_intent_k=8, filters=None):
        self.calls.append(("retrieve_with_coverage", budget))
        return []


def test_movement_singular_and_plural_route_intent():
    assert is_route_or_movement_query("trace his movement")
    assert is_route_or_movement_query("trace his movements")
    assert is_route_or_movement_query(POMPEY_QUERY)
    assert not is_route_or_movement_query("senate treaty")


def test_pompey_query_triggers_coverage_branch():
    retriever = _CoverageSpyRetriever()
    state = AgentState(session_id="g5m2a", user_query=POMPEY_QUERY, requested_output="historical_route")
    AgentToolRegistry(retriever, object()).execute(
        "search_historical_evidence",
        {"query": POMPEY_QUERY, "top_k": 10},
        state,
    )
    assert retriever.calls == [("retrieve_with_coverage", 20)]


def test_hispania_italia_legacy_edge_rejected_for_caesar_query():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text=(
            "He did not think that Caesar had yet arrived in Italy from Spain, "
            "and even if he were there he did not suspect that his rival would cross the Ionian sea."
        ),
        textual_basis=(
            "He did not think that Caesar had yet arrived in Italy from Spain, "
            "and even if he were there he did not suspect that his rival would cross the Ionian sea."
        ),
        source_place="Hispania",
        destination_place="Italia",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Plutarch", work="Lives")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_rhodanus_italia_still_rejected_for_caesar_query():
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


def test_current_episode_edge_with_anchor_overlap_still_admitted():
    claim = HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text="Caesar crossed the Adriatic and landed in Epirus near Apollonia.",
        textual_basis="Caesar crossed the Adriatic and landed in Epirus near Apollonia.",
        source_place="Epirus",
        destination_place="Apollonia",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = [ev("ev1", claim.text, author="Caesar", work="Civil War")]
    episode, detail = classify_legacy_claim_episode(claim, {evidence[0].id: evidence[0]}, (CAESAR_QUERY,))
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True
