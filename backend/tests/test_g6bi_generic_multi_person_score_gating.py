"""G6BI: generic multi-person score gating without historical-name dependency."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence

KNOWN_ROUTE_QUERY = "Trace Caesar and Pompey from Greece toward Egypt."
UNSEEN_ROUTE_QUERY = "Trace Demetrius and Seleucus from Greece toward Egypt."
ARISTON_ROUTE_QUERY = "Trace Ariston and Bion from Greece toward Egypt."

KNOWN_ROUTE_TEXT = "Caesar and Pompey marched from Greece toward Egypt with the army."
UNSEEN_ROUTE_TEXT = "Demetrius and Seleucus marched from Greece toward Egypt with the army."
ARISTON_ROUTE_TEXT = "Ariston and Bion marched from Greece toward Egypt with the army."

KNOWN_NARRATIVE_TEXT = "Caesar and Pompey debated policy at court."
UNSEEN_NARRATIVE_TEXT = "Demetrius and Seleucus debated policy at court."
ARISTON_NARRATIVE_TEXT = "Ariston and Bion debated policy at court."


def _evidence(identifier: str, text: str, *, vector_rank: int = 5) -> Evidence:
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt=text,
        text=text,
        score=0.5,
        metadata={
            "document_id": "doc",
            "semantic_candidate": True,
            "vector_rank": vector_rank,
        },
    )


def _score(query: str, text: str) -> dict:
    ranked = rerank_evidence(query, [_evidence("item", text)], pool_relative=False)
    return ranked[0].metadata["retrieval_ranking"]


def test_known_vs_unseen_joint_route_scores_equivalent():
    known = _score(KNOWN_ROUTE_QUERY, KNOWN_ROUTE_TEXT)
    unseen = _score(UNSEEN_ROUTE_QUERY, UNSEEN_ROUTE_TEXT)
    assert known["person_support"] == pytest.approx(unseen["person_support"])
    assert known["entity_support"] == pytest.approx(unseen["entity_support"])
    assert known["passage_local_support"] == pytest.approx(unseen["passage_local_support"])
    assert known["semantic_relevance"] == pytest.approx(unseen["semantic_relevance"])
    assert known["final_score"] == pytest.approx(unseen["final_score"])


def test_known_vs_unseen_joint_narrative_scores_equivalent():
    known = _score(KNOWN_ROUTE_QUERY, KNOWN_NARRATIVE_TEXT)
    unseen = _score(UNSEEN_ROUTE_QUERY, UNSEEN_NARRATIVE_TEXT)
    assert known["final_score"] == pytest.approx(unseen["final_score"])
    assert known["entity_support"] <= 0.04
    assert unseen["entity_support"] <= 0.04
    assert known["passage_local_support"] <= 0.25
    assert unseen["passage_local_support"] <= 0.25


def test_ariston_bion_route_matches_known_unseen_semantics():
    ariston = _score(ARISTON_ROUTE_QUERY, ARISTON_ROUTE_TEXT)
    unseen = _score(UNSEEN_ROUTE_QUERY, UNSEEN_ROUTE_TEXT)
    assert ariston["final_score"] == pytest.approx(unseen["final_score"])


def test_route_corroborated_full_person_receives_full_amplification():
    score = _score(KNOWN_ROUTE_QUERY, KNOWN_ROUTE_TEXT)
    assert score["person_support"] == 0.08
    assert score["entity_support"] == 0.04
    assert score["passage_local_support"] >= 0.6
    assert score["final_score"] >= 0.9


def test_narrative_only_full_person_is_capped():
    score = _score(KNOWN_ROUTE_QUERY, KNOWN_NARRATIVE_TEXT)
    assert score["person_support"] == 0.08
    assert score["entity_support"] <= 0.04
    assert score["passage_local_support"] <= 0.25


def test_single_subject_route_preserved():
    query = "Trace Caesar from Greece toward Egypt."
    text = "Caesar marched from Greece toward Egypt."
    score = _score(query, text)
    assert score["person_support"] >= 0.04
    assert score["final_score"] >= 0.9


def test_partial_person_on_multi_person_query():
    query = KNOWN_ROUTE_QUERY
    text = "Caesar marched from Greece toward Egypt."
    score = _score(query, text)
    assert score["person_support"] == 0.04
    assert score["entity_support"] <= 0.04


def test_wrong_actor_stays_below_subject_route():
    subject = _evidence("subject", KNOWN_ROUTE_TEXT, vector_rank=5)
    wrong = _evidence("wrong", "Brutus marched from Greece toward Egypt.", vector_rank=6)
    ranked = rerank_evidence(KNOWN_ROUTE_QUERY, [wrong, subject], pool_relative=False)
    details = {item.id: item.metadata["retrieval_ranking"] for item in ranked}
    assert details["subject"]["final_score"] > details["wrong"]["final_score"]
    assert ranked[0].id == "subject"


def test_marcus_valerius_not_treated_as_coordinated_multi_person():
    query = "Trace Marcus Valerius from Rome to Capua."
    text = "Marcus Valerius marched from Rome to Capua."
    score = _score(query, text)
    assert score["person_support"] >= 0.04
    assert score["final_score"] >= 0.9


def test_marcus_valerius_possessive_route_preserved():
    query = "Trace Marcus Valerius's route from Rome to Capua."
    text = "Marcus Valerius marched from Rome to Capua."
    score = _score(query, text)
    assert score["person_support"] >= 0.04
    assert score["final_score"] >= 0.9


def test_joint_route_beats_joint_narrative():
    route = _score(KNOWN_ROUTE_QUERY, KNOWN_ROUTE_TEXT)
    narrative = _score(KNOWN_ROUTE_QUERY, KNOWN_NARRATIVE_TEXT)
    assert route["final_score"] > narrative["final_score"]


@pytest.mark.integration
def test_lucullus_002_remains_final(production_retriever):
    from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, DEFAULT_RAW_OBSERVATION_K
    from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query

    query = (
        "Trace Lucullus's campaign movements against Mithridates from Pontus through "
        "Armenia and into Asia Minor."
    )
    target = "eb82a760397f57671a3f607ef6ae8d21:681:1589"
    channels = [RetrievalIntent("CANONICAL", query), *decompose_movement_query(query)]
    pool = {
        item.id: item
        for ch in channels
        for item in production_retriever.retrieve_candidates(ch.query, DEFAULT_RAW_OBSERVATION_K)
    }
    assert target in pool
    ranked = rerank_evidence(query, list(pool.values()), pool_relative=False)
    rank = next(i for i, item in enumerate(ranked, 1) if item.id == target)
    final = target in {
        item.id for item in production_retriever.retrieve_with_coverage(query, DEFAULT_COVERAGE_BUDGET)
    }
    assert rank <= 25
    assert final


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from pathlib import Path

        import chromadb

        from backend.app.core.config import settings
        from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
        from backend.app.rag.http_store import ChromaHttpEvidenceStore, build_production_retriever
        from backend.app.rag.query_bridge import HistoricalQueryBridge
        from backend.app.rag.retriever import ChromaHistoricalRetriever

        provider = SentenceTransformerEmbeddingProvider(
            settings.rag_embedding_model,
            settings.rag_embedding_device,
            settings.rag_embedding_batch_size,
        )
        for path in [
            Path(__file__).resolve().parents[0].parent / "data" / "chroma_server_roman_republic_v2",
            Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
        ]:
            if not path.exists():
                continue
            client = chromadb.PersistentClient(path=str(path))
            if settings.rag_collection in [collection.name for collection in client.list_collections()]:
                collection = client.get_collection(settings.rag_collection)
                if collection.count() > 1000:
                    retriever = ChromaHistoricalRetriever(
                        ChromaHttpEvidenceStore(collection, provider),
                        HistoricalQueryBridge(settings.rag_query_bridge_enabled),
                    )
                    retriever.retrieve("Rome", 1)
                    return retriever
        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"production chroma unavailable: {exc}")
