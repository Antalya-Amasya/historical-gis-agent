"""G6BK: structural person-phrase boundary for coordinated scoring."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.query_roles import analyze_query

SINGLE_NAMES = (
    "Alexander Severus",
    "Philip Arrhidaeus",
    "Seleucus Nicator",
    "Demetrius Poliorcetes",
    "Hannibal Barca",
    "Scipio Africanus",
    "Pyrrhus of Epirus",
)


def _evidence(text: str) -> Evidence:
    return Evidence(
        id="item",
        author="Author",
        work="Work",
        locator="section",
        excerpt=text,
        text=text,
        score=0.5,
        metadata={"document_id": "doc", "semantic_candidate": True, "vector_rank": 5},
    )


def _score(query: str, text: str) -> dict:
    ranked = rerank_evidence(query, [_evidence(text)], pool_relative=False)
    return ranked[0].metadata["retrieval_ranking"]


def _route_query(name: str) -> str:
    return f"Trace {name} from Rome to Capua."


@pytest.mark.parametrize("name", SINGLE_NAMES)
def test_non_roman_single_names_not_coordinated(name: str):
    roles = analyze_query(_route_query(name))
    assert roles.person_coordination_detected is False


@pytest.mark.parametrize("name", SINGLE_NAMES)
def test_non_roman_single_full_name_beats_partial(name: str):
    query = _route_query(name)
    parts = name.split()
    full = _score(query, f"{name} marched from Rome to Capua.")
    partial = _score(query, f"{parts[-1]} marched from Rome to Capua.")
    assert full["final_score"] > partial["final_score"]
    assert full["entity_support"] > partial["entity_support"]


def test_long_roman_single_names_not_coordinated():
    for query in (
        "Trace Gaius Julius Caesar from Rome to Brundisium.",
        "Trace Marcus Tullius Cicero from Rome to Capua.",
    ):
        roles = analyze_query(query)
        assert roles.person_coordination_detected is False


def test_gaius_julius_caesar_full_name_beats_partial():
    query = "Trace Gaius Julius Caesar from Rome to Brundisium."
    full = _score(query, "Gaius Julius Caesar marched from Rome to Brundisium.")
    partial = _score(query, "Caesar marched from Rome to Brundisium.")
    assert full["final_score"] > partial["final_score"]


def test_marcus_and_lucius_is_coordinated():
    query = "Trace Marcus and Lucius from Rome to Capua."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is True


def test_marcus_and_lucius_route_uses_coordinated_scoring():
    query = "Trace Marcus and Lucius from Rome to Capua."
    route = _score(query, "Marcus and Lucius marched from Rome to Capua.")
    assert route["entity_support"] == 0.04


def test_gaius_and_marcus_is_coordinated():
    query = "Trace Gaius and Marcus during the campaign from Rome toward Brundisium."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is True


def test_marcus_valerius_and_lucius_cornelius_is_coordinated():
    query = "Trace Marcus Valerius and Lucius Cornelius from Rome to Capua."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is True


def test_demetrius_and_seleucus_nicator_is_coordinated():
    query = "Trace Demetrius Poliorcetes and Seleucus Nicator through Syria."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is True


def test_marcus_valerius_single_not_coordinated():
    query = "Trace Marcus Valerius from Rome to Capua."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is False


def test_marcus_valerius_possessive_single_not_coordinated():
    query = "Trace Marcus Valerius's route from Rome to Capua."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is False


def test_alexander_severus_possessive_single_not_coordinated():
    query = "Trace Alexander Severus's route from Syria to Rome."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is False


def test_marcus_valerius_single_route_preserved():
    query = "Trace Marcus Valerius from Rome to Capua."
    score = _score(query, "Marcus Valerius marched from Rome to Capua.")
    assert score["final_score"] >= 0.9


def test_coordinated_narrative_capped():
    query = "Trace Marcus and Lucius from Rome to Capua."
    narrative = _score(query, "Marcus and Lucius debated policy at court.")
    assert narrative["entity_support"] <= 0.04
    assert narrative["final_score"] <= 0.3


def test_coordinated_route_beats_narrative():
    query = "Trace Marcus and Lucius from Rome to Capua."
    route = _score(query, "Marcus and Lucius marched from Rome to Capua.")
    narrative = _score(query, "Marcus and Lucius debated policy at court.")
    assert route["final_score"] > narrative["final_score"]


def test_wrong_actor_below_subject_route():
    query = "Trace Marcus and Lucius from Rome to Capua."
    subject = _evidence("Marcus and Lucius marched from Rome to Capua.")
    wrong = Evidence(
        id="wrong",
        author="Author",
        work="Work",
        locator="section",
        excerpt="Brutus marched from Rome to Capua.",
        text="Brutus marched from Rome to Capua.",
        score=0.5,
        metadata={"document_id": "doc", "semantic_candidate": True, "vector_rank": 6},
    )
    ranked = rerank_evidence(query, [wrong, subject], pool_relative=False)
    assert ranked[0].id == "item"


def test_caesar_pursuit_of_pompey_reports_non_coordination():
    query = "Trace Caesar's pursuit of Pompey from Greece toward Egypt."
    roles = analyze_query(query)
    assert roles.person_coordination_detected is False


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
