"""G6BG: passage-local person specificity / score-coupling fix."""
from __future__ import annotations

import pytest

from backend.app.models import Evidence
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.query_roles import analyze_query

LUCULLUS_QUERY = (
    "Trace Lucullus's campaign movements against Mithridates from Pontus through "
    "Armenia and into Asia Minor."
)


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


def _ranking(query: str, *items: Evidence) -> dict[str, dict]:
    ranked = rerank_evidence(query, list(items), pool_relative=False)
    return {item.id: item.metadata["retrieval_ranking"] for item in ranked}


def test_g6be_role_map_remains_correct():
    roles = analyze_query(LUCULLUS_QUERY)
    assert roles.person_terms == frozenset({"lucullus", "mithridates"})
    assert {"pontus", "armenia", "asia", "minor"} <= roles.location_terms
    assert not (roles.person_terms & {"asia", "minor"})


def test_person_complete_without_route_not_amplified_in_passage_local():
    weak = _evidence(
        "weak-person-only",
        "Lucullus and Mithridates exchanged embassies about terms while the court deliberated.",
    )
    score = _ranking(LUCULLUS_QUERY, weak)["weak-person-only"]
    assert score["person_support"] == 0.08
    assert score["action_support"] == 0.0
    assert score["location_support"] == 0.0
    assert score["entity_support"] == 0.04
    assert score["passage_local_support"] <= 0.25
    assert score["semantic_relevance"] <= 0.25


def test_route_passage_outranks_person_only_narrative():
    narrative = _evidence(
        "narrative",
        "Lucullus and Mithridates disputed command while both armies remained near the court.",
        vector_rank=6,
    )
    route = _evidence(
        "route",
        "Lucullus marched through Armenia toward the passes leading into Asia Minor after Mithridates withdrew.",
        vector_rank=5,
    )
    ranked = rerank_evidence(LUCULLUS_QUERY, [narrative, route], pool_relative=False)
    details = {item.id: item.metadata["retrieval_ranking"] for item in ranked}
    assert details["route"]["final_score"] > details["narrative"]["final_score"]
    assert ranked[0].id == "route"


def test_strong_subject_movement_route_remains_high():
    route = _evidence(
        "strong-route",
        "Lucullus marched through Armenia toward Asia Minor while Mithridates retreated from Pontus.",
    )
    score = _ranking(LUCULLUS_QUERY, route)["strong-route"]
    assert score["person_support"] >= 0.04
    assert score["action_support"] > 0
    assert score["location_support"] > 0
    assert score["passage_local_support"] >= 0.6
    assert score["final_score"] >= 0.9


def test_wrong_actor_route_does_not_beat_subject_route():
    wrong_actor = _evidence(
        "wrong-actor",
        "Mithridates marched through Armenia into Pontus before Lucullus could intervene.",
        vector_rank=6,
    )
    subject_route = _evidence(
        "subject-route",
        "Lucullus marched through Armenia toward Asia Minor after Mithridates withdrew from Pontus.",
        vector_rank=5,
    )
    ranked = rerank_evidence(LUCULLUS_QUERY, [wrong_actor, subject_route], pool_relative=False)
    assert ranked[0].id == "subject-route"


def test_multi_person_narrative_loses_to_multi_person_movement():
    narrative = _evidence(
        "narrative",
        "Lucullus confronted Mithridates politically in council without advancing the army.",
        vector_rank=6,
    )
    movement = _evidence(
        "movement",
        "Lucullus marched through Armenia after Mithridates fled toward Pontus.",
        vector_rank=5,
    )
    ranked = rerank_evidence(LUCULLUS_QUERY, [narrative, movement], pool_relative=False)
    assert ranked[0].id == "movement"


def test_single_subject_route_control():
    route = _evidence(
        "single-subject",
        "Lucullus marched from Pontus through Armenia into the highlands of Asia Minor.",
    )
    score = _ranking(LUCULLUS_QUERY, route)["single-subject"]
    assert score["person_support"] >= 0.04
    assert score["action_support"] > 0
    assert score["location_support"] > 0
    assert score["final_score"] >= 0.9


@pytest.mark.integration
def test_lucullus_002_common_rank_recovers(production_retriever):
    from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, DEFAULT_RAW_OBSERVATION_K
    from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query

    target = "eb82a760397f57671a3f607ef6ae8d21:681:1589"
    channels = [RetrievalIntent("CANONICAL", LUCULLUS_QUERY), *decompose_movement_query(LUCULLUS_QUERY)]
    pool = {
        item.id: item
        for ch in channels
        for item in production_retriever.retrieve_candidates(ch.query, DEFAULT_RAW_OBSERVATION_K)
    }
    assert target in pool
    ranked = rerank_evidence(LUCULLUS_QUERY, list(pool.values()), pool_relative=False)
    rank = next(i for i, item in enumerate(ranked, 1) if item.id == target)
    score = next(item for item in ranked if item.id == target).metadata["retrieval_ranking"]["final_score"]
    final = target in {
        item.id for item in production_retriever.retrieve_with_coverage(LUCULLUS_QUERY, DEFAULT_COVERAGE_BUDGET)
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
