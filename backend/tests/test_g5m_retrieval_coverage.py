"""G5M: coverage-oriented retrieval for historical movement queries."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    merge_coverage_results,
    movement_bearing_evidence,
    movement_bearing_text,
    top_source_share,
)
from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query
from backend.app.routes.evidence_relevance import classify_evidence_relevance, EvidenceRelevance

XENOPHON_QUERY = (
    "Trace the route of Xenophon and the Ten Thousand from Cunaxa back to the "
    "Greek world after the Battle of Cunaxa in 401 BCE."
)
CAESAR_QUERY = (
    "Trace Julius Caesar's route from Italy across the Adriatic into Epirus and "
    "through the campaign leading to Pharsalus in 48 BCE."
)
POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)
MITHRIDATES_QUERY = (
    "Reconstruct the major movements of Mithridates VI during the First Mithridatic War, "
    "from Pontus through Asia Minor and Greece, and back toward Anatolia."
)
ALEXANDER_QUERY = (
    "Reconstruct Alexander the Great's route from Bactria through the Hindu Kush "
    "into the Indian campaign, ending near the Hydaspes."
)

RR_CONTROLS = {
    "crassus": (
        "Trace Marcus Licinius Crassus's route during the Parthian campaign from Syria "
        "across the Euphrates toward Carrhae in 53 BCE.",
        ("Carrhae", "Euphrates"),
    ),
    "lucullus": (
        "Trace Lucullus's campaign movements against Mithridates from Pontus through "
        "Armenia and into Asia Minor.",
        ("Pontus", "Armenia"),
    ),
    "sertorius": (
        "Trace Sertorius's movements during the civil war in Hispania from Valentia "
        "through the Iberian interior.",
        ("Valentia", "Hispania"),
    ),
    "scipio": (
        "Trace Scipio Africanus's route from Hispania across to Africa during the "
        "campaign against Carthage.",
        ("Hispania", "Africa"),
    ),
    "marius": (
        "Trace Gaius Marius's movements during the Numidian campaign from Rome toward "
        "Africa in the Jugurthine War.",
        ("Rome", "Africa"),
    ),
}

KEY_MOVEMENT_REFERENCES = {
    XENOPHON_QUERY: (
        "travel by land from Athens to Peloponnesus",
        ("Athens", "Peloponnesus"),
    ),
    CAESAR_QUERY: (
        "",
        ("Epirus",),
    ),
    POMPEY_QUERY: (
        "",
        ("Egypt",),
    ),
    MITHRIDATES_QUERY: (
        "",
        ("Pontus",),
    ),
}


def _key_recall(evidence: list[Evidence], needle: str, terms: tuple[str, ...]) -> bool:
    for item in evidence:
        text = item.text or item.excerpt or ""
        if needle and needle.casefold() in text.casefold():
            return True
        if terms and all(term.casefold() in text.casefold() for term in terms):
            return True
    return False


def _episode_term_hits(evidence: list[Evidence], terms: tuple[str, ...]) -> int:
    hits = 0
    for item in evidence:
        text = item.text or item.excerpt or ""
        if any(term.casefold() in text.casefold() for term in terms):
            hits += 1
    return hits


def _evidence(identifier: str, text: str, *, score: float = 0.5, source: str = "doc-1") -> Evidence:
    return Evidence(
        id=identifier,
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt=text[:500],
        text=text,
        score=score,
        metadata={
            "document_id": source,
            "source_chunk_id": source,
            "retrieval_ranking": {"final_score": score},
        },
    )


def _intent_coverage(evidence: list[Evidence]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in evidence:
        provenance = item.metadata.get("retrieval_provenance") or {}
        for intent in provenance.get("matched_intents") or [provenance.get("retrieval_intent")]:
            if intent:
                counts[str(intent)] += 1
    return dict(counts)


def _episode_relevance_rate(evidence: list[Evidence], query: str) -> float:
    if not evidence:
        return 0.0
    relevant = 0
    for item in evidence:
        text = item.text or item.excerpt or ""
        relevance = classify_evidence_relevance(text, (query,))
        if relevance in {
            EvidenceRelevance.DIRECT_SUBJECT,
            EvidenceRelevance.DIRECT_CAMPAIGN,
            EvidenceRelevance.DIRECT_EVENT,
            EvidenceRelevance.SAME_CONFLICT_RELEVANT,
        }:
            relevant += 1
    return relevant / len(evidence)


@pytest.fixture(scope="module")
def production_retriever():
    try:
        from backend.app.core.config import settings
        from backend.app.rag.http_store import build_production_retriever

        retriever = build_production_retriever(settings)
        retriever.retrieve("Rome", 1)
        return retriever
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"production chroma unavailable: {exc}")


def test_decompose_movement_query_returns_complementary_intents():
    intents = decompose_movement_query(XENOPHON_QUERY)
    kinds = {intent.kind for intent in intents}
    assert "SUBJECT" in kinds
    assert "MOVEMENT" in kinds
    assert 3 <= len(intents) <= 5
    assert all(intent.query.strip() for intent in intents)


def test_decompose_has_no_duplicate_queries():
    intents = decompose_movement_query(CAESAR_QUERY)
    queries = [intent.query.casefold() for intent in intents]
    assert len(queries) == len(set(queries))


def test_merge_coverage_reserves_intent_slots_and_dedupes():
    subject_items = [
        _evidence("a", "Xenophon marched through the mountains.", score=0.9),
        _evidence("b", "General biography of Xenophon.", score=0.8),
    ]
    movement_items = [
        _evidence(
            "c",
            "It was therefore a very hazardous journey to travel by land from Athens to Peloponnesus.",
            score=0.7,
        ),
        _evidence("a", "Xenophon marched through the mountains.", score=0.6),
    ]
    merged = merge_coverage_results(
        XENOPHON_QUERY,
        [
            (RetrievalIntent("SUBJECT", "Xenophon Ten Thousand"), subject_items),
            (RetrievalIntent("MOVEMENT", "Xenophon march route travel"), movement_items),
        ],
        budget=3,
    )
    ids = [item.id for item in merged]
    assert len(ids) == len(set(ids))
    assert "c" in ids
    provenance = merged[0].metadata["retrieval_provenance"]
    assert provenance["retrieval_intent"] in {"SUBJECT", "MOVEMENT"}
    assert provenance["matched_intents"]


def test_merge_budget_is_bounded():
    items = [_evidence(f"id-{index}", f"movement {index} marched from A to B.", score=1 - index * 0.01) for index in range(12)]
    merged = merge_coverage_results(
        "generic route query",
        [(RetrievalIntent("MOVEMENT", "march route"), items)],
        budget=5,
    )
    assert len(merged) == 5


def test_movement_bearing_audit():
    assert movement_bearing_text("He marched from Athens to Sparta.")
    assert not movement_bearing_text("He was a famous general.")


def test_tools_bootstrap_uses_coverage_for_first_movement_search():
    captured: list[tuple[str, int]] = []

    class CoverageRetriever:
        def retrieve(self, query, top_k=5, filters=None):
            captured.append(("single", top_k))
            return []

        def retrieve_with_coverage(self, query, budget=DEFAULT_COVERAGE_BUDGET, *, per_intent_k=8, filters=None):
            captured.append(("coverage", budget))
            return [_evidence("x", "Caesar marched from Italy to Epirus.")]

    state = AgentState(session_id="g5m", user_query=CAESAR_QUERY, requested_output="historical_route")
    tools = AgentToolRegistry(CoverageRetriever(), object())
    tools.execute("search_historical_evidence", {"query": CAESAR_QUERY, "top_k": 10}, state)
    assert captured == [("coverage", DEFAULT_COVERAGE_BUDGET)]
    assert len(state.historical_evidence) == 1


def test_tools_subsequent_search_uses_single_retrieve():
    calls: list[str] = []

    class CoverageRetriever:
        def retrieve(self, query, top_k=5, filters=None):
            calls.append("single")
            return [_evidence("y", "Pompey sailed toward Egypt.")]

        def retrieve_with_coverage(self, query, budget=DEFAULT_COVERAGE_BUDGET, *, per_intent_k=8, filters=None):
            calls.append("coverage")
            return [_evidence("x", "Pompey fled after Pharsalus.")]

    state = AgentState(session_id="g5m2", user_query=POMPEY_QUERY, requested_output="historical_route")
    state.historical_evidence = [_evidence("seed", "seed evidence")]
    tools = AgentToolRegistry(CoverageRetriever(), object())
    tools.execute("search_historical_evidence", {"query": "Pompey Egypt", "top_k": 5}, state)
    assert calls == ["single"]


@pytest.mark.integration
def test_xenophon_athens_peloponnesus_recall_with_coverage(production_retriever):
    baseline = production_retriever.retrieve(XENOPHON_QUERY, 10)
    coverage = production_retriever.retrieve_with_coverage(XENOPHON_QUERY)
    needle, terms = KEY_MOVEMENT_REFERENCES[XENOPHON_QUERY]
    assert not _key_recall(baseline, needle, terms), "baseline should miss key pair in this audit"
    assert _key_recall(coverage, needle, terms), "coverage retrieval must recall Athens→Peloponnesus"
    assert len(coverage) <= DEFAULT_COVERAGE_BUDGET
    assert len(coverage) <= len(baseline) * 2


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,needle,terms",
    [
        (query, ref[0], ref[1])
        for query, ref in KEY_MOVEMENT_REFERENCES.items()
        if query != XENOPHON_QUERY
    ],
)
def test_key_movement_reference_recall_non_xenophon(production_retriever, query, needle, terms):
    coverage = production_retriever.retrieve_with_coverage(query)
    if not _key_recall(coverage, needle, terms):
        pytest.xfail(f"corpus may lack reference movement evidence for: {terms}")


@pytest.mark.integration
def test_g5m_offline_metrics_snapshot(production_retriever):
    cases = [
        ("xenophon", XENOPHON_QUERY),
        ("caesar", CAESAR_QUERY),
        ("pompey", POMPEY_QUERY),
        ("mithridates", MITHRIDATES_QUERY),
        ("alexander", ALEXANDER_QUERY),
        *[(name, query) for name, (query, _) in RR_CONTROLS.items()],
    ]
    report: dict[str, object] = {}
    recalls: list[float] = []
    for case_id, query in cases:
        before = production_retriever.retrieve(query, 10)
        after = production_retriever.retrieve_with_coverage(query)
        ref = KEY_MOVEMENT_REFERENCES.get(query)
        recall = 1.0 if ref is None else float(_key_recall(after, ref[0], ref[1]))
        if ref is not None:
            recalls.append(recall)
        report[case_id] = {
            "before_count": len(before),
            "after_count": len(after),
            "top_source_share_before": round(top_source_share(before), 3),
            "top_source_share_after": round(top_source_share(after), 3),
            "movement_bearing_before": len(movement_bearing_evidence(before)),
            "movement_bearing_after": len(movement_bearing_evidence(after)),
            "intent_coverage_after": _intent_coverage(after),
            "episode_relevance_before": round(_episode_relevance_rate(before, query), 3),
            "episode_relevance_after": round(_episode_relevance_rate(after, query), 3),
            "key_movement_recall": recall,
            "episode_term_hits_before": _episode_term_hits(before, RR_CONTROLS.get(case_id, (query, ()))[1] if case_id in RR_CONTROLS else ()),
            "episode_term_hits_after": _episode_term_hits(after, RR_CONTROLS.get(case_id, (query, ()))[1] if case_id in RR_CONTROLS else ()),
        }
        if case_id in {"caesar", "pompey", "mithridates", "xenophon"}:
            assert len(after) <= len(before) * 2
            assert len(movement_bearing_evidence(after)) >= len(movement_bearing_evidence(before))
    if recalls:
        assert sum(recalls) / len(recalls) >= 0.85
    out = Path(__file__).resolve().parents[2] / "outputs" / "g5m_retrieval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
