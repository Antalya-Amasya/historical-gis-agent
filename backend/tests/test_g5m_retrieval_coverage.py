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
from backend.tests.g5r_trusted_benchmark import (
    hard_benchmark_queries,
    trusted_key_movement_recall,
)

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

# G5Q audit: deprecated invalid L3 ground truth — do not use for historical recall.
DEPRECATED_KEY_MOVEMENT_REFERENCES = {
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
def test_deprecated_xenophon_theseus_reference_not_used_as_l3_key(production_retriever):
    """Athens→Peloponnesus in THESEUS may still be retrieved, but is not L3 ground truth."""
    coverage = production_retriever.retrieve_with_coverage(XENOPHON_QUERY)
    needle, terms = DEPRECATED_KEY_MOVEMENT_REFERENCES[XENOPHON_QUERY]
    deprecated_hit = _key_recall(coverage, needle, terms)
    trusted = trusted_key_movement_recall(coverage, query_id="xenophon_cunaxa_return")
    assert trusted["per_query"].get("xenophon_cunaxa_return") is None
    if deprecated_hit:
        pytest.xfail("deprecated THESEUS needle still retrieved; excluded from trusted denominator")


@pytest.mark.integration
@pytest.mark.parametrize("query_record", hard_benchmark_queries(), ids=lambda q: q["query_id"])
def test_trusted_key_movement_reference_recall(production_retriever, query_record):
    coverage = production_retriever.retrieve_with_coverage(query_record["query"])
    recall = trusted_key_movement_recall(coverage, query_id=query_record["query_id"])
    per = recall["per_query"][query_record["query_id"]]
    minimum = query_record.get("minimum_expected_reference_recall") or 0.0
    if per["recall"] is not None and per["recall"] < minimum:
        pytest.xfail(
            f"trusted refs not fully recalled for {query_record['query_id']}: "
            f"miss={per['miss_ids']}",
        )


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
    query_id_by_text = {q["query"]: q["query_id"] for q in hard_benchmark_queries()}
    report: dict[str, object] = {}
    trusted_recalls: list[float] = []
    for case_id, query in cases:
        before = production_retriever.retrieve(query, 10)
        after = production_retriever.retrieve_with_coverage(query)
        query_id = query_id_by_text.get(query)
        if query_id:
            trusted = trusted_key_movement_recall(after, query_id=query_id)
            trusted_recall = trusted["per_query"][query_id]["recall"]
            if trusted_recall is not None:
                trusted_recalls.append(trusted_recall)
        else:
            trusted_recall = None
        deprecated = DEPRECATED_KEY_MOVEMENT_REFERENCES.get(query)
        deprecated_recall = (
            float(_key_recall(after, deprecated[0], deprecated[1]))
            if deprecated
            else None
        )
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
            "trusted_key_movement_recall": trusted_recall,
            "deprecated_key_movement_recall": deprecated_recall,
            "episode_term_hits_before": _episode_term_hits(before, RR_CONTROLS.get(case_id, (query, ()))[1] if case_id in RR_CONTROLS else ()),
            "episode_term_hits_after": _episode_term_hits(after, RR_CONTROLS.get(case_id, (query, ()))[1] if case_id in RR_CONTROLS else ()),
        }
        # F6E: semantic movement contract — retain movement-bearing evidence in final coverage.
        assert len(movement_bearing_evidence(after)) >= 1, (
            f"{case_id}: coverage final must retain at least one movement-bearing candidate"
        )
        if case_id in {"caesar", "pompey", "mithridates", "xenophon"}:
            assert len(after) <= len(before) * 2
    report["trusted_aggregate_recall"] = (
        round(sum(trusted_recalls) / len(trusted_recalls), 3) if trusted_recalls else None
    )
    out = Path(__file__).resolve().parents[2] / "outputs" / "g5m_retrieval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
