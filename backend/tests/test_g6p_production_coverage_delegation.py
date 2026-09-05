"""G6P: production wrapper must delegate retrieve_with_coverage to the inner retriever."""
from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.coverage_retrieval import DEFAULT_COVERAGE_BUDGET, DEFAULT_PER_INTENT_K
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.evidence import SemanticRouteEvidenceRetriever

POMPEY_QUERY = (
    "Trace Pompey's movements after the defeat at Pharsalus, from Greece through "
    "the eastern Mediterranean until his arrival in Egypt in 48 BCE."
)


def _evidence(identifier: str = "ev-1") -> Evidence:
    return Evidence(
        id=identifier,
        author="Author",
        work="Work",
        locator="section",
        excerpt="He marched from Greece toward Egypt.",
        text="He marched from Greece toward Egypt.",
        score=0.5,
        metadata={"document_id": "doc"},
    )


class CoverageDelegateSpy:
    def __init__(self):
        self.coverage_calls: list[dict] = []
        self.retrieve_calls: list[dict] = []
        self.coverage_result = [_evidence()]

    def retrieve_with_coverage(
        self,
        query: str,
        budget: int = DEFAULT_COVERAGE_BUDGET,
        *,
        per_intent_k: int = DEFAULT_PER_INTENT_K,
        filters: dict[str, str] | None = None,
    ) -> list[Evidence]:
        self.coverage_calls.append(
            {
                "query": query,
                "budget": budget,
                "per_intent_k": per_intent_k,
                "filters": filters,
            }
        )
        return list(self.coverage_result)

    def retrieve(self, query: str, top_k: int = 5, filters: dict[str, str] | None = None) -> list[Evidence]:
        self.retrieve_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return []


def test_bug_base_class_fallback_is_not_selected_after_fix():
    assert SemanticRouteEvidenceRetriever.retrieve_with_coverage is not HistoricalRetriever.retrieve_with_coverage


def test_wrapper_delegates_coverage_to_inner_retriever():
    wrapper = SemanticRouteEvidenceRetriever()
    delegate = CoverageDelegateSpy()
    wrapper._retriever = delegate  # type: ignore[attr-defined]

    result = wrapper.retrieve_with_coverage(
        POMPEY_QUERY,
        budget=18,
        per_intent_k=7,
        filters={"author": "Plutarch"},
    )

    assert len(delegate.coverage_calls) == 1
    assert delegate.coverage_calls[0] == {
        "query": POMPEY_QUERY,
        "budget": 18,
        "per_intent_k": 7,
        "filters": {"author": "Plutarch"},
    }
    assert delegate.retrieve_calls == []
    assert result == delegate.coverage_result


def test_wrapper_does_not_use_base_retrieve_fallback_for_coverage():
    wrapper = SemanticRouteEvidenceRetriever()
    delegate = CoverageDelegateSpy()
    wrapper._retriever = delegate  # type: ignore[attr-defined]

    wrapper.retrieve_with_coverage(POMPEY_QUERY, filters={"book": "Lives"})

    assert delegate.coverage_calls
    assert not any(call.get("top_k") == DEFAULT_COVERAGE_BUDGET for call in delegate.retrieve_calls)


def test_agent_tool_path_uses_wrapper_coverage_delegation():
    delegate = CoverageDelegateSpy()
    wrapper = SemanticRouteEvidenceRetriever()
    wrapper._retriever = delegate  # type: ignore[attr-defined]

    state = AgentState(
        session_id="g6p",
        user_query=POMPEY_QUERY,
        requested_output="historical_route",
    )
    AgentToolRegistry(wrapper, object()).execute(
        "search_historical_evidence",
        {"query": POMPEY_QUERY, "top_k": 10, "author": "Plutarch"},
        state,
    )

    assert len(delegate.coverage_calls) == 1
    assert delegate.coverage_calls[0]["query"] == POMPEY_QUERY
    assert delegate.coverage_calls[0]["filters"] == {"author": "Plutarch"}
    assert delegate.retrieve_calls == []
