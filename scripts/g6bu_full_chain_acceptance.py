"""G6BU: offline trusted full-chain acceptance harness."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.agent.tools import AgentToolRegistry
from backend.app.core.config import settings
from backend.app.models import AgentState
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    DEFAULT_RAW_OBSERVATION_K,
    merge_coverage_results,
    select_qualified_local_proposals,
)
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query
from backend.tests.g5r_trusted_benchmark import (
    hard_benchmark_queries,
    references_for_query,
    trusted_key_movement_recall,
)

STAGE_ORDER = (
    "query",
    "retrieval",
    "evidence",
    "event",
    "geography",
    "relation",
    "route",
    "presentation",
)

CASE_LABELS = {
    "caesar_adriatic": "Caesar",
    "pompey_post_pharsalus_egypt": "Pompey",
    "mithridates_first_war": "Mithridates",
    "lucullus_mithridatic": "Lucullus",
    "alexander_hydaspes": "Alexander",
}


class OfflineGeography:
    def call(self, tool: str, arguments: dict) -> dict:
        from geography_mcp.service import GeographyService

        service = GeographyService()
        if tool == "resolve_ancient_place":
            return service.resolve_ancient_place_payload(arguments["name"])
        if tool == "calculate_distance":
            return service.calculate_distance_payload(arguments["point_a"], arguments["point_b"])
        if tool == "get_elevation":
            return service.get_elevation_payload(arguments["latitude"], arguments["longitude"])
        if tool == "get_elevation_profile":
            return service.get_elevation_profile_payload(arguments["points"])
        raise ValueError(f"Unsupported geography tool: {tool}")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def build_retriever():
    import chromadb

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
        ROOT / "backend" / "data" / "chroma_server_roman_republic_v2",
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


def retrieval_trace(retriever, query: str) -> dict:
    channels = [RetrievalIntent("CANONICAL", query), *decompose_movement_query(query)]
    raw_ids: set[str] = set()
    proposal_ids: set[str] = set()
    for intent in channels:
        pool = retriever._collect_candidates(intent.query, semantic_k=60, lexical_k=60)
        raw_ids.update(item.id for item in pool)
        ranked = rerank_evidence(intent.query, pool, pool_relative=False)
        proposal_ids.update(item.id for item in select_qualified_local_proposals(ranked, 60))
    intent_results = [
        (intent, retriever.retrieve_candidates(intent.query, DEFAULT_RAW_OBSERVATION_K))
        for intent in channels
    ]
    union_items = merge_coverage_results(query, intent_results, DEFAULT_COVERAGE_BUDGET)
    final_items = retriever.retrieve_with_coverage(query, DEFAULT_COVERAGE_BUDGET)
    return {
        "raw_count": len(raw_ids),
        "proposal_count": len(proposal_ids),
        "union_count": len(union_items),
        "final_count": len(final_items),
        "final_ids": [item.id for item in final_items],
        "raw_ids": sorted(raw_ids),
        "proposal_ids": sorted(proposal_ids),
        "union_ids": [item.id for item in union_items],
    }


def sibling_equivalents(final_ids: list[str], trusted_ids: set[str]) -> list[dict]:
    siblings: list[dict] = []
    for trusted in trusted_ids:
        prefix = trusted.rsplit(":", 1)[0]
        for candidate in final_ids:
            if candidate != trusted and candidate.startswith(prefix + ":"):
                siblings.append({"trusted_id": trusted, "equivalent_sibling_id": candidate, "label": "equivalent_sibling"})
    return siblings


def stage_status(name: str, payload: dict) -> str:
    return payload["status"]


def classify_query(query: str) -> dict:
    status = "PASS" if query.strip() else "FAIL"
    return {"status": status, "query": query}


def classify_retrieval(trace: dict, recall: dict, refs: list[dict]) -> dict:
    expected = {ref["evidence_id"] for ref in refs}
    final_ids = trace["final_ids"]
    exact_hits = sorted(set(final_ids) & expected)
    siblings = sibling_equivalents(final_ids, expected)
    per_query = recall["per_query"]
    if trace["final_count"] == 0:
        status = "FAIL"
    elif exact_hits or float(per_query.get("recall") or 0) >= 0.5:
        status = "PASS"
    elif siblings or trace["final_count"] > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "final_count": trace["final_count"],
        "exact_trusted_hits": exact_hits,
        "equivalent_siblings": siblings,
        "trusted_recall": per_query,
        "raw_count": trace["raw_count"],
        "proposal_count": trace["proposal_count"],
        "union_count": trace["union_count"],
    }


def classify_evidence(evidence: list, recall: dict) -> dict:
    if not evidence:
        return {"status": "FAIL", "usable_count": 0}
    recall_value = recall.get("aggregate_recall")
    status = "PASS" if evidence else "FAIL"
    if evidence and recall_value is not None and recall_value < 0.5:
        status = "PARTIAL"
    return {
        "status": status,
        "usable_count": len(evidence),
        "sources": sorted({item.work for item in evidence}),
        "evidence_ids": [item.id for item in evidence[:20]],
    }


def classify_event(events: list, diagnostics: dict) -> dict:
    extraction = diagnostics.get("extraction") or {}
    reason_codes = list(extraction.get("reason_codes") or [])
    movement_events = [event for event in events if getattr(event.event_type, "value", str(event.event_type)) == "MOVEMENT"]
    if not events:
        status = "FAIL"
    elif movement_events:
        status = "PASS" if "PLACE_UNRESOLVED" not in reason_codes else "PARTIAL"
    else:
        status = "PARTIAL"
    return {
        "status": status,
        "event_count": len(events),
        "movement_event_count": len(movement_events),
        "reason_codes": reason_codes,
        "events": [
            {
                "id": event.id,
                "summary": event.summary,
                "event_type": getattr(event.event_type, "value", str(event.event_type)),
                "evidence_refs": list(event.evidence_refs),
            }
            for event in events[:10]
        ],
    }


def classify_geography(diagnostics: dict) -> dict:
    place = diagnostics.get("place_resolution") or {}
    resolved = int(place.get("resolved_place_count") or 0)
    unresolved = int(place.get("unresolved_place_count") or 0)
    unlocated = int(place.get("unlocated_place_count") or 0)
    unavailable = int(place.get("unavailable_place_count") or 0)
    ambiguous = int(place.get("ambiguous_count") or 0)
    if resolved == 0 and unresolved + unlocated + unavailable + ambiguous == 0:
        status = "FAIL"
    elif unresolved or ambiguous or unavailable:
        status = "PARTIAL"
    else:
        status = "PASS"
    return {
        "status": status,
        "resolved": resolved,
        "unresolved": unresolved,
        "unlocated": unlocated,
        "unavailable": unavailable,
        "ambiguous": ambiguous,
        "reason_codes": list(place.get("reason_codes") or []),
    }


def classify_relation(route_diagnostics: dict | None) -> dict:
    if not route_diagnostics:
        return {"status": "SKIPPED", "relation_count": 0}
    trace = route_diagnostics.get("provenance_trace") or {}
    edges = trace.get("final_edges") or []
    event_first = trace.get("event_first") or {}
    ordering_count = int(event_first.get("ordering_relation_count") or 0)
    reason_codes = list(route_diagnostics.get("reason_codes") or [])
    relation_count = len(edges) if edges else ordering_count
    if edges:
        status = "PASS"
    elif ordering_count > 0:
        status = "PARTIAL"
    elif "PARTIAL_ROUTE" in reason_codes or "INSUFFICIENT_ORDERING" in reason_codes:
        status = "PARTIAL"
    elif route_diagnostics.get("route_source") == "none":
        status = "FAIL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "relation_count": relation_count,
        "reason_codes": reason_codes,
        "relations": edges[:10],
    }


def classify_route(route, route_diagnostics: dict | None) -> dict:
    if route is None:
        return {"status": "FAIL", "route_status": "FAIL", "reason_codes": list((route_diagnostics or {}).get("reason_codes") or [])}
    ordered = len(route.ordered_points)
    components = len(route.route_components or [])
    branches = len(route.branch_relations or [])
    reason_codes = list((route_diagnostics or {}).get("reason_codes") or [])
    if ordered >= 2:
        status = "PASS"
    elif components or branches or "PARTIAL_ROUTE" in reason_codes:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "status": status,
        "route_status": status,
        "ordered_points": ordered,
        "components": components,
        "branches": branches,
        "reason_codes": reason_codes,
        "evidence_refs": list(route.evidence_refs[:20]),
    }


def classify_presentation(presentation: dict | None, route_diagnostics: dict | None, route_stage: dict) -> dict:
    if route_stage["status"] == "FAIL":
        return {"status": "SKIPPED", "reason": "route_failed"}
    gis = (route_diagnostics or {}).get("gis_reconstruction") or {}
    if presentation:
        gis_status = str(gis.get("status") or "COMPLETE")
        status = "PASS" if gis_status == "COMPLETE" else "PARTIAL"
        return {"status": status, "gis_status": gis_status, "pipeline": gis.get("pipeline")}
    if gis.get("attempted"):
        return {"status": "PARTIAL", "gis_status": gis.get("status"), "reason_code": gis.get("reason_code")}
    return {"status": "SKIPPED", "reason": "SKIPPED_LIVE_LLM_DEPENDENCY"}


def first_divergence(stages: dict) -> str:
    for stage in STAGE_ORDER:
        status = stages[stage]["status"]
        if status != "PASS":
            return stage.upper()
    return "NONE"


def deepest_successful_stage(stages: dict) -> str:
    deepest = "NONE"
    for stage in STAGE_ORDER:
        if stages[stage]["status"] == "PASS":
            deepest = stage.upper()
        else:
            break
    return deepest


def run_case(retriever, tools: AgentToolRegistry, query_def: dict) -> dict:
    query_id = query_def["query_id"]
    query = query_def["query"]
    refs = references_for_query(query_id)
    trace = retrieval_trace(retriever, query)
    recall = trusted_key_movement_recall(
        [item for item in retriever.retrieve_with_coverage(query, DEFAULT_COVERAGE_BUDGET)],
        query_id=query_id,
    )

    state = AgentState(
        session_id=f"g6bu-{query_id}",
        user_query=query,
        requested_output="historical_route",
    )
    tools.execute("search_historical_evidence", {"query": query, "top_k": 20}, state)
    route_args = {
        "event_id": query_id,
        "name": query_def.get("subject") or query_id,
        "period": "unspecified",
    }
    build_result, _ = tools.execute("build_historical_route", route_args, state)
    route_diagnostics = state.historical_route_diagnostics or {}

    stages = {
        "query": classify_query(query),
        "retrieval": classify_retrieval(trace, recall, refs),
        "evidence": classify_evidence(state.historical_evidence, recall),
        "event": classify_event(state.historical_events, state.historical_event_diagnostics or {}),
        "geography": classify_geography(state.historical_event_diagnostics or {}),
        "relation": classify_relation(route_diagnostics),
        "route": classify_route(state.historical_route, route_diagnostics),
        "presentation": classify_presentation(
            state.historical_route_presentation,
            route_diagnostics,
            classify_route(state.historical_route, route_diagnostics),
        ),
    }
    route_status = stages["route"]["route_status"]
    return {
        "id": query_id,
        "label": CASE_LABELS.get(query_id, query_id),
        "query": query,
        "stages": stages,
        "first_divergence": first_divergence(stages),
        "deepest_successful_stage": deepest_successful_stage(stages),
        "route_status": route_status,
        "build_success": bool(build_result.get("success")),
    }


def aggregate_metrics(cases: list[dict]) -> dict:
    route_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    divergence_counts = {stage.upper(): 0 for stage in STAGE_ORDER}
    divergence_counts["NONE"] = 0
    for case in cases:
        route_counts[case["route_status"]] = route_counts.get(case["route_status"], 0) + 1
        divergence_counts[case["first_divergence"]] = divergence_counts.get(case["first_divergence"], 0) + 1
    return {"route_counts": route_counts, "first_divergence": divergence_counts}


def print_table(cases: list[dict]) -> None:
    header = (
        f"{'Case':<12} {'Retrieval':<10} {'Evidence':<10} {'Event':<8} {'Geography':<10} "
        f"{'Relation':<10} {'Route':<8} {'Presentation':<12} {'First divergence'}"
    )
    print(header)
    for case in cases:
        stages = case["stages"]
        print(
            f"{case['label']:<12} "
            f"{stages['retrieval']['status']:<10} "
            f"{stages['evidence']['status']:<10} "
            f"{stages['event']['status']:<8} "
            f"{stages['geography']['status']:<10} "
            f"{stages['relation']['status']:<10} "
            f"{stages['route']['status']:<8} "
            f"{stages['presentation']['status']:<12} "
            f"{case['first_divergence']}"
        )


def main() -> int:
    head = git_head()
    retriever = build_retriever()
    tools = AgentToolRegistry(retriever, OfflineGeography())
    cases = [run_case(retriever, tools, query_def) for query_def in hard_benchmark_queries()]
    metrics = aggregate_metrics(cases)
    payload = {"head": head, "cases": cases, "metrics": metrics}
    out_path = ROOT / "outputs" / "g6bu_full_chain_acceptance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_table(cases)
    print()
    print(f"PASS routes: {metrics['route_counts'].get('PASS', 0)}/5")
    print(f"PARTIAL routes: {metrics['route_counts'].get('PARTIAL', 0)}/5")
    print(f"FAIL routes: {metrics['route_counts'].get('FAIL', 0)}/5")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
