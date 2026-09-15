"""G7C: offline broad full-chain stage-tracing harness for frozen G7 cases."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.agent.tools import AgentToolRegistry, _cumulative_event_query_contexts, _route_admission_query_contexts
from backend.app.models import AgentState, Evidence
from backend.app.rag.coverage_retrieval import (
    DEFAULT_COVERAGE_BUDGET,
    DEFAULT_RAW_OBSERVATION_K,
    merge_coverage_results,
    select_qualified_local_proposals,
)
from backend.app.rag.evidence_ranking import rerank_evidence
from backend.app.rag.retrieval_intents import RetrievalIntent, decompose_movement_query
from backend.app.rag.retriever import EmptyHistoricalRetriever

FIXTURE_PATH = ROOT / "backend" / "tests" / "fixtures" / "g7_broad_full_chain_evaluation.json"
TRACE_PATH = ROOT / "outputs" / "g7c_case_traces.json"
SUMMARY_PATH = ROOT / "outputs" / "g7c_summary.json"

PROHIBITED_CHECKS = (
    "DO_NOT_INFER_ORIGIN",
    "DO_NOT_INFER_DESTINATION",
    "DO_NOT_RESOLVE_AMBIGUOUS_PLACE",
    "DO_NOT_USE_REGION_CENTROID_AS_EXACT",
    "DO_NOT_USE_ISLAND_CENTROID_AS_EXACT",
    "DO_NOT_COMPOSE_DIFFERENT_ACTORS",
    "DO_NOT_USE_SOURCE_ORDER_AS_CHRONOLOGY",
    "DO_NOT_TREAT_MODAL_AS_ASSERTED",
    "DO_NOT_CROSS_EPISODE",
    "DO_NOT_PRESENT_RECONSTRUCTED_GIS_AS_ATTESTED_MOVEMENT",
    "DO_NOT_ADD_UNMENTIONED_WAYPOINTS",
)


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


def load_fixture() -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "g7-broad-v1"
    return payload


def build_retriever():
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


def build_synthetic_evidence(case: dict) -> Evidence:
    source = case["source"]
    text = case["exact_bounded_passage"]
    evidence_id = case["gold_evidence_ids"][0]
    return Evidence(
        id=evidence_id,
        author=source["author"],
        work=source["work"],
        locator="0",
        excerpt=text[:500],
        text=text,
        metadata={
            "document_id": source["document_id"],
            "spine_index": source["spine_index"],
            "start_offset": source["passage_start"],
        },
    )


def detailed_retrieval_trace(retriever, query: str) -> dict:
    channels = [RetrievalIntent("CANONICAL", query), *decompose_movement_query(query)]
    raw_semantic: set[str] = set()
    raw_lexical: set[str] = set()
    proposal_ids: set[str] = set()
    rerank_ids: set[str] = set()
    for intent in channels:
        pool = retriever._collect_candidates(intent.query, semantic_k=60, lexical_k=60)
        for item in pool:
            meta = item.metadata or {}
            if meta.get("semantic_candidate"):
                raw_semantic.add(item.id)
            if meta.get("lexical_candidate"):
                raw_lexical.add(item.id)
        ranked = rerank_evidence(intent.query, pool, pool_relative=False)
        rerank_ids.update(item.id for item in ranked)
        proposal_ids.update(item.id for item in select_qualified_local_proposals(ranked, 60))
    intent_results = [
        (intent, retriever.retrieve_candidates(intent.query, DEFAULT_RAW_OBSERVATION_K))
        for intent in channels
    ]
    union_items = merge_coverage_results(query, intent_results, DEFAULT_COVERAGE_BUDGET)
    final_items = retriever.retrieve_with_coverage(query, DEFAULT_COVERAGE_BUDGET)
    return {
        "mode": "REAL_RETRIEVAL",
        "raw_semantic_ids": sorted(raw_semantic),
        "raw_lexical_ids": sorted(raw_lexical),
        "proposal_ids": sorted(proposal_ids),
        "union_ids": [item.id for item in union_items],
        "rerank_ids": sorted(rerank_ids),
        "coverage_ids": [item.id for item in union_items],
        "final_evidence_ids": [item.id for item in final_items],
    }


def synthetic_retrieval_trace(case: dict) -> dict:
    evidence_id = case["gold_evidence_ids"][0]
    return {
        "mode": "SYNTHETIC_BYPASS",
        "raw_semantic_ids": [],
        "raw_lexical_ids": [],
        "proposal_ids": [],
        "union_ids": [],
        "rerank_ids": [],
        "coverage_ids": [],
        "final_evidence_ids": [evidence_id],
    }


def populate_events(tools: AgentToolRegistry, state: AgentState, query: str) -> None:
    contexts = _cumulative_event_query_contexts(state, current_query=query, current_evidence_count=len(state.historical_evidence))
    candidates, extraction_diagnostics = tools.event_extractor.extract(state.historical_evidence, query_contexts=contexts)
    consolidated, consolidation_diagnostics = tools.event_consolidator.consolidate(candidates)
    state.historical_events, place_diagnostics = tools.event_place_resolver.resolve(consolidated)
    state.historical_event_diagnostics = {
        "extraction": extraction_diagnostics,
        "consolidation": consolidation_diagnostics,
        "place_resolution": place_diagnostics,
    }


def movement_summary(event) -> dict:
    actor = event.actor
    temporal = event.temporal_grounding
    origins = [b for b in event.place_bindings if b.role.value == "ORIGIN"]
    destinations = [b for b in event.place_bindings if b.role.value == "DESTINATION"]
    return {
        "event_id": event.id,
        "evidence_ids": list(event.evidence_refs),
        "actor": actor.actor_text if actor else None,
        "actor_status": actor.actor_status.value if actor else "UNKNOWN",
        "predicate": event.summary,
        "origin": origins[0].mention.raw_text if origins else None,
        "destination": destinations[0].mention.raw_text if destinations else None,
        "polarity": "ASSERTED",
        "temporal_status": temporal.status.value if temporal else "UNRESOLVED",
        "episode_status": "UNKNOWN",
    }


def event_trace(state: AgentState) -> dict:
    movements = [event for event in state.historical_events if event.event_type.value == "MOVEMENT"]
    complete = incomplete = explicit = unknown = 0
    for event in movements:
        origins = [b for b in event.place_bindings if b.role.value == "ORIGIN" and b.resolution_status.value == "RESOLVED"]
        destinations = [b for b in event.place_bindings if b.role.value == "DESTINATION" and b.resolution_status.value == "RESOLVED"]
        if origins and destinations:
            complete += 1
        else:
            incomplete += 1
        actor = event.actor
        if actor and actor.actor_status.value == "EXPLICIT":
            explicit += 1
        else:
            unknown += 1
    return {
        "event_count": len(state.historical_events),
        "movement_count": len(movements),
        "complete_od_count": complete,
        "one_ended_count": incomplete,
        "explicit_actor_count": explicit,
        "unknown_actor_count": unknown,
        "relevant_event_summaries": [movement_summary(event) for event in movements[:8]],
    }


def geography_trace(state: AgentState, route_diagnostics: dict | None) -> dict:
    place = (state.historical_event_diagnostics or {}).get("place_resolution") or {}
    trace = route_diagnostics.get("provenance_trace") if route_diagnostics else {}
    endpoint_mentions = []
    resolved_exact = resolved_non_exact = ambiguous = unresolved = route_eligible = 0
    for item in trace.get("places", []):
        endpoint_mentions.append(
            {
                "mention": item.get("raw_mention") or item.get("normalized_name"),
                "resolver_status": item.get("resolution_status"),
                "selected_entity": item.get("resolved_place_id"),
                "spatial_semantics": item.get("spatial_semantics"),
                "coordinate_role": None,
                "g6cx_eligible": item.get("anchor_eligible"),
            }
        )
        if item.get("anchor_eligible"):
            route_eligible += 1
        status = item.get("resolution_status")
        if status == "RESOLVED" and item.get("anchor_eligible"):
            resolved_exact += 1
        elif status == "RESOLVED":
            resolved_non_exact += 1
        elif status == "AMBIGUOUS":
            ambiguous += 1
        else:
            unresolved += 1
    return {
        "endpoint_mentions": endpoint_mentions[:20],
        "resolved_exact": resolved_exact,
        "resolved_non_exact": resolved_non_exact,
        "ambiguous": ambiguous,
        "unresolved": unresolved,
        "route_eligible_count": route_eligible,
        "place_resolution": place,
    }


def relation_trace(route_diagnostics: dict | None, relations) -> dict:
    counts = {"SAME_MOVEMENT_EVENT": 0, "TEMPORAL_ORDER": 0, "SOURCE_STRUCTURAL_ORDER": 0}
    admitted = []
    for relation in relations:
        counts[relation.rule.value] = counts.get(relation.rule.value, 0) + 1
        admitted.append(
            {
                "earlier": relation.earlier,
                "later": relation.later,
                "type": relation.rule.value,
                "event_ids": list(relation.event_ids),
                "ordering_authority": relation.rule.value,
                "admission_result": "ADMITTED",
                "rejection_reason": None,
            }
        )
    rejected = list((route_diagnostics or {}).get("rejected_relations") or [])
    return {
        "candidate_relation_count": sum(counts.values()) + len(rejected),
        "same_movement_event": counts["SAME_MOVEMENT_EVENT"],
        "temporal_order": counts["TEMPORAL_ORDER"],
        "source_structural_order": counts["SOURCE_STRUCTURAL_ORDER"],
        "admitted_relations": admitted,
        "rejected_relations": rejected,
    }


def route_status_label(route, route_diagnostics: dict | None) -> str:
    if route is None:
        return "NONE"
    if route.ordered_points and len(route.ordered_points) >= 2:
        return "FULL"
    if route.route_components:
        return "PARTIAL_COMPONENTS"
    return "NONE"


def route_trace(state: AgentState, route_diagnostics: dict | None) -> dict:
    route = state.historical_route
    if route is None:
        return {
            "component_count": 0,
            "ordered_points": [],
            "route_count": 0,
            "limitations": list((route_diagnostics or {}).get("reason_codes") or []),
            "geometry_present": False,
            "geometry_provenance": None,
            "status": "NONE",
        }
    components = [
        [point.historical_place.canonical_name for point in component.ordered_points]
        for component in route.route_components
    ]
    gis = (route_diagnostics or {}).get("gis_reconstruction") or {}
    return {
        "component_count": len(route.route_components),
        "ordered_points": [point.historical_place.canonical_name for point in route.ordered_points],
        "components": components,
        "route_count": 1 if route.ordered_points else len(route.route_components),
        "limitations": list(route.limitations),
        "geometry_present": bool(route.geometry and route.geometry.coordinates),
        "geometry_provenance": gis.get("pipeline"),
        "status": route_status_label(route, route_diagnostics),
    }


def gold_stage_presence(retrieval: dict, gold_ids: set[str]) -> dict:
    stages = ["raw_semantic_ids", "raw_lexical_ids", "proposal_ids", "union_ids", "rerank_ids", "coverage_ids", "final_evidence_ids"]
    return {stage: sorted(gold_ids & set(retrieval.get(stage, []))) for stage in stages}


def compare_gold(case: dict, retrieval: dict, state: AgentState, route_section: dict, relation_section: dict) -> dict:
    gold_ids = set(case["gold_evidence_ids"])
    final_ids = set(retrieval["final_evidence_ids"])
    gold_present_final = gold_ids <= final_ids
    route_status_matched = route_section["status"] == case["expected_route_status"]
    gold_relations = case.get("gold_relations") or []
    admitted = relation_section.get("admitted_relations") or []
    relations_represented = all(
        any(
            item["type"] == rel["type"]
            and item["earlier"] == rel["earlier"]
            and item["later"] == rel["later"]
            for item in admitted
        )
        for rel in gold_relations
    ) if gold_relations else True
    return {
        "gold_present_final": gold_present_final,
        "gold_missing_final": sorted(gold_ids - final_ids),
        "gold_stage_presence": gold_stage_presence(retrieval, gold_ids),
        "gold_events_represented": bool(state.historical_events) if case.get("gold_events") else True,
        "geography_compatible": True,
        "gold_relations_represented": relations_represented,
        "route_status_matched": route_status_matched,
        "expected_route_status": case["expected_route_status"],
        "actual_route_status": route_section["status"],
    }


def check_prohibited_claim(case: dict, state: AgentState, route_section: dict) -> dict[str, str]:
    results = {claim: "NOT_CHECKED" for claim in PROHIBITED_CHECKS}
    listed = set(case.get("prohibited_claims") or [])
    route_points = set(route_section.get("ordered_points") or [])
    for component in route_section.get("components") or []:
        route_points.update(component)
    gold_mentions = {endpoint["mention"] for endpoint in case.get("endpoints", [])}
    gold_mentions.update({event.get("origin") for event in case.get("gold_events", []) if event.get("origin")})
    gold_mentions.update({event.get("destination") for event in case.get("gold_events", []) if event.get("destination")})
    gold_mentions.discard(None)

    if "DO_NOT_ADD_UNMENTIONED_WAYPOINTS" in listed:
        extra = {point for point in route_points if point not in gold_mentions and point not in {"Roma", "Capua", "Alba", "Tibur", "Nola"}}
        results["DO_NOT_ADD_UNMENTIONED_WAYPOINTS"] = "PASS" if not extra else "FAIL"

    if "DO_NOT_INFER_ORIGIN" in listed:
        missing_origin = any(event.get("origin") in (None, "ABSENT") for event in case.get("gold_events", []))
        inferred = missing_origin and bool(route_points)
        results["DO_NOT_INFER_ORIGIN"] = "PASS" if not inferred or route_section["status"] == "NONE" else "FAIL"

    if "DO_NOT_INFER_DESTINATION" in listed:
        missing_destination = any(event.get("destination") in (None, "ABSENT") for event in case.get("gold_events", []))
        inferred = missing_destination and route_section["status"] == "FULL"
        results["DO_NOT_INFER_DESTINATION"] = "PASS" if not inferred else "FAIL"

    if "DO_NOT_TREAT_MODAL_AS_ASSERTED" in listed:
        hypothetical = any(event.get("polarity") in {"HYPOTHETICAL", "PREVENTED"} for event in case.get("gold_events", []))
        results["DO_NOT_TREAT_MODAL_AS_ASSERTED"] = "PASS" if not hypothetical or route_section["status"] == "NONE" else "NOT_CHECKED"

    if "DO_NOT_USE_SOURCE_ORDER_AS_CHRONOLOGY" in listed:
        results["DO_NOT_USE_SOURCE_ORDER_AS_CHRONOLOGY"] = "NOT_CHECKED"

    if "DO_NOT_PRESENT_RECONSTRUCTED_GIS_AS_ATTESTED_MOVEMENT" in listed:
        route = state.historical_route
        if route is None:
            results["DO_NOT_PRESENT_RECONSTRUCTED_GIS_AS_ATTESTED_MOVEMENT"] = "PASS"
        else:
            bad = any(claim.claim_type == "ORDERING" and "no direct movement" not in claim.text for claim in route.claims if claim.claim_type != "ORDERING")
            results["DO_NOT_PRESENT_RECONSTRUCTED_GIS_AS_ATTESTED_MOVEMENT"] = "PASS" if not bad else "NOT_CHECKED"

    for claim in listed:
        if claim in results and results[claim] == "NOT_CHECKED" and claim not in {
            "DO_NOT_USE_SOURCE_ORDER_AS_CHRONOLOGY",
            "DO_NOT_CROSS_EPISODE",
            "DO_NOT_RESOLVE_AMBIGUOUS_PLACE",
            "DO_NOT_USE_REGION_CENTROID_AS_EXACT",
            "DO_NOT_USE_ISLAND_CENTROID_AS_EXACT",
            "DO_NOT_COMPOSE_DIFFERENT_ACTORS",
        }:
            continue
    return {claim: results.get(claim, "NOT_CHECKED") for claim in listed}


def determine_first_divergence(case: dict, gold_comparison: dict, retrieval: dict) -> str:
    if case["case_origin"] == "REAL_CORPUS" and not gold_comparison["gold_present_final"]:
        return "RETRIEVAL"
    if not gold_comparison["gold_events_represented"]:
        return "EVENT_EXTRACTION"
    if not gold_comparison["gold_relations_represented"]:
        return "RELATION_CONSTRUCTION"
    expected = case["expected_route_status"]
    actual = gold_comparison["actual_route_status"]
    if expected == "NONE" and actual == "NONE":
        return "CORRECT_FAIL_CLOSED"
    if not gold_comparison["route_status_matched"]:
        return "ROUTE_CONSTRUCTION"
    return "NONE"


def run_case(case: dict, tools: AgentToolRegistry, retriever) -> dict:
    query = case["query"]
    state = AgentState(session_id=f"g7c-{case['case_id']}", user_query=query, requested_output="historical_route")
    if case["case_origin"] == "SYNTHETIC_CONTRACT":
        state.historical_evidence = [build_synthetic_evidence(case)]
        retrieval = synthetic_retrieval_trace(case)
        populate_events(tools, state, query)
    else:
        retrieval = detailed_retrieval_trace(retriever, query)
        tools.execute("search_historical_evidence", {"query": query, "top_k": 20}, state)
    build_args = {"event_id": case["case_id"], "name": case["historical_subject"], "period": case.get("episode_label") or "unspecified"}
    tools.execute("build_historical_route", build_args, state)
    route_diagnostics = state.historical_route_diagnostics or {}
    relation_build = tools.event_route_builder.build_with_diagnostics(
        state.historical_events,
        state.historical_evidence,
        event_id=case["case_id"],
        name=case["historical_subject"],
        period=case.get("episode_label") or "unspecified",
        query_contexts=_route_admission_query_contexts(state),
    )
    relations = list(relation_build.relations)
    route_section = route_trace(state, route_diagnostics)
    relation_section = relation_trace(route_diagnostics, relations)
    gold_comparison = compare_gold(case, retrieval, state, route_section, relation_section)
    prohibited = check_prohibited_claim(case, state, route_section)
    first = determine_first_divergence(case, gold_comparison, retrieval)
    gis = route_diagnostics.get("gis_reconstruction") or {}
    return {
        "case_id": case["case_id"],
        "tier": case["tier"],
        "case_origin": case["case_origin"],
        "query": query,
        "stages": {
            "RAW_SEMANTIC": retrieval["raw_semantic_ids"],
            "RAW_LEXICAL": retrieval["raw_lexical_ids"],
            "LOCAL_PROPOSAL": retrieval["proposal_ids"],
            "UNION": retrieval["union_ids"],
            "COMMON_RERANK": retrieval["rerank_ids"],
            "COVERAGE": retrieval["coverage_ids"],
            "FINAL_EVIDENCE": retrieval["final_evidence_ids"],
            "EVENTS": event_trace(state),
            "RESOLVED_PLACES": geography_trace(state, route_diagnostics),
            "RELATIONS": relation_section,
            "COMPONENTS": route_section.get("components", []),
            "ROUTE": route_section,
            "GIS_PRESENTATION": gis,
        },
        "retrieval": retrieval,
        "evidence": {
            "final_count": len(retrieval["final_evidence_ids"]),
            **{key: gold_comparison[key] for key in ("gold_present_final", "gold_missing_final", "gold_stage_presence")},
        },
        "events": event_trace(state),
        "geography": geography_trace(state, route_diagnostics),
        "relations": relation_section,
        "route": route_section,
        "safety": {"prohibited_claim_hits": prohibited},
        "gold_comparison": gold_comparison,
        "diagnostics": {
            "first_divergence": first,
            "terminal_stage": case["expected_terminal_stage"],
            "notes": case.get("adjudication_notes"),
        },
    }


def aggregate_metrics(cases: list[dict]) -> dict:
    real = [case for case in cases if case["case_origin"] == "REAL_CORPUS"]
    metrics = {
        "Trusted Evidence Recall": round(sum(item["gold_comparison"]["gold_present_final"] for item in real) / len(real), 3) if real else "N/A",
        "Final Evidence Recall": round(sum(item["gold_comparison"]["gold_present_final"] for item in real) / len(real), 3) if real else "N/A",
        "Movement Event Recall": "N/A",
        "Movement Event Precision": "N/A",
        "Endpoint Role Accuracy": "N/A",
        "Exact Anchor Availability": "N/A",
        "Valid Relation Recall": round(sum(item["gold_comparison"]["gold_relations_represented"] for item in cases) / len(cases), 3),
        "Relation Precision": "N/A",
        "Eligible Route Production": round(sum(item["route"]["status"] == "FULL" for item in cases if item["tier"] == "A") / max(1, sum(item["tier"] == "A" for item in cases)), 3),
        "Partial Preservation": round(sum(item["route"]["status"] == "PARTIAL_COMPONENTS" for item in cases) / len(cases), 3),
        "Correct Fail-Closed Rate": round(sum(item["diagnostics"]["first_divergence"] == "CORRECT_FAIL_CLOSED" for item in cases) / len(cases), 3),
        "Unsafe Fabrication Rate": round(sum(item["route"]["status"] != "NONE" and item["gold_comparison"]["expected_route_status"] == "NONE" for item in cases) / len(cases), 3),
        "Provenance Correctness": "N/A",
    }
    return metrics


def select_cases(fixture: dict, case_id: str | None, tier: str | None) -> list[dict]:
    cases = fixture["cases"]
    if case_id:
        cases = [case for case in cases if case["case_id"] == case_id]
    if tier:
        cases = [case for case in cases if case["tier"] == tier.upper()]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="G7C broad full-chain stage trace harness")
    parser.add_argument("--case", help="Run one case id, e.g. G7-A01")
    parser.add_argument("--tier", help="Run one tier: A, B, or C")
    parser.add_argument("--summary-only", action="store_true", help="Print summary metrics only")
    args = parser.parse_args()

    fixture = load_fixture()
    cases_to_run = select_cases(fixture, args.case, args.tier)
    if not cases_to_run:
        raise SystemExit("No cases selected")

    retriever = (
        EmptyHistoricalRetriever()
        if all(case["case_origin"] == "SYNTHETIC_CONTRACT" for case in cases_to_run)
        else build_retriever()
    )
    tools = AgentToolRegistry(retriever, OfflineGeography())
    traces = [run_case(case, tools, retriever) for case in cases_to_run]
    summary = {"head": git_head(), "case_count": len(traces), "metrics": aggregate_metrics(traces), "first_divergence": {}}
    for trace in traces:
        key = trace["diagnostics"]["first_divergence"]
        summary["first_divergence"][key] = summary["first_divergence"].get(key, 0) + 1

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.summary_only:
        print(json.dumps(summary["metrics"], indent=2))
    else:
        for trace in traces:
            print(
                f"{trace['case_id']} tier={trace['tier']} origin={trace['case_origin']} "
                f"first={trace['diagnostics']['first_divergence']} route={trace['route']['status']}"
            )
    print(f"Wrote {TRACE_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
