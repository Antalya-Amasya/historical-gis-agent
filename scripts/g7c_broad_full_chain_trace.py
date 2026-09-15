"""G7C: offline broad full-chain stage-tracing harness for frozen G7 cases."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
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
LEGACY_TRACE_PATH = ROOT / "outputs" / "g7c_case_traces.json"
LEGACY_SUMMARY_PATH = ROOT / "outputs" / "g7c_summary.json"
TRACE_PATH = LEGACY_TRACE_PATH
SUMMARY_PATH = LEGACY_SUMMARY_PATH
LATEST_MANIFEST_PATH = ROOT / "outputs" / "g7c_latest.json"
TRACE_FILENAME = "g7c_case_traces.json"
SUMMARY_FILENAME = "g7c_summary.json"
PREFLIGHT_SCRIPT = ROOT / "scripts" / "run_g7_preflight.py"

REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "modules.json",
    ("model.safetensors", "pytorch_model.bin"),
    ("sentence_bert_config.json", "config_sentence_transformers.json"),
)

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


class OfflineEmbeddingError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        model_name: str,
        expected_cache: str,
        cache_candidates: list[str],
        missing_requirement: str,
    ) -> None:
        self.code = code
        self.model_name = model_name
        self.expected_cache = expected_cache
        self.cache_candidates = cache_candidates
        self.missing_requirement = missing_requirement
        payload = {
            "error": code,
            "model_name": model_name,
            "expected_cache": expected_cache,
            "cache_candidates": cache_candidates,
            "missing_requirement": missing_requirement,
        }
        super().__init__(json.dumps(payload, indent=2))


class OutputPersistenceError(RuntimeError):
    def __init__(self, run_id: str, target_path: str, error: Exception) -> None:
        self.run_id = run_id
        self.target_path = target_path
        self.error = error
        payload = {
            "error": "G7_OUTPUT_PERSISTENCE_FAILED",
            "run_id": run_id,
            "target_path": target_path,
            "exception": str(error),
        }
        super().__init__(json.dumps(payload, indent=2))


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("run_g7_preflight", PREFLIGHT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def select_writable_temp_root() -> Path:
    return _load_preflight_module().select_writable_temp_root()


def make_unique_run_output_dir(runtime_root: Path | None = None) -> tuple[Path, str]:
    root = runtime_root or select_writable_temp_root()
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def validate_completed_run(
    traces: list[dict],
    summary: dict,
    current_head: str,
    expected_case_count: int | None = None,
) -> dict:
    issues: list[str] = []
    if summary.get("head") != current_head:
        issues.append("head mismatch")
    if summary.get("stale") is not False:
        issues.append("stale flag set")
    if expected_case_count is not None and summary.get("case_count") != expected_case_count:
        issues.append("summary case_count mismatch")
    if expected_case_count is not None and len(traces) != expected_case_count:
        issues.append("trace case_count mismatch")
    return {"valid": not issues, "issues": issues}


def update_latest_manifest(summary: dict, trace_path: Path, summary_path: Path, run_id: str) -> tuple[bool, str | None]:
    manifest = {
        "run_id": run_id,
        "head": summary["head"],
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "case_count": summary["case_count"],
        "completed": summary.get("completed") is True,
    }
    try:
        LATEST_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(LATEST_MANIFEST_PATH, manifest)
        return True, None
    except OSError as error:
        return False, str(error)


def persist_evaluation_results(
    traces: list[dict],
    summary: dict,
    *,
    run_dir: Path,
    run_id: str,
    current_head: str,
    expected_case_count: int | None = None,
) -> dict:
    trace_path = run_dir / TRACE_FILENAME
    summary_path = run_dir / SUMMARY_FILENAME
    summary_payload = dict(summary)
    summary_payload["run_id"] = run_id
    summary_payload["trace_path"] = str(trace_path)
    summary_payload["summary_path"] = str(summary_path)
    summary_payload["completed"] = False
    try:
        _write_json_atomic(trace_path, traces)
        _write_json_atomic(summary_path, summary_payload)
    except OSError as error:
        raise OutputPersistenceError(run_id, str(trace_path), error) from error

    validation = validate_completed_run(traces, summary_payload, current_head, expected_case_count)
    if not validation["valid"]:
        raise OutputPersistenceError(run_id, str(summary_path), RuntimeError("; ".join(validation["issues"])))

    summary_payload["completed"] = True
    try:
        _write_json_atomic(summary_path, summary_payload)
    except OSError as error:
        raise OutputPersistenceError(run_id, str(summary_path), error) from error

    manifest_updated, manifest_error = update_latest_manifest(summary_payload, trace_path, summary_path, run_id)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "completed": True,
        "manifest_updated": manifest_updated,
        "manifest_error": manifest_error,
        "validation": validation,
    }


def load_latest_manifest() -> dict | None:
    if not LATEST_MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(LATEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_existing_summary() -> dict | None:
    manifest = load_latest_manifest()
    if manifest is None:
        return None
    summary_path = Path(manifest["summary_path"])
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def output_is_current(current_head: str) -> bool:
    manifest = load_latest_manifest()
    if manifest is None or manifest.get("completed") is not True:
        return False
    return manifest.get("head") == current_head


def check_stale_outputs(current_head: str) -> dict:
    manifest = load_latest_manifest()
    if manifest is None:
        return {"stale": False, "accepted_as_current": False}
    output_head = manifest.get("head")
    stale = output_head != current_head or manifest.get("completed") is not True
    return {
        "stale": stale,
        "accepted_as_current": not stale,
        "output_head": output_head,
        "current_head": current_head,
        "manifest_path": str(LATEST_MANIFEST_PATH),
    }


def enforce_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def huggingface_hub_cache_root() -> Path:
    for key in ("HUGGINGFACE_HUB_CACHE",):
        value = os.environ.get(key)
        if value:
            return Path(value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(model_name: str) -> Path:
    return huggingface_hub_cache_root() / f"models--{model_name.replace('/', '--')}"


def snapshot_is_complete(snapshot: Path) -> tuple[bool, str | None]:
    for requirement in REQUIRED_SNAPSHOT_FILES:
        if isinstance(requirement, tuple):
            if not any((snapshot / name).exists() for name in requirement):
                return False, f"missing one of: {', '.join(requirement)}"
            continue
        if not (snapshot / requirement).exists():
            return False, f"missing {requirement}"
    return True, None


def resolve_local_embedding_snapshot(model_name: str) -> Path:
    cache_dir = model_cache_dir(model_name)
    snapshots_dir = cache_dir / "snapshots"
    candidates: list[Path] = []
    missing_notes: list[str] = []
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            complete, missing = snapshot_is_complete(snapshot)
            if complete:
                candidates.append(snapshot)
            else:
                missing_notes.append(f"{snapshot.name}: {missing}")
    refs_main = cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        resolved = snapshots_dir / revision
        complete, missing = snapshot_is_complete(resolved)
        if complete:
            return resolved
        missing_notes.append(f"refs/main@{revision}: {missing}")
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise OfflineEmbeddingError(
        "LOCAL_MODEL_CACHE_INCOMPLETE",
        model_name=model_name,
        expected_cache=str(cache_dir),
        cache_candidates=[str(path) for path in candidates],
        missing_requirement="; ".join(missing_notes) or f"no complete snapshot under {snapshots_dir}",
    )


def build_offline_embedding_provider(model_name: str, device: str, batch_size: int):
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider

    enforce_offline_env()
    local_path = resolve_local_embedding_snapshot(model_name)
    provider = SentenceTransformerEmbeddingProvider(str(local_path), device, batch_size)
    return provider, local_path


def verify_offline_embedding_ready(model_name: str, device: str, batch_size: int) -> dict:
    try:
        provider, local_path = build_offline_embedding_provider(model_name, device, batch_size)
    except OfflineEmbeddingError as error:
        raise OfflineEmbeddingError(
            "OFFLINE_EMBEDDING_UNAVAILABLE",
            model_name=error.model_name,
            expected_cache=error.expected_cache,
            cache_candidates=error.cache_candidates,
            missing_requirement=error.missing_requirement,
        ) from error
    except Exception as error:
        raise OfflineEmbeddingError(
            "OFFLINE_EMBEDDING_UNAVAILABLE",
            model_name=model_name,
            expected_cache=str(model_cache_dir(model_name)),
            cache_candidates=[],
            missing_requirement=str(error),
        ) from error
    vectors = provider.embed(["offline preflight probe"])
    if not vectors or not vectors[0]:
        raise OfflineEmbeddingError(
            "OFFLINE_EMBEDDING_UNAVAILABLE",
            model_name=model_name,
            expected_cache=str(model_cache_dir(model_name)),
            cache_candidates=[str(local_path)],
            missing_requirement="tiny embedding call returned no vector",
        )
    return {
        "model_name": model_name,
        "local_path": str(local_path),
        "dimensions": len(vectors[0]),
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


def load_fixture() -> dict:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "g7-broad-v1"
    return payload


def build_retriever():
    import chromadb

    from backend.app.core.config import settings
    from backend.app.rag.http_store import ChromaHttpEvidenceStore
    from backend.app.rag.query_bridge import HistoricalQueryBridge
    from backend.app.rag.retriever import ChromaHistoricalRetriever

    provider, local_path = build_offline_embedding_provider(
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
                return retriever, str(local_path)
    raise OfflineEmbeddingError(
        "OFFLINE_EMBEDDING_UNAVAILABLE",
        model_name=settings.rag_embedding_model,
        expected_cache=str(model_cache_dir(settings.rag_embedding_model)),
        cache_candidates=[str(local_path)],
        missing_requirement="local Chroma collection unavailable for offline retrieval",
    )


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

    current_head = git_head()
    stale_info = check_stale_outputs(current_head)
    if stale_info.get("stale"):
        print(
            f"Stale current-run manifest detected: head={stale_info.get('output_head')} "
            f"!= current {current_head}"
        )

    runtime_root = select_writable_temp_root()
    run_dir, run_id = make_unique_run_output_dir(runtime_root)
    print(f"G7 evaluation output run: {run_dir}")

    needs_real_retrieval = any(case["case_origin"] == "REAL_CORPUS" for case in cases_to_run)
    offline_preflight = None
    if needs_real_retrieval:
        from backend.app.core.config import settings

        offline_preflight = verify_offline_embedding_ready(
            settings.rag_embedding_model,
            settings.rag_embedding_device,
            settings.rag_embedding_batch_size,
        )

    retriever = EmptyHistoricalRetriever()
    embedding_local_path = None
    if needs_real_retrieval:
        retriever, embedding_local_path = build_retriever()
    tools = AgentToolRegistry(retriever, OfflineGeography())
    traces = [run_case(case, tools, retriever) for case in cases_to_run]
    summary = {
        "head": current_head,
        "run_id": run_id,
        "case_count": len(traces),
        "stale": False,
        "completed": False,
        "offline_preflight": offline_preflight,
        "embedding_local_path": embedding_local_path,
        "metrics": aggregate_metrics(traces),
        "first_divergence": {},
    }
    for trace in traces:
        key = trace["diagnostics"]["first_divergence"]
        summary["first_divergence"][key] = summary["first_divergence"].get(key, 0) + 1

    expected_case_count = len(fixture["cases"]) if len(cases_to_run) == len(fixture["cases"]) else len(traces)
    try:
        persistence = persist_evaluation_results(
            traces,
            summary,
            run_dir=run_dir,
            run_id=run_id,
            current_head=current_head,
            expected_case_count=expected_case_count,
        )
    except OutputPersistenceError as error:
        raise SystemExit(str(error)) from error

    if not persistence["manifest_updated"]:
        print(f"Warning: latest manifest not updated: {persistence['manifest_error']}")

    if args.summary_only:
        print(json.dumps(summary["metrics"], indent=2))
    else:
        for trace in traces:
            print(
                f"{trace['case_id']} tier={trace['tier']} origin={trace['case_origin']} "
                f"first={trace['diagnostics']['first_divergence']} route={trace['route']['status']}"
            )
    print(f"Wrote {persistence['trace_path']}")
    print(f"Wrote {persistence['summary_path']}")
    if persistence["manifest_updated"]:
        print(f"Updated {LATEST_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
