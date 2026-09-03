"""State-based route orchestration gates for live agent execution."""
from __future__ import annotations

import json

from backend.app.models import (
    AgentState,
    AgentToolHistoryEntry,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEventType,
)

PRE_ROUTE_UPSTREAM_TOOLS = frozenset({
    "search_historical_evidence", "calculate_distance", "get_elevation", "get_elevation_profile",
})
PRE_ROUTE_SUPPRESSION_MESSAGE = (
    "Route-ready state is established from accumulated Evidence and historical events. "
    "Do not call search_historical_evidence again. Call build_historical_route, then submit_grounded_answer."
)
ROUTE_READY_TOOL_NAMES = frozenset({"build_historical_route", "submit_grounded_answer"})
_EVENT_FIELDS = (
    "id", "name", "event_type", "summary", "period", "temporal_grounding",
    "place_mentions", "evidence_refs", "grounding_status", "limitations",
)
_RESOLVED_STATUSES = {
    EventPlaceResolutionStatus.RESOLVED,
    EventPlaceResolutionStatus.AMBIGUOUS,
    EventPlaceResolutionStatus.UNLOCATED,
    EventPlaceResolutionStatus.UNAVAILABLE,
}


def resolve_call_fingerprint(arguments: dict) -> str:
    name = str(arguments.get("name", "")).strip().casefold()
    period = str(arguments.get("period", "")).strip().casefold()
    return f"resolve:{name}:{period}"


def route_is_ready(state: AgentState) -> bool:
    if state.requested_output != "historical_route" or not state.historical_evidence:
        return False
    movement = [event for event in state.historical_events if event.event_type is HistoricalEventType.MOVEMENT]
    if not movement:
        return False
    if any(
        EventPlaceRole.ORIGIN in {mention.role for mention in event.place_mentions}
        and EventPlaceRole.DESTINATION in {mention.role for mention in event.place_mentions}
        for event in movement
    ):
        return True
    resolved = sum(
        1 for event in state.historical_events for binding in event.place_bindings
        if binding.resolution_status is EventPlaceResolutionStatus.RESOLVED and binding.place is not None
    )
    return resolved >= 2 or any(event.place_bindings for event in movement)


def should_suppress_pre_route_tool(tool_name: str, state: AgentState) -> bool:
    return route_is_ready(state) and tool_name in PRE_ROUTE_UPSTREAM_TOOLS


def route_ready_tool_schemas(schemas: list[dict]) -> list[dict]:
    return [schema for schema in schemas if schema.get("name") in ROUTE_READY_TOOL_NAMES]


def visible_historical_events(state: AgentState, *, limit: int = 8) -> list[dict]:
    evidence_ids = {item.id for item in state.historical_evidence}
    if not evidence_ids:
        return []
    ranked: list[tuple[int, dict]] = []
    for event in state.historical_events:
        refs = set(event.evidence_refs)
        if refs and not refs & evidence_ids:
            continue
        serialized = event.model_dump(mode="json")
        payload = {key: serialized.get(key) for key in _EVENT_FIELDS}
        score = int(event.event_type is HistoricalEventType.MOVEMENT) * 4
        roles = {mention.role for mention in event.place_mentions}
        score += int(EventPlaceRole.ORIGIN in roles and EventPlaceRole.DESTINATION in roles) * 3
        score += int(bool(event.place_bindings))
        ranked.append((score, payload))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:limit]]


def lookup_prior_resolve(arguments: dict, state: AgentState) -> tuple[dict, str] | None:
    fingerprint = resolve_call_fingerprint(arguments)
    for entry in reversed(state.tool_history):
        if entry.tool_name == "resolve_ancient_place" and resolve_call_fingerprint(entry.arguments) == fingerprint:
            return {
                "found": entry.success,
                "status": "resolved" if entry.success else "prior_failure",
                "duplicate_resolve_suppressed": True,
                "prior_outcome": entry.outcome,
            }, f"resolve_ancient_place duplicate suppressed ({entry.outcome})"
    requested_name = str(arguments.get("name", "")).strip().casefold()
    if not requested_name:
        return None
    for place in reversed(state.resolved_places):
        if place.canonical_name.strip().casefold() == requested_name:
            payload = place.model_dump(mode="json")
            payload.update({"found": True, "status": "already_resolved", "duplicate_resolve_suppressed": True})
            return payload, f"resolve_ancient_place reused resolved place {place.canonical_name}"
    for event in state.historical_events:
        for binding in event.place_bindings:
            mention = binding.mention
            names = {(mention.raw_text or "").strip().casefold(), (mention.canonical_hint or "").strip().casefold()}
            if requested_name in names and binding.resolution_status in _RESOLVED_STATUSES and binding.place is not None:
                payload = binding.place.model_dump(mode="json")
                payload.update({
                    "found": binding.resolution_status is EventPlaceResolutionStatus.RESOLVED,
                    "status": binding.resolution_status.value.lower(),
                    "duplicate_resolve_suppressed": True,
                })
                return payload, f"resolve_ancient_place reused event binding {binding.resolution_status.value.lower()}"
    return None


def prior_tool_attempt_key(tool_name: str, arguments: dict) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'), ensure_ascii=False)}"


def duplicate_attempt_payload(entry: AgentToolHistoryEntry) -> tuple[dict, str]:
    summary = f"duplicate cache hit; reuse prior result: {entry.result_summary}"
    return {
        "success": entry.success,
        "result": {
            "status": "duplicate",
            "previous_outcome": entry.outcome,
            "previous_summary": entry.result_summary,
        },
        "summary": summary,
        "duration_ms": 0,
    }, summary
