"""Top-level route result status contract for frontend consumption (V1C)."""
from __future__ import annotations

from enum import Enum

from backend.app.models import AgentState, HistoricalRoute


class RouteResultStatus(str, Enum):
    FULL_ROUTE = "FULL_ROUTE"
    PARTIAL = "PARTIAL"
    NO_ROUTE = "NO_ROUTE"
    ERROR = "ERROR"


_RUNTIME_ERROR_STATUSES = frozenset({"provider_error", "tool_failure"})


def _is_full_route(route: HistoricalRoute, reason_codes: list[str]) -> bool:
    if "PARTIAL_ROUTE" in reason_codes:
        return False
    if route.branch_relations:
        return False
    if len(route.route_components) > 1:
        return False
    if route.route_components and not route.ordered_points:
        return False
    return len(route.ordered_points) >= 2


def derive_route_result_status(state: AgentState) -> RouteResultStatus | None:
    if state.requested_output != "historical_route":
        return None
    if state.status in _RUNTIME_ERROR_STATUSES:
        return RouteResultStatus.ERROR

    diagnostics = state.historical_route_diagnostics or {}
    reason_codes = list(diagnostics.get("reason_codes") or [])
    route = state.historical_route

    if route is not None:
        if _is_full_route(route, reason_codes):
            return RouteResultStatus.FULL_ROUTE
        return RouteResultStatus.PARTIAL

    if state.historical_route_presentation:
        return RouteResultStatus.PARTIAL

    return RouteResultStatus.NO_ROUTE
