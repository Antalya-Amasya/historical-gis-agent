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


def _route_is_structurally_incomplete(route: HistoricalRoute, diagnostics: dict) -> bool:
    """True when the produced HistoricalRoute itself is incomplete."""
    if len(route.ordered_points) < 2:
        return True
    if route.branch_relations:
        return True
    if len(route.route_components) > 1:
        return True
    if route.route_components and not route.ordered_points:
        return True
    if int(diagnostics.get("contradictory_relation_count") or 0) > 0:
        return True
    return False


def _is_full_route(route: HistoricalRoute, diagnostics: dict) -> bool:
    return not _route_is_structurally_incomplete(route, diagnostics)


def derive_route_result_status(state: AgentState) -> RouteResultStatus | None:
    if state.requested_output != "historical_route":
        return None
    if state.status in _RUNTIME_ERROR_STATUSES:
        return RouteResultStatus.ERROR

    diagnostics = state.historical_route_diagnostics or {}
    route = state.historical_route

    if route is not None:
        if _is_full_route(route, diagnostics):
            return RouteResultStatus.FULL_ROUTE
        return RouteResultStatus.PARTIAL

    if state.historical_route_presentation:
        return RouteResultStatus.PARTIAL

    return RouteResultStatus.NO_ROUTE
