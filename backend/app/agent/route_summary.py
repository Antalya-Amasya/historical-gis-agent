"""Summarize an already admitted route. This is not a historical reasoning step."""
from __future__ import annotations

from backend.app.models import AgentState, HistoricalRoute, HistoricalTravelMode


def admitted_route_summary(state: AgentState) -> str | None:
    route = state.historical_route
    if route is None:
        return None
    names = _endpoint_names(route)
    actor = _explicit_actor(state)
    mode = _explicit_mode(route)
    partial = _is_partial(state, route)
    segments = _segment_roles(state)
    has_geometry = bool(segments) or state.historical_route_presentation is not None
    subject = f"{actor}'s movement" if actor else "movement"
    if mode is HistoricalTravelMode.SEA:
        subject = f"a sea {subject}" if not actor else f"{actor}'s sea movement"
    elif mode is HistoricalTravelMode.LAND:
        subject = f"an overland {subject}" if not actor else f"{actor}'s overland movement"
    if len(names) >= 2:
        movement = f"Sources support {subject} from {names[0]} to {names[-1]}."
    elif names:
        movement = f"Sources support {subject} involving {names[0]}. A complete endpoint pair is not available."
    else:
        movement = "A historical route record exists, but it does not contain a complete evidence-backed endpoint pair."
    if partial:
        movement = f"This result is partial. {movement}"
    if not has_geometry:
        geometry = "No simulated map geometry is attached."
    else:
        geometry = _geometry_sentence(segments)
    evidence = _evidence_sentence(route)
    return " ".join(part for part in (movement, geometry, evidence) if part)


def _endpoint_names(route: HistoricalRoute) -> list[str]:
    points = list(route.ordered_points)
    if not points and route.route_components:
        points = [point for component in route.route_components for point in component.ordered_points]
    names: list[str] = []
    for point in points:
        name = point.historical_place.canonical_name
        if name and (not names or names[-1] != name):
            names.append(name)
    return names


def _explicit_actor(state: AgentState) -> str | None:
    diagnostics = state.historical_route_diagnostics or {}
    actors = []
    for component in diagnostics.get("observation_components") or []:
        if not isinstance(component, dict):
            continue
        if component.get("actor_status") == "EXPLICIT" and component.get("actor_text"):
            actors.append(str(component["actor_text"]))
    unique = list(dict.fromkeys(actors))
    return unique[0] if len(unique) == 1 else None


def _explicit_mode(route: HistoricalRoute) -> HistoricalTravelMode | None:
    modes = {
        claim.travel_mode
        for claim in route.claims
        if claim.travel_mode in {HistoricalTravelMode.LAND, HistoricalTravelMode.SEA}
    }
    return next(iter(modes)) if len(modes) == 1 else None


def _is_partial(state: AgentState, route: HistoricalRoute) -> bool:
    diagnostics = state.historical_route_diagnostics or {}
    if str(diagnostics.get("canonical_completeness") or "") == "PARTIAL":
        return True
    if route.branch_relations or len(route.route_components) > 1:
        return True
    presentation = state.historical_route_presentation or {}
    road = presentation.get("road_network") or {}
    if road.get("route_status") == "PARTIAL":
        return True
    return any(
        (feature.get("properties") or {}).get("segment_role") == "failed_gap"
        for feature in (presentation.get("geojson") or {}).get("features") or []
        if isinstance(feature, dict)
    )


def _segment_roles(state: AgentState) -> set[str]:
    presentation = state.historical_route_presentation or {}
    roles = set()
    for feature in (presentation.get("geojson") or {}).get("features") or []:
        if isinstance(feature, dict):
            role = (feature.get("properties") or {}).get("segment_role")
            if role:
                roles.add(str(role))
    for leg in (presentation.get("road_network") or {}).get("legs") or []:
        if isinstance(leg, dict) and leg.get("simulation_route_mode"):
            roles.add(str(leg["simulation_route_mode"]))
    return roles


def _geometry_sentence(roles: set[str]) -> str:
    if "direct_water_edge" in roles or "SEA" in roles:
        coast = " Simulated coastal access is not a historical embarkation point." if "simulated_coastal_access" in roles else ""
        return (
            "The displayed water path is a simulated reconstruction from geographic data, "
            "not a documented exact sailing track." + coast
        )
    if "terrain_candidate" in roles or "TERRAIN" in roles:
        return (
            "The displayed path is a plausible terrain reconstruction, "
            "not a documented exact itinerary."
        )
    if "roman_road" in roles or "LAND" in roles or "access_connector" in roles:
        return (
            "The displayed path is a plausible reconstruction using ancient-road data, "
            "not a documented exact itinerary."
        )
    return (
        "The displayed path is a plausible geographic reconstruction, "
        "not a documented exact itinerary."
    )


def _evidence_sentence(route: HistoricalRoute) -> str:
    refs = [ref for ref in route.evidence_refs if ref]
    if not refs:
        return ""
    shown = ", ".join(refs[:4])
    return f"Evidence references: {shown}."
