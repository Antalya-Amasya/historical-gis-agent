from backend.app.agent.route_summary import admitted_route_summary
from backend.app.models import (
    AgentState,
    GeoJsonLineString,
    HistoricalClaim,
    HistoricalPlace,
    HistoricalRoute,
    HistoricalRoutePoint,
    HistoricalTravelMode,
)


def _point(name: str) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(id=name, canonical_name=name, latitude=1, longitude=2, source="fixture", confidence=0.8),
        event_summary="admitted", evidence_refs=["e1"], confidence=0.8,
    )


def _state(route: HistoricalRoute, presentation=None, diagnostics=None) -> AgentState:
    state = AgentState(session_id="summary")
    state.requested_output = "historical_route"
    state.historical_route = route
    state.historical_route_presentation = presentation
    state.historical_route_diagnostics = diagnostics or {}
    return state


def test_land_summary_uses_admitted_endpoints_and_disclaimer():
    route = HistoricalRoute(
        id="r", event_id="e", name="Pictor", period="historical",
        ordered_points=[_point("Delphi"), _point("Roma")],
        geometry=GeoJsonLineString(coordinates=[]), historical_confidence=0.7, evidence_refs=["e1"],
    )
    reply = admitted_route_summary(_state(
        route,
        presentation={"geojson": {"features": [{"properties": {"segment_role": "roman_road"}}]}},
        diagnostics={"observation_components": [{"actor_status": "EXPLICIT", "actor_text": "Quintus Fabius Pictor"}]},
    ))
    assert "Quintus Fabius Pictor's movement from Delphi to Roma" in reply
    assert "ancient-road data" in reply
    assert "not a documented exact itinerary" in reply
    assert "Evidence references: e1." in reply
    assert "sailed" not in reply.casefold()


def test_sea_summary_keeps_coastal_access_simulated():
    route = HistoricalRoute(
        id="r", event_id="e", name="Libo", period="historical",
        ordered_points=[_point("Orikon"), _point("Brundisium")],
        geometry=GeoJsonLineString(coordinates=[]), historical_confidence=0.7,
        claims=[HistoricalClaim(id="c", claim_type="ORDERING", text="sailed", confidence=0.8, travel_mode=HistoricalTravelMode.SEA)],
    )
    reply = admitted_route_summary(_state(route, presentation={"geojson": {"features": [
        {"properties": {"segment_role": "simulated_coastal_access"}},
        {"properties": {"segment_role": "direct_water_edge"}},
    ]}}))
    assert "sea movement from Orikon to Brundisium" in reply
    assert "simulated reconstruction" in reply
    assert "not a historical embarkation point" in reply
    assert "exact sailing track" in reply


def test_summary_does_not_invent_an_endpoint_pair():
    route = HistoricalRoute(
        id="r", event_id="e", name="gap", period="historical",
        ordered_points=[], geometry=GeoJsonLineString(coordinates=[]), historical_confidence=0.2,
    )
    reply = admitted_route_summary(_state(route))
    assert "complete evidence-backed endpoint pair" in reply
    assert "displayed path" not in reply
