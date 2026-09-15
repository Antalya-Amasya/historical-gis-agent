"""V1C: top-level route_result_status contract for chat/API responses."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import backend.app.main as main
from backend.app.models import AgentState, ChatResponse, Evidence
from backend.app.route_result_status import RouteResultStatus, derive_route_result_status
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.tests.test_g4d_route_preservation import structured_route

EID = "9d81ae3d968b243004d968dd420cd9d6:915:1212"
LIBO_QUERY = "Trace the route from Oricum to Brundisium."


def _route_state(**overrides) -> AgentState:
    state = AgentState(
        session_id="status-test",
        requested_output="historical_route",
        status="completed",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_full_route_from_complete_ordered_waypoints():
    route = structured_route()
    status = derive_route_result_status(
        _route_state(historical_route=route, historical_route_diagnostics={"reason_codes": []}),
    )
    assert status is RouteResultStatus.FULL_ROUTE


def test_partial_from_component_fragments():
    route = structured_route(components=True)
    status = derive_route_result_status(
        _route_state(
            historical_route=route,
            historical_route_diagnostics={"reason_codes": ["PARTIAL_ROUTE"]},
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_no_route_from_insufficient_ordering_diagnostics():
    status = derive_route_result_status(
        _route_state(
            historical_route=None,
            historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_ORDERING"]},
        ),
    )
    assert status is RouteResultStatus.NO_ROUTE


def test_error_from_provider_failure():
    status = derive_route_result_status(
        _route_state(status="provider_error", historical_route=None),
    )
    assert status is RouteResultStatus.ERROR


def test_error_from_tool_failure():
    status = derive_route_result_status(
        _route_state(status="tool_failure", historical_route=None),
    )
    assert status is RouteResultStatus.ERROR


def test_non_route_requests_return_none():
    status = derive_route_result_status(
        AgentState(session_id="answer", requested_output="answer", status="completed"),
    )
    assert status is None


def test_partial_from_presentation_without_route():
    status = derive_route_result_status(
        _route_state(
            historical_route=None,
            historical_route_presentation={"route": {"route_id": "fragment"}},
            historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_ORDERING"]},
        ),
    )
    assert status is RouteResultStatus.PARTIAL


def test_no_route_and_error_remain_distinct():
    no_route = derive_route_result_status(
        _route_state(
            status="completed",
            historical_route=None,
            historical_route_diagnostics={"reason_codes": ["INSUFFICIENT_PLACES"]},
        ),
    )
    error = derive_route_result_status(_route_state(status="provider_error"))
    assert no_route is RouteResultStatus.NO_ROUTE
    assert error is RouteResultStatus.ERROR
    assert no_route is not error


def test_libo_real_control_maps_to_full_route():
    from backend.app.rag.lexical_index import derive_passages
    from backend.app.routes.events import (
        EvidenceGroundedHistoricalEventExtractor,
        HistoricalEventConsolidator,
    )
    from backend.app.routes.event_places import HistoricalEventPlaceResolver
    from geography_mcp.service import GeographyService

    parent = EID.split(":")[0]
    db = Path(r"C:/D/python/202608231533/data/chroma_server_roman_republic_v2/chroma.sqlite3")
    text = sqlite3.connect(db).execute(
        """
        SELECT em.string_value
        FROM embedding_metadata em
        JOIN embeddings e ON e.id = em.id
        WHERE em.key = 'chroma:document' AND e.embedding_id LIKE ?
        LIMIT 1
        """,
        (parent + "%",),
    ).fetchone()[0]
    passage = next(
        item
        for item in derive_passages(
            parent, text, {"source_chunk_id": parent, "document_id": parent}
        )
        if item.id == EID
    )
    evidence = Evidence(
        id=passage.id,
        author="Caesar",
        work="Civil War",
        locator="XXIII",
        excerpt=passage.text[:500],
        text=passage.text,
        metadata=dict(passage.metadata),
    )

    class Geography:
        def call(self, tool: str, arguments: dict) -> dict:
            return GeographyService().resolve_ancient_place_payload(arguments["name"])

    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _ = extractor.extract([evidence], query_contexts=(LIBO_QUERY,))
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    events, _ = HistoricalEventPlaceResolver(Geography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        [evidence],
        event_id="v1c-libo",
        name="Libo",
        period="49 BCE",
        query_contexts=(LIBO_QUERY,),
    )
    assert outcome.route is not None
    status = derive_route_result_status(
        _route_state(
            historical_route=outcome.route,
            historical_route_diagnostics=outcome.diagnostics,
        ),
    )
    assert status is RouteResultStatus.FULL_ROUTE


def test_chat_response_serializes_route_result_status(monkeypatch):
    captured = structured_route()

    class StubAgent:
        def respond(self, message: str, state: AgentState):
            state.requested_output = "historical_route"
            state.status = "completed"
            state.historical_route = captured
            state.historical_route_diagnostics = {"reason_codes": []}
            return "Route ready.", state

    monkeypatch.setattr(main, "agent", StubAgent())
    response = TestClient(main.app).post(
        "/api/v1/agent/chat",
        json={"session_id": "v1c-serialize", "message": "Trace the route from Oricum to Brundisium."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route_result_status"] == "FULL_ROUTE"
    assert body["state"]["historical_route"]["id"] == captured.id
    assert body["state"]["historical_route"]["ordered_points"]
    parsed = ChatResponse.model_validate(body)
    assert parsed.route_result_status == "FULL_ROUTE"


def test_chat_response_error_status_serializes(monkeypatch):
    class StubAgent:
        def respond(self, message: str, state: AgentState):
            state.requested_output = "historical_route"
            state.status = "provider_error"
            state.historical_route = None
            return "The configured language-model provider is unavailable.", state

    monkeypatch.setattr(main, "agent", StubAgent())
    response = TestClient(main.app).post(
        "/api/v1/agent/chat",
        json={"session_id": "v1c-error", "message": "Trace a historical route."},
    )
    assert response.status_code == 200
    assert response.json()["route_result_status"] == "ERROR"
    assert response.json()["state"]["historical_route"] is None
