import pytest

from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.grid import GridPoint
from backend.app.candidate_routes.location import (
    CoordinateResolutionError,
    LocationConfidence,
    ResolvedCoordinate,
)
from backend.app.candidate_routes.models import CandidateRouteAnchor
from backend.app.historical_campaign_presentation_service import CaesarCampaignPresentationFactory
from backend.app.rag.campaign_route_adapter import CampaignChainWaypointAdapter, MockHistoricalCoordinateResolver
from backend.app.rag.event_chain_builder import HistoricalCampaignChain, HistoricalCampaignStep


def chain() -> HistoricalCampaignChain:
    steps = [
        HistoricalCampaignStep(order=1, event_id="helvetii", title="Helvetii migration", involved_places=["Helvetii"], period="58 BCE", corpus_id="caesar", source_book="1", source_chapter="I", evidence_refs=["c-1"], confidence=0.8),
        HistoricalCampaignStep(order=2, event_id="bibracte", title="Battle of Bibracte", involved_places=["Bibracte"], period="58 BCE", corpus_id="caesar", source_book="1", source_chapter="XXIII", evidence_refs=["c-2"], confidence=0.8),
        HistoricalCampaignStep(order=3, event_id="alesia", title="Siege of Alesia", involved_places=["Alesia"], period="52 BCE", corpus_id="caesar", source_book="7", source_chapter="LXVIII", evidence_refs=["c-3"], confidence=0.85),
    ]
    return HistoricalCampaignChain(chain_id="caesar", title="Caesar campaign", corpus_id="caesar", period="58-52 BCE", description="Configured order", event_ids=[step.event_id for step in steps], steps=steps)


def test_campaign_chain_adapter_preserves_explicit_order_and_provenance() -> None:
    annotation = HistoricalAnnotation(
        id="helvetii", title="Reviewed Helvetii event", event_type=HistoricalEventType.CAMPAIGN,
        period="58 BCE", description="Supplied annotation", source_refs=["c-1"],
        external_references=[HistoricalExternalReference(id="reading", reference_type=HistoricalExternalReferenceType.ENCYCLOPEDIA, title="Reading", url="https://example.invalid/reading")],
    )
    graph = CampaignChainWaypointAdapter().to_waypoint_graph(chain(), annotations=[annotation])
    assert [waypoint.id for waypoint in graph.waypoints] == ["helvetii", "bibracte", "alesia"]
    assert graph.waypoints[1].involved_places == ["Bibracte"]
    assert graph.waypoints[2].source_book == "7" and graph.waypoints[2].source_chapter == "LXVIII"
    assert graph.waypoints[0].evidence_refs == ["c-1"]
    assert graph.waypoints[0].annotation.external_references[0].id == "reading"


def test_strict_campaign_coordinate_resolver_rejects_missing_mapping() -> None:
    resolver = MockHistoricalCoordinateResolver({
        "helvetii": ResolvedCoordinate(GridPoint(0, 0), LocationConfidence.APPROXIMATE, display_coordinate=(6.1, 46.2)),
    })
    with pytest.raises(CoordinateResolutionError, match="no configured coordinate"):
        resolver.resolve(CandidateRouteAnchor(historical_place_id="alesia", canonical_name="Alesia", evidence_refs=["c-3"]))


def test_caesar_campaign_pipeline_returns_line_points_panels_and_provenance() -> None:
    response = CaesarCampaignPresentationFactory().build()
    assert response.route.route_id == "caesar-gallic-campaign-candidate"
    assert response.geojson["features"][0]["geometry"]["type"] == "LineString"
    points = response.geojson["features"][1:]
    assert len(points) == 4 and all(feature["geometry"]["type"] == "Point" for feature in points)
    assert response.waypoints[2].source_book == "7" and response.waypoints[2].source_chapter == "LXVIII"
    assert all(panel.waypoint_id for panel in response.knowledge_panels)
    assert all(feature["properties"]["knowledge_panel_id"] for feature in points)
