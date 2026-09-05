"""G6S: contextual exclusion must not prove unique historical place identity."""
from __future__ import annotations

from backend.app.geography.place_disambiguation import (
    PlaceResolutionContext,
    ResolvedCoMention,
    filter_resolution_candidates,
)
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.event_places import HistoricalEventPlaceResolver


def _ctx(**kwargs) -> PlaceResolutionContext:
    return PlaceResolutionContext(**kwargs)


def _candidate(pleiades_id: str, name: str) -> dict:
    return {
        "pleiades_id": pleiades_id,
        "canonical_name": name,
        "place_types": ["settlement"],
        "coordinate_available": True,
        "matched_name_details": [],
    }


def _place(pleiades_id: str, name: str, latitude: float, longitude: float) -> HistoricalPlace:
    return HistoricalPlace(
        id=f"pleiades-{pleiades_id}",
        canonical_name=name,
        latitude=latitude,
        longitude=longitude,
        source="test",
        confidence=0.6,
        source_id=pleiades_id,
        spatial_semantics=PlaceSpatialSemantics.UNKNOWN,
    )


def _filter(
    *,
    places: tuple[HistoricalPlace, ...],
    candidates: tuple[dict, ...],
    context: PlaceResolutionContext,
    status: str = "AMBIGUOUS",
):
    return filter_resolution_candidates(
        status=status,
        places=places,
        candidates=candidates,
        context=context,
    )


def test_a_basin_exclusion_survivor_is_not_unique():
    near = _place("1", "Apollonia Epirus", 40.72, 19.45)
    far = _place("2", "Apollonia Cyrenaica", 32.9, 21.3)
    candidates = (_candidate("1", near.canonical_name), _candidate("2", far.canonical_name))
    context = _ctx(
        period="48 BCE",
        place_role="DESTINATION",
        resolved_co_mentions=(ResolvedCoMention("Epirus anchor", 39.5, 20.5),),
    )

    status, places, surviving, diagnostics = _filter(
        places=(near, far), candidates=candidates, context=context,
    )

    assert len(candidates) == 2
    assert len(surviving) == 1
    assert status == "AMBIGUOUS"
    assert len(places) == 1
    assert places[0].canonical_name == near.canonical_name
    assert any("geo_basin_incompatible" in item["failed_filters"] for item in diagnostics if not item["passed"])


def test_b_genuine_singleton_remains_unique():
    place = _place("981509", "Asia (Roman province)", 38.46, 27.77)
    candidate = _candidate("981509", place.canonical_name)
    context = _ctx(period="88 BCE", place_role="DESTINATION")

    status, places, surviving, _ = _filter(
        places=(place,),
        candidates=(candidate,),
        context=context,
        status="UNIQUE",
    )

    assert status == "UNIQUE"
    assert places == (place,)
    assert surviving == (candidate,)


def test_c_context_ranks_without_proving_unique():
    near = _place("1", "Apollonia Epirus", 40.72, 19.45)
    far = _place("2", "Apollonia Cyrenaica", 32.9, 21.3)
    candidates = (_candidate("1", near.canonical_name), _candidate("2", far.canonical_name))
    context = _ctx(
        period="48 BCE",
        place_role="DESTINATION",
        resolved_co_mentions=(ResolvedCoMention("Epirus anchor", 39.5, 20.5),),
    )

    status, places, surviving, _ = _filter(
        places=(near, far), candidates=candidates, context=context,
    )

    assert status == "AMBIGUOUS"
    assert surviving[0]["canonical_name"] == near.canonical_name
    assert places[0].canonical_name == near.canonical_name


def test_d_cross_basin_survivor_does_not_become_unique():
    italian = _place("10", "Alpha Port", 40.6, 17.9)
    african = _place("20", "Alpha Port Africa", 32.8, 13.2)
    candidates = (_candidate("10", italian.canonical_name), _candidate("20", african.canonical_name))
    context = _ctx(
        period="48 BCE",
        place_role="DESTINATION",
        resolved_co_mentions=(
            ResolvedCoMention("Brundisium", 40.64, 17.94),
            ResolvedCoMention("Epirus anchor", 39.5, 20.5),
        ),
    )

    status, _, surviving, diagnostics = _filter(
        places=(italian, african), candidates=candidates, context=context,
    )

    assert status == "AMBIGUOUS"
    assert status != "UNIQUE"
    assert any("geo_basin_incompatible" in item["failed_filters"] for item in diagnostics if not item["passed"])


def test_e_mention_order_does_not_promote_heuristic_unique():
    near = _place("1", "Apollonia Epirus", 40.72, 19.45)
    far = _place("2", "Apollonia Cyrenaica", 32.9, 21.3)
    candidates = (_candidate("1", near.canonical_name), _candidate("2", far.canonical_name))

    context_a = _ctx(
        period="48 BCE",
        place_role="DESTINATION",
        resolved_co_mentions=(ResolvedCoMention("Epirus anchor", 39.5, 20.5),),
    )
    context_b = _ctx(
        period="48 BCE",
        place_role="DESTINATION",
        resolved_co_mentions=(ResolvedCoMention("Epirus anchor", 39.5, 20.5), ResolvedCoMention("Brundisium", 40.64, 17.94)),
    )

    status_a, _, _, _ = _filter(places=(near, far), candidates=candidates, context=context_a)
    status_b, _, _, _ = _filter(places=(near, far), candidates=candidates, context=context_b)

    assert status_a == "AMBIGUOUS"
    assert status_b == "AMBIGUOUS"


def test_f_existing_not_found_and_unavailable_statuses_regress():
    context = _ctx(period="88 BCE", place_role="DESTINATION")

    status, places, candidates, _ = filter_resolution_candidates(
        status="NOT_FOUND",
        places=(),
        candidates=(),
        context=context,
    )
    assert status == "NOT_FOUND"
    assert places == ()
    assert candidates == ()

    status, places, candidates, _ = filter_resolution_candidates(
        status="UNAVAILABLE",
        places=(),
        candidates=(),
        context=None,
    )
    assert status == "UNAVAILABLE"
    assert places == ()
    assert candidates == ()


class _AmbiguousGeographyClient:
    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        return {
            "found": False,
            "status": "AMBIGUOUS",
            "candidate_count": 2,
            "alternatives": [{"canonical_name": "Apollonia Epirus"}, {"canonical_name": "Apollonia Cyrenaica"}],
        }


def test_route_anchor_not_created_from_heuristic_ambiguity():
    mention = HistoricalEventPlaceMention(
        raw_text="Apollonia",
        canonical_hint="Apollonia",
        role=EventPlaceRole.DESTINATION,
        evidence_refs=["ev1"],
    )
    event = HistoricalEvent(
        id="move-1",
        name="move-1",
        summary="He marched to Apollonia in 48 BCE.",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        place_mentions=[mention],
        source_statements=["He marched to Apollonia in 48 BCE."],
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
    )

    resolved, diagnostics = HistoricalEventPlaceResolver(_AmbiguousGeographyClient()).resolve([event])
    binding = resolved[0].place_bindings[0]

    assert binding.resolution_status is EventPlaceResolutionStatus.AMBIGUOUS
    assert binding.place is None
    assert diagnostics["ambiguous_count"] == 1

    anchors, anchor_diagnostics = project_event_anchors(resolved, [
        Evidence(id="ev1", author="Source", work="Work", locator="1", excerpt=event.summary, text=event.summary),
    ])
    assert anchors == []
    assert anchor_diagnostics == ["AMBIGUOUS_PLACE:move-1"]
