from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import (
    AgentState,
    Evidence,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule

COORDINATES = {"Genava": (46.2, 6.1), "Lutetia": (48.9, 2.35), "Alesia": (47.5, 4.5), "Bibracte": (46.9, 4.0)}


def evidence(identifier, document="caesar_gallic_war", spine=None, offset=None):
    metadata = {"document_id": document}
    if spine is not None:
        metadata.update({"spine_index": spine, "start_offset": offset})
    return Evidence(id=identifier, author="Julius Caesar", work="Gallic War", locator="Book I", excerpt="text", text="text", metadata=metadata)


def place(name):
    latitude, longitude = COORDINATES[name]
    return HistoricalPlace(id=name.lower(), canonical_name=name, latitude=latitude, longitude=longitude, source="test registry", confidence=0.8, coordinate_role="exact_site")


def binding(name, role, refs):
    mention = HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=list(refs))
    return HistoricalEventPlaceBinding(mention=mention, place=place(name), role=role, resolution_status=EventPlaceResolutionStatus.RESOLVED, evidence_refs=list(refs), resolver_provenance="registry")


def event(identifier, bindings, refs, *, years=None, precision=TemporalPrecision.YEAR):
    grounding = HistoricalEventTemporalGrounding()
    if years is not None:
        start, end = years
        grounding = HistoricalEventTemporalGrounding(raw_expression=f"{start}", normalized_start=str(start), normalized_end=str(end), precision=precision, evidence_refs=list(refs), status=TemporalGroundingStatus.EVIDENCE_GROUNDED)
    return HistoricalEvent(id=identifier, name=identifier, summary=f"{identifier} summary", event_type=HistoricalEventType.MOVEMENT, evidence_refs=list(refs), place_bindings=bindings, temporal_grounding=grounding)


def build(events, items):
    return EventAnchorRouteBuilder().build_with_diagnostics(events, items, event_id="caesar", name="Caesar", period="58 BCE")


def names(outcome):
    return [point.historical_place.canonical_name for point in outcome.route.ordered_points]


def movement(identifier, origin, destination, refs, **kwargs):
    return event(identifier, [binding(origin, EventPlaceRole.ORIGIN, refs), binding(destination, EventPlaceRole.DESTINATION, refs)], refs, **kwargs)


def site(identifier, name, refs, **kwargs):
    return event(identifier, [binding(name, EventPlaceRole.EVENT_SITE, refs)], refs, **kwargs)


def test_same_movement_event_orders_origin_before_destination_and_keeps_provenance():
    outcome = build([movement("march", "Genava", "Lutetia", ["a"])], [evidence("a")])
    assert names(outcome) == ["Genava", "Lutetia"]
    assert [relation.rule for relation in outcome.relations] == [OrderingRule.SAME_MOVEMENT_EVENT]
    assert outcome.diagnostics["ordering_provenance"] == [{
        "earlier": "Genava", "later": "Lutetia", "rule": "SAME_MOVEMENT_EVENT",
        "event_ids": ["march"], "evidence_refs": ["a"],
        "historical_authority": "ATTESTED_MOVEMENT_ORDERING",
        "connection_semantics": "ALGORITHMIC_GIS_RECONSTRUCTION_REQUIRED",
    }]
    claim = outcome.route.claims[0]
    assert (claim.claim_type, claim.source_place, claim.destination_place) == ("ORDERING", "Genava", "Lutetia")
    assert claim.supporting_evidence_ids == ["a"] and outcome.diagnostics["reason_codes"] == []


def test_reverse_direction_of_a_movement_event_is_never_inferred():
    outcome = build([movement("march", "Lutetia", "Genava", ["a"])], [evidence("a")])
    assert names(outcome) == ["Lutetia", "Genava"]


def test_two_unrelated_anchors_without_ordering_authority_fail_closed():
    events = [site("first", "Genava", ["a"]), site("second", "Lutetia", ["b"])]
    outcome = build(events, [evidence("a"), evidence("b", document="one"), ])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_single_event_site_is_not_a_route():
    outcome = build([site("battle", "Alesia", ["a"])], [evidence("a")])
    assert outcome.route is None and outcome.diagnostics["anchor_count"] == 1
    assert outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_PLACES"]


def test_multiple_event_sites_are_not_chained_by_their_existence():
    events = [site("one", "Genava", ["a"]), site("two", "Lutetia", ["b"]), site("three", "Alesia", ["c"])]
    outcome = build(events, [evidence("a", document="x"), evidence("b", document="y"), evidence("c", document="z")])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_comparable_evidence_grounded_temporal_values_order_separate_events():
    events = [site("later", "Lutetia", ["b"], years=(-52, -52)), site("earlier", "Genava", ["a"], years=(-58, -58))]
    outcome = build(events, [evidence("a"), evidence("b")])
    assert names(outcome) == ["Genava", "Lutetia"]
    assert [relation.rule for relation in outcome.relations] == [OrderingRule.TEMPORAL_ORDER]
    claim = outcome.route.claims[0]
    assert claim.claim_type == "WAYPOINT_ORDERING" and claim.movement_relation is None
    assert "no direct movement is asserted" in claim.text
    assert outcome.diagnostics["ordering_provenance"][0]["historical_authority"] == "EVIDENCE_GROUNDED_WAYPOINT_ORDERING"
    assert outcome.diagnostics["ordering_provenance"][0]["connection_semantics"] == "ALGORITHMIC_GIS_RECONSTRUCTION_REQUIRED"


def test_overlapping_temporal_values_do_not_force_an_order():
    events = [site("one", "Genava", ["a"], years=(-58, -50)), site("two", "Lutetia", ["b"], years=(-55, -52))]
    outcome = build(events, [evidence("a", spine=1, offset=100), evidence("b", spine=1, offset=200)])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_same_source_structural_positions_order_events_without_retrieval_rank():
    events = [site("later", "Lutetia", ["b"]), site("earlier", "Genava", ["a"])]
    outcome = build(events, [evidence("b", spine=1, offset=200), evidence("a", spine=1, offset=100)])
    assert names(outcome) == ["Genava", "Lutetia"]
    assert [relation.rule for relation in outcome.relations] == [OrderingRule.SOURCE_STRUCTURAL_ORDER]


def test_structural_positions_from_different_documents_are_not_compared():
    events = [site("one", "Genava", ["a"]), site("two", "Lutetia", ["b"])]
    outcome = build(events, [evidence("a", document="caesar_gallic_war", spine=1, offset=100), evidence("b", document="livy_history", spine=1, offset=200)])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_list_order_alone_never_establishes_a_route():
    events = [site("one", "Genava", ["a"]), site("two", "Lutetia", ["b"])]
    items = [evidence("a"), evidence("b")]
    assert build(events, items).route is None
    assert build(list(reversed(events)), list(reversed(items))).route is None


def test_geographic_proximity_never_establishes_a_route():
    near = [site("one", "Alesia", ["a"]), site("two", "Bibracte", ["b"])]
    outcome = build(near, [evidence("a", document="x"), evidence("b", document="y")])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_roman_road_metadata_never_establishes_waypoints_or_chronology():
    first, second = evidence("a", document="x"), evidence("b", document="y")
    first.metadata["roman_road"] = "audited-road-a"
    second.metadata["roman_road"] = "audited-road-a"
    outcome = build([site("one", "Genava", ["a"]), site("two", "Lutetia", ["b"])], [first, second])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_ORDERING"]


def test_unproven_gap_yields_partial_route_instead_of_a_bridged_itinerary():
    events = [movement("march", "Genava", "Lutetia", ["a"]), site("gap", "Alesia", ["c"], years=None)]
    outcome = build(events, [evidence("a"), evidence("c", document="other")])
    assert names(outcome) == ["Genava", "Lutetia"]
    assert outcome.diagnostics["reason_codes"] == ["PARTIAL_ROUTE"]
    assert outcome.diagnostics["distinct_place_count"] == 3 and outcome.diagnostics["ordered_place_count"] == 2


def test_accepted_ordering_relation_keeps_event_and_evidence_provenance():
    outcome = build([movement("march", "Genava", "Lutetia", ["a", "b"])], [evidence("a"), evidence("b")])
    relation = outcome.relations[0]
    assert relation.event_ids == ("march",) and relation.evidence_refs == ("a", "b")
    assert all(point.claim_ids and point.evidence_refs for point in outcome.route.ordered_points)
    assert outcome.route.claims[0].source_documents == ["caesar_gallic_war"]


def test_unresolved_place_fails_closed_rather_than_dropping_to_an_anchorless_route():
    unresolved = HistoricalEventPlaceBinding(mention=HistoricalEventPlaceMention(raw_text="Lutetia", role=EventPlaceRole.DESTINATION, evidence_refs=["a"]), place=None, role=EventPlaceRole.DESTINATION, resolution_status=EventPlaceResolutionStatus.UNRESOLVED, evidence_refs=["a"])
    outcome = build([event("march", [binding("Genava", EventPlaceRole.ORIGIN, ["a"]), unresolved], ["a"])], [evidence("a")])
    assert outcome.route is None and outcome.diagnostics["reason_codes"] == ["INSUFFICIENT_PLACES"]


class Geography:
    def __init__(self, missing=()):
        self.missing = set(missing)

    def call(self, tool, arguments):
        name = arguments["name"]
        if name in self.missing or name not in {"Melodunum", "Lutetia"}:
            return {"found": False}
        latitude, longitude = {"Melodunum": (48.5, 2.7), "Lutetia": (48.9, 2.35)}[name]
        return {"found": True, "id": name.lower(), "canonical_name": name, "latitude": latitude, "longitude": longitude, "source": "test registry", "confidence": 0.8, "coordinate_role": "exact_site"}


class Retriever:
    def __init__(self, items):
        self.items = items

    def retrieve(self, query, top_k=5, filters=None):
        return self.items[:top_k]


def test_event_first_path_is_preferred_when_it_can_build_a_route():
    state = AgentState(session_id="event-first")
    state.historical_evidence = [evidence("a")]
    state.historical_events = [movement("march", "Genava", "Lutetia", ["a"])]
    result, _ = AgentToolRegistry(Retriever([]), Geography()).execute("build_historical_route", {"event_id": "caesar", "name": "Caesar", "period": "58 BCE"}, state)
    assert result["result"]["route"]["id"] == "caesar-event-anchor-route"
    assert state.historical_route_diagnostics["route_source"] == "event_anchor"


def test_legacy_strict_movement_fallback_remains_usable_without_event_anchors():
    item = Evidence(id="caesar-lutetia", author="Julius Caesar", work="Gallic War", locator="Book I", excerpt="He reached Melodunum and then led his army to Lutetia.", text="He reached Melodunum and then led his army to Lutetia.", metadata={"document_id": "caesar_gallic_war", "spine_index": 2, "start_offset": 100})
    state = AgentState(session_id="legacy")
    tools = AgentToolRegistry(Retriever([item]), Geography())
    tools.execute("search_historical_evidence", {"query": "Caesar", "top_k": 5}, state)
    result, _ = tools.execute("build_historical_route", {"event_id": "caesar", "name": "Caesar", "period": "52 BCE"}, state)
    assert [point["historical_place"]["canonical_name"] for point in result["result"]["route"]["ordered_points"]] == ["Melodunum", "Lutetia"]
    assert state.historical_route_diagnostics["route_source"] == "legacy_movement_claims"


def test_event_anchors_and_legacy_claims_are_never_merged_to_close_a_gap():
    item = Evidence(id="legacy", author="Julius Caesar", work="Gallic War", locator="Book I", excerpt="He reached Melodunum and then led his army to Lutetia.", text="He reached Melodunum and then led his army to Lutetia.", metadata={"document_id": "caesar_gallic_war"})
    state = AgentState(session_id="mixed")
    state.historical_evidence = [item, evidence("a")]
    state.historical_events = [site("isolated", "Alesia", ["a"])]
    result, _ = AgentToolRegistry(Retriever([]), Geography()).execute("build_historical_route", {"event_id": "caesar", "name": "Caesar", "period": "52 BCE"}, state)
    ordered = [point["historical_place"]["canonical_name"] for point in result["result"]["route"]["ordered_points"]]
    assert ordered == ["Melodunum", "Lutetia"] and "Alesia" not in ordered
    assert state.historical_route_diagnostics["route_source"] == "legacy_movement_claims"
    assert state.historical_route_diagnostics["event_anchor_diagnostics"]["reason_codes"] == ["INSUFFICIENT_PLACES"]
