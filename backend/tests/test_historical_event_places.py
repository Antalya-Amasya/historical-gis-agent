from backend.app.models import (
    Evidence, EventPlaceResolutionStatus, EventPlaceRole, HistoricalEvent, HistoricalEventPlaceMention,
    HistoricalPlace, HistoricalEventType, PlaceSpatialSemantics,
)
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def place(identifier: str, name: str, semantic: PlaceSpatialSemantics, role: str = "exact_site") -> HistoricalPlace:
    return HistoricalPlace(
        id=identifier, canonical_name=name, latitude=1, longitude=2, source="Audited registry",
        source_id=identifier, source_url=f"https://example.invalid/{identifier}", confidence=.9,
        coordinate_role=role, spatial_semantics=semantic,
        spatial_semantics_provenance="audited test registry",
    )


class Geography:
    def __init__(self, entries: dict[str, HistoricalPlace] | None = None, ambiguous: set[str] | None = None):
        self.entries, self.ambiguous, self.calls = entries or {}, ambiguous or set(), []

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        self.calls.append(name)
        if name in self.ambiguous:
            return {"found": False, "ambiguous": True}
        value = self.entries.get(name)
        return {"found": False} if value is None else {"found": True, **value.model_dump(mode="json")}


def events(*items: Evidence):
    return EvidenceGroundedHistoricalEventExtractor().extract(list(items))[0]


def test_exact_settlement_event_site_is_resolved_with_provenance():
    result, diagnostics = HistoricalEventPlaceResolver(Geography({"Settlement X": place("x", "Settlement X", PlaceSpatialSemantics.SETTLEMENT)})).resolve(
        events(evidence("e", "The armies fought the battle at Settlement X.")),
    )
    binding = result[0].place_bindings[0]
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert binding.place and binding.place.coordinate_role == "exact_site"
    assert binding.role.value == "EVENT_SITE" and "Audited registry / x" in (binding.resolver_provenance or "")
    assert diagnostics["exact_site_count"] == 1


def test_river_and_mountain_region_preserve_non_exact_semantics():
    geo = Geography({
        "River Y": place("river", "River Y", PlaceSpatialSemantics.RIVER, "representative_point"),
        "Alpes": place("alpes", "Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid"),
    })
    result, diagnostics = HistoricalEventPlaceResolver(geo).resolve(events(
        evidence("river", "The army fought the battle near River Y."),
        evidence("mountain", "The army marched near Alpes."),
    ))
    bindings = [binding for event in result for binding in event.place_bindings]
    assert {binding.place.spatial_semantics for binding in bindings if binding.place} == {PlaceSpatialSemantics.RIVER, PlaceSpatialSemantics.MOUNTAIN_REGION}
    assert all("not an exact" in binding.limitations[0].casefold() for binding in bindings)
    assert diagnostics["representative_point_count"] == 1 and diagnostics["regional_count"] == 1
    assert "NON_EXACT_SPATIAL_SEMANTICS" in diagnostics["reason_codes"]


def test_unknown_and_ambiguous_places_never_receive_coordinates():
    result, diagnostics = HistoricalEventPlaceResolver(Geography(ambiguous={"Alias Z"})).resolve(events(
        evidence("unknown", "The battle occurred at Unknown Q."),
        evidence("ambiguous", "The battle occurred at Alias Z."),
    ))
    bindings = [binding for event in result for binding in event.place_bindings]
    assert {binding.resolution_status for binding in bindings} == {EventPlaceResolutionStatus.UNRESOLVED, EventPlaceResolutionStatus.AMBIGUOUS}
    assert all(binding.place is None for binding in bindings)
    assert diagnostics["unresolved_place_count"] == diagnostics["ambiguous_count"] == 1


def test_unlocated_and_unavailable_statuses_are_preserved():
    class StatusGeography:
        def call(self, tool: str, arguments: dict) -> dict:
            name = arguments["name"]
            if name == "Unlocated Q":
                return {"found": False, "status": "UNLOCATED", "candidate_count": 1, "candidates": []}
            if name == "Broken Index":
                return {"found": False, "status": "UNAVAILABLE", "reason": "index missing"}
            return {"found": False, "status": "NOT_FOUND"}

    result, diagnostics = HistoricalEventPlaceResolver(StatusGeography()).resolve(events(
        evidence("unlocated", "The army halted at Unlocated Q."),
        evidence("unavailable", "The army halted at Broken Index."),
        evidence("missing", "The army halted at Missing Q."),
    ))
    statuses = {binding.mention.raw_text: binding.resolution_status for event in result for binding in event.place_bindings}
    assert statuses["Unlocated Q"] is EventPlaceResolutionStatus.UNLOCATED
    assert statuses["Broken Index"] is EventPlaceResolutionStatus.UNAVAILABLE
    assert statuses["Missing Q"] is EventPlaceResolutionStatus.UNRESOLVED
    assert diagnostics["unlocated_place_count"] == 1
    assert diagnostics["unavailable_place_count"] == 1
    assert diagnostics["unresolved_place_count"] == 1
    assert {"PLACE_UNLOCATED", "PLACE_UNAVAILABLE", "PLACE_UNRESOLVED"}.issubset(set(diagnostics["reason_codes"]))


def test_multiple_origin_destination_bindings_remain_distinct_and_create_no_route():
    geo = Geography({
        "Place A": place("a", "Place A", PlaceSpatialSemantics.SETTLEMENT),
        "Place B": place("b", "Place B", PlaceSpatialSemantics.SETTLEMENT),
    })
    result, _ = HistoricalEventPlaceResolver(geo).resolve(events(evidence("move", "The army marched from Place A to Place B.")))
    event = result[0]
    assert [binding.role.value for binding in event.place_bindings] == ["ORIGIN", "DESTINATION"]
    assert len(event.places) == 2 and not hasattr(event, "historical_route")
    assert geo.calls == ["Place A", "Place B"]


def test_alias_boundary_safety_and_duplicate_entity_bindings_preserve_refs():
    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _ = extractor.extract([
        evidence("a", "The battle at Rome ended."),
        evidence("b", "Later the battle at Roma ended."),
        evidence("c", "The army fought for liberty at Place C."),
    ])
    assert all("Iberus" not in [mention.canonical_hint for mention in event.place_mentions] for event in candidates)
    geo = Geography({"Roma": place("roma", "Roma", PlaceSpatialSemantics.SETTLEMENT)})
    result, _ = HistoricalEventPlaceResolver(geo).resolve(candidates[:2])
    assert all(event.place_bindings[0].place and event.place_bindings[0].place.id == "roma" for event in result)


def test_caesar_actor_is_not_extracted_as_an_event_site():
    candidates = events(evidence("caesar", "The battle was fought at Caesar."))
    assert len(candidates) == 1
    assert all(mention.raw_text != "Caesar" for mention in candidates[0].place_mentions)


def test_duplicate_mentions_in_one_event_are_deduplicated_without_losing_evidence_refs():
    event = HistoricalEvent(
        id="duplicate", name="Battle", summary="fixture", event_type=HistoricalEventType.BATTLE,
        place_mentions=[
            HistoricalEventPlaceMention(raw_text="Rome", canonical_hint="Roma", role=EventPlaceRole.EVENT_SITE, evidence_refs=["a"]),
            HistoricalEventPlaceMention(raw_text="Roma", canonical_hint="Roma", role=EventPlaceRole.EVENT_SITE, evidence_refs=["b"]),
        ],
    )
    result, diagnostics = HistoricalEventPlaceResolver(Geography({"Roma": place("roma", "Roma", PlaceSpatialSemantics.SETTLEMENT)})).resolve([event])
    assert len(result[0].place_bindings) == 1
    assert result[0].place_bindings[0].evidence_refs == ["a", "b"]
    assert diagnostics["resolved_place_count"] == 2
