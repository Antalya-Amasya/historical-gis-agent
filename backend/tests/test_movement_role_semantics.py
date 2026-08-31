from backend.app.models import Evidence, EventPlaceRole
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_aliases import HistoricalPlaceAlias


ALIASES = (
    HistoricalPlaceAlias("A", ("a",)),
    HistoricalPlaceAlias("B", ("b",)),
    HistoricalPlaceAlias("C", ("c",)),
)


def evidence(identifier, text, **metadata):
    return Evidence(id=identifier, author="Source", work="Work", locator="section", excerpt=text, text=text, metadata=metadata)


def extractor():
    return HistoricalPlaceMentionExtractor(ALIASES)


def claims(text):
    return extractor().movement_claims([evidence("e1", text)], event_id="event")


def test_explicit_from_to_preserves_direction_and_exact_evidence_ref():
    claim = claims("The army marched from A to B.")[0]
    assert (claim.source_place, claim.destination_place, claim.movement_relation) == ("A", "B", "from_to")
    assert claim.supporting_evidence_ids == ["e1"]


def test_leave_arrive_and_return_direction_are_syntax_driven():
    departed = claims("The army left A and arrived at B.")[0]
    returned = claims("The army returned from B to A.")[0]
    assert (departed.source_place, departed.destination_place) == ("A", "B")
    assert (returned.source_place, returned.destination_place) == ("B", "A")


def test_multi_place_movement_preserves_origin_intermediate_and_destination_roles():
    event = EvidenceGroundedHistoricalEventExtractor(extractor()).extract(
        [evidence("e1", "The army left A, crossed B, and arrived at C.")]
    )[0][0]
    roles = {item.canonical_hint: item.role for item in event.place_mentions}
    claim = claims("The army left A, crossed B, and arrived at C.")[0]
    assert roles == {"A": EventPlaceRole.ORIGIN, "B": EventPlaceRole.RELATED_PLACE, "C": EventPlaceRole.DESTINATION}
    assert (claim.source_place, claim.destination_place) == ("A", "C")


def test_crossing_precedes_arrival_only_when_the_sentence_orders_them_that_way():
    claim = claims("The army crossed B before arriving at C.")[0]
    assert (claim.source_place, claim.destination_place) == ("B", "C")
    # An arrival followed by retrospective crossing cannot be inverted into B -> A.
    assert not [item for item in claims("They came to C after leaving A, having crossed B.") if item.source_place]


def test_captured_complex_structure_never_reverses_crossing_into_departure():
    places = HistoricalPlaceMentionExtractor((
        HistoricalPlaceAlias("New Carthage", ("new carthage",)),
        HistoricalPlaceAlias("Alps", ("alps",)),
        HistoricalPlaceAlias("Italy", ("italy",)),
    ))
    text = "They came to Italy after leaving New Carthage, having crossed the Alps."
    values = places.movement_claims([evidence("livy", text)], event_id="event")
    event = EvidenceGroundedHistoricalEventExtractor(places).extract([evidence("livy", text)])[0][0]
    roles = {item.canonical_hint: item.role for item in event.place_mentions}
    assert not [item for item in values if item.source_place and item.destination_place]
    assert roles["New Carthage"] is EventPlaceRole.ORIGIN
    assert roles["Alps"] is EventPlaceRole.RELATED_PLACE
    assert roles["Italy"] is EventPlaceRole.DESTINATION


def test_article_prefixed_origin_is_emitted_without_promoting_unresolved_place():
    event = EvidenceGroundedHistoricalEventExtractor(extractor()).extract(
        [evidence("druentia", "From the Druentia, the army arrived at B.")]
    )[0][0]
    by_raw = {item.raw_text: item for item in event.place_mentions}
    assert by_raw["Druentia"].role is EventPlaceRole.ORIGIN
    assert by_raw["Druentia"].canonical_hint is None
    assert by_raw["B"].role is EventPlaceRole.DESTINATION


def test_unrelated_mentions_and_external_metadata_cannot_create_direction():
    assert claims("A and B were discussed near C.") == []
    ranked = evidence("ranked", "The army marched from A to B.", rank=999, latitude=91, longitude=999, actor="unrelated")
    claim = extractor().movement_claims([ranked], event_id="event")[0]
    assert (claim.source_place, claim.destination_place) == ("A", "B")
    # Retrieval rank, coordinates, and actor identity are not direction inputs.
    assert claim.text == "The army marched from A to B."
