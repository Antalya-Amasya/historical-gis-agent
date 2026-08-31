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


def test_march_infinitive_preserves_destination_without_inventing_origin():
    event = EvidenceGroundedHistoricalEventExtractor(extractor()).extract(
        [evidence("alesia", "The army immediately began to march to Alesia.")]
    )[0][0]
    assert event.event_type.value == "MOVEMENT"
    roles = {item.raw_text: item.role for item in event.place_mentions}
    assert roles == {"Alesia": EventPlaceRole.DESTINATION}
    assert EventPlaceRole.ORIGIN not in roles.values()


def test_army_at_place_does_not_create_movement_direction():
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract(
        [evidence("at-b", "The army was at B.")]
    )
    assert not any(
        mention.role in {EventPlaceRole.ORIGIN, EventPlaceRole.DESTINATION}
        for event in events
        for mention in event.place_mentions
    )


def test_co_occurring_places_without_movement_do_not_create_direction():
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract(
        [evidence("ab", "A and B were discussed near C.")]
    )
    assert events == []


def test_unrelated_mentions_and_external_metadata_cannot_create_direction():
    assert claims("A and B were discussed near C.") == []
    ranked = evidence("ranked", "The army marched from A to B.", rank=999, latitude=91, longitude=999, actor="unrelated")
    claim = extractor().movement_claims([ranked], event_id="event")[0]
    assert (claim.source_place, claim.destination_place) == ("A", "B")
    # Retrieval rank, coordinates, and actor identity are not direction inputs.
    assert claim.text == "The army marched from A to B."


def _movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([evidence("e1", text)])
    movement = next(event for event in events if event.event_type.value == "MOVEMENT")
    return {mention.raw_text: mention.role for mention in movement.place_mentions}


def test_march_to_narbo_preserves_single_destination():
    roles = _movement_roles(
        "Caesar thought that the march to Narbo ought to take the precedence of all his other plans."
    )
    assert roles.get("Narbo") is EventPlaceRole.DESTINATION
    assert EventPlaceRole.ORIGIN not in roles.values()


def test_attributive_gallic_custom_does_not_create_destination():
    roles = _movement_roles(
        "Archers from the Rutheni, and horse from the Gauls, with a long train of baggage, "
        "according to the Gallic custom of travelling, had arrived there."
    )
    assert "Gallic" not in roles or roles["Gallic"] is not EventPlaceRole.DESTINATION
    assert EventPlaceRole.DESTINATION not in roles.values()


def test_according_to_mannert_does_not_create_destination():
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([
        evidence("mannert", "According to Mannert, they derived their origin from the shattered remains of the army.")
    ])
    assert not any(
        mention.role is EventPlaceRole.DESTINATION and mention.raw_text == "Mannert"
        for event in events
        for mention in event.place_mentions
    )


def test_command_to_marius_does_not_create_destination():
    events, _ = EvidenceGroundedHistoricalEventExtractor(extractor()).extract([
        evidence("marius", "Quintus Cæpio was made equal in command to Marius, and became his colleague.")
    ])
    assert not any(
        mention.role is EventPlaceRole.DESTINATION and mention.raw_text == "Marius"
        for event in events
        for mention in event.place_mentions
    )


def test_brought_to_afranius_without_movement_link_does_not_create_destination():
    roles = _movement_roles(
        "Intelligence was brought to Afranius that the great convoys, which were on their march to Caesar, had halted at the river."
    )
    assert roles.get("Afranius") is not EventPlaceRole.DESTINATION


def test_troop_provenance_from_does_not_create_origin():
    roles = _movement_roles(
        "Archers from the Rutheni, and horse from the Gauls, with a long train of baggage, "
        "according to the Gallic custom of travelling, had arrived there."
    )
    assert roles.get("Rutheni") is not EventPlaceRole.ORIGIN
    assert roles.get("Gauls") is not EventPlaceRole.ORIGIN
