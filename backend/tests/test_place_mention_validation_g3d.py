from backend.app.geography.place_registry import resolve_with_status
from backend.app.models import (
    Evidence,
    EventPlaceResolutionStatus,
    PlaceMentionValidationClass,
)
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_mention_validation import validate_broad_place_mention
from geography_mcp.service import GeographyService


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


class RecordingGeography:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        self.calls.append(name)
        payload = GeographyService().resolve_ancient_place_payload(name)
        return payload


def extract_mentions(text: str):
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([evidence("e1", text)])
    return events[0].place_mentions if events else []


def resolve_mentions(text: str, geo: RecordingGeography | None = None):
    geo = geo or RecordingGeography()
    events, _ = EvidenceGroundedHistoricalEventExtractor().extract([evidence("e1", text)])
    resolved, diagnostics = HistoricalEventPlaceResolver(geo).resolve(events)
    return resolved, diagnostics, geo


def test_philip_person_context_is_not_sent_to_geography():
    mentions = extract_mentions("The Romans paid no attention to Philip, the Macedonian, when he began war.")
    philip = next(mention for mention in mentions if mention.raw_text == "Philip")
    assert philip.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
    assert philip.validation_reason == "NON_PLACE_PERSON_CONTEXT"
    _, _, geo = resolve_mentions("The Romans paid no attention to Philip, the Macedonian, when he began war.")
    assert "Philip" not in geo.calls


def test_scipio_possessive_person_context_is_not_sent_to_geography():
    text = "The army marched to Scipio's son before winter."
    mentions = extract_mentions(text)
    scipio = next(mention for mention in mentions if mention.raw_text == "Scipio")
    assert scipio.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
    _, _, geo = resolve_mentions(text)
    assert "Scipio" not in geo.calls


def test_ethnonym_actor_context_blocks_geography_for_romans_and_athenians():
    for text, surface in (
        ("The army marched to the Romans.", "Romans"),
        ("The army withdrew from the Athenians.", "Athenians"),
    ):
        mention = next(item for item in extract_mentions(text) if item.raw_text == surface)
        assert mention.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
        assert mention.validation_reason == "NON_PLACE_ETHNONYM_CONTEXT"
        _, _, geo = resolve_mentions(text)
        assert surface not in geo.calls


def test_rome_remains_eligible_for_geography():
    mentions = extract_mentions("The army marched to Rome.")
    rome = next(mention for mention in mentions if mention.raw_text == "Rome")
    assert rome.validation_class in {
        PlaceMentionValidationClass.UNKNOWN,
        PlaceMentionValidationClass.GEOGRAPHIC_PLACE_CANDIDATE,
    }
    resolved, diagnostics, geo = resolve_mentions("The army marched to Rome.")
    assert "Roma" in geo.calls
    binding = resolved[0].place_bindings[0]
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert binding.place is not None


def test_asia_and_alexandria_remain_eligible_for_geography():
    for text, surface in (
        ("The army marched into Asia.", "Asia"),
        ("The armies fought at Alexandria.", "Alexandria"),
    ):
        mention = next(item for item in extract_mentions(text) if item.raw_text == surface)
        assert mention.validation_class is not PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
        _, _, geo = resolve_mentions(text)
        assert surface in geo.calls


def test_greece_remains_eligible_for_curated_geography():
    mentions = extract_mentions("The army marched to Greece with provisions.")
    greece = next(mention for mention in mentions if mention.raw_text == "Greece")
    assert greece.validation_class is PlaceMentionValidationClass.GEOGRAPHIC_PLACE_CANDIDATE
    resolved, _, geo = resolve_mentions("The army marched to Greece with provisions.")
    assert "Greece" in geo.calls
    binding = next(binding for binding in resolved[0].place_bindings if binding.mention.raw_text == "Greece")
    assert binding.resolution_status is EventPlaceResolutionStatus.RESOLVED


def test_ocr_artifact_single_letter_is_not_sent_to_geography():
    mentions = extract_mentions("The campaign ended in B.c. 203 near the coast.")
    assert all(mention.raw_text != "B" for mention in mentions) or all(
        mention.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
        for mention in mentions
        if mention.raw_text == "B"
    )


def test_boilerplate_metadata_context_is_not_sent_to_geography():
    text = "This eBook is for the use of anyone anywhere in the United States and most other parts of the world."
    pattern = EvidenceGroundedHistoricalEventExtractor._PLACE_PATTERN
    match = next(pattern.finditer(text))
    validation = validate_broad_place_mention(match.group("place"), text, match)
    assert validation.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
    assert validation.reason == "NON_PLACE_BOILERPLATE"


def test_unknown_proper_noun_remains_eligible_for_geography():
    text = "The army halted at Obscurium."
    mention = extract_mentions(text)[0]
    assert mention.validation_class is PlaceMentionValidationClass.UNKNOWN
    _, _, geo = resolve_mentions(text)
    assert "Obscurium" in geo.calls


def test_sulla_movement_claims_remain_unchanged():
    text = (
        "Sulla left them and sailed for Greece, and thence passed on to Italy "
        "with the greater part of his army."
    )
    claim = HistoricalPlaceMentionExtractor().movement_claims([evidence("sulla", text)], event_id="event")[0]
    assert (claim.source_place, claim.destination_place, claim.movement_relation) == (
        "Greece", "Italia", "thence_passed_on_to",
    )


def test_adjectival_roman_territory_is_not_sent_to_geography():
    mentions = extract_mentions("The army advanced into the Roman territory.")
    roman = next(mention for mention in mentions if mention.raw_text == "Roman")
    assert roman.validation_class is PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE
    assert roman.validation_reason == "NON_PLACE_ADJECTIVAL_CONTEXT"


def test_direct_resolve_smoke_cases_remain_stable():
    for surface, expected_status in (
        ("Rome", "UNIQUE"),
        ("Greece", "CURATED"),
        ("Asia", "AMBIGUOUS"),
        ("Alexandria", "AMBIGUOUS"),
    ):
        assert resolve_with_status(surface).status == expected_status
