"""Deterministic validation for broad prepositional place-like mentions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.app.models import PlaceMentionValidationClass


class PlaceMentionValidationReason(str, Enum):
    NON_PLACE_OCR_ARTIFACT = "NON_PLACE_OCR_ARTIFACT"
    NON_PLACE_BOILERPLATE = "NON_PLACE_BOILERPLATE"
    NON_PLACE_ETHNONYM_CONTEXT = "NON_PLACE_ETHNONYM_CONTEXT"
    NON_PLACE_PERSON_CONTEXT = "NON_PLACE_PERSON_CONTEXT"
    NON_PLACE_ADJECTIVAL_CONTEXT = "NON_PLACE_ADJECTIVAL_CONTEXT"
    NON_PLACE_INSTITUTION_CONTEXT = "NON_PLACE_INSTITUTION_CONTEXT"


_ATTRIBUTIVE_AFTER_PLACE = re.compile(
    r"\s+(?:custom|war|manner|style|fashion|tradition|practice|people|triumph|riches|"
    r"territory|territories|cities|language|tongue|cities|soldiers|troops|army|fleet|"
    r"provinces|lands|coast|shores)\b",
    re.IGNORECASE,
)
_BOILERPLATE_METADATA_CONTEXT = re.compile(
    r"\b(?:ebook|etext|gutenberg|anyone anywhere|copyright|royalt(?:y|ies)|"
    r"literary archive foundation|project gutenberg)\b",
    re.IGNORECASE,
)
_ETHNONYM_SUFFIX = re.compile(r"(ians|ans|ites|enes|eeks|mans)\b", re.IGNORECASE)
_PERSON_APPOSITIVE = re.compile(
    r",\s*the\s+(?:Macedonian|Roman|Great|Younger|Elder|younger|elder|Achaemenid|"
    r"Seleucid|Ptolemaic|Syrian|Athenian|Spartan|Persian|Egyptian|African)\b",
    re.IGNORECASE,
)
_PERSON_POSSESSIVE = re.compile(
    r"'s\s+(?:son|daughter|wife|brother|sister|father|mother|friend|legate|general|army)\b",
    re.IGNORECASE,
)
_INSTITUTION_NOUNS = frozenset({"senate", "assembly"})


@dataclass(frozen=True)
class PlaceMentionValidation:
    validation_class: PlaceMentionValidationClass
    reason: str | None = None


def validate_broad_place_mention(
    surface: str,
    sentence: str,
    match: re.Match[str],
    *,
    canonical_hint: str | None = None,
) -> PlaceMentionValidation:
    """Conservative high-confidence non-place gate for broad _PLACE_PATTERN hits."""
    try:
        if canonical_hint:
            return PlaceMentionValidation(PlaceMentionValidationClass.GEOGRAPHIC_PLACE_CANDIDATE)

        if len(surface) == 1 and surface.isalpha():
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_OCR_ARTIFACT.value,
            )

        match_text = match.group(0)
        place_start = match.start("place")
        place_end = match.end("place")
        after_place = sentence[place_end:]
        lowered = surface.casefold()

        if "project gutenberg" in lowered or "literary archive" in lowered:
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_BOILERPLATE.value,
            )
        if lowered == "united states" and _BOILERPLATE_METADATA_CONTEXT.search(sentence):
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_BOILERPLATE.value,
            )

        if lowered in _INSTITUTION_NOUNS:
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_INSTITUTION_CONTEXT.value,
            )
        if lowered == "people" and re.search(r"\bthe\s+People\b", match_text):
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_INSTITUTION_CONTEXT.value,
            )

        prefix = match_text[: place_start - match.start()]
        the_prefix = bool(re.search(r"\bthe\s+$", prefix, re.IGNORECASE))
        if the_prefix and _ETHNONYM_SUFFIX.search(surface):
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_ETHNONYM_CONTEXT.value,
            )

        if _ATTRIBUTIVE_AFTER_PLACE.match(after_place):
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_ADJECTIVAL_CONTEXT.value,
            )

        if _PERSON_APPOSITIVE.match(after_place) or _PERSON_POSSESSIVE.match(after_place):
            return PlaceMentionValidation(
                PlaceMentionValidationClass.NON_PLACE_HIGH_CONFIDENCE,
                PlaceMentionValidationReason.NON_PLACE_PERSON_CONTEXT.value,
            )

        return PlaceMentionValidation(PlaceMentionValidationClass.UNKNOWN)
    except Exception:
        return PlaceMentionValidation(PlaceMentionValidationClass.UNKNOWN)
