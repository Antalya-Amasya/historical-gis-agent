"""G6DQ: gate SOURCE_STRUCTURAL_ORDER on explicit bounded connectors."""
from __future__ import annotations

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventActorGrounding,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.event_route_orchestration import (
    EventAnchorRouteBuilder,
    OrderingRule,
)
from backend.app.routes.evidence_relevance import relation_admission_allowed
from backend.tests.test_g4b_route_components import relation

COORDS = {
    "Rhodes": (36.4, 28.2),
    "Patra": (38.2, 21.7),
    "Alpha": (45.0, 5.0),
    "Cyprus": (35.1, 33.3),
    "Capua": (41.08, 14.25),
    "Brundisium": (40.6, 17.9),
    "Pelusium": (31.04, 32.55),
}

QUERY_ARISTON = ("Trace Ariston's route in 48 BCE.",)


def place(name: str) -> HistoricalPlace:
    lat, lon = COORDS[name]
    return HistoricalPlace(
        id=name.lower(),
        canonical_name=name,
        latitude=lat,
        longitude=lon,
        source="test",
        confidence=0.8,
        coordinate_role="exact_site",
    )


def explicit_actor(name: str) -> HistoricalEventActorGrounding:
    return HistoricalEventActorGrounding(
        actor_text=name,
        actor_tokens=[name],
        actor_status=EventActorStatus.EXPLICIT,
    )


def unknown_actor() -> HistoricalEventActorGrounding:
    return HistoricalEventActorGrounding(actor_status=EventActorStatus.UNKNOWN)


def binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs),
        place=place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="test",
    )


def movement_event(
    identifier: str,
    statement: str,
    origin: str,
    destination: str,
    *,
    refs: list[str],
    actor: HistoricalEventActorGrounding,
    year: str | None = None,
) -> HistoricalEvent:
    grounding = HistoricalEventTemporalGrounding()
    if year is not None:
        grounding = HistoricalEventTemporalGrounding(
            raw_expression=f"{year} BCE",
            normalized_start=f"-{year}",
            normalized_end=f"-{year}",
            precision=TemporalPrecision.YEAR,
            evidence_refs=refs,
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        )
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        source_statements=[statement],
        place_bindings=[
            binding(origin, EventPlaceRole.ORIGIN, refs),
            binding(destination, EventPlaceRole.DESTINATION, refs),
        ],
        temporal_grounding=grounding,
        actor=actor,
    )


def passage_evidence(identifier: str, text: str, *, offset: int, document: str = "doc-1") -> Evidence:
    return Evidence(
        id=identifier,
        author="Plutarch",
        work="Lives",
        locator="1",
        excerpt=text[:500],
        text=text,
        metadata={"document_id": document, "spine_index": 1, "start_offset": offset},
    )


def _anchors_for(event: HistoricalEvent):
    from backend.app.routes.event_anchors import EventAnchor

    refs = tuple(sorted(set(event.evidence_refs)))
    return [
        EventAnchor(
            event_id=event.id,
            event_type=event.event_type.value,
            canonical_name=binding.place.canonical_name,
            role=binding.role,
            latitude=binding.place.latitude,
            longitude=binding.place.longitude,
            evidence_refs=refs,
            resolver_provenance="test",
            coordinate_role=binding.place.coordinate_role,
            limitations=(),
            period=None,
            place=binding.place,
            admission_type="MOVEMENT_WAYPOINT",
        )
        for binding in event.place_bindings
    ]


def _pair(
    first_stmt: str,
    second_stmt: str,
    *,
    connector: str | None = None,
    first_actor: HistoricalEventActorGrounding | None = None,
    second_actor: HistoricalEventActorGrounding | None = None,
    first_year: str | None = None,
    second_year: str | None = None,
    first_refs: list[str] | None = None,
    second_refs: list[str] | None = None,
    passage: str | None = None,
    first_offset: int = 100,
    second_offset: int = 200,
):
    actor = first_actor or explicit_actor("Ariston")
    second = second_actor or actor
    if connector and not second_stmt.lstrip().casefold().startswith(connector.casefold()):
        second_stmt = f"{connector} {second_stmt.lstrip()}"
    passage_text = passage or f"{first_stmt} {second_stmt}"
    refs_a = first_refs or ["ev-a"]
    refs_b = second_refs or ["ev-b"]
    e1 = movement_event(
        "e1",
        first_stmt,
        "Rhodes",
        "Patra",
        refs=refs_a,
        actor=actor,
        year=first_year,
    )
    e2 = movement_event(
        "e2",
        second_stmt,
        "Alpha",
        "Cyprus",
        refs=refs_b,
        actor=second,
        year=second_year,
    )
    evidence = {
        refs_a[0]: passage_evidence(refs_a[0], passage_text, offset=first_offset),
        refs_b[0]: passage_evidence(refs_b[0], passage_text, offset=second_offset),
    }
    return e1, e2, evidence


def _build_pair(first_stmt: str, second_stmt: str, **kwargs):
    e1, e2, evidence_by_id = _pair(first_stmt, second_stmt, **kwargs)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2],
        list(evidence_by_id.values()),
        event_id="g6dq",
        name="Ariston",
        period="48 BCE",
        query_contexts=QUERY_ARISTON,
    )
    return outcome, e1, e2, evidence_by_id


def _inter_order(e1, e2, evidence_by_id):
    by_event = {
        e1.id: _anchors_for(e1),
        e2.id: _anchors_for(e2),
    }
    return EventAnchorRouteBuilder._inter_event_order(
        e1.id,
        e2.id,
        by_event,
        {e1.id: e1, e2.id: e2},
        evidence_by_id,
    )


def test_plain_adjacent_source_order_is_unsafe_before_g6dq():
    e1, e2, evidence_by_id = _pair(
        "Ariston left Rhodes for Patra.",
        "Ariston sailed from Alpha to Cyprus.",
    )
    assert _inter_order(e1, e2, evidence_by_id) is None


@pytest.mark.parametrize("connector", ["Then", "Afterward", "Afterwards", "Thereafter"])
def test_explicit_connector_authorizes_structural_order(connector: str):
    ordered = _inter_order(*_pair(
        "Ariston left Rhodes for Patra.",
        "Ariston sailed from Alpha to Cyprus.",
        connector=connector,
        passage=(
            f"Ariston left Rhodes for Patra. "
            f"{connector} Ariston sailed from Alpha to Cyprus."
        ),
    )[:3])
    assert ordered == ("e1", "e2", OrderingRule.SOURCE_STRUCTURAL_ORDER)


def test_different_explicit_actor_rejected():
    outcome, *_ = _build_pair(
        "Ariston left Rhodes for Patra.",
        "Bion sailed from Alpha to Cyprus.",
        connector="Then",
        second_actor=explicit_actor("Bion"),
    )
    assert OrderingRule.SOURCE_STRUCTURAL_ORDER not in {item.rule for item in outcome.relations}


def test_explicit_then_unknown_second_actor_rejected():
    outcome, *_ = _build_pair(
        "Ariston left Rhodes for Patra.",
        "he sailed from Alpha to Cyprus.",
        connector="Then",
        second_actor=unknown_actor(),
    )
    assert _inter_order(*_pair(
        "Ariston left Rhodes for Patra.",
        "Then he sailed from Alpha to Cyprus.",
        second_actor=unknown_actor(),
    )[:3]) is None


def test_unknown_unknown_actors_rejected():
    e1, e2, evidence_by_id = _pair(
        "He left Rhodes for Patra.",
        "he sailed from Alpha to Cyprus.",
        connector="Then",
        first_actor=unknown_actor(),
        second_actor=unknown_actor(),
    )
    assert _inter_order(e1, e2, evidence_by_id) is None


def test_different_episode_rejected_after_connector():
    outcome, e1, e2, evidence_by_id = _build_pair(
        "In 200 BCE Ariston left Rhodes for Patra.",
        "Ariston sailed from Alpha to Cyprus.",
        connector="Then",
        first_year="200",
        second_year=None,
        passage="In 200 BCE Ariston left Rhodes for Patra. Then Ariston sailed from Alpha to Cyprus.",
    )
    ordered = _inter_order(e1, e2, evidence_by_id)
    assert ordered is not None
    rel = relation(
        "Patra",
        "Alpha",
        OrderingRule.SOURCE_STRUCTURAL_ORDER,
        refs=tuple(sorted(set(e1.evidence_refs) | set(e2.evidence_refs))),
        event_ids=(ordered[0], ordered[1]),
    )
    assert not relation_admission_allowed(
        rel,
        {e1.id: e1, e2.id: e2},
        evidence_by_id,
        ("Trace Ariston's route in 200 BCE.",),
        rule=rel.rule,
    )


def test_non_adjacent_statements_rejected():
    passage = (
        "Ariston left Rhodes for Patra. "
        "The weather remained calm. "
        "Then Ariston sailed from Alpha to Cyprus."
    )
    assert _inter_order(*_pair(
        "Ariston left Rhodes for Patra.",
        "Ariston sailed from Alpha to Cyprus.",
        connector="Then",
        passage=passage,
    )[:3]) is None


def test_retrospective_earlier_blocks_source_order():
    e1, e2, evidence_by_id = _pair(
        "Ariston sailed from Alpha to Cyprus.",
        "Earlier Ariston had left Rhodes for Patra.",
        first_offset=200,
        second_offset=100,
        passage="Ariston sailed from Alpha to Cyprus. Earlier Ariston had left Rhodes for Patra.",
    )
    assert _inter_order(e1, e2, evidence_by_id) is None


def test_while_not_authorized_by_this_phase():
    assert _inter_order(*_pair(
        "Ariston left Rhodes for Patra.",
        "While Ariston sailed from Alpha to Cyprus, the fleet held course.",
        connector="While",
    )[:3]) is None


def test_pompey_cyprus_and_pelusium_do_not_get_structural_order():
    from backend.app.routes.event_anchors import project_event_anchors

    cyprus = movement_event(
        "event-4b6ee897cebc",
        "Pompey sailed toward Cyprus.",
        "Rhodes",
        "Cyprus",
        refs=["ev-cyprus"],
        actor=unknown_actor(),
    )
    pelusium = movement_event(
        "event-bcd445242ec8",
        "He steered his course that way toward Pelusium.",
        "Alpha",
        "Pelusium",
        refs=["ev-pelusium"],
        actor=unknown_actor(),
    )
    evidence_by_id = {
        "ev-cyprus": passage_evidence(
            "ev-cyprus",
            "Pompey sailed toward Cyprus.",
            offset=100,
            document="pompey-doc",
        ),
        "ev-pelusium": passage_evidence(
            "ev-pelusium",
            "He steered his course that way toward Pelusium.",
            offset=200,
            document="pompey-doc",
        ),
    }
    anchors, _ = project_event_anchors([cyprus, pelusium], list(evidence_by_id.values()))
    by_event = {}
    for anchor in anchors:
        by_event.setdefault(anchor.event_id, []).append(anchor)
    assert EventAnchorRouteBuilder._inter_event_order(
        cyprus.id,
        pelusium.id,
        by_event,
        {cyprus.id: cyprus, pelusium.id: pelusium},
        evidence_by_id,
    ) is None
