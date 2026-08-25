from backend.app.candidate_routes.annotation import (
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.knowledge_panel import (
    EvidenceBackedSummary,
    EvidenceBackedSummaryProvider,
    HistoricalSummaryProvider,
    KnowledgePanelBuilder,
    SummaryEvidenceError,
    SummaryResult,
)
from backend.app.candidate_routes.location import LocationConfidence
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.view_model import HistoricalWaypointViewModel


def _view(*, evidence_refs=None, external_references=None):
    return HistoricalWaypointViewModel(
        id="battle_cannae",
        name="Cannae",
        location_confidence=LocationConfidence.EXACT,
        evidence_refs=list(["polybius_001", "livy_002"] if evidence_refs is None else evidence_refs),
        external_references=list([] if external_references is None else external_references),
    )


def test_evidence_backed_provider_binds_summary_and_used_refs():
    provider = EvidenceBackedSummaryProvider({
        "battle_cannae": EvidenceBackedSummary(
            summary_text="Explicit local summary.", evidence_refs=["polybius_001"], confidence=0.8
        )
    })

    panel = KnowledgePanelBuilder().build(_view(), evidence_summary_provider=provider)

    assert panel.summary == "Explicit local summary."
    assert panel.evidence_backed_summary is not None
    assert panel.evidence_backed_summary.evidence_refs == ["polybius_001"]
    assert panel.evidence_backed_summary.confidence == 0.8


class _InvalidEvidenceProvider(HistoricalSummaryProvider):
    def get_summary(self, waypoint_id, evidence_refs):
        return SummaryResult(text="Unsupported binding", used_evidence_refs=["outside"], confidence=0.7)


def test_provider_cannot_claim_evidence_absent_from_waypoint():
    try:
        KnowledgePanelBuilder().build(_view(), evidence_summary_provider=_InvalidEvidenceProvider())
    except SummaryEvidenceError as exc:
        assert "not attached" in str(exc)
    else:
        raise AssertionError("expected evidence binding validation to reject the provider result")


def test_provider_use_requires_waypoint_evidence():
    provider = EvidenceBackedSummaryProvider({})

    try:
        KnowledgePanelBuilder().build(_view(evidence_refs=[]), evidence_summary_provider=provider)
    except SummaryEvidenceError as exc:
        assert "without waypoint evidence" in str(exc)
    else:
        raise AssertionError("expected an evidence-less waypoint to be rejected")


class _RecordingProvider(HistoricalSummaryProvider):
    def __init__(self):
        self.calls = []

    def get_summary(self, waypoint_id, evidence_refs):
        self.calls.append((waypoint_id, list(evidence_refs)))
        return SummaryResult(text=None, used_evidence_refs=[], confidence=0.0)


def test_external_references_are_not_summary_provider_input():
    reference = HistoricalExternalReference(
        id="reading", reference_type=HistoricalExternalReferenceType.PAPER,
        title="Reading", url="https://example.invalid/reading",
    )
    provider = _RecordingProvider()

    panel = KnowledgePanelBuilder().build(
        _view(external_references=[reference]), evidence_summary_provider=provider
    )

    assert provider.calls == [("battle_cannae", ["polybius_001", "livy_002"])]
    assert panel.summary is None and panel.evidence_backed_summary is None


def test_evidence_backed_summary_does_not_change_ranking():
    route = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    provider = EvidenceBackedSummaryProvider({
        "battle_cannae": EvidenceBackedSummary(
            summary_text="Explicit local summary.", evidence_refs=["livy_002"], confidence=0.6
        )
    })
    KnowledgePanelBuilder().build(_view(), evidence_summary_provider=provider)
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()

    assert before == after
