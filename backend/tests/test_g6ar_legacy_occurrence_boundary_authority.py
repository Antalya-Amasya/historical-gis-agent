"""G6AR: explicit campaign/episode boundaries must break legacy occurrence fusion."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.extractor import _legacy_od_has_positive_authority

QUERY = "Trace Ariston's route."

S6_03 = (
    "Ariston crossed the Rhone, "
    "then in a different campaign Ariston entered the Alps."
)
ANOTHER_CAMPAIGN = (
    "Ariston crossed the Rhone. "
    "In another campaign Ariston entered the Alps."
)
LATER_CAMPAIGN = (
    "Ariston crossed the Rhone. "
    "In a later campaign Ariston entered the Alps."
)
PRONOUN_DIFFERENT = (
    "Ariston crossed the Rhone. "
    "In a different campaign he entered the Alps."
)
VALID_PRONOUN = (
    "Ariston crossed the Rhone, "
    "then he entered the Alps."
)
VALID_DIRECT = (
    "Ariston crossed the Rhone "
    "and then entered the Alps."
)
TEMPORAL = (
    "Ariston crossed the Rhone and returned to Gaul. "
    "Years later Ariston entered the Alps."
)
DIFFERENT_SUBJECT = (
    "Ariston crossed the Rhone, "
    "then Bion entered the Alps."
)
ORDINARY_CAMPAIGN = (
    "During the campaign Ariston crossed the Rhone, "
    "then he entered the Alps."
)
CAMPAIGN_ALPHA = (
    "During Campaign Alpha Ariston crossed the Rhone, "
    "then he entered the Alps."
)
SAME_CAMPAIGN = (
    "Ariston crossed the Rhone. "
    "In the same campaign he entered the Alps."
)


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Source",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


class Retriever(HistoricalRetriever):
    def __init__(self, items: list[Evidence]):
        self.items = items

    def retrieve(self, *_args, **_kwargs):
        return self.items


class Geography:
    places = {
        "Rhodanus": (43.3, 4.8),
        "Alpes": (43.7, 7.4),
    }

    def call(self, tool, arguments):
        assert tool == "resolve_ancient_place"
        name = arguments["name"]
        if name not in self.places:
            return {"found": False}
        lat, lon = self.places[name]
        return {
            "found": True,
            "id": f"fixture-{name}",
            "canonical_name": name,
            "latitude": lat,
            "longitude": lon,
            "source": "fixture",
            "source_id": name,
            "confidence": 0.8,
            "coordinate_role": "representative_point",
        }


def authority(text: str) -> bool:
    return _legacy_od_has_positive_authority(
        text,
        "Rhodanus",
        "Alpes",
        source_surface="Rhone",
        destination_surface="Alps",
    )


def execute_route(text: str, *, query: str = QUERY) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6ar",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence("ev1", text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6ar", "name": "G6AR", "period": "100 BCE"},
        state,
    )
    diagnostics = state.historical_route_diagnostics or {}
    trace = diagnostics.get("provenance_trace", {})
    return state, result, {"diagnostics": diagnostics, "trace": trace}


def route_edge_pairs(state: AgentState) -> list[tuple[str, str]]:
    if state.historical_route is None:
        return []
    names = [point.historical_place.canonical_name for point in state.historical_route.ordered_points]
    return list(zip(names, names[1:]))


def legacy_claim_endpoints(trace: dict) -> list[tuple[str | None, str | None]]:
    return [
        (claim.get("origin"), claim.get("destination"))
        for claim in trace.get("legacy", {}).get("claims", [])
        if claim.get("accepted")
    ]


def assert_no_rhone_alps_fusion(text: str):
    assert authority(text) is False
    state, result, meta = execute_route(text)
    assert ("Rhodanus", "Alpes") not in legacy_claim_endpoints(meta["trace"])
    assert ("Rhodanus", "Alpes") not in route_edge_pairs(state)
    assert result["result"]["route"] is None or ("Rhodanus", "Alpes") not in route_edge_pairs(state)


def assert_rhone_alps_preserved(text: str):
    assert authority(text) is True
    state, result, meta = execute_route(text)
    assert ("Rhodanus", "Alpes") in legacy_claim_endpoints(meta["trace"])
    assert route_edge_pairs(state) == [("Rhodanus", "Alpes")]
    assert result["result"]["route"] is not None


def test_a_exact_gpt6_s6_03_different_campaign():
    assert_no_rhone_alps_fusion(S6_03)


def test_b_another_campaign():
    assert_no_rhone_alps_fusion(ANOTHER_CAMPAIGN)


def test_c_later_campaign():
    assert_no_rhone_alps_fusion(LATER_CAMPAIGN)


def test_d_pronoun_with_different_campaign():
    assert_no_rhone_alps_fusion(PRONOUN_DIFFERENT)


def test_e_valid_pronoun_continuation():
    assert_rhone_alps_preserved(VALID_PRONOUN)


def test_f_valid_direct_same_subject_continuation():
    assert_rhone_alps_preserved(VALID_DIRECT)


def test_g_temporal_boundary_regression():
    assert authority(TEMPORAL) is False
    assert_no_rhone_alps_fusion(TEMPORAL)


def test_h_different_subject_regression():
    assert_no_rhone_alps_fusion(DIFFERENT_SUBJECT)


def test_i_ordinary_campaign_mention_does_not_break():
    assert_rhone_alps_preserved(ORDINARY_CAMPAIGN)


def test_j_campaign_alpha_descriptor_does_not_break():
    assert_rhone_alps_preserved(CAMPAIGN_ALPHA)


def test_k_same_campaign_not_treated_as_boundary():
    assert authority(SAME_CAMPAIGN) is True
