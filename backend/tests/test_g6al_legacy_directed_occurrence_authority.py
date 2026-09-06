"""G6AL: legacy fallback must preserve directed roles and movement occurrence identity."""

from __future__ import annotations

from backend.app.agent.tools import AgentToolRegistry
from backend.app.models import AgentState, Evidence
from backend.app.rag.retriever import HistoricalRetriever
from backend.app.routes.extractor import _legacy_od_has_positive_authority

QUERY_UNRESTRICTED = "Trace Ariston's route."
QUERY_ROME_CAPUA = "Trace Ariston from Rome to Capua."

CASE_A = (
    "Ariston did not march from Rome to Capua, "
    "but marched from Capua to Rome."
)
CASE_B = "Ariston crossed the Rhone, then Bion entered the Alps."
SAME_OCCURRENCE = "Ariston crossed the Rhone and then entered the Alps."
SEPARATE_OCCURRENCES = (
    "Ariston crossed the Rhone and returned to Gaul. "
    "Years later Ariston entered the Alps."
)
SIMPLE_DIRECTED = "Ariston marched from Rome to Capua."
TWO_OPPOSITE = (
    "Ariston marched from Rome to Capua, "
    "then later marched from Capua to Rome."
)
BION_PREFIX = "In 200 BCE Bion marched from Rome to Capua."
CONTRADICTORY_YEAR = "In 100 BCE Ariston marched from Rome to Capua."
QUERY_200 = "Trace Ariston's route in 200 BCE."


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
        "Roma": (41.9, 12.5),
        "Capua": (41.08, 14.25),
        "Rhodanus": (43.3, 4.8),
        "Alpes": (43.7, 7.4),
        "Gaul": (46.0, 2.0),
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


def execute_route(
    text: str,
    *,
    query: str = QUERY_UNRESTRICTED,
    evidence_id: str = "ev1",
) -> tuple[AgentState, dict, dict]:
    state = AgentState(
        session_id="g6al",
        user_query=query,
        requested_output="historical_route",
    )
    tools = AgentToolRegistry(Retriever([evidence(evidence_id, text)]), Geography())
    tools.execute("search_historical_evidence", {"query": query, "top_k": 5}, state)
    result, _ = tools.execute(
        "build_historical_route",
        {"event_id": "g6al", "name": "G6AL", "period": "100 BCE"},
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


def authority(
    sentence: str,
    origin: str,
    destination: str,
    *,
    origin_surface: str | None = None,
    destination_surface: str | None = None,
) -> bool:
    return _legacy_od_has_positive_authority(
        sentence,
        origin,
        destination,
        source_surface=origin_surface or origin,
        destination_surface=destination_surface or destination,
    )


def test_a_reverse_direction_production_path_denies_denied_pair():
    assert not authority(CASE_A, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert authority(CASE_A, "Capua", "Roma", origin_surface="Capua", destination_surface="Rome")

    state, _, meta = execute_route(CASE_A, query=QUERY_ROME_CAPUA)
    trace = meta["trace"]
    accepted = set(legacy_claim_endpoints(trace))

    assert ("Roma", "Capua") not in accepted
    assert ("Capua", "Roma") not in accepted
    assert ("Roma", "Capua") not in route_edge_pairs(state)


def test_b_different_subject_crossing_production_path_blocked():
    assert not authority(
        CASE_B,
        "Rhodanus",
        "Alpes",
        origin_surface="Rhone",
        destination_surface="Alps",
    )

    state, result, meta = execute_route(CASE_B)
    accepted = set(legacy_claim_endpoints(meta["trace"]))

    assert ("Rhodanus", "Alpes") not in accepted
    assert result["result"]["route"] is None or ("Rhodanus", "Alpes") not in route_edge_pairs(state)


def test_c_valid_same_occurrence_crossing_preserved():
    assert authority(
        SAME_OCCURRENCE,
        "Rhodanus",
        "Alpes",
        origin_surface="Rhone",
        destination_surface="Alps",
    )
    state, result, meta = execute_route(SAME_OCCURRENCE)
    assert ("Rhodanus", "Alpes") in set(legacy_claim_endpoints(meta["trace"]))
    assert route_edge_pairs(state) == [("Rhodanus", "Alpes")]
    assert result["result"]["route"] is not None


def test_d_same_subject_separate_occurrences_not_fused():
    assert not authority(
        SEPARATE_OCCURRENCES,
        "Rhodanus",
        "Alpes",
        origin_surface="Rhone",
        destination_surface="Alps",
    )


def test_e_ordinary_directed_movement_preserves_direction():
    assert authority(SIMPLE_DIRECTED, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert not authority(SIMPLE_DIRECTED, "Capua", "Roma", origin_surface="Capua", destination_surface="Rome")


def test_f_two_positive_opposite_movements_both_valid():
    assert authority(TWO_OPPOSITE, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert authority(TWO_OPPOSITE, "Capua", "Roma", origin_surface="Capua", destination_surface="Rome")


def test_g_canonical_direction_without_reverse():
    text = "Ariston crossed the Rhone and then entered the Alps."
    assert authority(text, "Rhodanus", "Alpes", origin_surface="Rhone", destination_surface="Alps")
    assert not authority(text, "Alpes", "Rhodanus", origin_surface="Alps", destination_surface="Rhone")


def test_suffix_equivalence_matches_prefix_subject():
    prefix = "In 200 BCE Ariston marched from Rome to Capua."
    suffix = "Ariston marched from Rome to Capua in 200 BCE."
    assert authority(prefix, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")
    assert authority(suffix, "Roma", "Capua", origin_surface="Rome", destination_surface="Capua")


def test_route_level_date_prefix_movement_preserved():
    text = "In 200 BCE Ariston marched from Rome to Capua."
    state, result, meta = execute_route(text)
    assert route_edge_pairs(state) == [("Roma", "Capua")]
    assert result["result"]["route"] is not None


def test_wrong_subject_still_rejects():
    state, _, meta = execute_route(BION_PREFIX)
    assert ("Roma", "Capua") not in set(legacy_claim_endpoints(meta["trace"]))


def test_query_year_contradiction_still_rejects():
    state, _, meta = execute_route(CONTRADICTORY_YEAR, query=QUERY_200)
    assert ("Roma", "Capua") not in set(legacy_claim_endpoints(meta["trace"]))
