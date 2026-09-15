"""G7B: fixture integrity checks for the frozen broad evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "g7_broad_full_chain_evaluation.json"
G5R_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "g5r_trusted_movement_benchmark.json"

EXPECTED_CASE_IDS = [
    "G7-A01", "G7-A02", "G7-A03", "G7-A04", "G7-A05", "G7-A06",
    "G7-B01", "G7-B02", "G7-B03", "G7-B04", "G7-B05", "G7-B06", "G7-B07",
    "G7-C01", "G7-C02", "G7-C03", "G7-C04", "G7-C05", "G7-C06", "G7-C07",
]
SYNTHETIC_CASE_IDS = {"G7-C03", "G7-C04", "G7-C05", "G7-C06", "G7-C07"}
DUPLICATE_EXCLUSIONS = {
    "G7-A03": "fdd7cc22f0101becb09038160ca5bb78:2960:3378",
    "G7-A04": "38b95cee7bff2d5b81ddc06db4248b05:3543:3890",
    "G7-B04": "2ce94d921bb70e8f668b4bc336ea5950:2174:2677",
    "G7-B07": "618aa3e294764249a2b48cf90b240a05:769:1295",
}
CANONICAL_A05_ID = "e18d56bc132575c68cdeb326348f9ec0:3341:3742"
COORDINATE_KEYS = {"latitude", "longitude", "coordinates", "geometry"}


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def g5r_evidence_ids() -> set[str]:
    payload = json.loads(G5R_FIXTURE_PATH.read_text(encoding="utf-8"))
    ids = {item["evidence_id"] for item in payload.get("references", []) if item.get("evidence_id")}
    for item in payload.get("deprecated_references", []):
        if item.get("evidence_id"):
            ids.add(item["evidence_id"])
    return ids


def test_fixture_loads_and_schema_version(fixture: dict):
    assert fixture["schema_version"] == "g7-broad-v1"


def test_aggregate_policy(fixture: dict):
    policy = fixture["aggregate_policy"]
    assert policy["real_case_count"] == 15
    assert policy["synthetic_case_count"] == 5
    assert policy["excluded_sets"] == ["G5R"]
    assert policy["unsafe_fabrication_target"] == 0


def test_case_counts_and_ids(fixture: dict):
    cases = fixture["cases"]
    assert len(cases) == 20
    case_ids = [case["case_id"] for case in cases]
    assert len(set(case_ids)) == 20
    assert case_ids == EXPECTED_CASE_IDS


def test_origin_and_tier_counts(fixture: dict):
    real = [case for case in fixture["cases"] if case["case_origin"] == "REAL_CORPUS"]
    synthetic = [case for case in fixture["cases"] if case["case_origin"] == "SYNTHETIC_CONTRACT"]
    assert len(real) == 15
    assert len(synthetic) == 5
    assert {case["case_id"] for case in synthetic} == SYNTHETIC_CASE_IDS

    tiers = {case["tier"] for case in fixture["cases"]}
    assert tiers == {"A", "B", "C"}
    assert sum(case["tier"] == "A" for case in fixture["cases"]) == 6
    assert sum(case["tier"] == "B" for case in fixture["cases"]) == 7
    assert sum(case["tier"] == "C" for case in fixture["cases"]) == 7


def test_real_cases_have_source_metadata_and_passage(fixture: dict):
    for case in fixture["cases"]:
        if case["case_origin"] != "REAL_CORPUS":
            continue
        source = case["source"]
        for key in (
            "author", "work", "document_id", "parent_id", "spine_index",
            "parent_start_offset", "parent_end_offset", "passage_start", "passage_end",
        ):
            assert key in source
        assert case["exact_bounded_passage"]
        assert case["gold_evidence_ids"]


def test_every_case_has_prohibited_claims(fixture: dict):
    for case in fixture["cases"]:
        assert case["prohibited_claims"]
        assert isinstance(case["prohibited_claims"], list)


def test_tier_a_full_cases_have_expected_waypoints(fixture: dict):
    for case in fixture["cases"]:
        if case["tier"] == "A" and case["expected_route_status"] == "FULL":
            assert case.get("expected_waypoints")


def test_canonical_a05_evidence_id(fixture: dict):
    a05 = next(case for case in fixture["cases"] if case["case_id"] == "G7-A05")
    assert CANONICAL_A05_ID in a05["gold_evidence_ids"]


def test_duplicate_exclusions_not_in_gold_evidence(fixture: dict):
    for case in fixture["cases"]:
        excluded = set(case.get("duplicate_evidence_ids_excluded", []))
        assert excluded.isdisjoint(set(case["gold_evidence_ids"]))
        expected = DUPLICATE_EXCLUSIONS.get(case["case_id"])
        if expected:
            assert expected in case.get("duplicate_evidence_ids_excluded", [])


def test_no_g5r_evidence_overlap(fixture: dict, g5r_evidence_ids: set[str]):
    gold_ids = {evidence_id for case in fixture["cases"] for evidence_id in case["gold_evidence_ids"]}
    assert gold_ids.isdisjoint(g5r_evidence_ids)


def test_synthetic_evidence_ids_match_passage_length(fixture: dict):
    for case in fixture["cases"]:
        if case["case_origin"] != "SYNTHETIC_CONTRACT":
            continue
        passage = case["exact_bounded_passage"]
        expected = f"g7-synthetic:{case['case_id']}:0:{len(passage)}"
        assert case["gold_evidence_ids"] == [expected]
        assert case["source"]["parent_id"] == f"g7-synthetic:{case['case_id']}"
        assert case["source"]["author"] == "SYNTHETIC"
        assert case["source"]["work"] == "G7 Contract Probes"
        assert case["source"]["document_id"] == "g7_synthetic_contracts"


def test_no_route_gold_coordinates(fixture: dict):
    for case in fixture["cases"]:
        waypoints = case.get("expected_waypoints")
        if waypoints is not None:
            assert all(not isinstance(item, dict) for item in waypoints)
        for endpoint in case.get("endpoints", []):
            assert COORDINATE_KEYS.isdisjoint(endpoint.keys())
            authority = endpoint.get("authority")
            if authority:
                assert COORDINATE_KEYS.isdisjoint(authority.keys())
