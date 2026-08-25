import json

from backend.app.candidate_routes import (
    ArmyProfile,
    CandidateRouteAnchor,
    CandidateRouteEngine,
    evaluate_route,
    route_to_geojson,
    serialize_route_evaluation,
)
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid


def profile(name, tolerance, supply_range):
    return ArmyProfile(
        name=name, movement_type="infantry", mountain_tolerance=tolerance,
        rough_terrain_penalty=1.5, supply_range=supply_range,
    )


def route(*, with_evidence=True, length=2):
    first = CandidateRouteAnchor(
        historical_place_id="anchor-a", canonical_name="Anchor A",
        evidence_refs=["evidence-a"] if with_evidence else [],
    )
    second = CandidateRouteAnchor(
        historical_place_id="anchor-b", canonical_name="Anchor B",
        evidence_refs=["evidence-b"] if with_evidence else [],
    )
    grid = SyntheticGrid.flat(length, 1)
    if length > 1:
        grid.set_cell(GridPoint(1, 0), elevation_m=100, terrain="mountain", terrain_multiplier=5)
    return CandidateRouteEngine().build_route(
        from_anchor=first, to_anchor=second, start=GridPoint(0, 0), goal=GridPoint(length - 1, 0), grid=grid,
        profile=ArmyProfile(),
    )


def test_candidate_route_score_serializes_to_stable_json():
    candidate = route()
    army = profile("roman_legion", 0.5, 30)
    score = candidate.evaluate(army)
    first = serialize_route_evaluation(candidate, score, army_profile=army, source_ids=["polybius-3"])
    second = serialize_route_evaluation(candidate, score, army_profile=army, source_ids=["polybius-3"])
    assert json.loads(json.dumps(first)) == first
    assert first == second
    assert first["score"]["total_cost"] == score.total_cost
    assert first["provenance"]["evidence_ids"] == ["evidence-a", "evidence-b"]
    assert first["provenance"]["source_ids"] == ["polybius-3"]


def test_geojson_feature_collection_has_line_and_frontend_properties():
    candidate = route()
    army = profile("roman_legion", 0.5, 30)
    geojson = route_to_geojson(candidate, candidate.evaluate(army), army)
    feature = geojson["features"][0]
    assert geojson["type"] == "FeatureCollection"
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "LineString"
    assert feature["properties"]["route_id"] == candidate.id
    assert feature["properties"]["army_profile"] == "roman_legion"
    assert "total_cost" in feature["properties"] and "confidence" in feature["properties"]


def test_profiles_produce_different_scores_for_same_candidate_route():
    candidate = route()
    roman = evaluate_route(candidate, profile("roman_legion", 0.5, 30))
    carthaginian = evaluate_route(candidate, profile("carthaginian_army", 0.8, 40))
    assert roman.route_id == carthaginian.route_id
    assert roman.candidate_route.geometry == carthaginian.candidate_route.geometry
    assert roman.score.total_cost != carthaginian.score.total_cost


def test_explanation_is_deterministic_and_no_evidence_still_has_algorithmic_output():
    candidate = route(with_evidence=False)
    first = evaluate_route(candidate)
    second = evaluate_route(candidate)
    assert first.explanation.model_dump() == second.explanation.model_dump()
    assert first.provenance.evidence_ids == []
    assert first.provenance.algorithmic_factors
    assert any(factor.type in {"terrain", "elevation", "algorithm"} for factor in first.explanation.factors)
