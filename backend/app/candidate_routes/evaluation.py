"""Pure-data bridge from candidate-route scores to frontend-ready structures."""
from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, Field

from .models import ArmyProfile, CandidateRoute, RouteScore


class RouteExplanationFactor(BaseModel):
    type: Literal["terrain", "elevation", "supply", "evidence", "algorithm"]
    impact: Literal["negative", "neutral"]
    reason: str


class RouteExplanation(BaseModel):
    """Deterministic explanation factors, never LLM-generated prose."""

    factors: list[RouteExplanationFactor] = Field(default_factory=list)


class RouteEvaluationProvenance(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    algorithmic_factors: list[str] = Field(default_factory=list)


class RouteEvaluationResult(BaseModel):
    route_id: str
    candidate_route: CandidateRoute
    army_profile: ArmyProfile | None = None
    score: RouteScore
    explanation: RouteExplanation
    provenance: RouteEvaluationProvenance


def build_route_explanation(route: CandidateRoute, score: RouteScore, profile: ArmyProfile | None) -> RouteExplanation:
    """Explain only observable cost/metric conditions from this local evaluation."""
    factors: list[RouteExplanationFactor] = []
    if score.terrain_cost > 0:
        factors.append(RouteExplanationFactor(type="terrain", impact="negative", reason="terrain and local slope increased route cost"))
    if route.metrics.elevation_gain_m > 0 or route.metrics.elevation_loss_m > 0:
        factors.append(RouteExplanationFactor(type="elevation", impact="negative", reason="elevation change contributed to movement cost"))
    if profile is not None and route.metrics.distance_km > profile.supply_range:
        factors.append(RouteExplanationFactor(type="supply", impact="negative", reason="route distance exceeds configured supply range"))
    if route.evidence_refs:
        factors.append(RouteExplanationFactor(type="evidence", impact="neutral", reason="anchors retain supplied evidence references"))
    if not factors:
        factors.append(RouteExplanationFactor(type="algorithm", impact="neutral", reason="algorithmic candidate route has no additional cost factors"))
    return RouteExplanation(factors=factors)


def evaluate_route(
    route: CandidateRoute,
    army_profile: ArmyProfile | None = None,
    *,
    source_ids: Sequence[str] = (),
) -> RouteEvaluationResult:
    """Score one existing candidate route and construct a JSON-ready provenance boundary."""
    score = route.evaluate(army_profile)
    explanation = build_route_explanation(route, score, army_profile)
    return RouteEvaluationResult(
        route_id=route.id,
        candidate_route=route,
        army_profile=army_profile,
        score=score,
        explanation=explanation,
        provenance=RouteEvaluationProvenance(
            evidence_ids=list(dict.fromkeys(route.evidence_refs)),
            source_ids=list(dict.fromkeys(source_ids)),
            algorithmic_factors=[factor.reason for factor in explanation.factors if factor.type != "evidence"],
        ),
    )


def serialize_route_evaluation(
    route: CandidateRoute,
    score: RouteScore,
    *,
    army_profile: ArmyProfile | None = None,
    source_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Return plain JSON-compatible data without runtime objects or internal search state."""
    result = evaluate_route(route, army_profile, source_ids=source_ids)
    # Preserve the supplied score when callers evaluate externally with an injected model.
    result = result.model_copy(update={"score": score})
    return {
        "route": {
            "id": route.id,
            "geometry": route.geometry.model_dump(mode="json"),
            "points": route.geometry.model_dump(mode="json")["coordinates"],
            "confidence": route.confidence,
            "provenance": route.provenance,
        },
        "army_profile": army_profile.model_dump(mode="json") if army_profile else None,
        "score": score.model_dump(mode="json"),
        "explanation": result.explanation.model_dump(mode="json"),
        "provenance": result.provenance.model_dump(mode="json"),
    }


def route_to_geojson(route: CandidateRoute, score: RouteScore, army_profile: ArmyProfile | None = None) -> dict[str, object]:
    """Map-SDK-free standard GeoJSON FeatureCollection for exactly one candidate line."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": route.geometry.model_dump(mode="json"),
                "properties": {
                    "route_id": route.id,
                    "total_cost": score.total_cost,
                    "army_profile": army_profile.name if army_profile else score.profile_name,
                    "confidence": route.confidence,
                },
            }
        ],
    }
