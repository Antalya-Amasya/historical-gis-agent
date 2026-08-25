"""Deterministic scoring and ordering for existing candidate routes."""
from __future__ import annotations

from .models import ArmyProfile, CandidateRoute, CandidateRouteSet, RankedRoute, RankedRouteSet, RankingProfile, RouteScore


class RouteRankingModel:
    """Ranks supplied candidate geometries; it never generates or recomputes a route."""

    @staticmethod
    def weighted_score(score: RouteScore, profile: RankingProfile) -> float:
        return (
            profile.distance_weight * score.distance_cost
            + profile.terrain_weight * score.terrain_cost
            + profile.historical_weight * score.historical_cost
        )

    def rank(
        self,
        route_set: CandidateRouteSet,
        *,
        army_profile: ArmyProfile | None = None,
        ranking_profile: RankingProfile | None = None,
    ) -> RankedRouteSet:
        ranking_profile = ranking_profile or RankingProfile()
        entries: list[tuple[float, float, str, CandidateRoute, RouteScore]] = []
        for route in route_set.routes:
            score = route.evaluate(army_profile)
            weighted = self.weighted_score(score, ranking_profile)
            # Route id is an explicit stable tie-breaker for reproducible results.
            entries.append((weighted, score.total_cost, route.id, route, score))
        entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        ranked = [
            RankedRoute(
                route=route,
                rank=index,
                score=score,
                ranking_score=weighted,
                reasons=self._reasons(score, ranking_profile),
            )
            for index, (weighted, _, _, route, score) in enumerate(entries, start=1)
        ]
        return RankedRouteSet(
            routes=ranked,
            ranking_profile=ranking_profile,
            generation_method=route_set.generation_method,
        )

    @staticmethod
    def _reasons(score: RouteScore, profile: RankingProfile) -> list[str]:
        reasons = ["deterministic weighted score from distance, terrain, and historical costs"]
        if profile.terrain_weight and score.terrain_cost:
            reasons.append(f"terrain component={score.terrain_cost:.3f}")
        if profile.historical_weight and score.historical_cost:
            reasons.append(f"historical movement component={score.historical_cost:.3f}")
        if not score.terrain_cost and not score.historical_cost:
            reasons.append("no terrain or historical penalty in this evaluation")
        return reasons
