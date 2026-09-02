"""Present evidence-backed route components as independent GIS fragments."""
from __future__ import annotations

from backend.app.candidate_routes.presentation import HistoricalRouteResponse, RoutePresentationFragment
from backend.app.models import Evidence, GeoJsonLineString, HistoricalRoute, HistoricalRouteComponent, HistoricalRouteIntent
from backend.app.route_orchestrator import HistoricalRouteOrchestrator, RouteOrchestrationError


class ComponentFragmentPresentationResult:
    def __init__(
        self,
        *,
        presentation: HistoricalRouteResponse | None = None,
        fragments: list[RoutePresentationFragment] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.presentation = presentation
        self.fragments = fragments or []
        self.diagnostics = diagnostics or {}


def _canonical_chain(component: HistoricalRouteComponent) -> tuple[str, ...]:
    return tuple(point.historical_place.canonical_name for point in component.ordered_points)


def _component_evidence_refs(component: HistoricalRouteComponent) -> list[str]:
    refs = set(component.evidence_refs)
    for point in component.ordered_points:
        refs.update(point.evidence_refs)
    return sorted(refs)


def _component_is_eligible(component: HistoricalRouteComponent) -> str | None:
    if len(component.ordered_points) < 2:
        return "INSUFFICIENT_POINTS"
    for point in component.ordered_points:
        if not point.evidence_refs:
            return "MISSING_EVIDENCE_REFS"
        place = point.historical_place
        if not (-90 <= place.latitude <= 90 and -180 <= place.longitude <= 180):
            return "INVALID_COORDINATES"
    return None


def _temporary_component_route(parent: HistoricalRoute, component: HistoricalRouteComponent) -> HistoricalRoute:
    points = component.ordered_points
    return HistoricalRoute(
        id=f"{parent.id}:{component.component_id}",
        event_id=parent.event_id,
        name=parent.name,
        period=parent.period,
        ordered_points=points,
        geometry=GeoJsonLineString(
            coordinates=[(point.historical_place.longitude, point.historical_place.latitude) for point in points],
        ),
        evidence_refs=_component_evidence_refs(component),
        assumptions=list(parent.assumptions),
        limitations=list(parent.limitations),
        historical_confidence=parent.historical_confidence,
        claims=list(parent.claims),
        route_components=[],
        branch_relations=[],
    )


def _tag_fragment_features(response: HistoricalRouteResponse, component_id: str) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    route_feature = dict(response.route_geojson)
    route_props = dict(route_feature.get("properties") or {})
    route_props.update({
        "component_id": component_id,
        "layer_type": "route_fragment",
        "fragment_status": "COMPLETE",
    })
    route_feature["properties"] = route_props
    features.append(route_feature)
    for feature in response.geojson.get("features", [])[1:]:
        copied = dict(feature)
        props = dict(copied.get("properties") or {})
        props["component_id"] = component_id
        copied["properties"] = props
        features.append(copied)
    return features


def _merge_fragment_presentations(
    successes: list[tuple[HistoricalRouteComponent, HistoricalRouteResponse]],
    failures: list[RoutePresentationFragment],
    skipped: list[RoutePresentationFragment],
) -> HistoricalRouteResponse:
    primary = successes[0][1]
    features: list[dict[str, object]] = []
    waypoints = []
    panels = []
    warnings: list[str] = []
    fragment_records: list[RoutePresentationFragment] = []
    for component, response in sorted(successes, key=lambda item: item[0].component_id):
        features.extend(_tag_fragment_features(response, component.component_id))
        waypoints.extend(response.waypoints)
        panels.extend(response.knowledge_panels)
        warnings.extend(response.location_warnings)
        fragment_records.append(RoutePresentationFragment(
            component_id=component.component_id,
            status="COMPLETE",
            evidence_refs=_component_evidence_refs(component),
            waypoints=[item.model_dump(mode="json") for item in response.waypoints],
            route_geojson=dict(response.route_geojson),
        ))
    fragment_records.extend(failures)
    fragment_records.extend(skipped)
    fragment_records.sort(key=lambda item: item.component_id)
    summary = primary.presentation_summary
    if summary is not None and (failures or len(successes) > 1):
        summary = summary.model_copy(update={
            "route_method": "component_fragment_terrain_reconstruction",
            "route_interpretation": (
                "Each fragment is an independent evidence-backed linear component. "
                "Fragment serialization order is not historical chronology."
            ),
            "limitations": [
                *summary.limitations,
                "Fragments are not connected across component boundaries.",
                *(["Some evidence-backed components could not be reconstructed safely."] if failures else []),
            ],
        })
    return HistoricalRouteResponse(
        route=primary.route,
        waypoints=waypoints,
        geojson={"type": "FeatureCollection", "features": features},
        route_geojson=primary.route_geojson,
        explanations=primary.explanations,
        location_warnings=sorted(set(warnings)),
        knowledge_panels=panels,
        presentation_summary=summary,
        fragments=fragment_records,
    )


class ComponentFragmentPresentationAdapter:
    def __init__(self, orchestrator: HistoricalRouteOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or HistoricalRouteOrchestrator()

    def present(
        self,
        intent: HistoricalRouteIntent,
        route: HistoricalRoute,
        evidence: list[Evidence],
        *,
        global_chain: tuple[str, ...] | None = None,
    ) -> ComponentFragmentPresentationResult:
        components = sorted(route.route_components, key=lambda item: item.component_id)
        seen_chains: set[tuple[str, ...]] = set()
        successes: list[tuple[HistoricalRouteComponent, HistoricalRouteResponse]] = []
        failures: list[RoutePresentationFragment] = []
        skipped: list[RoutePresentationFragment] = []
        component_results: list[dict[str, object]] = []
        skipped_duplicate_global = 0

        for component in components:
            chain = _canonical_chain(component)
            if global_chain is not None and chain == global_chain:
                skipped_duplicate_global += 1
                skipped.append(RoutePresentationFragment(
                    component_id=component.component_id,
                    status="SKIPPED",
                    evidence_refs=_component_evidence_refs(component),
                    reason_code="DUPLICATE_GLOBAL_CHAIN",
                ))
                component_results.append({
                    "component_id": component.component_id,
                    "status": "SKIPPED",
                    "reason_code": "DUPLICATE_GLOBAL_CHAIN",
                })
                continue
            if chain in seen_chains:
                skipped.append(RoutePresentationFragment(
                    component_id=component.component_id,
                    status="SKIPPED",
                    evidence_refs=_component_evidence_refs(component),
                    reason_code="DUPLICATE_COMPONENT_CHAIN",
                ))
                component_results.append({
                    "component_id": component.component_id,
                    "status": "SKIPPED",
                    "reason_code": "DUPLICATE_COMPONENT_CHAIN",
                })
                continue
            seen_chains.add(chain)
            ineligible = _component_is_eligible(component)
            if ineligible is not None:
                skipped.append(RoutePresentationFragment(
                    component_id=component.component_id,
                    status="SKIPPED",
                    evidence_refs=_component_evidence_refs(component),
                    reason_code=ineligible,
                ))
                component_results.append({
                    "component_id": component.component_id,
                    "status": "SKIPPED",
                    "reason_code": ineligible,
                })
                continue
            try:
                response = self.orchestrator.present(
                    intent,
                    _temporary_component_route(route, component),
                    evidence,
                )
            except RouteOrchestrationError as exc:
                failures.append(RoutePresentationFragment(
                    component_id=component.component_id,
                    status="FAILED",
                    evidence_refs=_component_evidence_refs(component),
                    reason_code=type(exc).__name__,
                ))
                component_results.append({
                    "component_id": component.component_id,
                    "status": "FAILED",
                    "reason_code": type(exc).__name__,
                })
                continue
            except (ValueError, KeyError) as exc:
                failures.append(RoutePresentationFragment(
                    component_id=component.component_id,
                    status="FAILED",
                    evidence_refs=_component_evidence_refs(component),
                    reason_code=type(exc).__name__,
                ))
                component_results.append({
                    "component_id": component.component_id,
                    "status": "FAILED",
                    "reason_code": type(exc).__name__,
                })
                continue
            successes.append((component, response))
            component_results.append({
                "component_id": component.component_id,
                "status": "COMPLETE",
            })

        presented = len(successes)
        failed = len(failures)
        total = len(components)
        status = "COMPLETE" if presented and not failed else "PARTIAL" if presented else "FAILED"
        diagnostics = {
            "attempted": True,
            "pipeline": "terrain_component_fragments",
            "status": status,
            "components_total": total,
            "components_presented": presented,
            "components_failed": failed,
            "components_skipped": len(skipped),
            "component_results": component_results,
        }
        if global_chain is not None:
            diagnostics["components_skipped_duplicate"] = skipped_duplicate_global
        if not successes:
            return ComponentFragmentPresentationResult(
                presentation=None,
                fragments=[*failures, *skipped],
                diagnostics=diagnostics,
            )
        presentation = _merge_fragment_presentations(successes, failures, skipped)
        return ComponentFragmentPresentationResult(
            presentation=presentation,
            fragments=presentation.fragments,
            diagnostics=diagnostics,
        )
