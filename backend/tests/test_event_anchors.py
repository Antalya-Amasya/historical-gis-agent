from backend.app.models import Evidence, EventGroundingStatus, EventPlaceResolutionStatus, EventPlaceRole, HistoricalEvent, HistoricalEventPlaceBinding, HistoricalEventPlaceMention, HistoricalEventType, HistoricalPlace, PlaceSpatialSemantics, TemporalGroundingStatus
from backend.app.routes.event_anchors import project_event_anchors

def ev(): return Evidence(id="e",author="a",work="w",locator="l",excerpt="x")
def binding(role):
 p=HistoricalPlace(id="p",canonical_name="Place",latitude=1,longitude=2,source="Pleiades",confidence=.8,coordinate_role="exact_site",spatial_semantics=PlaceSpatialSemantics.SETTLEMENT)
 m=HistoricalEventPlaceMention(raw_text="Place",role=role,evidence_refs=["e"])
 return HistoricalEventPlaceBinding(mention=m,place=p,role=role,resolution_status=EventPlaceResolutionStatus.RESOLVED,evidence_refs=["e"],resolver_provenance="resolver",limitations=[])
def event(bindings,refs=["e"], **kwargs): return HistoricalEvent(id="x",name="x",summary="x",event_type=HistoricalEventType.MOVEMENT,evidence_refs=refs,place_bindings=bindings, source_statements=["The army campaigned at Place."], **kwargs)
def test_movement_projects_origin_destination_with_event_provenance():
 a,d=project_event_anchors([event([binding(EventPlaceRole.ORIGIN),binding(EventPlaceRole.DESTINATION)])],[ev()])
 assert [x.role for x in a]==[EventPlaceRole.ORIGIN,EventPlaceRole.DESTINATION] and {x.event_id for x in a}=={"x"} and d==[]
def test_site_projects_but_related_and_invalid_evidence_do_not():
 a,_=project_event_anchors([event([binding(EventPlaceRole.EVENT_SITE),binding(EventPlaceRole.RELATED_PLACE)])],[ev()]); assert len(a)==1 and a[0].role is EventPlaceRole.EVENT_SITE
 a,d=project_event_anchors([event([binding(EventPlaceRole.ORIGIN)],[])],[ev()]); assert not a and d==["INVALID_EVIDENCE_PROVENANCE:x"]

def test_unresolved_ambiguous_and_missing_coordinates_fail_closed():
 b=binding(EventPlaceRole.ORIGIN); b.place=None; b.resolution_status=EventPlaceResolutionStatus.UNRESOLVED
 a,d=project_event_anchors([event([b])],[ev()]); assert not a and d==["UNRESOLVED_PLACE:x"]
 b=binding(EventPlaceRole.ORIGIN); b.resolution_status=EventPlaceResolutionStatus.AMBIGUOUS
 a,d=project_event_anchors([event([b])],[ev()]); assert not a and d==["AMBIGUOUS_PLACE:x"]
 b=binding(EventPlaceRole.ORIGIN); b.place.latitude=None
 a,d=project_event_anchors([event([b])],[ev()]); assert not a and d==["MISSING_COORDINATE:x"]


def test_related_place_becomes_contextual_only_for_evidence_grounded_route_intent():
    related = binding(EventPlaceRole.RELATED_PLACE)
    anchors, _ = project_event_anchors([event([related])], [ev()])
    assert anchors == []
    anchors, diagnostics = project_event_anchors([event([related])], [ev()], allow_contextual_related_places=True)
    assert len(anchors) == 1 and anchors[0].admission_type == "CONTEXTUAL_WAYPOINT" and diagnostics == []


def test_contextual_related_place_keeps_geography_evidence_event_and_temporal_guards():
    related = binding(EventPlaceRole.RELATED_PLACE)
    related.place = None
    related.resolution_status = EventPlaceResolutionStatus.UNRESOLVED
    anchors, diagnostics = project_event_anchors([event([related])], [ev()], allow_contextual_related_places=True)
    assert anchors == [] and diagnostics == ["UNRESOLVED_PLACE:x"]

    missing_refs = binding(EventPlaceRole.RELATED_PLACE)
    missing_refs.evidence_refs = []
    anchors, diagnostics = project_event_anchors([event([missing_refs])], [ev()], allow_contextual_related_places=True)
    assert anchors == [] and diagnostics == ["CONTEXTUAL_EVIDENCE_PROVENANCE_INVALID:x"]

    weak_event = event([binding(EventPlaceRole.RELATED_PLACE)], grounding_status=EventGroundingStatus.INSUFFICIENT_GROUNDING)
    anchors, diagnostics = project_event_anchors([weak_event], [ev()], allow_contextual_related_places=True)
    assert anchors == [] and diagnostics == ["CONTEXTUAL_EVENT_ASSOCIATION_INSUFFICIENT:x"]

    conflict = event([binding(EventPlaceRole.RELATED_PLACE)])
    conflict.temporal_grounding.status = TemporalGroundingStatus.CONFLICT
    anchors, diagnostics = project_event_anchors([conflict], [ev()], allow_contextual_related_places=True)
    assert anchors == [] and diagnostics == ["CONTEXTUAL_TEMPORAL_CONFLICT:x"]
