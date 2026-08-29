from backend.app.models import Evidence, EventPlaceResolutionStatus, EventPlaceRole, HistoricalEvent, HistoricalEventPlaceBinding, HistoricalEventPlaceMention, HistoricalEventType, HistoricalPlace
from backend.app.routes.event_anchors import project_event_anchors

def ev(): return Evidence(id="e",author="a",work="w",locator="l",excerpt="x")
def binding(role):
 p=HistoricalPlace(id="p",canonical_name="Place",latitude=1,longitude=2,source="Pleiades",confidence=.8,coordinate_role="representative_point")
 m=HistoricalEventPlaceMention(raw_text="Place",role=role,evidence_refs=["e"])
 return HistoricalEventPlaceBinding(mention=m,place=p,role=role,resolution_status=EventPlaceResolutionStatus.RESOLVED,evidence_refs=["e"],resolver_provenance="resolver",limitations=["representative"])
def event(bindings,refs=["e"]): return HistoricalEvent(id="x",name="x",summary="x",event_type=HistoricalEventType.MOVEMENT,evidence_refs=refs,place_bindings=bindings)
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
