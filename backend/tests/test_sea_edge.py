from backend.app.gis.sea import SeaEdgeStatus, build_sea_edge
from backend.app.gis.surface import MockSurfaceClassifier, SurfaceClassification, SurfaceType
from backend.app.gis.transport import SearchState, TransportMode
def s(x, mode=TransportMode.SEA): return SearchState(str(x),0,x,mode)
def test_sea_edge_requires_sea_and_all_water_samples():
 c=MockSurfaceClassifier(SurfaceClassification(SurfaceType.WATER,"fixture"))
 for x in (0,.01,.02): c.register(0,x,SurfaceType.WATER)
 assert build_sea_edge(s(0),s(.02),c,max_sampling_interval_m=1200).status is SeaEdgeStatus.AVAILABLE
 c.register(0,.01,SurfaceType.LAND)
 assert build_sea_edge(s(0),s(.02),c,max_sampling_interval_m=1200).status is SeaEdgeStatus.UNAVAILABLE
 assert build_sea_edge(s(0,TransportMode.LAND),s(.02),c).status is SeaEdgeStatus.UNAVAILABLE

def test_geometry_edges_cost_status_and_fail_closed_inputs():
 c=MockSurfaceClassifier(SurfaceClassification(SurfaceType.WATER,"fixture"))
 for a,b in ((s(0),s(.02)),(SearchState('n',0,0,TransportMode.SEA),SearchState('n2',.02,0,TransportMode.SEA)),(SearchState('d',1,1,TransportMode.SEA),SearchState('d2',1.02,1.02,TransportMode.SEA)),(SearchState('am',10,179,TransportMode.SEA),SearchState('am2',10,-179,TransportMode.SEA))):
  e=build_sea_edge(a,b,c,max_sampling_interval_m=5000); assert e.status is SeaEdgeStatus.AVAILABLE and e.physical_distance_km>0 and e.cost_breakdown.total_cost==e.physical_distance_m
 assert build_sea_edge(s(0),s(0),c).status is SeaEdgeStatus.AVAILABLE
 assert build_sea_edge(SearchState('x',0,0,TransportMode.SEA),SearchState('y',0,180,TransportMode.SEA),c).status is SeaEdgeStatus.UNAVAILABLE
 import pytest
 with pytest.raises(ValueError): build_sea_edge(s(0),s(.02),c,max_sampling_interval_m=0)
 with pytest.raises(ValueError): build_sea_edge(s(0),s(.02),c,max_sampling_interval_m=-1)
