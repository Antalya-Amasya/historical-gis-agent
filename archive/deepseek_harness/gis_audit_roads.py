"""READ-ONLY: load the Itiner-e road graph once and measure connectivity statistics.
Writes nothing; does not touch Chroma."""
import sys, statistics
from collections import Counter, defaultdict
sys.path.insert(0, r"C:\D\python\historical-gis-cursor")
from backend.app.roads.itiner_e import RomanRoadGraph

PATH = r"C:\D\python\historical-gis-cursor\data\raw\itiner_e\itinere_roads_zenodo_17122148.geojson"
g = RomanRoadGraph.load(PATH)
s = g.stats
print("=== RomanRoadGraphStats ===")
print(f"  node_count              {s.node_count}")
print(f"  edge_count              {s.edge_count}")
print(f"  component_count         {s.component_count}")
print(f"  largest_component_nodes {s.largest_component_nodes}")
print(f"  snap tolerance (m)      {s.tolerance_m}")
print(f"  largest component share {100.0*s.largest_component_nodes/max(1,s.node_count):.1f}% of nodes")

# degree from adjacency (edge incidence per node)
deg = sorted(len(v) for v in g.adjacency.values())
print("\n=== DEGREE (incident edges per node) ===")
print(f"  n={len(deg)} min={deg[0]} median={statistics.median(deg)} mean={statistics.mean(deg):.2f} max={deg[-1]}")
print(f"  isolated (degree 0): {sum(1 for d in deg if d == 0)}")
print(f"  degree==1 (leaf): {sum(1 for d in deg if d == 1)} ({100.0*sum(1 for d in deg if d==1)/len(deg):.1f}%)")

# components
comp_sizes = Counter(g.component_ids.values())
print("\n=== COMPONENTS ===")
print(f"  distinct component ids: {len(comp_sizes)}")
for cid, n in comp_sizes.most_common(8):
    print(f"    {n:>6} nodes  {cid[:40]}")
print(f"  singleton components: {sum(1 for n in comp_sizes.values() if n == 1)}")

# edge lengths
L = sorted(e.length_m for e in g.edges.values())
def pct(v, p): return v[min(len(v)-1, round((len(v)-1)*p))] if v else 0
print("\n=== EDGE LENGTH (m) ===")
print(f"  n={len(L)} total_km={sum(L)/1000:.1f}")
print(f"  min={L[0]:.1f} P25={pct(L,.25):.1f} median={statistics.median(L):.1f} P75={pct(L,.75):.1f} "
      f"P90={pct(L,.9):.1f} P99={pct(L,.99):.1f} max={L[-1]:.1f}")
print(f"  edges < 100 m: {sum(1 for x in L if x < 100)} ({100.0*sum(1 for x in L if x<100)/len(L):.1f}%)")
print(f"  edges > 20 km: {sum(1 for x in L if x > 20000)} ({100.0*sum(1 for x in L if x>20000)/len(L):.1f}%)")

# duplicates / self-loops / invalid geometry
pairs = Counter(tuple(sorted((e.source_node_id, e.target_node_id))) for e in g.edges.values())
parallel = sum(c - 1 for c in pairs.values() if c > 1)
selfloops = sum(1 for e in g.edges.values() if e.source_node_id == e.target_node_id)
zero_geom = sum(1 for e in g.edges.values() if len(e.segment.geometry) < 2)
zero_len = sum(1 for e in g.edges.values() if e.length_m <= 0)
print("\n=== INTEGRITY ===")
print(f"  parallel/duplicate node pairs: {parallel}")
print(f"  self-loops:                    {selfloops}")
print(f"  edges with <2 geometry points: {zero_geom}")
print(f"  zero-length edges:             {zero_len}")
print(f"  distinct node pairs:           {len(pairs)}")

# road type / chronology distribution if present
try:
    types = Counter(getattr(e.segment, "road_type", None) for e in g.edges.values())
    print("\n=== ROAD TYPE (top 10) ===")
    for k, v in types.most_common(10):
        print(f"    {v:>6}  {k}")
except Exception as e:
    print("road_type unavailable:", e)
print("\nDONE")
