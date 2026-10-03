"""READ-ONLY retrieval diagnostics v2: semantic-only vs lexical-only vs hybrid vs coverage,
with first-relevant-rank. Fixes the earlier shape bug (store.query returns a dict)."""
import json, re, sys, urllib.request

sys.path.insert(0, r"C:\D\python\historical-gis-cursor")
from backend.app.rag.http_store import build_production_retriever
from backend.app.core.config import settings

BASE = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database"
COL = "06c01e2d-cacc-4594-9c7f-4596b0339c09"
def post(p, pl):
    r = urllib.request.Request(f"{BASE}/{p}", data=json.dumps(pl).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=180) as x:
        return json.loads(x.read().decode())

ids, docs, metas = [], [], []
off = 0
while True:
    res = post(f"{COL and f'collections/{COL}/get'}", {"limit": 500, "offset": off,
              "include": ["documents", "metadatas"]})
    g = res.get("ids") or []
    if not g: break
    ids += g; docs += res.get("documents") or []; metas += res.get("metadatas") or []
    off += len(g)
    if len(g) < 500: break
N = len(docs)
print(f"corpus {N}")

retriever = build_production_retriever(settings)
store = retriever.store

def _flat(v):
    """Chroma query returns one result list per query embedding -> unwrap outer level."""
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
        return v[0]
    return v or []

def sem_items(q, k):
    """semantic channel returns a dict of parallel arrays; return list of (chunk_id, source_chunk_id, distance)."""
    r = store.query(q, k)
    if not isinstance(r, dict):
        return []
    cids = _flat(r.get("ids"))
    mds = _flat(r.get("metadatas"))
    ds = _flat(r.get("distances"))
    out = []
    for i, cid in enumerate(cids):
        md = mds[i] if i < len(mds) and isinstance(mds[i], dict) else {}
        d = ds[i] if i < len(ds) else None
        out.append((cid, md.get("source_chunk_id") or cid, d))
    return out

def chunk_of(ev):
    md = getattr(ev, "metadata", {}) or {}
    return md.get("source_chunk_id") or (getattr(ev, "id", "") or "").split(":")[0]

CASES = [
  ("A exact person + route", "Caesar marched from Rome into Gaul",
   r"Caesar.{0,400}?(?:marched|set out|led).{0,400}?Gaul"),
  ("B alias person", "Pompeius campaign in the east",
   r"Pompey.{0,300}(?:east|Asia|Cilicia|Mithridates)"),
  ("C unseen/rare person", "Lucullus marched against Tigranes",
   r"Lucullus.{0,300}(?:Tigranes|Armenia|marched)"),
  ("D place alias Oricum", "Oricum to Brundisium crossing",
   r"(?:Oricum|Brundisium)"),
  ("E multi-episode person", "Hannibal in Italy after Cannae",
   r"Hannibal.{0,300}(?:Italy|Cannae)"),
  ("F negative movement", "entry into the city was prevented",
   r"(?:prevented|blocked).{0,200}(?:entering|entrance|entry)|(?:entering|entry).{0,200}(?:prevented|blocked)"),
  ("G wrong actor distractor", "Antony marched to Mutina",
   r"Antony.{0,300}(?:Mutina|marched)"),
  ("H same place wrong campaign", "Scipio in Africa",
   r"Scipio.{0,300}Africa"),
  ("I rare proper noun", "Setovia besieged", r"Setovia"),
  ("J common-name ambiguity", "the consul marched",
   r"\bconsul\b.{0,200}(?:marched|set out)"),
]

def relevant(pattern):
    rx = re.compile(pattern, re.I | re.S)
    return {ids[i] for i in range(N) if rx.search(docs[i] or "")}

def fr_chunks(pairs, rel):
    for k, (cid, chunk, _d) in enumerate(pairs, 1):
        if chunk in rel or cid in rel:
            return k
    return None

def fr_evs(evs, rel):
    for k, ev in enumerate(evs, 1):
        if chunk_of(ev) in rel:
            return k
    return None

print(f"\n{'case':30} {'#rel':>5} {'sem':>5} {'lex':>5} {'hyb':>5} {'cov':>5}   top-3 semantic chunk prefixes")
rows = []
for label, query, pattern in CASES:
    rel = relevant(pattern)
    s = sem_items(query, 20)
    try:
        lex = store.lexical_candidates(query, 20)
    except Exception as e:
        lex = []; print("lex err", e)
    hyb = retriever.retrieve(query, 10)
    cov = retriever.retrieve_with_coverage(query)
    sr = fr_chunks(s, rel)
    lr = fr_evs(lex, rel)
    hr = fr_evs(hyb, rel)
    cr = fr_evs(cov, rel)
    tops = [c[:8] for _i, c, _d in s[:3]]
    rows.append((label, len(rel), sr, lr, hr, cr))
    print(f"{label:30} {len(rel):>5} {str(sr):>5} {str(lr):>5} {str(hr):>5} {str(cr):>5}   {tops}")

print("\n=== bridge A/B (enabled vs disabled) on a CJK query ===")
from backend.app.rag.query_bridge import HistoricalQueryBridge
q = "\u51ef\u6492\u7684\u884c\u519b\u8def\u7ebf"   # 凯撒的行军路线
on = HistoricalQueryBridge(True); off = HistoricalQueryBridge(False)
print("bridge ON :", on.transform(q).retrieval_query)
print("bridge OFF:", off.transform(q).retrieval_query)
for label, bridge in (("ON", on), ("OFF", off)):
    rq = bridge.transform(q).retrieval_query
    lex = store.lexical_candidates(rq, 20)
    s = sem_items(rq, 20)
    print(f"  {label}: lexical candidates={len(lex)}  semantic chunks={len(s)}")

print("\n=== case F detailed: negative-movement retrieval ===")
for query in ["prevented from entering the city", "did not reach the city",
              "failed to take the town", "Marcus attempted to enter but was prevented"]:
    s = sem_items(query, 20)
    lex = store.lexical_candidates(query, 20)
    print(f"  q={query!r}: sem={len(s)} lex={len(lex)}")

print("\nJSON", json.dumps([{"case": r[0], "n_rel": r[1], "sem": r[2], "lex": r[3],
                             "hybrid": r[4], "coverage": r[5]} for r in rows]))
