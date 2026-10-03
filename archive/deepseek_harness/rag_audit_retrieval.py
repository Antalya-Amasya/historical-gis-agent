"""READ-ONLY retrieval diagnostics: semantic-only vs lexical-only vs hybrid,
first-relevant-rank, and whether reranking demotes relevant evidence.
Uses the production retriever against the live Chroma. No writes, no LLM calls."""
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

# ---- build a small ground-truth-by-construction set: fetch chunks matching a regex,
# then ask which of them retrieval surfaces, and at what rank.
ids, docs, metas = [], [], []
off = 0
while True:
    res = post(f"collections/{COL}/get", {"limit": 500, "offset": off, "include": ["documents", "metadatas"]})
    g = res.get("ids") or []
    if not g: break
    ids += g; docs += res.get("documents") or []; metas += res.get("metadatas") or []
    off += len(g)
    if len(g) < 500: break
N = len(docs)
print(f"corpus {N}")

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
  ("I rare proper noun", "Setovia besieged",
   r"Setovia"),
  ("J common-name ambiguity", "the consul marched",
   r"\bconsul\b.{0,200}(?:marched|set out)"),
]

retriever = build_production_retriever(settings)
print("retriever:", type(retriever).__name__)
inner = getattr(retriever, "retriever", None) or getattr(retriever, "_retriever", None)
print("inner:", type(inner).__name__ if inner else None)
store = getattr(inner, "store", None) or getattr(retriever, "store", None)
print("store:", type(store).__name__ if store else None)

# probe the primitives directly
sem_fn = getattr(store, "query", None)
lex_fn = getattr(store, "lexical_candidates", None)
print("store.query:", bool(sem_fn), " store.lexical_candidates:", bool(lex_fn))

def relevant_ids(pattern):
    rx = re.compile(pattern, re.I | re.S)
    return {ids[i] for i in range(N) if rx.search(docs[i] or "")}

def first_rank(order, rel):
    """order items are Evidence passages; match on source_chunk_id metadata."""
    for k, item in enumerate(order, 1):
        iid = getattr(item, "id", "") or ""
        chunk = (getattr(item, "metadata", {}) or {}).get("source_chunk_id") or iid.split(":")[0]
        if chunk in rel or iid in rel:
            return k
    return None

def top_ids(order, n=3):
    out = []
    for item in order[:n]:
        out.append((getattr(item, "metadata", {}) or {}).get("source_chunk_id") or (getattr(item, "id", "") or "").split(":")[0])
    return out

print(f"\n{'case':30} {'#rel':>5} {'sem':>5} {'lex':>5} {'hybrid':>7} {'coverage':>9}")
rows = []
for label, query, pattern in CASES:
    rel = relevant_ids(pattern)
    sem_order = sem_fn(query, 20) if sem_fn else []
    lex_order = lex_fn(query, 20) if lex_fn else []
    try:
        hyb = retriever.retrieve(query, 10)
    except Exception as e:
        hyb = []
        print("hybrid error", e)
    try:
        cov = retriever.retrieve_with_coverage(query)
    except Exception as e:
        cov = []
        print("coverage error", e)
    sr, lr = first_rank(sem_order, rel), first_rank(lex_order, rel)
    hr, cr = first_rank(hyb, rel), first_rank(cov, rel)
    rows.append((label, len(rel), sr, lr, hr, cr, query, pattern))
    print(f"{label:30} {len(rel):>5} {str(sr):>5} {str(lr):>5} {str(hr):>7} {str(cr):>9}")

print("\nJSON")
print(json.dumps([{"case": r[0], "n_rel": r[1], "sem": r[2], "lex": r[3], "hybrid": r[4], "coverage": r[5]} for r in rows]))
