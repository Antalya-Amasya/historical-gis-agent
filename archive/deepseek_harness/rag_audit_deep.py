"""READ-ONLY: quantify actor/antecedent splitting, boilerplate contamination,
metadata-less entity coverage, and duplicate windows."""
import json, re, random, urllib.request
from collections import Counter

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
    res = post(f"collections/{COL}/get", {"limit": 500, "offset": off, "include": ["documents", "metadatas"]})
    g = res.get("ids") or []
    if not g: break
    ids += g; docs += res.get("documents") or []; metas += res.get("metadatas") or []
    off += len(g)
    if len(g) < 500: break
N = len(docs)
SENT = re.compile(r"(?<=[.!?])\s+")
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
PRON = re.compile(r"\b(?:he|she|him|her|his|hers|they|them|their)\b", re.I)
def sents(d): return [s.strip() for s in SENT.split(d or "") if s.strip()]

print("=== ACTOR/ANTECEDENT SPLITTING (the chunk-boundary risk) ===")
first_sent_cont = first_sent_pron = first_proper_pos = 0
proper_offset = []
gap_pron_to_name = []
for d in docs:
    ss = sents(d)
    if not ss: continue
    f = ss[0]
    # does the first sentence carry a resolvable proper name?
    if PROPER.search(f): first_proper_pos += 1
    else: first_sent_cont += 1
    if PRON.search(f): first_sent_pron += 1
    m = PROPER.search(d)
    if m: proper_offset.append(m.start())
    # distance from first pronoun to the first proper noun AFTER it
    p = PRON.search(d)
    if p:
        m2 = PROPER.search(d, p.end())
        if m2: gap_pron_to_name.append(m2.start() - p.start())
print(f"  chunks whose FIRST sentence has NO proper name : {first_sent_cont:>5} {100.0*first_sent_cont/N:5.1f}%")
print(f"  chunks whose FIRST sentence contains a pronoun : {first_sent_pron:>5} {100.0*first_sent_pron/N:5.1f}%")
if proper_offset:
    proper_offset.sort()
    print(f"  chars before 1st proper name: P50={proper_offset[len(proper_offset)//2]} P90={proper_offset[int(len(proper_offset)*0.9)]}")
if gap_pron_to_name:
    gap_pron_to_name.sort()
    print(f"  chars from a pronoun to the NEXT proper name: P50={gap_pron_to_name[len(gap_pron_to_name)//2]} "
          f"P90={gap_pron_to_name[int(len(gap_pron_to_name)*0.9)]} max={gap_pron_to_name[-1]}")

print("\n=== BOILERPLATE / NON-NARRATIVE CONTAMINATION ===")
BOILER = re.compile(r"project\s+gutenberg|gutenberg-tm|plain\s+vanilla|royalty\s+fee|"
                    r"copyright|trademark|licen[cs]e|disclaimer|donations?", re.I)
BOILER_STRONG = re.compile(r"project\s+gutenberg|gutenberg-tm|plain\s+vanilla|royalty\s+fee", re.I)
FRONTMATTER = re.compile(r"^\s*(?:table\s+of\s+contents|contents|preface|introduction|index|"
                         r"bibliography|notes?\s+on\s+the\s+text|translator)", re.I)
b_any = b_strong = b_front = 0
b_heavy = 0
for d in docs:
    t = d or ""
    hits = len(BOILER.findall(t))
    if hits: b_any += 1
    if BOILER_STRONG.search(t): b_strong += 1
    if FRONTMATTER.search(t): b_front += 1
    # heavy = boilerplate density high
    if hits >= 5: b_heavy += 1
print(f"  any boilerplate/licence token      : {b_any:>5} {100.0*b_any/N:5.1f}%")
print(f"  Project-Gutenberg licence text     : {b_strong:>5} {100.0*b_strong/N:5.1f}%   <- pure non-corpus noise")
print(f"  front-matter opener (TOC/preface)  : {b_front:>5} {100.0*b_front/N:5.1f}%")
print(f"  >=5 boilerplate tokens (heavy)     : {b_heavy:>5} {100.0*b_heavy/N:5.1f}%")

print("\n=== TINY / DEGENERATE CHUNKS ===")
for thresh in (10, 50, 100, 200):
    c = sum(1 for d in docs if len(d or "") < thresh)
    print(f"  < {thresh:>4} chars : {c:>5} {100.0*c/N:5.1f}%")
print("  samples:")
for i in sorted(range(N), key=lambda i: len(docs[i] or ""))[:12]:
    print(f"    len={len(docs[i]):>4} doc={metas[i].get('document_id')} spine={metas[i].get('spine_index')} "
          f"off={metas[i].get('start_offset')}-{metas[i].get('end_offset')} {docs[i][:40]!r}")

print("\n=== STRUCTURED ENTITY METADATA IN THE INDEX? ===")
allkeys = set()
for m in metas:
    if isinstance(m, dict): allkeys |= set(m)
entity_like = [k for k in sorted(allkeys) if re.search(
    r"person|place|event|campaign|episode|date|year|time|polarity|negat|movement|"
    r"actor|origin|destination|entity|ner|gazett|pleiad", k, re.I)]
print(f"  metadata keys matching entity/event/date/polarity patterns: {entity_like or 'NONE'}")
print(f"  ALL metadata keys: {sorted(allkeys)}")

print("\n=== PLACE-NAME STRING VARIANT COLLISION (alias need) ===")
variants = {
    "Rome/Roma": (r"\bRome\b", r"\bRoma\b"),
    "Carthage/Carthago": (r"\bCarthage\b", r"\bCarthago\b"),
    "Oricum/Orikon": (r"\bOricum\b", r"\bOrikon\b"),
    "Brundisium/Brindisi": (r"\bBrundisium\b", r"\bBrindisi\b"),
    "Massilia/Massalia": (r"\bMassilia\b", r"\bMassalia\b"),
}
for label, (a, b) in variants.items():
    ca = sum(1 for d in docs if re.search(a, d or "", re.I))
    cb = sum(1 for d in docs if re.search(b, d or "", re.I))
    print(f"  {label:22} first={ca:>5}  second={cb:>5}")

print("\n=== PERSON NAME SURFACE-VARIANT COLLISION ===")
for label, (a, b) in {
    "Caesar / Caius Iulius": (r"\bCaesar\b", r"\b(?:Caius|Gaius)\s+Iulius\b"),
    "Pompey / Pompeius": (r"\bPompey\b", r"\bPompeius\b"),
    "Hannibal / Annibal": (r"\bHannibal\b", r"\bAnnibal\b"),
    "Mark Antony / Antonius": (r"\b(?:Mark\s+)?Antony\b", r"\bAntonius\b"),
}.items():
    ca = sum(1 for d in docs if re.search(a, d or "", re.I))
    cb = sum(1 for d in docs if re.search(b, d or "", re.I))
    print(f"  {label:26} first={ca:>5}  second={cb:>5}")

print("\n=== DUPLICATE WINDOW / ID SPACE ===")
by_doc = Counter(m.get("document_id") for m in metas)
print(f"  distinct document_id: {len(by_doc)}")
for k, v in by_doc.most_common():
    print(f"    {v:>5}  {k}")
# same document_id AND same start_offset
pairs = Counter((m.get("document_id"), m.get("start_offset")) for m in metas)
dupe_spans = {k: v for k, v in pairs.items() if v > 1}
print(f"  (document_id,start_offset) collisions: {len(dupe_spans)}")
print("\nDONE")
