"""READ-ONLY corpus semantic-coherence audit: does a chunk carry WHO/WHAT/FROM/TO/WHEN
locally, or are movement components split across chunk boundaries?"""
import json
import re
import random
import urllib.request
from collections import Counter

BASE = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database"
COL = "06c01e2d-cacc-4594-9c7f-4596b0339c09"


def post(path, payload):
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


ids, docs, metas = [], [], []
off = 0
while True:
    res = post(f"collections/{COL}/get", {"limit": 500, "offset": off,
                                          "include": ["documents", "metadatas"]})
    got = res.get("ids") or []
    if not got:
        break
    ids += got; docs += res.get("documents") or []; metas += res.get("metadatas") or []
    off += len(got)
    if len(got) < 500:
        break

MOVEMENT = re.compile(r"\b(?:marched|marching|sailed|sailing|advanced|crossed|crossing|departed|"
                      r"set\s+out|set\s+sail|entered|entering|arrived|reached|proceeded|moved|"
                      r"retreated|withdrew|invaded|besieged)\b", re.I)
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
PRONOUN = re.compile(r"\b(?:he|she|they|him|her|them|his|their|it)\b", re.I)
FROMTO = re.compile(r"\b(?:from|to|toward|towards|into|out\s+of|across)\b", re.I)
LOCATIVE = re.compile(r"\b(?:from|to|at|into|toward|towards|near|through|across)\s+(?:the\s+)?"
                      r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})")
YEAR = re.compile(r"\b(?:[0-9]{3,4}\s*(?:BC|BCE|AD|CE)|consulship|year\s+of)\b", re.I)
# narrative sentence openers that suggest the sentence continues a prior subject
CONTINUATION_OPENER = re.compile(
    r"^\s*(?:he|she|they|him|her|them|his|her|their|it|thereupon|thence|then|next|"
    r"after\s+this|accordingly|whereupon|meanwhile|but\s+he|and\s+he)\b", re.I)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

random.seed(20261002)
N = len(docs)


def sentences(d):
    return [s.strip() for s in SENT_SPLIT.split(d or "") if s.strip()]


def stats_for(name, idxs):
    n = len(idxs)
    if not n:
        print(f"\n[{name}] no records"); return None
    has_mv = has_proper = has_pronoun = has_fromto = has_year = 0
    mv_top = proper_top = 0            # movement in FIRST sentence only
    cont_opener = 0
    loc_count = Counter()
    sent_counts = []
    for i in idxs:
        d = docs[i]
        ss = sentences(d)
        sent_counts.append(len(ss))
        mv = MOVEMENT.search(d)
        pr = PROPER.findall(d)
        if mv: has_mv += 1
        if pr: has_proper += 1
        if PRONOUN.search(d): has_pronoun += 1
        if FROMTO.search(d): has_fromto += 1
        if YEAR.search(d): has_year += 1
        if ss:
            if MOVEMENT.search(ss[0]): mv_top += 1
            if PROPER.search(ss[0]): proper_top += 1
            if CONTINUATION_OPENER.search(ss[0]): cont_opener += 1
        for m in LOCATIVE.finditer(d):
            loc_count[m.group(1).split()[0]] += 1
    print(f"\n[{name}]  n={n}  median_sentences={sorted(sent_counts)[n//2]}")
    print(f"   movement verb present      {has_mv:>5}  {100.0*has_mv/n:5.1f}%")
    print(f"   movement in 1st sentence   {mv_top:>5}  {100.0*mv_top/n:5.1f}%   <- self-contained opening")
    print(f"   proper noun present        {has_proper:>5}  {100.0*has_proper/n:5.1f}%")
    print(f"   proper noun in 1st sentence{proper_top:>5}  {100.0*proper_top/n:5.1f}%")
    print(f"   pronoun present            {has_pronoun:>5}  {100.0*has_pronoun/n:5.1f}%")
    print(f"   continuation-style opener  {cont_opener:>5}  {100.0*cont_opener/n:5.1f}%   <- likely depends on PRIOR chunk")
    print(f"   from/to/into present       {has_fromto:>5}  {100.0*has_fromto/n:5.1f}%")
    print(f"   explicit year/BC marker    {has_year:>5}  {100.0*has_year/n:5.1f}%")
    print(f"   distinct place-ish tokens  {len(loc_count):>5}   top: {loc_count.most_common(6)}")
    return dict(n=n, has_mv=has_mv, mv_top=mv_top, cont=cont_opener, proper=has_proper)


print(f"corpus size {N}")
print(f"total sentences {sum(len(sentences(d)) for d in docs)}")

# ---- groups ----
# movement-heavy: many movement verbs
mv_counts = [(len(MOVEMENT.findall(docs[i] or "")), i) for i in range(N)]
mv_counts.sort(reverse=True)
mv_heavy = [i for _, i in mv_counts[:200]]

# multi-person
person_counts = [(len(set(PROPER.findall(docs[i] or ""))), i) for i in range(N)]
person_counts.sort(reverse=True)
multi_person = [i for _, i in person_counts[:200]]

# negative
NEG = re.compile(r"\b(?:prevented|blocked|abandoned|aborted|planned\s+to|intended\s+to|attempted\s+to|"
                 r"did\s+not|never|failed\s+to|in\s+vain|unsuccess\w*|without\s+success)\b", re.I)
neg_idx = [i for i in range(N) if NEG.search(docs[i] or "")]

# adjacency risk: continuation opener AND no proper noun in first sentence
adj_risk = []
for i in range(N):
    ss = sentences(docs[i])
    if ss and CONTINUATION_OPENER.search(ss[0]) and not PROPER.search(ss[0]):
        adj_risk.append(i)
print(f"\nrecords whose FIRST sentence is a continuation opener with NO proper noun: {len(adj_risk)}  ({100.0*len(adj_risk)/N:.1f}%)")

rnd = random.sample(range(N), 200)
stats_for("RANDOM 200", rnd)
stats_for("MOVEMENT-HEAVY 200", mv_heavy)
stats_for("MULTI-PERSON 200", multi_person)
stats_for("NEGATIVE 200 (sampled from %d)" % len(neg_idx), random.sample(neg_idx, min(200, len(neg_idx))))
stats_for("ADJACENCY-RISK 200", adj_risk[:200])

# ---- cross-boundary continuity: does a chunk START mid-sentence? ----
print("\n=== BOUNDARY INTEGRITY ===")
starts_lower = 0
starts_midword = 0
for d in docs:
    s = (d or "").lstrip()
    if not s:
        continue
    if s[0].islower():
        starts_lower += 1
    if s[:1] and not re.match(r"[A-Z0-9\"“'(\[]", s):
        starts_midword += 1
print(f"  chunks starting with a lowercase char: {starts_lower} ({100.0*starts_lower/N:.1f}%)  <- suggests mid-sentence cut")
print(f"  chunks not starting with cap/digit/quote: {starts_midword} ({100.0*starts_midword/N:.1f}%)")
ends_no_term = 0
for d in docs:
    s = (d or "").rstrip()
    if s and s[-1] not in '.!?"”\')]':
        ends_no_term += 1
print(f"  chunks NOT ending in sentence punctuation: {ends_no_term} ({100.0*ends_no_term/N:.1f}%)  <- suggests truncated tail")

# ---- longest / shortest samples ----
order = sorted(range(N), key=lambda i: len(docs[i] or ""))
print("\n=== SHORTEST 3 ===")
for i in order[:3]:
    print(f"  len={len(docs[i])} id={ids[i]} doc_id={metas[i].get('document_id')} "
          f"spine={metas[i].get('spine_index')} off={metas[i].get('start_offset')}-{metas[i].get('end_offset')}")
    print(f"     {docs[i][:220]!r}")
print("\n=== LONGEST 2 ===")
for i in order[-2:]:
    print(f"  len={len(docs[i])} id={ids[i]} doc_id={metas[i].get('document_id')} spine={metas[i].get('spine_index')}")
    print(f"     head {docs[i][:160]!r}")
    print(f"     tail {docs[i][-160:]!r}")

print("\n=== SAMPLE NEGATIVE-MOVEMENT CHUNK ===")
if neg_idx:
    i = neg_idx[0]
    print(f"  id={ids[i]} doc_id={metas[i].get('document_id')} spine={metas[i].get('spine_index')}")
    print(f"  {docs[i][:700]!r}")

print("\nDONE")
