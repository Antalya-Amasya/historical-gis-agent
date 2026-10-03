"""READ-ONLY corpus audit. Fetches all records from the live Chroma collection
and computes schema coverage, chunk statistics, duplication and language stats.
Writes nothing to the repo and nothing to Chroma."""
import json
import sys
import urllib.request
from collections import Counter, defaultdict

BASE = "http://127.0.0.1:8002/api/v2/tenants/default_tenant/databases/default_database"
COL = "06c01e2d-cacc-4594-9c7f-4596b0339c09"


def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}/{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


ids, docs, metas = [], [], []
offset = 0
BATCH = 500
while True:
    res = post(f"collections/{COL}/get", {
        "limit": BATCH, "offset": offset,
        "include": ["documents", "metadatas"],
    })
    got = res.get("ids") or []
    if not got:
        break
    ids.extend(got)
    docs.extend(res.get("documents") or [])
    metas.extend(res.get("metadatas") or [])
    offset += len(got)
    if len(got) < BATCH:
        break

print(f"TOTAL_RECORDS_FETCHED {len(ids)}")
print(f"DISTINCT_IDS {len(set(ids))}")

# ---------------- metadata schema ----------------
keys = Counter()
for m in metas:
    if isinstance(m, dict):
        for k in m:
            keys[k] += 1

print("\n=== METADATA FIELD COVERAGE ===")
print(f"{'field':34} {'populated':>9} {'pct':>7} {'uniq':>7}  samples")
n = len(metas)
rows = []
for k, cnt in keys.most_common():
    vals = [m.get(k) for m in metas if isinstance(m, dict) and k in m]
    types = Counter(type(v).__name__ for v in vals)
    uniq = len(set(map(str, vals)))
    sample = [str(v)[:40] for v in vals[:3]]
    rows.append((k, cnt, 100.0 * cnt / n, uniq, dict(types), sample))
    print(f"{k:34} {cnt:>9} {100.0*cnt/n:>6.1f}% {uniq:>7}  {sample}")

# ---------------- id format ----------------
print("\n=== ID FORMAT ===")
pref = Counter()
for i in ids:
    parts = i.split(":")
    pref[parts[0][:38] if len(parts) > 1 else "(no colon)"] += 1
print("distinct id prefixes:", len(pref))
for p, c in pref.most_common(12):
    print(f"   {c:>6}  {p}")
lens = [len(i) for i in ids]
lens.sort()
print(f"id length min/median/max: {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")

# ---------------- chunk sizes ----------------
def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

char_lens = sorted(len(d or "") for d in docs)
word_lens = sorted(len((d or "").split()) for d in docs)
print("\n=== CHUNK LENGTH (characters) ===")
for p in (0, 10, 25, 50, 75, 90, 95, 100):
    print(f"  P{p:<3} {pct(char_lens, p):>10.0f}")
print(f"  mean {sum(char_lens)/len(char_lens):>10.1f}")
print("=== CHUNK LENGTH (whitespace words) ===")
for p in (0, 10, 25, 50, 75, 90, 95, 100):
    print(f"  P{p:<3} {pct(word_lens, p):>10.0f}")
print(f"  mean {sum(word_lens)/len(word_lens):>10.1f}")
print(f"total chars {sum(char_lens)}  total words {sum(word_lens)}")

# ---------------- duplication ----------------
print("\n=== DUPLICATION ===")
exact = Counter(d for d in docs)
dups = {k: v for k, v in exact.items() if v and v > 1}
print(f"distinct documents: {len(exact)}  (records {len(ids)})")
print(f"exact-duplicate document groups: {len(dups)}  extra records: {sum(v-1 for v in dups.values())}")
for k, v in sorted(dups.items(), key=lambda x: -x[1])[:5]:
    print(f"   x{v}  {str(k)[:90]!r}")

# overlap analysis: shared 8-gram shingles between consecutive ids
def shingles(t, w=8):
    toks = (t or "").lower().split()
    return {" ".join(toks[i:i+w]) for i in range(max(0, len(toks) - w + 1))}

print("\n=== ADJACENT-ORDER OVERLAP (sample 300 consecutive pairs) ===")
overlaps = []
for a, b in list(zip(docs, docs[1:]))[:300]:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        continue
    inter = len(sa & sb)
    overlaps.append(inter / max(1, min(len(sa), len(sb))))
if overlaps:
    overlaps.sort()
    print(f"  pairs measured {len(overlaps)}")
    print(f"  mean Jaccard-ish overlap {sum(overlaps)/len(overlaps):.3f}")
    print(f"  median {pct(overlaps,50):.3f}  P90 {pct(overlaps,90):.3f}  max {overlaps[-1]:.3f}")
    print(f"  pairs with >0.5 overlap: {sum(1 for o in overlaps if o > 0.5)}")

# ---------------- language ----------------
def cjk(t):
    return sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff")

print("\n=== LANGUAGE MIX (per-record classification) ===")
lang = Counter()
for d in docs:
    d = d or ""
    c = cjk(d)
    if c > 10 and c / max(1, len(d)) > 0.2:
        lang["chinese"] += 1
    elif c > 0:
        lang["mixed_cjk"] += 1
    else:
        lang["latin_script"] += 1
for k, v in lang.most_common():
    print(f"  {k:16} {v:>6}  {100.0*v/len(docs):.1f}%")

# ---------------- negative-movement language ----------------
print("\n=== NEGATIVE / NON-COMPLETION LANGUAGE IN CORPUS TEXT (regex counts) ===")
import re
PATTERNS = {
    "prevented": r"\bprevent(?:ed|ing)\b",
    "blocked": r"\bblock(?:ed|ing)\b",
    "abandoned/aborted": r"\b(?:abandon(?:ed|ing)?|abort(?:ed|ing)?)\b",
    "planned/intended": r"\b(?:planned|intended|proposed)\s+to\b",
    "attempted to": r"\battempt(?:ed|ing)?\s+to\b",
    "did not / never + move": r"\b(?:did\s+not|didn't|never)\s+(?:enter|march|sail|move|reach|advance)\b",
    "failed to reach": r"\bfail(?:ed|ing)?\s+to\s+(?:reach|enter|take)\b",
    "in vain": r"\bin\s+vain\b",
    "unsuccessful": r"\bunsuccess\w*\b",
    "without success": r"\bwithout\s+success\b",
}
for name, pat in PATTERNS.items():
    rx = re.compile(pat, re.I)
    hits = sum(1 for d in docs if rx.search(d or ""))
    print(f"  {name:26} {hits:>6} records  {100.0*hits/len(docs):>5.1f}%")
any_neg = sum(1 for d in docs if any(re.search(p, d or "", re.I) for p in PATTERNS.values()))
print(f"  {'ANY negative marker':26} {any_neg:>6} records  {100.0*any_neg/len(docs):>5.1f}%")

# ---------------- movement language ----------------
print("\n=== MOVEMENT LANGUAGE (regex counts) ===")
MV = {
    "marched": r"\bmarched\b", "sailed": r"\bsailed\b", "crossed": r"\bcrossed\b",
    "advanced": r"\badvanced\b", "set out/departed": r"\b(?:set\s+out|departed|set\s+sail)\b",
    "arrived/reached": r"\b(?:arrived|reached)\b", "entered": r"\bentered\b",
}
for name, pat in MV.items():
    rx = re.compile(pat, re.I)
    hits = sum(1 for d in docs if rx.search(d or ""))
    print(f"  {name:22} {hits:>6} records  {100.0*hits/len(docs):>5.1f}%")
any_mv = sum(1 for d in docs if any(re.search(p, d or "", re.I) for p in MV.values()))
print(f"  {'ANY movement verb':22} {any_mv:>6} records  {100.0*any_mv/len(docs):>5.1f}%")

# ---------------- source distribution ----------------
print("\n=== SOURCE DISTRIBUTION (from metadata) ===")
for key in ("work", "author", "book", "document_id", "source", "source_id", "document"):
    if key in keys:
        c = Counter(str(m.get(key)) for m in metas if isinstance(m, dict))
        print(f"\n-- {key}: {len(c)} distinct --")
        for v, ct in c.most_common(15):
            print(f"   {ct:>6}  {v[:80]}")

# ---------------- sentence counts ----------------
print("\n=== STRUCTURE PROXY ===")
sent = [len(re.findall(r"[.!?]+\s", (d or "") + " ")) + 1 for d in docs]
sent.sort()
print(f"sentence-ish units per chunk: min {sent[0]} median {pct(sent,50):.0f} P90 {pct(sent,90):.0f} max {sent[-1]}")
print(f"chunks with <=1 sentence: {sum(1 for s in sent if s <= 1)}  ({100.0*sum(1 for s in sent if s<=1)/len(sent):.1f}%)")

json.dump({"ids": ids, "count": len(ids)}, open("_rag_ids.json", "w"))
print("\nDONE")
