"""READ-ONLY chunk-boundary forensics on the LIVE corpus.
Reconstructs each section from chunk offsets (start_offset is section-relative),
then measures real windowing, overlap, and boundary quality per layer."""
import json, re, statistics, urllib.request
from collections import Counter, defaultdict

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
    res = post(f"collections/{COL}/get", {"limit": 500, "offset": off,
              "include": ["documents", "metadatas"]})
    g = res.get("ids") or []
    if not g: break
    ids += g; docs += res.get("documents") or []; metas += res.get("metadatas") or []
    off += len(g)
    if len(g) < 500: break
N = len(docs)
print(f"records {N}")

# group chunks by (document_id, spine_index, section_index) = one SECTION
sec = defaultdict(list)
for i in range(N):
    m = metas[i]
    key = (m.get("document_id"), str(m.get("spine_index")), str(m.get("section_index")))
    sec[key].append(i)
print(f"distinct sections: {len(sec)}")

chunks_per_section = Counter(len(v) for v in sec.values())
print("\n=== CHUNKS PER SECTION (does windowing actually happen?) ===")
for k in sorted(chunks_per_section):
    print(f"  {k:>3} chunk(s): {chunks_per_section[k]:>5} sections")
multi = sum(1 for v in sec.values() if len(v) > 1)
print(f"  sections producing >1 chunk: {multi} ({100.0*multi/len(sec):.1f}%)")

# isolate sections that actually windowed
print("\n=== WINDOWED SECTIONS: overlap + boundary evidence ===")
ov_samples = []
midstart = midend = 0
wind_chunks = 0
size_hist = Counter()
for key, idxs in sec.items():
    if len(idxs) < 2:
        continue
    ordered = sorted(idxs, key=lambda i: int(metas[i].get("start_offset") or 0))
    recomposed = docs[ordered[0]]
    for a, b in zip(ordered, ordered[1:]):
        sa = int(metas[a].get("start_offset") or 0)
        sb = int(metas[b].get("start_offset") or 0)
        ea = int(metas[a].get("end_offset") or 0)
        step = sb - sa
        ta, tb = docs[a], docs[b]
        # real textual overlap: does tb's prefix appear in ta's tail?
        best = 0
        lo = max(0, len(ta) - 800)
        for L in range(50, min(700, len(tb)) + 1, 25):
            if tb[:L] and tb[:L] in ta[lo:]:
                best = L
        ov_samples.append((step, len(ta), best))
    for i in ordered:
        wind_chunks += 1
        size_hist[len(docs[i])] += 1

if ov_samples:
    steps = [s for s, _la, _o in ov_samples]
    print(f"  adjacent pairs inside windowed sections: {len(ov_samples)}")
    print(f"  start_offset stride: min={min(steps)} median={statistics.median(steps)} max={max(steps)}")
    ovs = [o for _s, _la, o in ov_samples]
    print(f"  measured textual overlap: min={min(ovs)} median={statistics.median(ovs)} max={max(ovs)}")
    print(f"  pairs with >=50 char real overlap: {sum(1 for o in ovs if o>=50)}/{len(ovs)}")
print(f"  chunks inside windowed sections: {wind_chunks}")
if size_hist:
    sizes = sorted(size_hist.elements())
    print(f"  their char sizes: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")

print("\n=== SECTION SIZE DISTRIBUTION (reconstructed) ===")
sec_lens = []
for key, idxs in sec.items():
    ordered = sorted(idxs, key=lambda i: int(metas[i].get("start_offset") or 0))
    total = max(int(metas[i].get("end_offset") or 0) for i in ordered)
    sec_lens.append(total)
sec_lens.sort()
def pct(v, p):
    if not v: return 0
    return v[min(len(v)-1, round((len(v)-1)*p))]
for p in (0, 25, 50, 75, 90, 95, 100):
    print(f"  P{p:<3} {pct(sec_lens, p/100):>8}")
print(f"  sections > 3400 chars (so windowing triggers): {sum(1 for s in sec_lens if s>3400)} ({100.0*sum(1 for s in sec_lens if s>3400)/len(sec_lens):.1f}%)")
print(f"  sections <= 3400 chars (single chunk): {sum(1 for s in sec_lens if s<=3400)}")

print("\n=== MID-SENTENCE STARTS (where do they come from?) ===")
SENT_END = re.compile(r"[.!?][\"'\u201d\u2019)]?\s*$")
single_start_mid = 0
wind_start_mid = 0
single_n = wind_n = 0
for key, idxs in sec.items():
    ordered = sorted(idxs, key=lambda i: int(metas[i].get("start_offset") or 0))
    for pos, i in enumerate(ordered):
        t = docs[i] or ""
        first = t.lstrip()[:1]
        mid = bool(first) and first.islower()
        if len(idxs) == 1:
            single_n += 1; single_start_mid += mid
        else:
            wind_n += 1; wind_start_mid += mid
print(f"  SINGLE-chunk sections: {single_start_mid}/{single_n} start lowercase ({100.0*single_start_mid/max(1,single_n):.1f}%)")
print(f"  WINDOWED  chunks     : {wind_start_mid}/{wind_n} start lowercase ({100.0*wind_start_mid/max(1,wind_n):.1f}%)")

print("\n=== PARAGRAPH BOUNDARIES AVAILABLE IN STORED TEXT? ===")
nl = sum(1 for d in docs if "\n" in (d or ""))
nl2 = sum(1 for d in docs if "\n\n" in (d or ""))
print(f"  chunks containing any newline: {nl} ({100.0*nl/N:.1f}%)")
print(f"  chunks containing blank line : {nl2} ({100.0*nl2/N:.1f}%)")
print("  (parser joins <p> elements with '\\n' then _text() collapses whitespace)")

print("\n=== SENTENCE PUNCTUATION QUALITY PER AUTHOR ===")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
by_author = defaultdict(list)
for i in range(N):
    by_author[metas[i].get("author")].append(docs[i] or "")
for author, texts in sorted(by_author.items(), key=lambda x: -len(x[1])):
    joined = " ".join(texts)
    sents = [s for s in SENT_SPLIT.split(joined) if s.strip()]
    if not sents: continue
    lens = sorted(len(s) for s in sents)
    # suspicious: split produced a very short fragment (abbreviation artifact)
    tiny = sum(1 for s in sents if len(s) < 25)
    print(f"  {author:16} n={len(texts):>5} sentences={len(sents):>7} median_len={lens[len(lens)//2]:>4} "
          f"P95={lens[int(len(lens)*0.95)]:>5} tiny(<25ch)={100.0*tiny/len(sents):>5.1f}%")

print("\n=== ABBREVIATION / ROMAN NUMERAL HAZARDS IN TEXT ===")
for name, pat in [("B.C./A.D.", r"\bB\.C\.|\bA\.D\."), ("cons./cos.", r"\b(?:cons|cos)\."),
                  ("ch./chap.", r"\b(?:ch|chap)\."), ("Roman numeral suffix", r"\b[IVXLCDM]{2,}\b"),
                  ("quote-heavy", r"[\u201c\u201d]")]:
    c = sum(1 for d in docs if re.search(pat, d or ""))
    print(f"  {name:22} {c:>6} records ({100.0*c/N:5.1f}%)")
print("\nDONE")
