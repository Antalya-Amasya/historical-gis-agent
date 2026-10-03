"""READ-ONLY chunking simulation. Re-parses the 16 source EPUBs (structure only;
never embeds, never writes Chroma) and simulates competing chunking designs on
the REAL text, measuring boundary integrity, antecedent recovery and index growth."""
import re, sys, statistics, hashlib
from collections import Counter
from pathlib import Path

REPO = r"C:\D\python\historical-gis-cursor"
sys.path.insert(0, REPO)
from backend.app.rag.ingestion.corpus_registry import CorpusRegistry
from backend.app.rag.ingestion.generic_epub import load_epub_sections

INCOMING = Path(r"C:\data\historical-gis-runtime\data\historical_sources\incoming")
REGISTRY = Path(r"C:\data\historical-gis-runtime\data\historical_sources\corpus.json")
print("registry:", REGISTRY, REGISTRY.exists())

reg = CorpusRegistry.from_file(REGISTRY)
# The registry filenames contain CJK brackets that do not round-trip through the console,
# so pair documents to real files by stable author/work prefix instead of exact filename.
DISK = sorted(INCOMING.glob("*.epub"))
def find_disk(doc):
    author = doc.author.split()[0].lower()
    key = "caesar" if "caesar" in author else author
    cands = [p for p in DISK if key in p.name.lower()]
    if not cands:
        return None
    if doc.volume:
        import re as _re
        vol = str(doc.volume).replace("\u2013", "-").replace("\u2014", "-")
        if vol.isdigit():
            for p in cands:
                if _re.search(rf"Vol\.?\s*{vol}\b", p.name, _re.I):
                    return p
            return None
        m = _re.search(r"Books\s+([0-9]+)", vol)
        if m:
            lo = m.group(1)
            for p in cands:
                if _re.search(rf"Books\s+{lo}\s*-", p.name, _re.I):
                    return p
        return None
    plain = [p for p in cands if "Vol." not in p.name and "Books" not in p.name]
    return plain[0] if plain else (cands[0] if len(cands) == 1 else None)
by_name = {}
for d in reg.documents:
    p = find_disk(d)
    if p is not None:
        by_name[p.name] = d
print("registry docs:", len(reg.documents), " matched to files:", len(by_name), " disk epub:", len(DISK))
for p in DISK:
    if p.name not in by_name:
        print("   UNMATCHED:", p.name)

# Re-parse with a paragraph-preserving variant of the same traversal (read-only, in-memory).
from xml.etree import ElementTree
from zipfile import ZipFile
import backend.app.rag.ingestion.generic_epub as ge

def section_paragraphs(path: Path, document):
    """Same traversal as load_epub_sections but keeps <p>-level blocks separate."""
    structure = ge.epub_structure(path)
    by_target = {(x.normalized_href, x.fragment): x for x in structure.navigation}
    by_item = {x.normalized_href: x for x in structure.navigation if x.fragment is None}
    labels = {x: None for x in ge._LEVELS}
    out = []          # (heading, [paragraph,...])
    with ZipFile(path) as archive:
        for spine_index, spine_item in enumerate(structure.spine):
            if not spine_item.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                root = ElementTree.fromstring(archive.read(spine_item))
            except Exception:
                continue
            active = by_item.get(spine_item); parts = []; heading = None
            def apply(entry):
                nonlocal active, labels
                if entry is not None:
                    active = entry; inherited = {x: None for x in ge._LEVELS}
                    for label in entry.navigation_path:
                        inherited = ge._labels(label, inherited)
                    labels = inherited
            def flush():
                nonlocal parts
                if parts:
                    out.append((heading, [p for p in parts if p.strip()]))
                parts = []
            apply(active)
            for element in root.iter():
                identifier = element.attrib.get("id") or element.attrib.get("name")
                target = by_target.get((spine_item, identifier)) if identifier else None
                tag = ge._name(element.tag)
                if target and parts: flush()
                apply(target)
                if tag not in ge._TEXT: continue
                value = " ".join("".join(element.itertext()).split())
                if not value: continue
                if tag in ge._HEADINGS:
                    if parts: flush()
                    heading = value
                else:
                    parts.append(value)
            flush()
    return out

# ---------------- gather real paragraphs ----------------
docs = []
groups = []          # list of paragraph lists (one per source section)
for path in sorted(INCOMING.glob("*.epub")):
    doc = by_name.get(path.name)
    if doc is None or not doc.enabled:
        continue
    try:
        secs = section_paragraphs(path, doc)
    except Exception as e:
        print("  PARSE FAIL", path.name, type(e).__name__); continue
    n = 0
    for heading, paras in secs:
        plist = [p for p in paras if p.strip()]
        if not plist: continue
        groups.append(plist)
        docs.extend(plist); n += len(plist)
    print(f"  {doc.document_id:36} paragraphs={n}")
print(f"\nTOTAL PARAGRAPHS {len(docs)}  chars={sum(len(d) for d in docs)}  sections={len(groups)}")

# ---------------- paragraph statistics ----------------
plens = sorted(len(d) for d in docs)
def pct(v, p):
    return v[min(len(v) - 1, round((len(v) - 1) * p))] if v else 0
print("\n=== PARAGRAPH LENGTH (chars) ===")
for p in (0, 10, 25, 50, 75, 90, 95, 100):
    print(f"  P{p:<3} {pct(plens, p/100):>7}")
print(f"  mean {statistics.mean(plens):.0f}")
print(f"  <200ch: {sum(1 for x in plens if x<200)} ({100.0*sum(1 for x in plens if x<200)/len(plens):.1f}%)")
print(f"  >2000ch: {sum(1 for x in plens if x>2000)} ({100.0*sum(1 for x in plens if x>2000)/len(plens):.1f}%)")
print(f"  >4000ch: {sum(1 for x in plens if x>4000)} ({100.0*sum(1 for x in plens if x>4000)/len(plens):.1f}%)")

SENT = re.compile(r"(?<=[.!?])\s+")
slens = []
for d in docs:
    slens += [len(s) for s in SENT.split(d) if s.strip()]
slens.sort()
print(f"\n=== SENTENCE LENGTH (chars), n={len(slens)} ===")
for p in (0, 25, 50, 75, 90, 95, 100):
    print(f"  P{p:<3} {pct(slens, p/100):>7}")

# ---------------- antecedent distance (Phase 7) ----------------
ANAPHOR = re.compile(r"^(?:he|she|they|him|her|them|his|their|it|the\s+consul|the\s+army|the\s+general|"
                     r"the\s+enemy|from\s+there|thence|thereupon|whereupon|accordingly|meanwhile)\b", re.I)
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
# build a sentence stream per paragraph, tracking distance to the last named entity
dist_same = dist_prev = dist_2 = dist_para = dist_far = 0
total_anaph = 0
for d in docs:
    sents = [s.strip() for s in SENT.split(d) if s.strip()]
    last_name_idx = None
    last_name_para = None
    for si, s in enumerate(sents):
        if ANAPHOR.match(s):
            total_anaph += 1
            if PROPER.search(s):
                dist_same += 1
            elif last_name_idx == si - 1:
                dist_prev += 1
            elif last_name_idx == si - 2:
                dist_2 += 1
            elif last_name_idx is not None:
                dist_para += 1
            else:
                dist_far += 1
        if PROPER.search(s):
            last_name_idx = si
print(f"\n=== ANTECEDENT DISTANCE (anaphoric sentence openers) ===")
print(f"  total anaphoric openers: {total_anaph}")
if total_anaph:
    for lbl, v in (("name in SAME sentence", dist_same), ("previous sentence", dist_prev),
                   ("2 sentences back", dist_2), ("3+ sentences back (same para)", dist_para),
                   ("no antecedent in paragraph", dist_far)):
        print(f"  {lbl:36} {v:>6}  {100.0*v/total_anaph:5.1f}%")

# ---------------- movement span (Phase 8) ----------------
MOVE = re.compile(r"\b(?:marched|marching|sailed|sailing|advanced|crossed|crossing|departed|set\s+out|"
                  r"set\s+sail|entered|entering|arrived|reached|proceeded|retreated|withdrew|invaded)\b", re.I)
spans = Counter(); span_chars = []
for d in docs:
    sents = [s.strip() for s in SENT.split(d) if s.strip()]
    for si, s in enumerate(sents):
        if not MOVE.search(s): continue
        has_actor = bool(PROPER.search(s))
        has_from = bool(re.search(r"\bfrom\b", s, re.I))
        has_to = bool(re.search(r"\b(?:to|toward|towards|into)\b", s, re.I))
        need = 1
        if not (has_actor and has_from and has_to):
            # widen until complete or 4 sentences
            for w in range(2, 5):
                lo = max(0, si - (w - 1)); window = " ".join(sents[lo:si + w])
                if PROPER.search(window) and re.search(r"\bfrom\b", window, re.I) and \
                   re.search(r"\b(?:to|toward|towards|into)\b", window, re.I):
                    need = w; break
            else:
                need = 0
        spans[need] += 1
        span_chars.append(len(" ".join(sents[si:si + max(1, need)])))
tot = sum(spans.values())
print(f"\n=== MOVEMENT SPAN: sentences needed for actor+from+to (n={tot}) ===")
for k in sorted(spans):
    lbl = "not complete in 4" if k == 0 else f"{k} sentence(s)"
    print(f"  {lbl:22} {spans[k]:>7}  {100.0*spans[k]/tot:5.1f}%")
if span_chars:
    span_chars.sort()
    print(f"  span chars: P50={pct(span_chars,.5):.0f} P90={pct(span_chars,.9):.0f} P95={pct(span_chars,.95):.0f}")

# ---------------- fixed-size simulation with sentence snapping (Phase 9) ----------------
print("\n=== SIMULATED CHUNKERS (sentence-aware snapping) ===")
def chunk_sentence_aware(text_sents, target, overlap_sents):
    """greedy pack sentences up to target, snap to sentence boundary, overlap by N sentences."""
    out = []; i = 0; n = len(text_sents)
    while i < n:
        buf = []; size = 0; j = i
        while j < n and (size + len(text_sents[j]) <= target or not buf):
            buf.append(text_sents[j]); size += len(text_sents[j]) + 1; j += 1
        out.append(" ".join(buf))
        if j >= n: break
        i = max(i + 1, j - overlap_sents)
    return out

# operate per paragraph so we never merge across paragraph boundaries (upper bound on quality)
print(f"{'design':34} {'chunks':>8} {'mid-sent start':>15} {'P50 chars':>10} {'x of 7230':>10}")
baseline_chunks = 7230
for target, ov in ((4000, 1), (2000, 1), (1600, 1), (1200, 1), (800, 1), (2000, 0), (1200, 0)):
    total = 0; mid = 0; sizes = []
    for d in docs:
        sents = [s.strip() for s in SENT.split(d) if s.strip()]
        if not sents: continue
        cs = chunk_sentence_aware(sents, target, ov)
        total += len(cs)
        for c in cs:
            sizes.append(len(c))
            if c[:1].islower(): mid += 1
    sizes.sort()
    print(f"{'sentence-aware '+str(target)+'/'+str(ov):34} {total:>8} {100.0*mid/max(1,total):>14.1f}% "
          f"{pct(sizes,.5):>10.0f} {total/baseline_chunks:>9.2f}x")

# paragraph-as-chunk and paragraph accumulation (accumulate ACROSS paragraphs within a section)
def accum(plist, budget):
    out = []; buf = ""
    for p in plist:
        if buf and len(buf) + len(p) + 1 > budget:
            out.append(buf); buf = p
        else:
            buf = (buf + "\n" + p).strip() if buf else p
    if buf: out.append(buf)
    return out

for budget in (None, 1200, 2000, 4000):
    total = 0; sizes = []; mid = 0
    for plist in groups:
        cs = plist if budget is None else accum(plist, budget)
        total += len(cs); sizes += [len(c) for c in cs]; mid += sum(1 for c in cs if c[:1].islower())
    sizes.sort()
    lbl = "paragraph-as-chunk" if budget is None else f"paragraph-accum<={budget}"
    print(f"{lbl:34} {total:>8} {100.0*mid/max(1,total):>14.1f}% {pct(sizes,.5):>10.0f} {total/baseline_chunks:>9.2f}x")

print("\nDONE")
