"""READ-ONLY: how much context is needed to recover anaphoric antecedents, and
what would sibling/parent expansion buy? Operates on re-parsed source text only."""
import re, sys, statistics
from collections import Counter
from pathlib import Path

REPO = r"C:\D\python\historical-gis-cursor"
sys.path.insert(0, REPO)
from backend.app.rag.ingestion.corpus_registry import CorpusRegistry
from backend.app.rag.ingestion.generic_epub import load_epub_sections
import backend.app.rag.ingestion.generic_epub as ge_word

INCOMING = Path(r"C:\data\historical-gis-runtime\data\historical_sources\incoming")
REG = Path(r"C:\data\historical-gis-runtime\data\historical_sources\corpus.json")
reg = CorpusRegistry.from_file(REG)
DISK = sorted(INCOMING.glob("*.epub"))
def find_disk(doc):
    toks = [t for t in re.split(r"\W+", doc.author.lower()) if t]
    key = "caesar" if "caesar" in toks else toks[0]
    cands = [p for p in DISK if key in p.name.lower()]
    if not cands: return None
    if doc.volume:
        vol = str(doc.volume).replace("\u2013", "-").replace("\u2014", "-")
        if vol.isdigit():
            for p in cands:
                if re.search(rf"Vol\.?\s*{vol}\b", p.name, re.I): return p
            return None
        m = re.search(r"Books\s+([0-9]+)", vol)
        if m:
            for p in cands:
                if re.search(rf"Books\s+{m.group(1)}\s*-", p.name, re.I): return p
        return None
    plain = [p for p in cands if "Vol." not in p.name and "Books" not in p.name]
    return plain[0] if plain else (cands[0] if len(cands) == 1 else None)
by_name = {}
for d in reg.documents:
    p = find_disk(d)
    if p: by_name[p.name] = d
print("matched:", len(by_name), "of", len(DISK))

# Use the PRODUCTION parser to get real section boundaries (no paragraph preservation)
SENT = re.compile(r"(?<=[.!?])\s+")
ANAPHOR = re.compile(r"^(?:he|she|they|him|her|them|his|their|it|the\s+consul|the\s+army|the\s+general|"
                     r"the\s+enemy|the\s+senate|from\s+there|thence|thereupon|whereupon|accordingly|meanwhile)\b", re.I)
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Per section: sentences with running char offsets
sec_sents = []          # list of (section_key, [(sent, start, end)], section_text)
for path in sorted(INCOMING.glob("*.epub")):
    doc = by_name.get(path.name)
    if not doc: continue
    try:
        secs, _ = load_epub_sections(path, doc)
    except Exception as e:
        print("  parse fail", path.name, type(e).__name__); continue
    for s in secs:
        text = s.text
        sents = [(m.group().strip(), m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*", text) if m.group().strip()]
        if sents: sec_sents.append(((doc.document_id, s.spine_index, s.section_index), sents, text))
print("sections:", len(sec_sents), "sentences:", sum(len(s[1]) for s in sec_sents))

# ---- Phase 7 refined: antecedent distance measured WITHIN section, in chars ----
print("\n=== ANTECEDENT RECOVERY vs CONTEXT WINDOW (chars before the anaphor) ===")
need = Counter()
tot = 0
for key, sents, text in sec_sents:
    last_name_pos = None
    for i, (sent, a, b) in enumerate(sents):
        if ANAPHOR.match(sent):
            tot += 1
            if PROPER.search(sent):
                need["same sentence"] += 1; continue
            # need text back to the last sentence containing a proper name
            back = None
            for j in range(i - 1, max(-1, i - 40), -1):
                if PROPER.search(sents[j][0]):
                    back = a - sents[j][0 if False else 1]
                    back = a - sents[j][1]
                    break
            if back is None:
                need["unresolved in section"] += 1
            elif back <= 200:
                need["<=200 chars"] += 1
            elif back <= 400:
                need["<=400 chars"] += 1
            elif back <= 800:
                need["<=800 chars"] += 1
            elif back <= 1200:
                need["<=1200 chars"] += 1
            elif back <= 2000:
                need["<=2000 chars"] += 1
            else:
                need[">2000 chars"] += 1
        if PROPER.search(sent):
            last_name_pos = b
print(f"  anaphoric openers: {tot}")
cum = 0
order = ["same sentence", "<=200 chars", "<=400 chars", "<=800 chars", "<=1200 chars", "<=2000 chars",
         ">2000 chars", "unresolved in section"]
for k in order:
    v = need.get(k, 0); cum += v
    print(f"  {k:24} {v:>7}  {100.0*v/tot:5.1f}%   cumulative {100.0*cum/tot:5.1f}%")

# ---- Phase 13/15: simulate chunkings and measure how often the antecedent lands in the SAME chunk ----
def pack(sents, target, overlap_sents=0, snap=True):
    """greedy sentence packing; returns list of (start_char, end_char, [(sent, a, b)])"""
    out = []; i = 0; n = len(sents)
    while i < n:
        j = i; size = 0; group = []
        while j < n:
            L = len(sents[j][0]) + 1
            if group and size + L > target: break
            group.append(sents[j]); size += L; j += 1
        out.append(group)
        if j >= n: break
        i = max(i + 1, j - overlap_sents)
    return out

print("\n=== SIMULATED: does the antecedent fall in the SAME chunk? ===")
print(f"{'design':30} {'chunks':>8} {'antecedent in same chunk':>26} {'mid-sent start':>15}")
for target, ov in ((4000, 0), (4000, 1), (2000, 1), (1200, 1), (800, 1), (1600, 1)):
    total_chunks = 0; recovered = 0; anaph_total = 0; mid = 0
    for key, sents, text in sec_sents:
        groups = pack(sents, target, ov)
        total_chunks += len(groups)
        # map sentence index -> group index
        gi = {}
        for k, g in enumerate(groups):
            for s in g:
                gi[id(s)] = k
        for k, g in enumerate(groups):
            first = g[0][0]
            if first[:1].islower(): mid += 1
        for i, (sent, a, b) in enumerate(sents):
            if not ANAPHOR.match(sent) or PROPER.search(sent): continue
            anaph_total += 1
            g = gi[id(sents[i])]
            # does this chunk contain a proper name (i.e. its own antecedent)?
            if any(PROPER.search(s[0]) for s in groups[g]):
                recovered += 1
            elif g > 0 and any(PROPER.search(s[0]) for s in groups[g - 1]):
                # previous SIBLING would rescue it
                recovered += 0.5
    print(f"{'sentence-aware '+str(target)+'/'+str(ov):30} {total_chunks:>8} "
          f"{100.0*recovered/max(1,anaph_total):>25.1f}% {100.0*mid/max(1,total_chunks):>14.1f}%")
print("  (0.5 credit = antecedent recoverable only from the PREVIOUS SIBLING chunk)")

print("\n=== SIBLING vs PARENT value ===")
for target, ov in ((4000, 1), (1200, 1)):
    same = prev = beyond = 0; n = 0
    for key, sents, text in sec_sents:
        groups = pack(sents, target, ov)
        gi = {}
        for k, g in enumerate(groups):
            for s in g: gi[id(s)] = k
        for i, (sent, a, b) in enumerate(sents):
            if not ANAPHOR.match(sent) or PROPER.search(sent): continue
            n += 1; g = gi[id(sents[i])]
            if any(PROPER.search(s[0]) for s in groups[g]): same += 1
            elif g > 0 and any(PROPER.search(s[0]) for s in groups[g-1]): prev += 1
            else: beyond += 1
    print(f"  target={target} ov={ov}: same-chunk {100.0*same/n:.1f}%  prev-sibling {100.0*prev/n:.1f}%  "
          f"needs 2+ back {100.0*beyond/n:.1f}%")
print("\nDONE")
