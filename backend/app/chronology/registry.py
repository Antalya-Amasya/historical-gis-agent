"""Deterministic, provenance-preserving ingestion for modern chronology EPUBs."""
from __future__ import annotations
import hashlib, json, re, html
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree

from backend.app.rag.ingestion.corpus_registry import CorpusDocument
from backend.app.rag.ingestion.generic_epub import epub_spine, load_epub_sections
from backend.app.routes.temporal import EvidenceTemporalResolver

AUTHORITY_ROLE = "MODERN_CHRONOLOGY_SOURCE"
VERSION = "chronology-registry-v1"

def extract_epub_document_blocks(raw: bytes, href: str) -> list[dict]:
    """Namespace-insensitive, body-first XHTML walker for Gutenberg-style EPUBs."""
    root = ElementTree.fromstring(raw)
    body = next((x for x in root.iter() if _tag(x.tag) == "body"), root)
    blocks=[]

    def text(node):
        return " ".join(html.unescape("".join(node.itertext())).split())

    def has_descendant_block(node):
        return any(_tag(child.tag) in {"p", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
                   for child in node.iter() if child is not node)

    def walk(node):
        name = _tag(node.tag)
        if name in {"script","style"}: return
        if name in {"h1","h2","h3","h4","h5","h6","p","blockquote"}:
            value=text(node)
            block_type = "heading" if name.startswith("h") else ("paragraph" if name == "p" else "blockquote")
            if value: blocks.append({"type":block_type,"text":value,"document_href":href,"block_index":len(blocks)})
            return
        children=list(node)
        if name == "div" and not has_descendant_block(node):
            value=text(node)
            if value: blocks.append({"type":"paragraph","text":value,"document_href":href,"block_index":len(blocks)})
            return
        for child in children: walk(child)
    walk(body); return blocks

def _tag(value: str) -> str: return value.rsplit("}", 1)[-1].lower()
def _metadata(path: Path) -> dict[str, str | None]:
    with ZipFile(path) as z:
        root = ElementTree.fromstring(z.read("META-INF/container.xml")); opf = next(x.attrib["full-path"] for x in root.iter() if _tag(x.tag)=="rootfile")
        package = ElementTree.fromstring(z.read(opf)); values = {}
        for item in package.iter():
            key = _tag(item.tag)
            if key in {"title","creator","language","publisher","date"} and item.text and key not in values: values[key] = " ".join(item.itertext()).strip()
        return values

def source(path: Path) -> dict:
    meta = _metadata(path); digest = hashlib.sha256(path.read_bytes()).hexdigest(); source_id = f"chronology-{digest[:16]}"
    return {"source_id":source_id,"title":meta.get("title") or path.stem,"author":meta.get("creator") or "Unknown","language":meta.get("language"),"publisher":meta.get("publisher"),"publication_date":meta.get("date"),"filename":path.name,"sha256":digest,"source_type":"SECONDARY_HISTORICAL_SYNTHESIS","authority_role":AUTHORITY_ROLE,"limitations":["Modern secondary synthesis; chronology authority is source-specific, not a claim of final scholarly consensus."]}

def _direct_blocks(path: Path) -> tuple[list[tuple[str, str | None, int, int, str]], dict]:
    """Return paragraph blocks in EPUB spine then DOM order, with diagnostics."""
    direct=[]
    diagnostics={"visible_blocks": 0, "paragraphs": 0, "headings": 0, "visible_characters": 0,
                 "bc_bce_marker_hits": 0, "ad_ce_marker_hits": 0}
    with ZipFile(path) as archive:
        for href in epub_spine(path):
            if not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            raw_bytes = archive.read(href)
            heading = None
            paragraph_index = 0
            for block in extract_epub_document_blocks(raw_bytes, href):
                diagnostics["visible_blocks"] += 1
                diagnostics["visible_characters"] += len(block["text"])
                diagnostics["bc_bce_marker_hits"] += len(re.findall(r"\b(?:B\.\s*C\.\s*\d+|\d+\s*B\.\s*C\.|\d+\s*(?:BC|BCE))", block["text"], re.I))
                diagnostics["ad_ce_marker_hits"] += len(re.findall(r"\b(?:A\.\s*D\.\s*\d+|\d+\s*A\.\s*D\.|\d+\s*(?:AD|CE))", block["text"], re.I))
                if block["type"] == "heading":
                    diagnostics["headings"] += 1
                    heading = block["text"]
                    continue
                diagnostics["paragraphs"] += 1
                direct.append((href, heading, block["block_index"], paragraph_index, block["text"]))
                paragraph_index += 1
    return direct, diagnostics


def records(path: Path) -> tuple[dict, list[dict], int, dict]:
    meta = source(path); document = CorpusDocument(document_id=meta["source_id"], author=meta["author"], work=meta["title"], language=meta["language"] or "en", source_type="chronology_source", filename=path.name, license="local", parser_hints={})
    sections, spine_count = load_epub_sections(path, document); resolver = EvidenceTemporalResolver(); result=[]
    # Gutenberg XHTML is often text-file XHTML without semantic <section> wrappers.
    # Fall back to paragraph-level visible body extraction while preserving spine-like hrefs.
    direct, diagnostics = _direct_blocks(path)
    iterable = direct or ((s.spine_item, s.heading, s.section_index, s.section_index, s.text) for s in sections)
    for href, heading, section_index, paragraph_index, paragraph in iterable:
        readings, _ = resolver.resolve(paragraph, meta["source_id"])
        for reading_index, reading in enumerate(readings):
            record_id = hashlib.sha256(f"{meta['source_id']}:{href}:{section_index}:{paragraph_index}:{reading_index}:{reading.raw_expression}".encode()).hexdigest()[:20]
            result.append({"record_id":f"chronology-{record_id}","canonical_event_label":heading,"raw_event_text":paragraph[:1200],"normalized_start":reading.normalized_start,"normalized_end":reading.normalized_end,"precision":reading.precision.value,"raw_temporal_expression":reading.raw_expression,"source_id":meta["source_id"],"source_locator":{"epub_document":href,"section_index":section_index,"heading":heading,"paragraph_index":paragraph_index},"source_excerpt":paragraph[:500],"evidence_refs":reading.evidence_refs,"grounding_status":reading.status.value,"limitations":["Date-bearing passage extraction only; no automatic HistoricalEvent identity matching."]})
    return meta, result, spine_count, diagnostics

def ingest(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True); all_records=[]; manifest=[]
    for path in sorted(input_dir.glob("*.epub")):
        meta, rows, spine_count, diagnostics = records(path); all_records.extend(rows); resolved=[r for r in rows if r["normalized_start"]]
        manifest.append({**meta,"content_document_count":spine_count, **diagnostics, "record_count":len(rows),"resolved_temporal_count":len(resolved),"unresolved_temporal_count":len(rows)-len(resolved),"earliest_normalized_year":min((int(r["normalized_start"]) for r in resolved),default=None),"latest_normalized_year":max((int(r["normalized_end"] or r["normalized_start"]) for r in resolved),default=None)})
    (output_dir/"records.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in all_records),encoding="utf-8")
    report={"registry_version":VERSION,"sources":manifest,"total_records":len(all_records)}
    (output_dir/"source_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
