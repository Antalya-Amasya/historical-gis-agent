"""Read-only corpus discovery, validation, and reporting. Never embeds or opens Chroma."""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .corpus_registry import CorpusDocument, CorpusRegistry
from .document_sections import DocumentSection
from .generic_epub import epub_structure, load_epub_sections
from .models import TextChunk

PARSER_VERSION = "generic-epub-navigation-v2"
CHUNKER_VERSION = "section-window-v1"
DEFAULT_CHUNK_CHARS = 4000
DEFAULT_OVERLAP_CHARS = 600


def document_chunks(sections: list[DocumentSection], document: CorpusDocument, *, size: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_OVERLAP_CHARS) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    step = max(1, size - min(overlap, size // 4))
    for section_index, section in enumerate(sections):
        for start in range(0, len(section.text), step):
            text = section.text[start:start + size].strip()
            if not text:
                continue
            end = start + len(text)
            identity = f"{document.document_id}:{section.spine_item}:{section.section_index}:{start}:{end}:{text}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            metadata = {
                "document_id": document.document_id, "author": document.author, "work": document.work,
                "volume": document.volume, "book": section.book, "chapter": section.chapter,
                "section": section.section, "heading": section.heading, "heading_level": section.heading_level,
                "navigation_path": list(section.navigation_path), "navigation_source": section.navigation_source,
                "source_file": section.source_file,
                "source_url": document.source_url, "source_type": document.source_type, "license": document.license,
                "spine_item": section.spine_item, "spine_index": section.spine_index,
                "section_index": section.section_index, "href": section.href, "fragment": section.fragment,
                "start_offset": start, "end_offset": end, "chunker_version": CHUNKER_VERSION,
                "parser_version": PARSER_VERSION,
            }
            chunks.append(TextChunk(chunk_id, text, metadata))
    return chunks


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * percentile))]


def _document_report(document: CorpusDocument, path: Path) -> dict[str, object]:
    entry: dict[str, object] = {"document_id": document.document_id, "filename": document.filename, "author": document.author, "work": document.work, "volume": document.volume, "warnings": [], "errors": []}
    try:
        structure = epub_structure(path)
        sections, spine_count = load_epub_sections(path, document)
        chunks = document_chunks(sections, document)
    except Exception as exc:
        entry.update({"spine_item_count": 0, "section_count": 0, "detected_book_count": 0, "detected_chapter_count": 0, "heading_coverage": 0.0, "structure_coverage_ratio": 0.0, "expected_chunk_count": 0, "chunk_chars": {"min": 0, "median": 0, "p95": 0, "max": 0}, "empty_chunk_count": 0, "oversized_chunk_count": 0, "duplicate_chunk_id_count": 0})
        entry["errors"].append(f"parser_failure:{type(exc).__name__}:{exc}")
        return entry
    books = {section.book for section in sections if section.book}
    volumes = {section.volume for section in sections if section.volume}
    chapters = {f"{section.book}:{section.chapter}" for section in sections if section.chapter}
    detected_sections = {f"{section.book}:{section.chapter}:{section.section}" for section in sections if section.section}
    lengths = [len(chunk.text) for chunk in chunks]
    duplicate_ids = sum(count - 1 for count in Counter(chunk.id for chunk in chunks).values() if count > 1)
    heading_coverage = sum(section.heading is not None for section in sections) / len(sections) if sections else 0.0
    structured = sum(section.book is not None or section.chapter is not None or section.section is not None for section in sections)
    structure_coverage = structured / len(sections) if sections else 0.0
    aligned = sum(section.navigation_source is not None for section in sections)
    entry.update({"spine_item_count": spine_count, "navigation_entry_count": len(structure.navigation), "navigation_source": structure.navigation_source, "section_count": len(sections), "detected_volume_count": len(volumes), "detected_book_count": len(books), "detected_chapter_count": len(chapters), "detected_section_count": len(detected_sections), "navigation_alignment_ratio": round(aligned / len(sections), 4) if sections else 0.0, "heading_coverage": round(heading_coverage, 4), "structure_coverage_ratio": round(structure_coverage, 4), "expected_chunk_count": len(chunks), "chunk_chars": {"min": min(lengths, default=0), "median": int(statistics.median(lengths)) if lengths else 0, "p95": _percentile(lengths, .95), "max": max(lengths, default=0)}, "empty_chunk_count": sum(not chunk.text.strip() for chunk in chunks), "oversized_chunk_count": sum(length > DEFAULT_CHUNK_CHARS for length in lengths), "duplicate_chunk_id_count": duplicate_ids})
    warnings = entry["warnings"]
    if not sections: warnings.append("no_text_sections")
    if not books: warnings.append("book_not_detected")
    if not chapters: warnings.append("chapter_not_detected")
    if heading_coverage < 0.5: warnings.append("low_heading_coverage")
    if structure_coverage < 0.5: warnings.append("low_structure_coverage")
    if duplicate_ids: warnings.append("duplicate_chunk_ids")
    return entry


def dry_run(incoming_dir: Path, registry_path: Path, report_path: Path | None = None) -> dict[str, object]:
    registry = CorpusRegistry.from_file(registry_path)
    discovered = sorted(incoming_dir.glob("*.epub"))
    by_filename = registry.by_filename()
    reports = []
    unregistered = []
    for path in discovered:
        document = by_filename.get(path.name)
        if document is None:
            unregistered.append(path.name)
            continue
        if document.enabled:
            reports.append(_document_report(document, path))
    registered_missing = [document.filename for document in registry.documents if document.enabled and not (incoming_dir / document.filename).exists()]
    report = {"mode": "dry_run", "generated_at": datetime.now(timezone.utc).isoformat(), "parser_version": PARSER_VERSION, "chunker_version": CHUNKER_VERSION, "planned_collection": "roman_republic_primary_sources_v1", "embedding_called": False, "chroma_written": False, "documents_discovered": len(discovered), "documents_registered": len(reports), "unregistered_files": unregistered, "registered_missing_files": registered_missing, "documents": reports, "totals": {"expected_chunk_count": sum(int(item["expected_chunk_count"]) for item in reports), "parser_failures": sum(bool(item["errors"]) for item in reports), "warning_count": sum(len(item["warnings"]) for item in reports), "navigation_entry_count": sum(int(item.get("navigation_entry_count", 0)) for item in reports)}}
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
