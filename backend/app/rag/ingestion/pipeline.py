from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .chunking import structure_aware
from .loaders import load_epub, load_pdf
from .structure import annotate_structure


SOURCE_CATALOG = {
    "polybius": {"author": "Polybius", "work": "Histories", "language": "en", "campaign": "Second Punic War"},
    "livy": {"author": "Livy", "work": "Ab Urbe Condita / History of Rome", "language": "en", "campaign": "Second Punic War"},
}


def source_metadata(pdf: Path) -> dict[str, object]:
    key = next(key for key in SOURCE_CATALOG if key in pdf.as_posix().lower())
    return SOURCE_CATALOG[key] | {
        "source_file": pdf.as_posix(),
        "source_type": "primary_source",
        "topic": "Hannibal; Second Punic War; Alpine Crossing",
        "fingerprint": sha256(pdf.read_bytes()).hexdigest(),
    }


def ingest_pdf(pdf: Path) -> tuple[list, dict[str, object], list]:
    started = perf_counter()
    pages = load_pdf(pdf)
    metadata = source_metadata(pdf)
    chunks = structure_aware(annotate_structure(pages), metadata)
    log = {
        "source_file": pdf.as_posix(),
        "pages_total": len(pages),
        "pages_extracted": sum(bool(page.text.strip()) for page in pages),
        "pages_empty": sum(page.extraction_quality == "empty" for page in pages),
        "pages_requires_ocr": sum(page.requires_ocr for page in pages),
        "chunks_created": len(chunks),
        "book_detection_result": sorted({record["book"] for record in annotate_structure(pages) if record["book"]}),
        "duration_seconds": round(perf_counter() - started, 3),
        "errors": [],
    }
    return pages, log, chunks


def write_processed(output: Path, pages: list, chunks: list, log: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pages.jsonl").open("w", encoding="utf-8") as stream:
        for page in pages:
            stream.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    (output / "metadata.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"corpus_id", "author", "work", "language", "source", "books"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"corpus manifest missing fields: {sorted(missing)}")
    return manifest

def ingest_epub(epub: Path, manifest_path: Path) -> tuple[list, dict[str, object], list]:
    """Ingest an offline EPUB with manifest-supplied provenance; PDF behaviour remains unchanged."""
    started = perf_counter()
    manifest = load_corpus_manifest(manifest_path)
    pages = load_epub(epub)
    metadata = {
        "author": manifest["author"], "work": manifest["work"], "language": manifest["language"],
        "source": manifest["source"], "corpus_id": manifest["corpus_id"],
        "provenance": "manifest_declared_epub", "source_reference": manifest["source"],
        "source_file": epub.as_posix(), "source_type": "primary_source_epub",
        "fingerprint": sha256(epub.read_bytes()).hexdigest(),
    }
    chunks = structure_aware(annotate_structure(pages), metadata)
    log = {"source_file": epub.as_posix(), "corpus_id": manifest["corpus_id"], "sections_total": len(pages), "chunks_created": len(chunks), "book_detection_result": sorted({chunk.metadata["book"] for chunk in chunks}), "duration_seconds": round(perf_counter() - started, 3), "errors": []}
    return pages, log, chunks
