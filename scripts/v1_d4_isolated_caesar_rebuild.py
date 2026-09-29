"""Isolated one-document rebuild smoke for Gutenberg 10657.

This script never writes roman_republic_primary_sources_v2 and never
connects to Chroma on port 8002 for ingestion. Download and persistence
stay under an explicit work directory (default: a temp folder).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.rag.ingestion.corpus_dry_run import (
    CHUNKER_VERSION,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    PARSER_VERSION,
    document_chunks,
)
from backend.app.rag.ingestion.corpus_registry import CorpusDocument
from backend.app.rag.ingestion.generic_epub import load_epub_sections

EDITION_PAGE = "https://www.gutenberg.org/ebooks/10657"
CATALOG_NOIMAGES = "/ebooks/10657.epub.noimages"
DOCUMENT_ID = "caesar_gallic_civil_wars"
COLLECTION = "v1_d4_caesar_10657_smoke"
OLD_LOCAL_SHA256 = "4840578daed0627bffcaad09feedd6322e22f41f3e7465040854f7f88ef97e77"
MODEL = "intfloat/multilingual-e5-small"


def _tag(name: str) -> str:
    return name.rsplit("}", 1)[-1].lower()


def fetch_catalog_html(edition_page: str) -> tuple[str, str]:
    req = urllib.request.Request(edition_page, headers={"User-Agent": "historical-gis-v1-d4-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as response:
        return response.geturl(), response.read().decode("utf-8", "replace")


def catalog_noimages_url(html: str, edition_page: str) -> str:
    if CATALOG_NOIMAGES not in html:
        raise SystemExit("catalog page does not advertise the no-images EPUB href /ebooks/10657.epub.noimages")
    from urllib.parse import urljoin
    return urljoin(edition_page, CATALOG_NOIMAGES)


def download_epub(url: str, dest: Path) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "historical-gis-v1-d4-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as response:
        data = response.read()
        dest.write_bytes(data)
        return data, response.geturl()


def opf_identity(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        opf = next(item.attrib["full-path"] for item in container.iter() if _tag(item.tag) == "rootfile")
        package = ElementTree.fromstring(archive.read(opf))
        fields = {"identifier": [], "title": [], "creator": []}
        for element in package.iter():
            name = _tag(element.tag)
            text = " ".join((element.text or "").split())
            if text and name in fields:
                fields[name].append(text)
    return {"opf": opf, **fields}


def caesar_document(filename: str) -> CorpusDocument:
    return CorpusDocument(
        document_id=DOCUMENT_ID,
        author="Julius Caesar",
        work="Gallic War + Civil War",
        language="en",
        source_type="primary_source",
        filename=filename,
        source_url=EDITION_PAGE,
        license="Public domain in the USA.",
        enabled=True,
        parser_hints={},
    )


def parse_twice(epub: Path, document: CorpusDocument) -> dict[str, object]:
    def once():
        sections, spine = load_epub_sections(epub, document)
        chunks = document_chunks(sections, document)
        return sections, spine, chunks

    first_sections, spine, first = once()
    second_sections, _, second = once()
    return {
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunk_chars": DEFAULT_CHUNK_CHARS,
        "overlap_chars": DEFAULT_OVERLAP_CHARS,
        "spine_count": spine,
        "section_count": len(first_sections),
        "chunk_count": len(first),
        "section_order_equal": [item.spine_item for item in first_sections] == [item.spine_item for item in second_sections],
        "section_text_equal": [item.text for item in first_sections] == [item.text for item in second_sections],
        "chunk_count_equal": len(first) == len(second),
        "chunk_ids_equal": [item.id for item in first] == [item.id for item in second],
        "chunk_text_equal": [item.text for item in first] == [item.text for item in second],
        "provenance_equal": [item.metadata for item in first] == [item.metadata for item in second],
        "chunk_ids": [item.id for item in first],
        "chunks": first,
    }


def ingest_isolated(work: Path, incoming: Path, document: CorpusDocument, chunks, provider) -> dict[str, object]:
    import chromadb
    from backend.app.rag.ingestion.lifecycle import RuntimeIdentity
    from backend.app.rag.ingestion.production_lifecycle import (
        ChromaHttpVectorStoreAdapter,
        CorpusDocumentSource,
        CorpusIngestionConfig,
        SentenceTransformerEmbeddingAdapter,
        build_production_handler,
    )

    chroma_dir = work / "chroma"
    state_dir = work / "state"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    names = [item.name for item in client.list_collections()]
    if COLLECTION in names:
        raise SystemExit(f"refusing to reuse existing collection {COLLECTION} in {chroma_dir}")
    collection = client.create_collection(COLLECTION, metadata={"experiment": "v1-d4", "hnsw:space": "cosine"})
    store = ChromaHttpVectorStoreAdapter(collection)
    source = CorpusDocumentSource([document], lambda _: chunks)
    config = CorpusIngestionConfig("127.0.0.1", 18002, COLLECTION, state_dir, "v1d4", (DOCUMENT_ID,), batch_size=32)
    embedding = SentenceTransformerEmbeddingAdapter(provider)
    handler, selected = build_production_handler(
        config,
        source,
        store,
        embedding,
        identity=RuntimeIdentity(os.getpid(), True, "v1-d4-isolated", datetime.now(timezone.utc).isoformat()),
    )
    handler.resume(selected, batch_size=32)
    records = collection.get(include=["documents", "metadatas"])
    return {
        "client": client,
        "collection": collection,
        "count": collection.count(),
        "ids": records["ids"],
        "texts": records["documents"],
        "metadatas": records["metadatas"],
        "document_ids": sorted({(meta or {}).get("document_id") for meta in records["metadatas"]}),
        "names": [item.name for item in client.list_collections()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated Gutenberg 10657 rebuild smoke")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TEMP", ".")) / "v1-d4-caesar-10657")
    parser.add_argument("--epub", type=Path)
    parser.add_argument("--ingest", action="store_true", help="Embed into an isolated PersistentClient collection")
    args = parser.parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    edition_url, html = fetch_catalog_html(EDITION_PAGE)
    download_url = catalog_noimages_url(html, edition_url)
    epub = args.epub or (work / "pg10657.epub.noimages")
    if args.epub:
        data = epub.read_bytes()
        resolved = str(epub)
    else:
        data, resolved = download_epub(download_url, epub)
    digest = hashlib.sha256(data).hexdigest()
    identity = opf_identity(epub)
    if "http://www.gutenberg.org/10657" not in identity["identifier"] and "10657" not in " ".join(identity["identifier"]):
        raise SystemExit(f"OPF identifier is not Gutenberg 10657: {identity['identifier']}")

    document = caesar_document(epub.name)
    incoming = work / "incoming"
    incoming.mkdir(exist_ok=True)
    incoming_epub = incoming / epub.name
    incoming_epub.write_bytes(data)
    (work / "registry.json").write_text(
        json.dumps({"registry_version": 1, "documents": [document.model_dump()]}, indent=2),
        encoding="utf-8",
    )
    parsed = parse_twice(incoming_epub, document)
    chunks = parsed.pop("chunks")
    report = {
        "edition_page": edition_url,
        "catalog_download_href": CATALOG_NOIMAGES,
        "download_url": download_url,
        "resolved_download_url": resolved,
        "bytes": len(data),
        "sha256": digest,
        "old_local_sha256": OLD_LOCAL_SHA256,
        "byte_equal_to_old_local": digest == OLD_LOCAL_SHA256,
        "opf": identity,
        "parse": parsed,
    }
    print(json.dumps({k: report[k] for k in report if k != "parse"}, indent=2))
    print("parse", {k: parsed[k] for k in parsed if k != "chunk_ids"})
    if not args.ingest:
        (work / "source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
    from backend.app.rag.http_store import canonical_e5_query

    provider = SentenceTransformerEmbeddingProvider(MODEL, "cpu", 16)
    first = ingest_isolated(work / "run_a", incoming, document, chunks, provider)
    second = ingest_isolated(work / "run_b", incoming, document, chunks, provider)
    query = "All Gaul is divided into three parts"
    hits = first["collection"].query(query_embeddings=provider.embed([canonical_e5_query(query)]), n_results=5, include=["documents", "metadatas"])
    report["ingest"] = {
        "collection": COLLECTION,
        "model": MODEL,
        "dimension": provider.dimensions,
        "run_a_count": first["count"],
        "run_b_count": second["count"],
        "ids_equal": sorted(first["ids"]) == sorted(second["ids"]),
        "document_ids": first["document_ids"],
        "retrieval_ids": hits["ids"][0],
        "retrieval_document_ids": [item.get("document_id") for item in hits["metadatas"][0]],
        "retrieval_source_urls": [item.get("source_url") for item in hits["metadatas"][0]],
    }
    (work / "source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ingest", report["ingest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
