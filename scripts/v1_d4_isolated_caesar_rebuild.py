"""Isolated one-document rebuild smoke.

Defaults preserve the V1-D4 Caesar / Gutenberg 10657 experiment. Optional
CLI flags select a second Gutenberg edition without changing production
ingestion. This script never writes roman_republic_primary_sources_v2 and
never connects to Chroma on port 8002 for ingestion.
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
from urllib.parse import urljoin
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

DEFAULT_EDITION_PAGE = "https://www.gutenberg.org/ebooks/10657"
DEFAULT_DOCUMENT_ID = "caesar_gallic_civil_wars"
DEFAULT_COLLECTION = "v1_d4_caesar_10657_smoke"
DEFAULT_OLD_LOCAL_SHA256 = "4840578daed0627bffcaad09feedd6322e22f41f3e7465040854f7f88ef97e77"
DEFAULT_AUTHOR = "Julius Caesar"
DEFAULT_WORK = "Gallic War + Civil War"
DEFAULT_QUERY = "All Gaul is divided into three parts"
MODEL = "intfloat/multilingual-e5-small"


def gutenberg_ebook_id(edition_page: str) -> str:
    return edition_page.rstrip("/").rsplit("/", 1)[-1]


def catalog_noimages_href(ebook_id: str) -> str:
    return f"/ebooks/{ebook_id}.epub.noimages"


def expected_opf_identifier(ebook_id: str) -> str:
    return f"http://www.gutenberg.org/{ebook_id}"


def opf_matches_edition(identity: dict[str, object], ebook_id: str) -> bool:
    joined = " ".join(identity.get("identifier") or [])
    return expected_opf_identifier(ebook_id) in joined or ebook_id in joined


def _tag(name: str) -> str:
    return name.rsplit("}", 1)[-1].lower()


def fetch_catalog_html(edition_page: str) -> tuple[str, str]:
    req = urllib.request.Request(edition_page, headers={"User-Agent": "historical-gis-v1-d4-smoke/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as response:
        return response.geturl(), response.read().decode("utf-8", "replace")


def catalog_noimages_url(html: str, edition_page: str) -> str:
    href = catalog_noimages_href(gutenberg_ebook_id(edition_page))
    if href not in html:
        raise SystemExit(f"catalog page does not advertise the no-images EPUB href {href}")
    return urljoin(edition_page, href)


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


def source_document(
    filename: str,
    *,
    document_id: str,
    author: str,
    work: str,
    source_url: str,
) -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        author=author,
        work=work,
        language="en",
        source_type="primary_source",
        filename=filename,
        source_url=source_url,
        license="Public domain in the USA.",
        enabled=True,
        parser_hints={},
    )


def caesar_document(filename: str) -> CorpusDocument:
    return source_document(
        filename,
        document_id=DEFAULT_DOCUMENT_ID,
        author=DEFAULT_AUTHOR,
        work=DEFAULT_WORK,
        source_url=DEFAULT_EDITION_PAGE,
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


def ingest_isolated(
    work: Path,
    incoming: Path,
    document: CorpusDocument,
    chunks,
    provider,
    *,
    collection_name: str,
) -> dict[str, object]:
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
    if collection_name in names:
        raise SystemExit(f"refusing to reuse existing collection {collection_name} in {chroma_dir}")
    collection = client.create_collection(
        collection_name,
        metadata={"experiment": "isolated-rebuild", "hnsw:space": "cosine"},
    )
    store = ChromaHttpVectorStoreAdapter(collection)
    source = CorpusDocumentSource([document], lambda _: chunks)
    config = CorpusIngestionConfig(
        "127.0.0.1",
        18002,
        collection_name,
        state_dir,
        "smoke",
        (document.document_id,),
        batch_size=32,
    )
    embedding = SentenceTransformerEmbeddingAdapter(provider)
    handler, selected = build_production_handler(
        config,
        source,
        store,
        embedding,
        identity=RuntimeIdentity(os.getpid(), True, "isolated-rebuild", datetime.now(timezone.utc).isoformat()),
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
    parser = argparse.ArgumentParser(description="Isolated Gutenberg one-document rebuild smoke")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TEMP", ".")) / "v1-d4-caesar-10657")
    parser.add_argument("--epub", type=Path)
    parser.add_argument("--ingest", action="store_true", help="Embed into an isolated PersistentClient collection")
    parser.add_argument("--edition-page", default=DEFAULT_EDITION_PAGE)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--old-sha256", default=DEFAULT_OLD_LOCAL_SHA256)
    parser.add_argument("--expected-sha256", help="Reject the downloaded file if SHA-256 does not match")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    args = parser.parse_args(argv)
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    ebook_id = gutenberg_ebook_id(args.edition_page)
    href = catalog_noimages_href(ebook_id)
    edition_url = args.edition_page
    download_url = None
    resolved = None

    epub = args.epub or (work / f"pg{ebook_id}.epub.noimages")
    if args.epub:
        data = epub.read_bytes()
        resolved = str(epub)
    else:
        edition_url, html = fetch_catalog_html(args.edition_page)
        download_url = catalog_noimages_url(html, edition_url)
        data, resolved = download_epub(download_url, epub)
    digest = hashlib.sha256(data).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit(f"input SHA-256 mismatch: got {digest}, expected {args.expected_sha256}")
    identity = opf_identity(epub)
    if not opf_matches_edition(identity, ebook_id):
        raise SystemExit(f"OPF identifier is not Gutenberg {ebook_id}: {identity['identifier']}")

    document = source_document(
        epub.name,
        document_id=args.document_id,
        author=args.author,
        work=args.work,
        source_url=args.edition_page,
    )
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
        "catalog_download_href": href,
        "download_url": download_url,
        "resolved_download_url": resolved,
        "bytes": len(data),
        "sha256": digest,
        "old_local_sha256": args.old_sha256,
        "byte_equal_to_old_local": digest == args.old_sha256,
        "opf": identity,
        "document_id": document.document_id,
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
    first = ingest_isolated(work / "run_a", incoming, document, chunks, provider, collection_name=args.collection)
    second = ingest_isolated(work / "run_b", incoming, document, chunks, provider, collection_name=args.collection)
    hits = first["collection"].query(
        query_embeddings=provider.embed([canonical_e5_query(args.query)]),
        n_results=5,
        include=["documents", "metadatas"],
    )
    report["ingest"] = {
        "collection": args.collection,
        "model": MODEL,
        "dimension": provider.dimensions,
        "run_a_count": first["count"],
        "run_b_count": second["count"],
        "ids_equal": sorted(first["ids"]) == sorted(second["ids"]),
        "text_equal": dict(zip(first["ids"], first["texts"])) == dict(zip(second["ids"], second["texts"])),
        "document_ids": first["document_ids"],
        "retrieval_query": args.query,
        "retrieval_ids": hits["ids"][0],
        "retrieval_document_ids": [item.get("document_id") for item in hits["metadatas"][0]],
        "retrieval_source_urls": [item.get("source_url") for item in hits["metadatas"][0]],
    }
    (work / "source_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ingest", report["ingest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
