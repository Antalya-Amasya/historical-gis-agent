"""Legacy direct ingestion path; unsafe for production lifecycle use.

The production CLI uses the managed lifecycle path in ``production_lifecycle``.
This module is retained only for historical/debug compatibility.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from ..embeddings.provider import SentenceTransformerEmbeddingProvider
from .corpus_dry_run import CHUNKER_VERSION, PARSER_VERSION, document_chunks
from .corpus_registry import CorpusRegistry
from .generic_epub import load_epub_sections

COLLECTION = "roman_republic_primary_sources_v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(chunk):
    value = dict(chunk.metadata)
    value["navigation_path_json"] = json.dumps(value.pop("navigation_path"), ensure_ascii=False)
    for key, item in list(value.items()):
        if item is None: value[key] = ""
    return value


def ingest(incoming: Path, registry_path: Path, chroma_path: Path, manifest_path: Path, *, batch_size: int = 16, host: str | None = None, port: int | None = None) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    registry = CorpusRegistry.from_file(registry_path)
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"documents": {}}
    provider = SentenceTransformerEmbeddingProvider("intfloat/multilingual-e5-small", "auto", batch_size)
    client = chromadb.HttpClient(host=host, port=port) if host and port else chromadb.PersistentClient(path=str(chroma_path))
    metadata = {"corpus_scope":"roman_republic_primary_sources","collection_version":"v1","embedding_provider":"sentence_transformers","embedding_model":provider.model_name,"embedding_dimension":provider.dimensions,"parser_version":PARSER_VERSION,"chunker_version":CHUNKER_VERSION,"document_count":16}
    collection = client.get_or_create_collection(COLLECTION, metadata=metadata)
    progress_path = manifest_path.with_name("ingestion_progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"documents": {}}
    results=[]; next_docs={}
    registered = {d.document_id for d in registry.documents if d.enabled}
    removed = sorted(set(old.get("documents", {})) - registered)
    for position, document in enumerate((d for d in registry.documents if d.enabled), 1):
        path=incoming / document.filename; file_hash=_hash(path); prior=old.get("documents",{}).get(document.document_id)
        if prior and prior.get("file_hash")==file_hash and prior.get("parser_version")==PARSER_VERSION and prior.get("chunker_version")==CHUNKER_VERSION and prior.get("embedding_model")==provider.model_name:
            next_docs[document.document_id]=prior; results.append({"document_id":document.document_id,"status":"skipped","chunk_count":prior["chunk_count"]}); continue
        sections,_=load_epub_sections(path,document); chunks=document_chunks(sections,document)
        completed = set(progress["documents"].get(document.document_id, {}).get("completed_chunk_ids", []))
        existing = collection.get(where={"document_id": document.document_id}, include=[])["ids"]
        if completed and completed != set(existing): raise RuntimeError(f"progress_collection_mismatch:{document.document_id}")
        if prior: collection.delete(where={"document_id":document.document_id})
        for start in range(0,len(chunks),batch_size):
            batch=[x for x in chunks[start:start+batch_size] if x.id not in completed]
            if not batch:
                continue
            collection.upsert(ids=[x.id for x in batch],documents=[x.text for x in batch],metadatas=[_metadata(x) for x in batch],embeddings=provider.embed([("passage: "+x.text) for x in batch]))
            probes=[batch[0],batch[len(batch)//2],batch[-1]]; read=collection.get(ids=[x.id for x in probes],include=["documents","metadatas"])
            if set(read["ids"]) != {x.id for x in probes}: raise RuntimeError(f"batch_readback_failed:{document.document_id}:{start}")
            completed.update(x.id for x in batch)
            progress["documents"][document.document_id]={"document_id":document.document_id,"expected_chunks":len(chunks),"completed_chunks":len(completed),"completed_batches":(start//batch_size)+1,"last_completed_chunk_id":batch[-1].id,"completed_chunk_ids":sorted(completed),"file_hash":file_hash,"parser_version":PARSER_VERSION,"chunker_version":CHUNKER_VERSION,"embedding_model":provider.model_name,"status":"in_progress"}
            temp=progress_path.with_suffix(".tmp"); temp.write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding="utf-8"); temp.replace(progress_path)
        record={"document_id":document.document_id,"file_hash":file_hash,"parser_version":PARSER_VERSION,"chunker_version":CHUNKER_VERSION,"embedding_model":provider.model_name,"chunk_count":len(chunks),"chunk_id_fingerprint":hashlib.sha256("".join(x.id for x in chunks).encode()).hexdigest(),"ingested_at":datetime.now(timezone.utc).isoformat(),"collection":COLLECTION,"status":"ingested"}
        next_docs[document.document_id]=record; results.append({"document_id":document.document_id,"status":"ingested","chunk_count":len(chunks),"position":position})
        progress["documents"][document.document_id]["status"]="completed"
        partial = {"collection": COLLECTION, "collection_path": str(chroma_path), "embedding_provider": "sentence_transformers", "embedding_model": provider.model_name, "embedding_device": provider.device, "embedding_dimension": provider.dimensions, "documents": next_docs, "removed_documents_reported": removed, "results": results, "vector_count": collection.count()}
        manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"document {position}/16 {document.document_id}: chunks={len(chunks)} upserted elapsed=complete", flush=True)
    report={"collection":COLLECTION,"collection_path":str(chroma_path),"embedding_provider":"sentence_transformers","embedding_model":provider.model_name,"embedding_device":provider.device,"embedding_dimension":provider.dimensions,"documents":next_docs,"removed_documents_reported":removed,"results":results,"vector_count":collection.count()}
    manifest_path.parent.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
