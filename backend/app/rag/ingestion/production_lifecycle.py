"""Dependency-injected production wiring for bounded corpus ingestion.

This module deliberately constructs no Chroma client and loads no embedding model.
The caller supplies both adapters, which makes the same lifecycle path testable offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .fake_lifecycle_runner import JsonStore, PersistedRunner
from .lifecycle import RuntimeIdentity, WriterLock, LifecycleError, read_json
from .lifecycle_commands import LifecycleCommandHandler
from .persisted_coordinator import PersistedIngestionCoordinator


@dataclass(frozen=True)
class CorpusIngestionConfig:
    host: str
    port: int
    collection_name: str
    state_dir: Path
    state_prefix: str
    document_ids: tuple[str, ...]
    batch_size: int = 128
    max_batches: int | None = None
    max_documents: int | None = None

    def state_path(self, name: str) -> Path:
        return self.state_dir / f"{name}_{self.state_prefix}.json"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / f"ingestion_writer_{self.state_prefix}.lock"


class ChromaHttpVectorStoreAdapter:
    """Adapter over an injected collection; never opens an HTTP connection itself."""
    def __init__(self, collection): self.collection = collection
    def get_document_ids(self, document_id): return set(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
    def upsert(self, chunks, vectors):
        if not vectors: raise LifecycleError('empty_embeddings')
        if len(chunks)!=len(vectors): raise LifecycleError('embedding_batch_size_mismatch')
        if any(not vector or not all(isinstance(value,(int,float)) for value in vector) for vector in vectors): raise LifecycleError('invalid_embedding_vector')
        self.collection.upsert(ids=[c.id for c in chunks], documents=[c.text for c in chunks], metadatas=[_metadata(c) for c in chunks], embeddings=vectors)
    def read_back(self, ids): return set(self.collection.get(ids=list(ids), include=[])["ids"])
    def count(self): return self.collection.count()
    def document_count(self, document_id): return len(self.get_document_ids(document_id))


class SentenceTransformerEmbeddingAdapter:
    """Adapter over an injected provider; provider construction remains outside this module."""
    def __init__(self, provider): self.provider = provider
    def embed(self, chunks): return self.provider.embed(["passage: " + chunk.text for chunk in chunks])


def _metadata(chunk):
    value=dict(chunk.metadata); value["navigation_path_json"]=json.dumps(value.pop("navigation_path", []),ensure_ascii=False)
    return {key: "" if item is None else item for key,item in value.items()}


class CorpusDocumentSource:
    """Stable ordered source with explicit, generic document selection."""
    def __init__(self, documents, chunk_loader): self.documents=tuple(documents); self.chunk_loader=chunk_loader
    def selected(self, document_ids):
        if not document_ids: raise ValueError("document_id_required")
        wanted=set(document_ids); selected=[document for document in self.documents if document.document_id in wanted]
        if {document.document_id for document in selected} != wanted: raise ValueError("unknown_document_id")
        return [self.chunk_loader(document) for document in selected]


def build_production_handler(config, source, store, embedding, *, identity, lock_exists=lambda _pid: False, events=None):
    """Build the real control plane with injected adapters, without invoking them."""
    runtime=JsonStore(config.state_path("ingestion_runtime")); progress=JsonStore(config.state_path("ingestion_progress")); manifest=JsonStore(config.state_path("ingestion_manifest")); lock=WriterLock(config.lock_path,lock_exists); trace=[] if events is None else events
    def runner_factory(): return PersistedRunner(embedding,store,progress,manifest,trace)
    coordinator=PersistedIngestionCoordinator(runner_factory,progress,manifest,store,runtime,lock,identity,config.collection_name)
    return LifecycleCommandHandler(runtime,progress,manifest,lock,store,coordinator=coordinator), source.selected(config.document_ids)


class ProductionLifecycleBackend:
    """Lazy facade used by the CLI: expensive factories are resume-only."""
    def __init__(self, config, source_factory, store_factory, embedding_factory, *, identity, lock_exists=lambda _pid: False):
        self.config,self.source_factory,self.store_factory,self.embedding_factory=config,source_factory,store_factory,embedding_factory
        self.identity,self.lock_exists=identity,lock_exists
        self.runtime=JsonStore(config.state_path("ingestion_runtime")); self.progress=JsonStore(config.state_path("ingestion_progress")); self.manifest=JsonStore(config.state_path("ingestion_manifest")); self.lock=WriterLock(config.lock_path,lock_exists)
    def status(self): return {'runtime':self.runtime.read(),'progress':self.progress.read(),'manifest':self.manifest.read(),'lock':self.lock.path.read_bytes() if self.lock.path.exists() else None}
    def stop(self):
        self.config.state_dir.mkdir(parents=True,exist_ok=True)
        state=self.runtime.read(); state.setdefault('writer',{})['stop_requested']=True; self.runtime.write(state)
    def resume(self, _unused=None, **kwargs):
        if self.lock.path.exists():
            owner=read_json(self.lock.path)
            if owner and self.lock_exists(owner.get('pid')): raise LifecycleError('writer_lock_active')
        self.config.state_dir.mkdir(parents=True,exist_ok=True)
        source=self.source_factory()
        documents=source.selected(self.config.document_ids)
        store=self.store_factory(); embedding=self.embedding_factory()
        handler,_=build_production_handler(self.config,source,store,embedding,identity=self.identity,lock_exists=self.lock_exists)
        return handler.resume(documents,batch_size=self.config.batch_size,max_batches=kwargs.get('max_batches',self.config.max_batches),max_documents=kwargs.get('max_documents',self.config.max_documents))


def build_production_lifecycle_backend(config, source_factory, store_factory, embedding_factory, *, identity, lock_exists=lambda _pid: False):
    return ProductionLifecycleBackend(config,source_factory,store_factory,embedding_factory,identity=identity,lock_exists=lock_exists)


def default_backend_factory(config, args):
    """Real deployment factory. All costly/external resources stay inside resume-only closures."""
    import os
    def source_factory():
        from .corpus_registry import CorpusRegistry
        from .corpus_dry_run import document_chunks
        from .generic_epub import load_epub_sections
        registry=CorpusRegistry.from_file(args.registry)
        def load(document):
            sections,_=load_epub_sections(args.incoming / document.filename,document)
            return document_chunks(sections,document)
        return CorpusDocumentSource(registry.documents,load)
    def store_factory():
        import chromadb
        client=chromadb.HttpClient(host=config.host,port=config.port)
        return ChromaHttpVectorStoreAdapter(client.get_collection(config.collection_name))
    def embedding_factory():
        from ..embeddings.provider import SentenceTransformerEmbeddingProvider
        return SentenceTransformerEmbeddingAdapter(SentenceTransformerEmbeddingProvider('intfloat/multilingual-e5-small','cpu',config.batch_size))
    return build_production_lifecycle_backend(config,source_factory,store_factory,embedding_factory,identity=RuntimeIdentity(os.getpid(),True,'roman-republic-ingestion',''))
