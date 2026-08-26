"""Canonical chunk accessors for lifecycle code."""
from typing import Protocol, Mapping
from .lifecycle import LifecycleError

class LifecycleChunk(Protocol):
    id: str
    text: str
    metadata: Mapping[str, object]

def get_document_id(chunk: LifecycleChunk) -> str:
    value=chunk.metadata.get('document_id')
    if not isinstance(value,str) or not value.strip(): raise LifecycleError('missing_document_id')
    return value

def validate_document_chunks(chunks):
    if not chunks: raise LifecycleError('missing_document_id')
    document_id=get_document_id(chunks[0])
    if any(get_document_id(chunk)!=document_id for chunk in chunks): raise LifecycleError('mixed_document_chunks')
    return document_id
