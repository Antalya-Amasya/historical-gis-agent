"""Data-driven inventory of source documents; deliberately separate from campaign ontology."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class CorpusDocument(BaseModel):
    document_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    work: str = Field(min_length=1)
    volume: str | None = None
    language: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    period_start_bce: int | None = None
    period_end_bce: int | None = None
    source_url: str | None = None
    license: str | None = None
    enabled: bool = True
    parser_hints: dict[str, object] = Field(default_factory=dict)


class CorpusRegistry(BaseModel):
    registry_version: int
    documents: list[CorpusDocument]

    @classmethod
    def from_file(cls, path: Path) -> "CorpusRegistry":
        registry = cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        ids = [document.document_id for document in registry.documents]
        filenames = [document.filename for document in registry.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("corpus registry has duplicate document_id values")
        if len(filenames) != len(set(filenames)):
            raise ValueError("corpus registry maps a filename more than once")
        return registry

    def by_filename(self) -> dict[str, CorpusDocument]:
        return {document.filename: document for document in self.documents}
