"""Historical corpus contract for offline evidence provenance isolation."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HistoricalCorpus(BaseModel):
    id: str
    name: str
    period: str
    region: str
    description: str
    source_refs: list[str] = Field(min_length=1)
