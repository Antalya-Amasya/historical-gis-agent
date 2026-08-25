"""Build evidence-preserving campaign chains from manually reviewed registry records."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .event_registry import HistoricalEventRegistry


class CampaignChainConfig(BaseModel):
    chain_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    period: str = Field(min_length=1)
    description: str = Field(min_length=1)
    events: list[str] = Field(min_length=1)


class HistoricalCampaignStep(BaseModel):
    order: int = Field(ge=1)
    event_id: str
    title: str
    involved_places: list[str] = Field(min_length=1)
    period: str
    corpus_id: str
    source_book: str
    source_chapter: str
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class HistoricalCampaignChain(BaseModel):
    chain_id: str
    title: str
    corpus_id: str
    period: str
    description: str
    event_ids: list[str] = Field(min_length=1)
    steps: list[HistoricalCampaignStep] = Field(min_length=1)


class HistoricalEventChainBuilder:
    """Copies only registry-approved event facts in caller-supplied configuration order."""

    def build(self, registry: HistoricalEventRegistry, config: CampaignChainConfig) -> HistoricalCampaignChain:
        if len(config.events) != len(set(config.events)):
            raise ValueError("campaign chain event ids must be unique")
        steps = []
        for order, event_id in enumerate(config.events, start=1):
            record = registry.get(event_id)
            if record.corpus_id != config.corpus_id:
                raise ValueError(f"event {event_id} belongs to corpus {record.corpus_id}, not {config.corpus_id}")
            steps.append(HistoricalCampaignStep(
                order=order, event_id=record.event_id, title=record.title,
                involved_places=list(record.involved_places), period=record.period,
                corpus_id=record.corpus_id, source_book=record.source_book,
                source_chapter=record.source_chapter, evidence_refs=list(record.evidence_refs),
                confidence=record.confidence,
            ))
        return HistoricalCampaignChain(
            chain_id=config.chain_id, title=config.title, corpus_id=config.corpus_id,
            period=config.period, description=config.description,
            event_ids=list(config.events), steps=steps,
        )

    def from_json(self, registry: HistoricalEventRegistry, path: Path) -> HistoricalCampaignChain:
        return self.build(registry, CampaignChainConfig.model_validate(json.loads(path.read_text(encoding="utf-8"))))
