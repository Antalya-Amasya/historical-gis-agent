"""JSON-backed historical campaign ontology for deterministic route-intent resolution."""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


class HistoricalCampaignEntity(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    type: str = Field(min_length=1)
    period: str = Field(min_length=1)
    parent_campaign: str | None = None
    description: str = Field(min_length=1)
    source_references: list[str] = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    match_terms: list[str] = Field(default_factory=list)
    route_type: str = Field(min_length=1)
    route_context_id: str = Field(min_length=1)
    evidence_available: bool = False


class HistoricalCampaignOntology:
    """Loads reviewed entity metadata; it performs no retrieval, inference, or geocoding."""

    def __init__(self, entities: list[HistoricalCampaignEntity]) -> None:
        self.entities = tuple(entities)
        self._by_id = {entity.id: entity for entity in self.entities}
        if len(self._by_id) != len(self.entities):
            raise ValueError("campaign ontology entity ids must be unique")

    @classmethod
    def from_file(cls, path: str | Path) -> "HistoricalCampaignOntology":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([HistoricalCampaignEntity.model_validate(item) for item in payload["entities"]])

    @classmethod
    def default(cls) -> "HistoricalCampaignOntology":
        return cls.from_file(Path(__file__).parent / "registries" / "campaigns.json")

    def get(self, entity_id: str | None) -> HistoricalCampaignEntity | None:
        return self._by_id.get(entity_id or "")

    def match(self, message: str) -> HistoricalCampaignEntity | None:
        query = self._normalize(message)
        scored: list[tuple[int, HistoricalCampaignEntity]] = []
        for entity in self.entities:
            if not entity.evidence_available:
                continue
            alias_scores = [len(self._normalize(alias)) for alias in entity.aliases if self._normalize(alias) in query]
            terms_match = bool(entity.match_terms) and all(self._normalize(term) in query for term in entity.match_terms)
            if alias_scores:
                scored.append((100 + max(alias_scores), entity))
            elif terms_match:
                scored.append((50 + sum(len(self._normalize(term)) for term in entity.match_terms), entity))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return scored[0][1]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", text.casefold())
