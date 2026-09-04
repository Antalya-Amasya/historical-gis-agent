"""Read-only passage-level lexical candidates derived from Chroma source chunks."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math, re
from typing import Any
from backend.app.rag.query_roles import (
    analyze_query,
    episode_context_terms,
    movement_scoring_terms,
    normalized_tokens,
    route_movement_query,
)

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_GENERIC_TERM_WEIGHT = 0.25

@dataclass(frozen=True)
class LexicalCandidate:
    id: str
    text: str
    metadata: dict[str, Any]
    lexical_score: float

def derive_passages(source_chunk_id: str, text: str, metadata: dict[str, Any]) -> list[LexicalCandidate]:
    """Stable two-sentence windows, each an exact source-text substring."""
    sentences=[(m.start(),m.end()) for m in _SENTENCE.finditer(text) if m.group().strip()]
    if not sentences:
        return [] if not text else [LexicalCandidate(f"{source_chunk_id}:0:{len(text)}",text,{**metadata,"source_chunk_id":source_chunk_id,"passage_start":0,"passage_end":len(text),"passage_index":0},0.0)]
    result=[]
    for i,(start,end) in enumerate(sentences):
        a=sentences[max(0,i-1)][0]
        if end-a>1200: a=start
        result.append(LexicalCandidate(f"{source_chunk_id}:{a}:{end}",text[a:end],{**metadata,"source_chunk_id":source_chunk_id,"passage_start":a,"passage_end":end,"passage_index":i},0.0))
    return result

class LexicalEvidenceIndex:
    """Immutable, read-only lexical view of source chunks.

    Passage text and metadata are deliberately not retained per passage.  The
    index stores a source identifier and offsets, then materializes exact text
    only for the small final lexical candidate set.
    """
    def __init__(self, collection):
        self.collection=collection
        self._sources=self._passages=self._postings=None
        self.passage_count=0

    def _build(self):
        if self._sources is not None: return
        rows=self.collection.get(include=["documents","metadatas"])
        sources={}; passages={}; postings=defaultdict(set)
        for sid,text,meta in zip(rows["ids"],rows["documents"],rows["metadatas"]):
            source_text, source_meta = text or "", dict(meta or {})
            sources[sid]=(source_text, source_meta)
            for p in derive_passages(sid,source_text,source_meta):
                passages[p.id]=(sid,p.metadata["passage_start"],p.metadata["passage_end"],p.metadata["passage_index"])
                for term in normalized_tokens(p.text+" "+str(source_meta.get("heading",""))): postings[term].add(p.id)
        # Publish only after every structure is complete: a failed build leaves
        # this instance retryable and never exposes a partial index.
        self._sources,self._passages,self._postings=sources,passages,dict(postings)
        self.passage_count=len(passages)

    def ensure_built(self) -> None:
        """Construct this immutable index once, if it has not been published."""
        self._build()

    def query(self, query: str, top_k: int, filters: dict[str,str]|None=None)->list[LexicalCandidate]:
        self.ensure_built()
        roles = analyze_query(query)
        query_norms = normalized_tokens(query)
        mov_terms = movement_scoring_terms(roles)
        scores=defaultdict(float); movement_scores=defaultdict(float); total=len(self._passages)
        def idf(term: str) -> float:
            post=self._postings.get(term,{})
            return math.log(1+(total+.5)/(len(post)+.5)) if post else 0.0
        core = roles.person_terms | roles.location_match_terms | roles.action_terms | roles.movement_inflection_terms
        for term in core:
            weight=idf(term)
            for ident in self._postings.get(term, {}):
                scores[ident]+=weight
                if term in mov_terms:
                    movement_scores[ident]+=weight
        for term in roles.generic_terms:
            weight=_GENERIC_TERM_WEIGHT * idf(term)
            for ident in self._postings.get(term, {}): scores[ident]+=weight
        if roles.person_terms:
            person_postings = [set(self._postings.get(term, {})) for term in roles.person_terms]
            person_ids = set().union(*person_postings) if person_postings else set()
            for term in roles.expanded_action_terms:
                weight=idf(term)
                for ident in self._postings.get(term, {}):
                    if ident in person_ids:
                        scores[ident]+=weight
                        movement_scores[ident]+=weight
        if route_movement_query(query_norms, roles):
            episode_terms = episode_context_terms(roles)
            if episode_terms:
                for ident, mov in movement_scores.items():
                    if mov <= 0:
                        continue
                    overlap = [term for term in episode_terms if ident in self._postings.get(term, {})]
                    if overlap:
                        scores[ident] += sum(idf(term) for term in overlap)
        out=[]
        for ident,score in scores.items():
            source_id,start,end,index=self._passages[ident]
            text,source_meta=self._sources[source_id]
            if filters and any(source_meta.get(k)!=v for k,v in filters.items() if v): continue
            metadata={**source_meta,"source_chunk_id":source_id,"passage_start":start,"passage_end":end,"passage_index":index}
            out.append(LexicalCandidate(ident,text[start:end],metadata,score))
        return sorted(out,key=lambda p:(-p.lexical_score,p.id))[:top_k]
