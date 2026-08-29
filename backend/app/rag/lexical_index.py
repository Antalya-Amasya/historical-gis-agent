"""Read-only passage-level lexical candidates derived from Chroma source chunks."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math, re
from typing import Any
from .evidence_ranking import normalized_tokens

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_ACTIONS = frozenset({"assassination", "assassinate", "assassinated", "murder", "murdered", "slain", "killed", "stabbed"})

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
    def __init__(self, collection): self.collection=collection; self._documents=self._postings=None; self.passage_count=0
    def _build(self):
        if self._documents is not None: return
        rows=self.collection.get(include=["documents","metadatas"]); documents={}; postings=defaultdict(dict)
        for sid,text,meta in zip(rows["ids"],rows["documents"],rows["metadatas"]):
            for p in derive_passages(sid,text or "",dict(meta or {})):
                documents[p.id]=p
                for term in normalized_tokens(p.text+" "+str(p.metadata.get("heading",""))): postings[term][p.id]=1
        self._documents,self._postings,self.passage_count=documents,dict(postings),len(documents)
    def query(self, query: str, top_k: int, filters: dict[str,str]|None=None)->list[LexicalCandidate]:
        self._build(); terms=normalized_tokens(query)
        if terms&_ACTIONS: terms|=_ACTIONS
        scores=defaultdict(float); total=len(self._documents)
        for term in terms:
            post=self._postings.get(term,{})
            weight=math.log(1+(total+.5)/(len(post)+.5)) if post else 0
            for ident in post: scores[ident]+=weight
        out=[]
        for ident,score in scores.items():
            p=self._documents[ident]
            if filters and any(p.metadata.get(k)!=v for k,v in filters.items() if v): continue
            out.append(LexicalCandidate(p.id,p.text,p.metadata,score))
        return sorted(out,key=lambda p:(-p.lexical_score,p.id))[:top_k]
