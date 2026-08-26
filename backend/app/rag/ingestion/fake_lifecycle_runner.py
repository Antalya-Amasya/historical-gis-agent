"""In-memory contract runner used only by lifecycle tests."""
from dataclasses import dataclass, field
from .lifecycle import DocumentState, LifecycleError, validate_resume
from .lifecycle import atomic_json, read_json
from .lifecycle_chunks import validate_document_chunks
from pathlib import Path
class JsonStore:
    def __init__(self,path:Path): self.path=path
    def read(self):
        value=read_json(self.path,{})
        for item in value.values():
            if isinstance(item,dict) and isinstance(item.get('ids'),list): item['ids']=set(item['ids'])
        return value
    def write(self,value):
        serial={k:({**v,'ids':sorted(v['ids'])} if isinstance(v,dict) and isinstance(v.get('ids'),set) else v) for k,v in value.items()}; atomic_json(self.path,serial)
@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str = ""
    metadata: dict = field(default_factory=dict)
    def __post_init__(self):
        if 'document_id' not in self.metadata: object.__setattr__(self,'metadata',{**self.metadata,'document_id':self.document_id})
@dataclass
class FakeEmbedding:
    embedded_chunk_ids:list[str]=field(default_factory=list)
    returned_vectors:list[list[float]]=field(default_factory=list)
    def embed(self,chunks):
        self.embedded_chunk_ids += [x.id for x in chunks]
        vectors=[[float(index+1),0.25,0.5] for index,_ in enumerate(chunks)]
        self.returned_vectors.extend(vectors); return vectors
@dataclass
class FakeStore:
    ids:set[str]=field(default_factory=set); upserted_chunk_ids:list[str]=field(default_factory=list); upserted_embeddings:list[list[float]]=field(default_factory=list); upsert_calls:int=0
    def upsert(self,chunks,vectors):
        self.upsert_calls+=1
        if not vectors: raise LifecycleError('empty_embeddings')
        if len(chunks)!=len(vectors): raise LifecycleError('embedding_batch_size_mismatch')
        self.ids.update(x.id for x in chunks); self.upserted_chunk_ids += [x.id for x in chunks]; self.upserted_embeddings.extend(vectors)
    def get(self,ids): return set(ids)&self.ids
    def get_document_ids(self,document_id):
        prefixed={item for item in self.ids if item.startswith(f"{document_id}:")}
        return prefixed if prefixed else set(self.ids)
    def read_back(self,ids): return self.get(ids)
class Runner:
    def __init__(self,embed,store,progress,manifest,events): self.embed,self.store,self.progress,self.manifest,self.events=embed,store,progress,manifest,events
    def run(self,chunks,batch_size,max_batches=None,stop=False,runtime_store=None,on_checkpoint=None):
        d=validate_document_chunks(chunks); state=self.progress.get(d,{"ids":set(),"state":DocumentState.PENDING}); done=set(state['ids']); expected={c.id for c in chunks}; server_ids=self.store.get_document_ids(d) if hasattr(self.store,'get_document_ids') else {x for x in self.store.ids if x in expected}; validate_resume(done,server_ids & expected); missing=[x for x in chunks if x.id not in done]; batches=0
        for i in range(0,len(missing),batch_size):
            if max_batches is not None and batches>=max_batches: break
            b=missing[i:i+batch_size]; vectors=self.embed.embed(b); self.events.append('embed')
            if not vectors: raise LifecycleError('empty_embeddings')
            if len(vectors)!=len(b): raise LifecycleError('embedding_batch_size_mismatch')
            if any(not vector or not all(isinstance(value,(int,float)) for value in vector) for vector in vectors): raise LifecycleError('invalid_embedding_vector')
            self.store.upsert(b,vectors); self.events.append('upsert')
            read_back=self.store.read_back([x.id for x in b]) if hasattr(self.store,'read_back') else self.store.get([x.id for x in b])
            if read_back!={x.id for x in b}: raise LifecycleError('readback_failed')
            self.events+=['readback','checkpoint']; done.update(x.id for x in b); self.progress[d]={"ids":done,"state":DocumentState.IN_PROGRESS}
            if hasattr(self,'progress_store'): self.progress_store.write(self.progress)
            batches+=1
            if on_checkpoint: on_checkpoint()
            if stop or (runtime_store and runtime_store.read().get('writer',{}).get('stop_requested')): break
        if len(done)==len(chunks): self.progress[d]['state']=DocumentState.COMPLETED; self.events.append('journal_completed'); self.manifest[d]='ingested'; self.events.append('manifest_written')
        return batches
class PersistedRunner(Runner):
    def __init__(self,embed,store,progress_store,manifest_store,events):
        self.progress_store,self.manifest_store=progress_store,manifest_store
        super().__init__(embed,store,progress_store.read(),manifest_store.read(),events)
    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs); self.progress_store.write(self.progress); self.manifest_store.write(self.manifest); return result
    def request_stop(self, runtime_store):
        state=runtime_store.read(); state.setdefault('writer',{})['stop_requested']=True; runtime_store.write(state)
    def managed_run(self,chunks,batch_size,runtime_store,lock,identity,collection,max_batches=None,on_checkpoint=None):
        lock.acquire(identity,collection); self.events += ['lock_acquired','writer_running']; state=runtime_store.read(); state.setdefault('writer',{})['status']='running'; runtime_store.write(state)
        try:
            result=self.run(chunks,batch_size,max_batches=max_batches,runtime_store=runtime_store,on_checkpoint=on_checkpoint)
            state=runtime_store.read(); state['writer']['status']='stopped'; state['writer']['stop_requested']=False; runtime_store.write(state); self.events.append('writer_stopped'); return result
        except Exception:
            state=runtime_store.read(); state.setdefault('writer',{})['status']='failed'; runtime_store.write(state); self.events.append('writer_failed'); raise
        finally:
            lock.release(identity); self.events.append('lock_released')
    def run_documents(self, documents, batch_size, max_documents=None, max_batches=None):
        completed=0
        for chunks in documents:
            if max_documents is not None and completed >= max_documents: break
            self.run(chunks,batch_size,max_batches=max_batches)
            if self.progress[validate_document_chunks(chunks)]['state'] is DocumentState.COMPLETED: completed += 1
            else: break
        return completed
