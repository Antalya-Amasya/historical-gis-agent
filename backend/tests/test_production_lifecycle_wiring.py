from pathlib import Path
import pytest
from backend.app.rag.ingestion.corpus_cli import build_parser, lifecycle_command, production_config, main
from backend.app.rag.ingestion.fake_lifecycle_runner import Chunk, FakeEmbedding, FakeStore
from backend.app.rag.ingestion.lifecycle import DocumentState, RuntimeIdentity, default_runtime
from backend.app.rag.ingestion.production_lifecycle import ChromaHttpVectorStoreAdapter, CorpusDocumentSource, SentenceTransformerEmbeddingAdapter, build_production_handler, build_production_lifecycle_backend

class Document:
    def __init__(self, document_id): self.document_id=document_id

def test_http_adapter_is_injected_and_has_no_client_side_effects():
    class Collection:
        def __init__(self): self.ids=set(); self.calls=[]
        def get(self, **kwargs):
            self.calls.append(('get',kwargs)); requested=kwargs.get('ids'); return {'ids':sorted(self.ids if requested is None else self.ids & set(requested))}
        def upsert(self, **kwargs): self.calls.append(('upsert',kwargs)); self.ids.update(kwargs['ids'])
        def count(self): return len(self.ids)
    collection=Collection(); adapter=ChromaHttpVectorStoreAdapter(collection); chunks=[Chunk('a','A')]
    assert adapter.get_document_ids('A')==set(); adapter.upsert(chunks,[[0.0]]); assert adapter.read_back(['a'])=={'a'} and adapter.document_count('A')==1 and adapter.count()==1
    assert collection.calls[0][0]=='get' and collection.calls[1][0]=='upsert'

def test_http_adapter_preserves_vectors_and_rejects_invalid_batches():
    class Collection:
        def upsert(self, **kwargs): self.kwargs=kwargs
    from backend.app.rag.ingestion.lifecycle import LifecycleError
    collection=Collection(); adapter=ChromaHttpVectorStoreAdapter(collection); vectors=[[.1,.2,.3]]; adapter.upsert([Chunk('a','A')],vectors)
    assert collection.kwargs['embeddings']==vectors
    with pytest.raises(LifecycleError,match='empty_embeddings'): adapter.upsert([Chunk('a','A')],[])
    with pytest.raises(LifecycleError,match='embedding_batch_size_mismatch'): adapter.upsert([Chunk('a','A'),Chunk('b','A')],vectors)

def test_embedding_adapter_wraps_injected_provider_only():
    class Provider:
        def __init__(self): self.values=[]
        def embed(self, values): self.values.extend(values); return [[1.0] for _ in values]
    provider=Provider(); assert SentenceTransformerEmbeddingAdapter(provider).embed([Chunk('a','A')])==[[1.0]]; assert provider.values==['passage: ']

def test_production_cli_wiring_is_appian_only_v2_and_bounded(tmp_path:Path):
    appian=Document('appian_roman_history_civil_wars'); other=Document('caesar_gallic_civil_wars'); loaded=[]
    def loader(document):
        loaded.append(document.document_id)
        return [Chunk(f'{document.document_id}:{x}',document.document_id) for x in 'abcdef']
    args=build_parser().parse_args(['--resume','--document-id','appian_roman_history_civil_wars','--max-batches','2','--max-documents','1','--collection','roman_republic_primary_sources_v2'])
    config=production_config(args,state_dir=tmp_path,default_collection='wrong')
    source=CorpusDocumentSource([appian,other],loader); store=FakeStore(); embedding=FakeEmbedding()
    handler,documents=build_production_handler(config,source,store,embedding,identity=RuntimeIdentity(1,True,'fake-writer','now'))
    handler.runtime.write({'server':{'host':config.host,'port':config.port,'persistence_path':'data/chroma_server_roman_republic_v2'},'writer':{'status':'stopped','stop_requested':False}})
    assert lifecycle_command(handler,args,documents,batch_size=2)==[('appian_roman_history_civil_wars','partial')]
    assert loaded==['appian_roman_history_civil_wars'] and embedding.embedded_chunk_ids==['appian_roman_history_civil_wars:a','appian_roman_history_civil_wars:b','appian_roman_history_civil_wars:c','appian_roman_history_civil_wars:d']
    assert not [item for item in store.ids if item.startswith('caesar_gallic_civil_wars:')] and handler.progress.read()['appian_roman_history_civil_wars']['state']==DocumentState.IN_PROGRESS and not handler.manifest.read()
    assert handler.progress.path.name=='ingestion_progress_v2.json' and handler.manifest.path.name=='ingestion_manifest_v2.json' and handler.lock.path.name=='ingestion_writer_v2.lock'

def test_document_source_requires_explicit_selection():
    source=CorpusDocumentSource([Document('A')],lambda _: [Chunk('a','A')])
    try: source.selected(())
    except ValueError as error: assert str(error)=='document_id_required'
    else: raise AssertionError('all-corpus selection must be rejected')

def test_main_lazy_factories_and_real_cli_chain(tmp_path:Path, monkeypatch):
    appian=Document('appian_roman_history_civil_wars'); calls={'http':0,'embedding':0,'legacy':0}; store=FakeStore(); embedded=FakeEmbedding()
    def source_factory(): return CorpusDocumentSource([appian],lambda _: [Chunk(f'appian_roman_history_civil_wars:{x}','appian_roman_history_civil_wars') for x in range(300)])
    def store_factory(): calls['http']+=1; return store
    def embedding_factory(): calls['embedding']+=1; return embedded
    def factory(config,args): return build_production_lifecycle_backend(config,source_factory,store_factory,embedding_factory,identity=RuntimeIdentity(1,True,'writer','now'))
    monkeypatch.chdir(tmp_path)
    # Status and stop construct neither real-resource adapter.
    assert main(['--status','--document-id','appian_roman_history_civil_wars'],backend_factory=factory)==0 and calls=={'http':0,'embedding':0,'legacy':0}
    assert main(['--stop','--document-id','appian_roman_history_civil_wars'],backend_factory=factory)==0 and calls=={'http':0,'embedding':0,'legacy':0}
    from backend.app.rag.ingestion.fake_lifecycle_runner import JsonStore
    JsonStore(tmp_path/'data/historical_sources/processed/ingestion_runtime_v2.json').write(default_runtime())
    assert main(['--resume','--document-id','appian_roman_history_civil_wars','--max-batches','2'],backend_factory=factory)==0
    assert calls=={'http':1,'embedding':1,'legacy':0} and embedded.embedded_chunk_ids==[f'appian_roman_history_civil_wars:{x}' for x in range(256)]

def test_main_resume_foreign_lock_and_selection_fail_before_factories(tmp_path:Path, monkeypatch):
    calls=[]; monkeypatch.chdir(tmp_path)
    def factory(config,args):
        backend=build_production_lifecycle_backend(config,lambda: CorpusDocumentSource([Document('A')],lambda _: []),lambda: calls.append('http'),lambda: calls.append('embedding'),identity=RuntimeIdentity(2,True,'self','now'),lock_exists=lambda pid:pid==9)
        backend.lock.acquire(RuntimeIdentity(9,True,'foreign','now'),'roman'); return backend
    import pytest
    with pytest.raises(Exception): main(['--resume','--document-id','A'],backend_factory=factory)
    assert calls==[]
    with pytest.raises(SystemExit): main(['--resume'],backend_factory=factory)
    assert calls==[]
