import pytest
from pathlib import Path
from backend.app.rag.ingestion.lifecycle import DocumentState, LifecycleError, RuntimeIdentity, may_manifest, transition, validate_resume
from backend.app.rag.ingestion.lifecycle import WriterLock, atomic_json, default_runtime, read_json, acknowledge_failed_prewrite
from backend.app.rag.ingestion.fake_lifecycle_runner import Chunk, FakeEmbedding, FakeStore, Runner, PersistedRunner, JsonStore
from backend.app.rag.ingestion.lifecycle_commands import LifecycleCommandHandler, KillSpy
from backend.app.rag.ingestion.corpus_cli import build_parser, lifecycle_command
from backend.app.rag.ingestion.persisted_coordinator import PersistedIngestionCoordinator
from backend.app.rag.ingestion.models import TextChunk
from backend.app.rag.ingestion.lifecycle_chunks import get_document_id

def test_legal_transitions():
    assert transition(DocumentState.PENDING,DocumentState.IN_PROGRESS)==DocumentState.IN_PROGRESS
    assert transition(DocumentState.IN_PROGRESS,DocumentState.COMPLETED)==DocumentState.COMPLETED
    with pytest.raises(LifecycleError): transition(DocumentState.PENDING,DocumentState.COMPLETED)
def test_manifest_requires_completed_and_count():
    assert may_manifest(DocumentState.COMPLETED,2,2)
    assert not may_manifest(DocumentState.IN_PROGRESS,2,2)
    assert not may_manifest(DocumentState.COMPLETED,2,1)
def test_resume_requires_exact_ids_and_skips_completed():
    done=validate_resume({'a','b'},{'a','b'}); assert [x for x in ['a','b','c'] if x not in done]==['c']
    with pytest.raises(LifecycleError): validate_resume({'a','b'},{'a'})
def test_unknown_pid_never_permits_stop():
    assert not RuntimeIdentity(None,False,None,None).can_stop()
    assert not RuntimeIdentity(1,False,'writer','now').can_stop()
    assert RuntimeIdentity(1,True,'writer','now').can_stop()
def test_runtime_atomic_and_malformed(tmp_path:Path):
    p=tmp_path/'runtime.json'; atomic_json(p,default_runtime()); assert read_json(p)['writer']['status']=='stopped'; p.write_text('{',encoding='utf8')
    with pytest.raises(LifecycleError): read_json(p)
def test_lock_active_stale_and_ownership(tmp_path:Path):
    p=tmp_path/'lock'; a=RuntimeIdentity(1,True,'a','now'); lock=WriterLock(p,lambda pid:pid==1); lock.acquire(a,'c')
    with pytest.raises(LifecycleError): lock.acquire(RuntimeIdentity(2,True,'b','now'),'c')
    with pytest.raises(LifecycleError): lock.release(RuntimeIdentity(2,True,'b','now'))
    lock.release(a); atomic_json(p,{'pid':9,'command':'gone'}); stale=WriterLock(p,lambda _:False)
    with pytest.raises(LifecycleError): stale.acquire(a,'c')
def test_runner_partial_resume_and_manifest_order():
    chunks=[Chunk(x,'d') for x in 'abcde']; e=FakeEmbedding(); s=FakeStore({'a','b'}); p={'d':{'ids':{'a','b'},'state':DocumentState.IN_PROGRESS}}; m={}; events=[]
    Runner(e,s,p,m,events).run(chunks,2,max_batches=1); assert e.embedded_chunk_ids==['c','d'] and s.upserted_chunk_ids==['c','d'] and 'd' not in m
    Runner(e,s,p,m,events).run(chunks,2); assert e.embedded_chunk_ids==['c','d','e'] and m['d']=='ingested'; assert events.index('journal_completed')<events.index('manifest_written')
def test_runner_completed_noop_and_mismatch():
    c=[Chunk('a','d')];e=FakeEmbedding();s=FakeStore({'a'});p={'d':{'ids':{'a'},'state':DocumentState.COMPLETED}};m={'d':'ingested'};Runner(e,s,p,m,[]).run(c,1);assert not e.embedded_chunk_ids
    with pytest.raises(LifecycleError): Runner(e,FakeStore(),p,{},[]).run(c,1)
def test_restart_from_file_only_processes_missing(tmp_path:Path):
    c=[Chunk(x,'d') for x in 'abcdef']; s=FakeStore(); p=JsonStore(tmp_path/'p.json');m=JsonStore(tmp_path/'m.json'); e1=FakeEmbedding(); PersistedRunner(e1,s,p,m,[]).run(c,2,max_batches=2)
    assert e1.embedded_chunk_ids==['a','b','c','d'] and not m.read()
    e2=FakeEmbedding(); PersistedRunner(e2,s,p,m,[]).run(c,2)
    assert e2.embedded_chunk_ids==['e','f'] and m.read()['d']=='ingested'
def test_max_documents_touches_only_completed_first_document(tmp_path:Path):
    a=[Chunk(x,'a') for x in 'ab']; b=[Chunk(x,'b') for x in 'cd']; e=FakeEmbedding();s=FakeStore();p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m')
    assert PersistedRunner(e,s,p,m,[]).run_documents([a,b],2,max_documents=1)==1
    assert e.embedded_chunk_ids==['a','b'] and set(p.read())=={'a'} and set(m.read())=={'a'}
def test_stop_request_is_only_persisted_flag_and_preserves_server(tmp_path:Path):
    runtime=JsonStore(tmp_path/'runtime'); before={'server':{'host':'127.0.0.1','port':8001},'writer':{'stop_requested':False}};runtime.write(before)
    r=PersistedRunner(FakeEmbedding(),FakeStore(),JsonStore(tmp_path/'p'),JsonStore(tmp_path/'m'),[]);r.request_stop(runtime); after=runtime.read()
    assert after['server']==before['server'] and after['writer']['stop_requested'] is True
def test_managed_runner_runtime_lock_order(tmp_path:Path):
    runtime=JsonStore(tmp_path/'r');runtime.write(default_runtime()); p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');events=[]; ident=RuntimeIdentity(1,True,'writer','now');lock=WriterLock(tmp_path/'lock',lambda _:False)
    PersistedRunner(FakeEmbedding(),FakeStore(),p,m,events).managed_run([Chunk('a','d')],1,runtime,lock,ident,'c')
    assert runtime.read()['writer']['status']=='stopped' and not (tmp_path/'lock').exists()
    assert [events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','writer_stopped','lock_released']]==sorted(events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','writer_stopped','lock_released'])
def test_persisted_stop_only_after_checkpoint(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');e=FakeEmbedding();s=FakeStore();lock=WriterLock(tmp_path/'l',lambda _:False); runner=PersistedRunner(e,s,p,m,[])
    runner.managed_run([Chunk(x,'d') for x in 'abcdef'],2,rt,lock,RuntimeIdentity(1,True,'w','n'),'c',on_checkpoint=lambda: runner.request_stop(rt))
    assert p.read()['d']['state']==DocumentState.IN_PROGRESS and len(p.read()['d']['ids'])==2 and not m.read() and rt.read()['writer']['status']=='stopped' and not rt.read()['writer']['stop_requested'] and not (tmp_path/'l').exists()
def test_failure_keeps_only_prior_checkpoint_and_releases_lock(tmp_path:Path):
    class Broken(FakeStore):
        def upsert(self,chunks,v):
            if chunks[0].id=='c': raise RuntimeError('boom')
            super().upsert(chunks,v)
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');lock=WriterLock(tmp_path/'l',lambda _:False)
    with pytest.raises(RuntimeError): PersistedRunner(FakeEmbedding(),Broken(),p,m,[]).managed_run([Chunk(x,'d') for x in 'abcd'],2,rt,lock,RuntimeIdentity(1,True,'w','n'),'c')
    assert p.read()['d']['ids']=={'a','b'} and not m.read() and rt.read()['writer']['status']=='failed' and not (tmp_path/'l').exists()
def test_foreign_active_lock_rejects_before_work(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');e=FakeEmbedding();s=FakeStore();lock=WriterLock(tmp_path/'l',lambda pid:pid==1); owner=RuntimeIdentity(1,True,'a','n');lock.acquire(owner,'c')
    with pytest.raises(LifecycleError): PersistedRunner(e,s,p,m,[]).managed_run([Chunk('x','d')],1,rt,lock,RuntimeIdentity(2,True,'b','n'),'c')
    assert not e.embedded_chunk_ids and not s.upserted_chunk_ids and rt.read()['writer']['status']=='stopped' and (tmp_path/'l').exists()
def test_cli_status_readonly_stop_safe_and_resume_validation(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');lock=WriterLock(tmp_path/'l');e=FakeEmbedding();s=FakeStore({'a'});r=PersistedRunner(e,s,p,m,[]);h=LifecycleCommandHandler(rt,p,m,lock,s,r); before=(tmp_path/'r').read_bytes();h.status();assert (tmp_path/'r').read_bytes()==before;h.stop();assert rt.read()['writer']['stop_requested'] and KillSpy().kill_calls==0
    p.write({'d':{'ids':['a','b'],'state':DocumentState.IN_PROGRESS}})
    with pytest.raises(LifecycleError): h.resume([Chunk('a','d'),Chunk('b','d')],batch_size=1)
def test_argparse_status_stop_and_budget_forwarding():
    args=build_parser().parse_args(['--resume','--max-batches','2','--max-documents','1']); seen={}
    class H:
        def resume(self,chunks,**kw): seen.update(kw);return 'ok'
    assert lifecycle_command(H(),args,[Chunk('a','d')])=='ok' and seen['max_batches']==2 and seen['max_documents']==1
def test_cli_partial_and_completed_noop(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');p.write({'d':{'ids':['a','b'],'state':DocumentState.IN_PROGRESS}});e=FakeEmbedding();s=FakeStore({'a','b'});r=PersistedRunner(e,s,p,m,[]);h=LifecycleCommandHandler(rt,p,m,WriterLock(tmp_path/'l'),s,r);args=build_parser().parse_args(['--resume'])
    lifecycle_command(h,args,[Chunk(x,'d') for x in 'abcd']); assert e.embedded_chunk_ids==['c','d'] and m.read()['d']=='ingested'
    e2=FakeEmbedding(); h2=LifecycleCommandHandler(rt,p,m,WriterLock(tmp_path/'l2'),s,PersistedRunner(e2,s,p,m,[])); assert lifecycle_command(h2,args,[Chunk(x,'d') for x in 'abcd'])=='already_completed' and not e2.embedded_chunk_ids
def test_cli_foreign_lock_rejects_without_work(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');lock=WriterLock(tmp_path/'l',lambda pid:pid==1);lock.acquire(RuntimeIdentity(1,True,'foreign','n'),'c'); before=(tmp_path/'l').read_bytes();e=FakeEmbedding();s=FakeStore();h=LifecycleCommandHandler(rt,p,m,lock,s,PersistedRunner(e,s,p,m,[]));args=build_parser().parse_args(['--resume'])
    with pytest.raises(LifecycleError): lifecycle_command(h,args,[Chunk('a','d')])
    assert not e.embedded_chunk_ids and not s.upserted_chunk_ids and (tmp_path/'l').read_bytes()==before
def test_coordinator_sequential_document_budget(tmp_path:Path):
    p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');s=FakeStore(); made=[]
    def factory(): e=FakeEmbedding();made.append(e);return PersistedRunner(e,s,p,m,[])
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime()); c=PersistedIngestionCoordinator(factory,p,m,s,rt,WriterLock(tmp_path/'l',lambda _:False),RuntimeIdentity(1,True,'w','n'),'c'); a=[Chunk(x,'a') for x in 'ab'];b=[Chunk(x,'b') for x in 'cd']
    assert c.resume([a,b],batch_size=2,max_documents=1)==[('a','completed')]; assert set(p.read())=={'a'}
    c.resume([a,b],batch_size=2,max_documents=1); assert set(m.read())=={'a','b'} and made[1].embedded_chunk_ids==['c','d']
def test_cli_handler_coordinator_wiring(tmp_path:Path):
    rt=JsonStore(tmp_path/'r');rt.write(default_runtime());p=JsonStore(tmp_path/'p');m=JsonStore(tmp_path/'m');s=FakeStore({'a','b'}); p.write({'d':{'ids':['a','b'],'state':DocumentState.IN_PROGRESS}}); embeds=[]
    def factory(): e=FakeEmbedding();embeds.append(e);return PersistedRunner(e,s,p,m,[])
    coordinator=PersistedIngestionCoordinator(factory,p,m,s,rt,WriterLock(tmp_path/'l',lambda _:False),RuntimeIdentity(1,True,'w','n'),'c'); h=LifecycleCommandHandler(rt,p,m,WriterLock(tmp_path/'l2'),s,coordinator=coordinator); args=build_parser().parse_args(['--resume','--max-documents','1'])
    lifecycle_command(h,args,[Chunk(x,'d') for x in 'abcd']); assert embeds[0].embedded_chunk_ids==['c','d'] and m.read()['d']=='ingested'

def test_production_textchunk_uses_metadata_identity_end_to_end():
    chunks=[TextChunk('a','text a',{'document_id':'appian_roman_history_civil_wars'}),TextChunk('b','text b',{'document_id':'appian_roman_history_civil_wars'})]
    embedding=FakeEmbedding(); store=FakeStore(); progress={}; manifest={}
    Runner(embedding,store,progress,manifest,[]).run(chunks,2)
    assert get_document_id(chunks[0])=='appian_roman_history_civil_wars' and embedding.embedded_chunk_ids==['a','b'] and manifest['appian_roman_history_civil_wars']=='ingested'

def test_chunk_identity_metadata_fails_closed_for_missing_and_mixed_documents():
    embedding=FakeEmbedding(); store=FakeStore()
    with pytest.raises(LifecycleError,match='missing_document_id'):
        Runner(embedding,store,{}, {},[]).run([TextChunk('a','text',{})],1)
    with pytest.raises(LifecycleError,match='mixed_document_chunks'):
        Runner(embedding,store,{}, {},[]).run([TextChunk('a','text',{'document_id':'A'}),TextChunk('b','text',{'document_id':'B'})],1)
    assert embedding.embedded_chunk_ids==[] and store.upserted_chunk_ids==[]

def test_embedding_vectors_are_propagated_and_invalid_batches_fail_before_upsert():
    chunks=[Chunk('a','A'),Chunk('b','A')]; embedding=FakeEmbedding(); store=FakeStore(); Runner(embedding,store,{}, {},[]).run(chunks,2)
    assert store.upserted_embeddings==embedding.returned_vectors
    class Empty:
        def embed(self,_): return []
    empty_store=FakeStore()
    with pytest.raises(LifecycleError,match='empty_embeddings'): Runner(Empty(),empty_store,{}, {},[]).run([Chunk('c','A')],1)
    assert empty_store.upsert_calls==0
    class Short:
        def embed(self,_): return [[1.0]]
    short_store=FakeStore()
    with pytest.raises(LifecycleError,match='embedding_batch_size_mismatch'): Runner(Short(),short_store,{}, {},[]).run(chunks,2)
    assert short_store.upsert_calls==0

def test_acknowledge_failed_prewrite_resets_only_empty_state(tmp_path:Path):
    runtime=JsonStore(tmp_path/'runtime'); progress=JsonStore(tmp_path/'progress'); manifest=JsonStore(tmp_path/'manifest'); lock=WriterLock(tmp_path/'lock')
    runtime.write({'writer':{'status':'failed'}})
    baseline=acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=0,queue_count=0)
    assert baseline==runtime.read() and baseline['writer']['status']=='stopped' and not baseline['writer']['stop_requested']
    for count,queue,error in [(1,0,'collection_not_empty'),(0,1,'queue_not_empty')]:
        runtime.write({'writer':{'status':'failed'}})
        with pytest.raises(LifecycleError,match=error): acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=count,queue_count=queue)
    progress.write({'A':{'ids':['a']}}); runtime.write({'writer':{'status':'failed'}})
    with pytest.raises(LifecycleError,match='progress_not_empty'): acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=0,queue_count=0)
    progress.write({}); manifest.write({'A':'ingested'}); runtime.write({'writer':{'status':'failed'}})
    with pytest.raises(LifecycleError,match='manifest_not_empty'): acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=0,queue_count=0)
    manifest.write({}); lock.acquire(RuntimeIdentity(1,True,'writer','now'),'c'); runtime.write({'writer':{'status':'failed'}})
    with pytest.raises(LifecycleError,match='lock_present'): acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=0,queue_count=0)
    lock.release(RuntimeIdentity(1,True,'writer','now')); runtime.write(default_runtime())
    with pytest.raises(LifecycleError,match='requires_failed_runtime'): acknowledge_failed_prewrite(runtime,progress,manifest,lock,collection_count=0,queue_count=0)

def _persisted_cli(tmp_path, store, events=None, *, lock_exists=lambda _: False):
    """Create a complete fake CLI control plane from the four persisted files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime=JsonStore(tmp_path/'runtime.json')
    if not (tmp_path/'runtime.json').exists(): runtime.write(default_runtime())
    progress=JsonStore(tmp_path/'progress.json'); manifest=JsonStore(tmp_path/'manifest.json')
    lock=WriterLock(tmp_path/'writer.lock',lock_exists); traces=[] if events is None else events; embeddings=[]
    def factory():
        embedding=FakeEmbedding(); embeddings.append(embedding)
        return PersistedRunner(embedding,store,progress,manifest,traces)
    coordinator=PersistedIngestionCoordinator(factory,progress,manifest,store,runtime,lock,RuntimeIdentity(11,True,'fake-writer','now'),'roman')
    return LifecycleCommandHandler(runtime,progress,manifest,lock,store,coordinator=coordinator), runtime, progress, manifest, lock, embeddings, traces

def test_four_file_persisted_cli_restart_missing_only_and_completed_noop(tmp_path:Path):
    a=[Chunk(x,'A') for x in 'abcdef']; args=build_parser().parse_args(['--resume','--max-batches','2'])
    handler,runtime,progress,manifest,lock,embeddings,_=_persisted_cli(tmp_path,FakeStore())
    assert lifecycle_command(handler,args,a,batch_size=2)==[('A','partial')]
    assert embeddings[0].embedded_chunk_ids==list('abcd') and progress.read()['A']=={'ids':set('abcd'),'state':DocumentState.IN_PROGRESS}
    assert not manifest.read() and runtime.read()['writer']['status']=='stopped' and not lock.path.exists()
    # Reconstruct every control-plane object from files; only the known physical vectors survive.
    del handler, runtime, progress, manifest, lock, embeddings
    handler,runtime,progress,manifest,lock,embeddings,events=_persisted_cli(tmp_path,FakeStore(set('abcd')))
    assert lifecycle_command(handler,build_parser().parse_args(['--resume']),a,batch_size=2)==[('A','completed')]
    assert embeddings[0].embedded_chunk_ids==list('ef') and manifest.read()=={'A':'ingested'} and progress.read()['A']['state']==DocumentState.COMPLETED
    assert runtime.read()['writer']['status']=='stopped' and not lock.path.exists()
    assert [events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','journal_completed','manifest_written','writer_stopped','lock_released']]==sorted(events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','journal_completed','manifest_written','writer_stopped','lock_released'])
    del handler, runtime, progress, manifest, lock, embeddings
    handler,_,_,_,_,embeddings,_=_persisted_cli(tmp_path,FakeStore(set('abcdef')))
    assert lifecycle_command(handler,build_parser().parse_args(['--resume']),a,batch_size=2)==[('A','already_completed')]
    assert embeddings==[]

def test_persisted_cli_sequential_max_documents_restart(tmp_path:Path):
    a=[Chunk(x,'A') for x in ('a1','a2')]; b=[Chunk(x,'B') for x in ('b1','b2')]; args=build_parser().parse_args(['--resume','--max-documents','1'])
    handler,_,progress,manifest,_,embeddings,_=_persisted_cli(tmp_path,FakeStore())
    assert lifecycle_command(handler,args,[a,b],batch_size=2)==[('A','completed')]
    assert embeddings[0].embedded_chunk_ids==['a1','a2'] and set(progress.read())=={'A'} and manifest.read()=={'A':'ingested'}
    del handler, progress, manifest, embeddings
    handler,_,progress,manifest,_,embeddings,_=_persisted_cli(tmp_path,FakeStore({'a1','a2'}))
    assert lifecycle_command(handler,args,[a,b],batch_size=2)==[('A','already_completed'),('B','completed')]
    assert len(embeddings)==1 and embeddings[0].embedded_chunk_ids==['b1','b2'] and set(progress.read())=={'A','B'} and manifest.read()=={'A':'ingested','B':'ingested'}

def test_persisted_failure_checkpoint_integrity_and_event_order(tmp_path:Path):
    class BrokenStore(FakeStore):
        def upsert(self,chunks,vectors):
            if chunks[0].id=='c': raise RuntimeError('upsert failure')
            super().upsert(chunks,vectors)
    events=[]; handler,runtime,progress,manifest,lock,embeddings,events=_persisted_cli(tmp_path,BrokenStore(),events)
    with pytest.raises(RuntimeError): lifecycle_command(handler,build_parser().parse_args(['--resume']),[Chunk(x,'A') for x in 'abcd'],batch_size=2)
    assert progress.read()['A']=={'ids':{'a','b'},'state':DocumentState.IN_PROGRESS} and not manifest.read()
    assert runtime.read()['writer']['status']=='failed' and not lock.path.exists()
    assert events.count('checkpoint')==1 and 'journal_completed' not in events and 'manifest_written' not in events
    assert [events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','writer_failed','lock_released']]==sorted(events.index(x) for x in ['lock_acquired','writer_running','embed','upsert','readback','checkpoint','writer_failed','lock_released'])
    assert embeddings[0].embedded_chunk_ids==list('abcd')

def test_persisted_cli_crash_after_upsert_before_checkpoint_fails_closed(tmp_path:Path):
    handler,_,progress,manifest,_,embeddings,_=_persisted_cli(tmp_path,FakeStore(set('abcd')))
    progress.write({'A':{'ids':['a','b'],'state':DocumentState.IN_PROGRESS}}); assert not manifest.read()
    with pytest.raises(LifecycleError,match='progress_collection_mismatch'):
        lifecycle_command(handler,build_parser().parse_args(['--resume']),[Chunk(x,'A') for x in 'abcd'],batch_size=2)
    assert embeddings==[]

def test_persisted_cooperative_stop_then_restart_missing_only(tmp_path:Path):
    chunks=[Chunk(x,'A') for x in 'abcdef']; events=[]; handler,runtime,progress,manifest,lock,embeddings,events=_persisted_cli(tmp_path,FakeStore(),events)
    # The callback models an external --stop write immediately after checkpoint one.
    coordinator=handler.coordinator; original_factory=coordinator.runner_factory
    def factory_with_stop():
        runner=original_factory(); original_run=runner.managed_run
        def run_with_stop(*args,**kwargs): return original_run(*args,on_checkpoint=lambda: runner.request_stop(runtime),**kwargs)
        runner.managed_run=run_with_stop; return runner
    coordinator.runner_factory=factory_with_stop
    assert lifecycle_command(handler,build_parser().parse_args(['--resume']),chunks,batch_size=2)==[('A','partial')]
    assert embeddings[0].embedded_chunk_ids==['a','b'] and progress.read()['A']=={'ids':{'a','b'},'state':DocumentState.IN_PROGRESS}
    assert not manifest.read() and runtime.read()['writer']['status']=='stopped' and not runtime.read()['writer']['stop_requested'] and not lock.path.exists()
    del handler, runtime, progress, manifest, lock, embeddings
    handler,_,progress,manifest,lock,embeddings,_=_persisted_cli(tmp_path,FakeStore({'a','b'}))
    assert lifecycle_command(handler,build_parser().parse_args(['--resume']),chunks,batch_size=2)==[('A','completed')]
    assert embeddings[0].embedded_chunk_ids==list('cdef') and progress.read()['A']['state']==DocumentState.COMPLETED and manifest.read()=={'A':'ingested'} and not lock.path.exists()

def test_persisted_cli_foreign_lock_and_status_stop_safety(tmp_path:Path):
    active=lambda pid:pid==99; store=FakeStore(); handler,runtime,progress,manifest,lock,embeddings,events=_persisted_cli(tmp_path,store,lock_exists=active)
    foreign=RuntimeIdentity(99,True,'foreign-writer','then'); lock.acquire(foreign,'roman'); before_lock=lock.path.read_bytes(); before=(runtime.path.read_bytes(), progress.path.exists(), manifest.path.exists())
    with pytest.raises(LifecycleError,match='writer_lock_active'):
        lifecycle_command(handler,build_parser().parse_args(['--resume']),[Chunk('a','A')],batch_size=1)
    assert embeddings==[] and store.upserted_chunk_ids==[] and lock.path.read_bytes()==before_lock and runtime.path.read_bytes()==before[0] and not before[1] and not before[2]
    before_status=(runtime.path.read_bytes(),lock.path.read_bytes()); assert lifecycle_command(handler,build_parser().parse_args(['--status']))['lock']==before_lock
    assert (runtime.path.read_bytes(),lock.path.read_bytes())==before_status and embeddings==[] and store.upserted_chunk_ids==[]
    # Use an independent handler without an active lock for the cooperative, zero-kill stop write.
    safe,rt2,p2,m2,l2,_,_=_persisted_cli(tmp_path/'safe',FakeStore()); safe.runner=PersistedRunner(FakeEmbedding(),FakeStore(),p2,m2,[])
    server_before=rt2.read()['server']; safe.stop()
    assert rt2.read()['writer']['stop_requested'] is True and rt2.read()['server']==server_before and not p2.path.exists() and not m2.path.exists() and not l2.path.exists()
