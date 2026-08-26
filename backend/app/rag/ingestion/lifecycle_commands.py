from .lifecycle import LifecycleError, validate_resume
from .lifecycle_chunks import validate_document_chunks
class KillSpy:
    def __init__(self): self.kill_calls=self.terminate_calls=self.server_stop_calls=0
class LifecycleCommandHandler:
    def __init__(self,runtime,progress,manifest,lock,store,runner=None,coordinator=None): self.runtime,self.progress,self.manifest,self.lock,self.store,self.runner,self.coordinator=runtime,progress,manifest,lock,store,runner,coordinator
    def status(self):
        return {'runtime':self.runtime.read(),'progress':self.progress.read(),'manifest':self.manifest.read(),'lock':self.lock.path.read_bytes() if self.lock.path.exists() else None}
    def stop(self): self.runner.request_stop(self.runtime)
    def resume(self,chunks,**kwargs):
        if self.lock.path.exists():
            from .lifecycle import read_json
            owner = read_json(self.lock.path)
            if owner and self.lock.exists(owner.get("pid")): raise LifecycleError("writer_lock_active")
        documents = chunks if chunks and isinstance(chunks[0], list) else [chunks]
        if self.coordinator:
            return self.coordinator.resume(documents,batch_size=kwargs.get('batch_size',128),max_batches=kwargs.get('max_batches'),max_documents=kwargs.get('max_documents'))
        d=validate_document_chunks(chunks); p=self.progress.read().get(d,{}); ids=set(p.get('ids',[])); server=self.store.get_document_ids(d) & {c.id for c in chunks}; validate_resume(ids,server)
        if p.get('state')=='completed' and self.manifest.read().get(d)=='ingested': return 'already_completed'
        return self.runner.run(chunks,**{key:value for key,value in kwargs.items() if key in {"batch_size","max_batches"}})
