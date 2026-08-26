"""Document-level, restart-safe orchestration over the single-document runner."""
from .lifecycle import DocumentState, LifecycleError, validate_resume
from .lifecycle_chunks import validate_document_chunks
class PersistedIngestionCoordinator:
    def __init__(self, runner_factory, progress, manifest, store, runtime, lock, identity, collection): self.runner_factory,self.progress,self.manifest,self.store,self.runtime,self.lock,self.identity,self.collection=runner_factory,progress,manifest,store,runtime,lock,identity,collection
    def resume(self, documents, *, batch_size, max_batches=None, max_documents=None):
        completed=0; outcomes=[]
        for chunks in documents:
            doc=validate_document_chunks(chunks); state=self.progress.read().get(doc,{}); expected={x.id for x in chunks}; ids=set(state.get('ids',[])); actual=self.store.get_document_ids(doc) & expected
            validate_resume(ids,actual)
            if state.get('state')==DocumentState.COMPLETED and self.manifest.read().get(doc)=='ingested': outcomes.append((doc,'already_completed')); continue
            if max_documents is not None and completed>=max_documents: break
            runner=self.runner_factory(); runner.managed_run(chunks,batch_size,self.runtime,self.lock,self.identity,self.collection,max_batches=max_batches); state=self.progress.read().get(doc,{})
            if state.get('state')==DocumentState.COMPLETED: completed+=1; outcomes.append((doc,'completed'))
            else: outcomes.append((doc,'partial')); break
        return outcomes
