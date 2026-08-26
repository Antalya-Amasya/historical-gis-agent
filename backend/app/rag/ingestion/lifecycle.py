"""Pure lifecycle controls for resumable corpus ingestion; no Chroma dependency."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path

class DocumentState(StrEnum):
    PENDING="pending"; IN_PROGRESS="in_progress"; COMPLETED="completed"; FAILED="failed"; STOPPED="stopped"

_ALLOWED={DocumentState.PENDING:{DocumentState.IN_PROGRESS,DocumentState.FAILED,DocumentState.STOPPED},DocumentState.IN_PROGRESS:{DocumentState.COMPLETED,DocumentState.FAILED,DocumentState.STOPPED},DocumentState.STOPPED:{DocumentState.IN_PROGRESS,DocumentState.FAILED},DocumentState.FAILED:{DocumentState.IN_PROGRESS},DocumentState.COMPLETED:set()}

class LifecycleError(RuntimeError): pass
def transition(current: DocumentState, target: DocumentState) -> DocumentState:
    if target not in _ALLOWED[current]: raise LifecycleError(f"illegal_transition:{current}:{target}")
    return target
def validate_resume(journal_ids: set[str], server_ids: set[str]) -> set[str]:
    if journal_ids != server_ids: raise LifecycleError("progress_collection_mismatch")
    return server_ids
def may_manifest(state: DocumentState, expected: int, server_count: int) -> bool:
    return state is DocumentState.COMPLETED and expected == server_count
@dataclass(frozen=True)
class RuntimeIdentity:
    pid: int | None; pid_verified: bool; command: str | None; started_at: str | None
    def can_stop(self) -> bool: return self.pid is not None and self.pid_verified and bool(self.command)

def default_runtime() -> dict:
    return {"server":{"pid":None,"pid_verified":False,"host":"127.0.0.1","port":8001,"persistence_path":None,"command":None,"started_at":None},"writer":{"pid":None,"pid_verified":False,"status":"stopped","command":None,"started_at":None,"stop_requested":False}}

def acknowledge_failed_prewrite(runtime_store, progress_store, manifest_store, lock, *, collection_count: int, queue_count: int) -> dict:
    """Acknowledge only a proven pre-write failure and atomically restore a clean baseline."""
    runtime=runtime_store.read()
    if runtime.get("writer",{}).get("status") != "failed": raise LifecycleError("prewrite_reset_requires_failed_runtime")
    if collection_count != 0: raise LifecycleError("prewrite_reset_collection_not_empty")
    if queue_count != 0: raise LifecycleError("prewrite_reset_queue_not_empty")
    if progress_store.read(): raise LifecycleError("prewrite_reset_progress_not_empty")
    if manifest_store.read(): raise LifecycleError("prewrite_reset_manifest_not_empty")
    if lock.path.exists(): raise LifecycleError("prewrite_reset_lock_present")
    baseline=default_runtime(); runtime_store.write(baseline)
    return baseline
def read_json(path: Path, default: dict | None=None) -> dict:
    if not path.exists(): return dict(default or {})
    try: value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: raise LifecycleError("malformed_state") from exc
    if not isinstance(value,dict): raise LifecycleError("malformed_state")
    return value
def atomic_json(path: Path, value: dict) -> None:
    temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8"); temp.replace(path)
class WriterLock:
    def __init__(self,path:Path,exists=lambda pid:False): self.path=path; self.exists=exists
    def acquire(self,identity:RuntimeIdentity,collection:str) -> None:
        current=read_json(self.path)
        if current:
            if self.exists(current.get("pid")): raise LifecycleError("writer_lock_active")
            raise LifecycleError("writer_lock_stale")
        atomic_json(self.path,{"pid":identity.pid,"pid_verified":identity.pid_verified,"command":identity.command,"collection":collection})
    def release(self,identity:RuntimeIdentity) -> None:
        current=read_json(self.path)
        if current.get("pid")!=identity.pid or current.get("command")!=identity.command: raise LifecycleError("writer_lock_not_owned")
        self.path.unlink()
