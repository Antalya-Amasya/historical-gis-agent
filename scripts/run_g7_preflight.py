"""G7 pre-flight: run mandatory tests with a verified writable pytest basetemp."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_FALLBACK_TMP_ROOT = ROOT / ".pytest_tmp"
PREFLIGHT_TESTS = (
    "backend/tests/test_g7_broad_fixture_contract.py",
    "backend/tests/test_g7c_broad_full_chain_trace_harness.py",
)
G7C_SCRIPT = ROOT / "scripts" / "g7c_broad_full_chain_trace.py"


class PreflightTempRootError(RuntimeError):
    pass


def candidate_temp_roots() -> list[Path]:
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Temp" / "historical-gis-g7")
    temp_dir = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_dir:
        candidates.append(Path(temp_dir) / "historical-gis-g7")
    candidates.extend([Path(r"C:\D\python\_g7_runtime_tmp"), REPO_FALLBACK_TMP_ROOT])
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _probe_temp_root(path: Path) -> tuple[bool, str]:
    probe_dir = path / f".probe-{uuid.uuid4().hex}"
    probe_file = probe_dir / "probe.txt"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_dir.mkdir()
        probe_file.write_text("ok", encoding="utf-8")
        if probe_file.read_text(encoding="utf-8") != "ok":
            return False, "probe read mismatch"
        probe_file.unlink()
        probe_dir.rmdir()
        return True, "ok"
    except OSError as error:
        return False, str(error)
    finally:
        if probe_file.exists():
            probe_file.unlink(missing_ok=True)
        if probe_dir.exists():
            probe_dir.rmdir()


def select_writable_temp_root() -> Path:
    failures: list[tuple[Path, str]] = []
    for candidate in candidate_temp_roots():
        usable, reason = _probe_temp_root(candidate)
        if usable:
            print(f"G7 pre-flight temp root: {candidate}")
            return candidate
        failures.append((candidate, reason))
    lines = ["G7_PREFLIGHT_NO_WRITABLE_TEMP_ROOT"]
    lines.extend(f"  {candidate}: {reason}" for candidate, reason in failures)
    raise PreflightTempRootError("\n".join(lines))


def make_fresh_basetemp(root: Path | None = None) -> Path:
    base = root or select_writable_temp_root()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = base / f"run-{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_g7c_module():
    spec = importlib.util.spec_from_file_location("g7c_broad_full_chain_trace", G7C_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify_offline_embedding_preflight() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend.app.core.config import settings

    module = _load_g7c_module()
    return module.verify_offline_embedding_ready(
        settings.rag_embedding_model,
        "cpu",
        settings.rag_embedding_batch_size,
    )


def run_preflight() -> int:
    root = select_writable_temp_root()
    basetemp = make_fresh_basetemp(root)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--basetemp={basetemp}",
        *PREFLIGHT_TESTS,
    ]
    print(f"G7 pre-flight basetemp: {basetemp}")
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        return result.returncode
    preflight = verify_offline_embedding_preflight()
    print(
        "Offline embedding pre-flight PASS "
        f"model={preflight['model_name']} dims={preflight['dimensions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_preflight())
