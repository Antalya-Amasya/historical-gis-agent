"""G7 pre-flight: run mandatory tests with a fresh repo-local pytest basetemp."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASETMP_ROOT = ROOT / ".pytest_tmp"
PREFLIGHT_TESTS = (
    "backend/tests/test_g7_broad_fixture_contract.py",
    "backend/tests/test_g7c_broad_full_chain_trace_harness.py",
)
G7C_SCRIPT = ROOT / "scripts" / "g7c_broad_full_chain_trace.py"


def make_fresh_basetemp() -> Path:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = BASETMP_ROOT / f"run-{run_id}"
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
    basetemp = make_fresh_basetemp()
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
