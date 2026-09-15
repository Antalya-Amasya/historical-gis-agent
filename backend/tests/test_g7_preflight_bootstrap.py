"""G7 pre-flight bootstrap: unique repo-local pytest basetemp selection."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts" / "run_g7_preflight.py"
BLOCKED_TESTS = (
    "backend/tests/test_g7c_broad_full_chain_trace_harness.py::test_incomplete_local_cache_fails_fast",
    "backend/tests/test_g7c_broad_full_chain_trace_harness.py::test_stale_output_head_is_not_current",
    "backend/tests/test_g7c_broad_full_chain_trace_harness.py::test_current_output_head_is_accepted",
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location("run_g7_preflight", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_fresh_basetemp_is_unique():
    launcher = _load_launcher()
    first = launcher.make_fresh_basetemp()
    second = launcher.make_fresh_basetemp()
    assert first != second
    assert first.parent == launcher.BASETMP_ROOT
    assert second.parent == launcher.BASETMP_ROOT


def test_stale_old_run_does_not_block_fresh_invocation():
    launcher = _load_launcher()
    stale = launcher.BASETMP_ROOT / "old-run"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "g7c_summary.json").write_text('{"head":"stale"}', encoding="utf-8")

    fresh = launcher.make_fresh_basetemp()
    assert fresh != stale
    assert stale.exists()

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env.setdefault("PYTHONPATH", str(ROOT))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--basetemp={fresh}",
            *BLOCKED_TESTS,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert stale.exists()
