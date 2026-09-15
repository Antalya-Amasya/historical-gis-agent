"""G7 pre-flight bootstrap: writable runtime temp root selection."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_probe_temp_root_verifies_write_read(tmp_path):
    launcher = _load_launcher()
    ok, reason = launcher._probe_temp_root(tmp_path / "probe-root")
    assert ok is True
    assert reason == "ok"


def test_selects_second_candidate_when_first_unwritable(tmp_path, monkeypatch):
    launcher = _load_launcher()
    first = tmp_path / "blocked"
    second = tmp_path / "writable"
    monkeypatch.setattr(launcher, "candidate_temp_roots", lambda: [first, second])
    real_probe = launcher._probe_temp_root

    def fake_probe(path: Path):
        if path == first:
            return False, "denied"
        return real_probe(path)

    monkeypatch.setattr(launcher, "_probe_temp_root", fake_probe)
    assert launcher.select_writable_temp_root() == second


def test_inaccessible_repo_pytest_tmp_does_not_block(tmp_path, monkeypatch):
    launcher = _load_launcher()
    preferred = tmp_path / "runtime"
    blocked_repo = launcher.REPO_FALLBACK_TMP_ROOT
    real_probe = launcher._probe_temp_root
    monkeypatch.setattr(launcher, "candidate_temp_roots", lambda: [blocked_repo, preferred])

    def fake_probe(path: Path):
        if path == blocked_repo:
            return False, "Permission denied"
        return real_probe(path)

    monkeypatch.setattr(launcher, "_probe_temp_root", fake_probe)
    assert launcher.select_writable_temp_root() == preferred


def test_make_fresh_basetemp_is_unique(tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "runtime-root"
    first = launcher.make_fresh_basetemp(root)
    second = launcher.make_fresh_basetemp(root)
    assert first != second
    assert first.parent == root
    assert second.parent == root


def test_fail_fast_when_all_candidates_unusable(monkeypatch):
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "candidate_temp_roots", lambda: [Path("/blocked-a"), Path("/blocked-b")])
    monkeypatch.setattr(launcher, "_probe_temp_root", lambda _path: (False, "denied"))
    with pytest.raises(launcher.PreflightTempRootError) as error:
        launcher.select_writable_temp_root()
    message = str(error.value)
    assert "G7_PREFLIGHT_NO_WRITABLE_TEMP_ROOT" in message
    assert "blocked-a" in message
    assert "blocked-b" in message


def test_run_preflight_passes_basetemp_to_pytest(monkeypatch, tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "selected-root"
    basetemp = root / "run-test"
    basetemp.mkdir(parents=True)
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(launcher, "select_writable_temp_root", lambda: root)
    monkeypatch.setattr(launcher, "make_fresh_basetemp", lambda _root=None: basetemp)
    monkeypatch.setattr(
        launcher,
        "verify_offline_embedding_preflight",
        lambda: {"model_name": "intfloat/multilingual-e5-small", "dimensions": 384},
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.run_preflight() == 0
    assert f"--basetemp={basetemp}" in captured["cmd"]


def test_stale_old_run_does_not_block_fresh_invocation(tmp_path):
    launcher = _load_launcher()
    root = tmp_path / "runtime-root"
    stale = root / "old-run"
    stale.mkdir(parents=True)
    (stale / "g7c_summary.json").write_text('{"head":"stale"}', encoding="utf-8")
    fresh = launcher.make_fresh_basetemp(root)
    assert fresh != stale
    assert stale.exists()

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env.setdefault("PYTHONPATH", str(ROOT))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--basetemp={fresh}", *BLOCKED_TESTS],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert stale.exists()
