"""G7 evaluation output persistence tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "g7c_broad_full_chain_trace.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("g7c_broad_full_chain_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unique_run_directory_is_used(tmp_path, monkeypatch):
    module = _load_module()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    run_dir, run_id = module.make_unique_run_output_dir(runtime_root)
    assert run_dir.parent == runtime_root / "results"
    assert run_id in str(run_dir)


def test_fixed_legacy_outputs_remain_untouched(tmp_path, monkeypatch):
    module = _load_module()
    legacy_dir = tmp_path / "outputs"
    legacy_dir.mkdir()
    legacy_trace = legacy_dir / "g7c_case_traces.json"
    legacy_summary = legacy_dir / "g7c_summary.json"
    legacy_trace.write_text('{"legacy": true}', encoding="utf-8")
    legacy_summary.write_text('{"head":"legacy"}', encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_TRACE_PATH", legacy_trace)
    monkeypatch.setattr(module, "LEGACY_SUMMARY_PATH", legacy_summary)

    run_dir = tmp_path / "runtime" / "results" / "run-test"
    run_dir.mkdir(parents=True)
    traces = [{"case_id": "G7-C03"}]
    summary = {"head": "head-1", "case_count": 1, "stale": False}
    result = module.persist_evaluation_results(
        traces,
        summary,
        run_dir=run_dir,
        run_id="run-test",
        current_head="head-1",
        expected_case_count=1,
    )

    assert legacy_trace.read_text(encoding="utf-8") == '{"legacy": true}'
    assert legacy_summary.read_text(encoding="utf-8") == '{"head":"legacy"}'
    assert Path(result["trace_path"]).exists()
    assert Path(result["summary_path"]).exists()


def test_locked_fixed_output_does_not_block_new_persistence(tmp_path, monkeypatch):
    module = _load_module()
    legacy_dir = tmp_path / "outputs"
    legacy_dir.mkdir()
    legacy_summary = legacy_dir / "g7c_summary.json"
    legacy_summary.write_text('{"head":"locked"}', encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_SUMMARY_PATH", legacy_summary)
    real_write = module._write_json_atomic

    def patched_write(path: Path, payload):
        if path == legacy_summary:
            raise PermissionError("legacy locked")
        return real_write(path, payload)

    monkeypatch.setattr(module, "_write_json_atomic", patched_write)
    run_dir = tmp_path / "runtime" / "results" / "run-new"
    run_dir.mkdir(parents=True)
    result = module.persist_evaluation_results(
        [{"case_id": "G7-C03"}],
        {"head": "head-1", "case_count": 1, "stale": False},
        run_dir=run_dir,
        run_id="run-new",
        current_head="head-1",
        expected_case_count=1,
    )
    assert result["completed"] is True
    assert legacy_summary.read_text(encoding="utf-8") == '{"head":"locked"}'


def test_trace_and_summary_share_run_id(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "runtime" / "results" / "run-shared"
    run_dir.mkdir(parents=True)
    result = module.persist_evaluation_results(
        [{"case_id": "G7-A01"}],
        {"head": "head-1", "case_count": 1, "stale": False},
        run_dir=run_dir,
        run_id="run-shared",
        current_head="head-1",
        expected_case_count=1,
    )
    summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert summary["run_id"] == "run-shared"
    assert summary["trace_path"] == result["trace_path"]


def test_incomplete_run_is_not_marked_completed(tmp_path):
    module = _load_module()
    run_dir = tmp_path / "runtime" / "results" / "run-incomplete"
    run_dir.mkdir(parents=True)
    with pytest.raises(module.OutputPersistenceError):
        module.persist_evaluation_results(
            [{"case_id": "G7-A01"}],
            {"head": "head-1", "case_count": 20, "stale": False},
            run_dir=run_dir,
            run_id="run-incomplete",
            current_head="head-1",
            expected_case_count=20,
        )
    summary_path = run_dir / module.SUMMARY_FILENAME
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed"] is False


def test_latest_pointer_failure_keeps_completed_run(tmp_path, monkeypatch):
    module = _load_module()
    manifest_path = tmp_path / "outputs" / "g7c_latest.json"
    monkeypatch.setattr(module, "LATEST_MANIFEST_PATH", manifest_path)
    real_write = module._write_json_atomic

    def patched_write(path: Path, payload):
        if path == manifest_path:
            raise PermissionError("manifest locked")
        return real_write(path, payload)

    monkeypatch.setattr(module, "_write_json_atomic", patched_write)
    run_dir = tmp_path / "runtime" / "results" / "run-complete"
    run_dir.mkdir(parents=True)
    result = module.persist_evaluation_results(
        [{"case_id": "G7-C03"}],
        {"head": "head-1", "case_count": 1, "stale": False},
        run_dir=run_dir,
        run_id="run-complete",
        current_head="head-1",
        expected_case_count=1,
    )
    assert result["completed"] is True
    assert result["manifest_updated"] is False
    assert Path(result["trace_path"]).exists()
    assert json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))["completed"] is True


def test_legacy_fixed_outputs_not_current_authority(tmp_path, monkeypatch):
    module = _load_module()
    legacy_summary = tmp_path / "outputs" / "g7c_summary.json"
    legacy_summary.parent.mkdir(parents=True)
    legacy_summary.write_text(
        json.dumps({"head": "legacy-head", "stale": False, "completed": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "LEGACY_SUMMARY_PATH", legacy_summary)
    monkeypatch.setattr(module, "LATEST_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    assert module.output_is_current("legacy-head") is False


def test_current_authority_comes_from_latest_manifest(tmp_path, monkeypatch):
    module = _load_module()
    run_dir = tmp_path / "runtime" / "results" / "run-current"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / module.SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps({"head": "same-head", "stale": False, "completed": True, "case_count": 1}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "outputs" / "g7c_latest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-current",
                "head": "same-head",
                "trace_path": str(run_dir / module.TRACE_FILENAME),
                "summary_path": str(summary_path),
                "case_count": 1,
                "completed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "LATEST_MANIFEST_PATH", manifest_path)
    assert module.output_is_current("same-head") is True
    assert module.check_stale_outputs("same-head")["accepted_as_current"] is True
