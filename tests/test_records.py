from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from toylife2 import records
from toylife2.results import ArrayResult
from toylife2.specs import RecordingSpec, RunSpec
from toylife2.workspaces import init_workspace


@pytest.fixture
def small_packs(monkeypatch):
    monkeypatch.setattr(records, "_source_archive", lambda path, source=None: records._write_bytes(
        path, b"source"))
    monkeypatch.setattr(records, "_table_archive", lambda root: _small_table_pack(root))


def _small_table_pack(root):
    path = root / "source" / "tables.zip"
    records._write_bytes(path, b"tables")
    return path


def test_record_checkpoint_finalization_integrity_and_export(tmp_path, small_packs):
    external = tmp_path / "input.npy"
    np.save(external, np.arange(4, dtype=np.uint8))
    spec = RunSpec(seed=7)
    root = records.begin_record(spec, tmp_path / "record", inputs={"target": external})
    relocated = tmp_path / "relocated.npy"
    relocated.write_bytes(external.read_bytes())
    same = records.begin_record(spec, tmp_path / "same", inputs={"target": relocated})
    assert records.inspect_record(root)["scientific_identity"] == (
        records.inspect_record(same)["scientific_identity"])
    np.save(external, np.arange(4, dtype=np.uint8) + 1)
    other = records.begin_record(spec, tmp_path / "other", inputs={"target": external})
    assert records.inspect_record(root)["scientific_identity"] != (
        records.inspect_record(other)["scientific_identity"])
    records.write_checkpoint(root, {"cursor": 3, "rng_state": {"state": 123}},
                             arrays={"population": np.arange(2)})
    checkpoint = records.read_checkpoint(root)
    assert checkpoint["state"]["cursor"] == 3
    np.testing.assert_array_equal(checkpoint["loaded_arrays"]["population"], [0, 1])
    values = ArrayResult(np.array([1, 2], dtype=np.uint8),
                         np.array(["complete", "resource_limit"]))
    with pytest.raises(ValueError, match="non-complete units"):
        records.finalize_record(root, arrays={"phenotype": values})
    assert records.verify_record(root)["ok"]
    records.finalize_record(root, arrays={"phenotype": values},
                            summaries={"count": 2, "valid": 1},
                            metadata={"execution": {"actual_backend": "cpu"}}, status="partial")
    assert records.verify_record(root)["ok"]
    archive = records.export_record(root, tmp_path / "portable.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert "record/source/tables.zip" in bundle.namelist()
    (root / "summary.json").write_text(json.dumps({"count": 3}), encoding="utf-8")
    assert not records.verify_record(root)["ok"]


def test_workspace_table_pack_is_content_addressed(tmp_path, monkeypatch):
    from toylife2 import tables

    source = tmp_path / "lookup"
    source.mkdir()
    (source / "a.txt").write_text("data", encoding="utf-8")
    monkeypatch.setattr(tables, "DEFAULT_SOURCE", source)
    monkeypatch.setattr(tables, "_source_files", lambda root: [root / "a.txt"])
    monkeypatch.setattr(records, "_source_archive", lambda path, source=None: records._write_bytes(
        path, b"source"))
    workspace = init_workspace(tmp_path / "workspace")
    first = workspace / "runs" / "first"
    second = workspace / "runs" / "second"
    records.begin_record(RunSpec(seed=1), first)
    records.begin_record(RunSpec(seed=2), second)
    link1 = first / "source" / "tables.zip"
    link2 = second / "source" / "tables.zip"
    assert link1.is_symlink() and link2.is_symlink()
    assert link1.resolve() == link2.resolve()
    assert records.verify_record(first)["ok"]
    with zipfile.ZipFile(link1) as archive:
        assert archive.read("a.txt") == b"data"


def test_full_trace_respects_byte_budget_and_reports_truncation(tmp_path, small_packs):
    spec = RunSpec(seed=5, recording=RecordingSpec(mode="full", trace_budget_mib=1))
    root = records.begin_record(spec, tmp_path / "record")
    result = records.append_trace(root, [{"event": 1}, {"payload": "x" * (2 * 1024 * 1024)},
                                         {"event": 2}])
    assert result == {"written": 1, "dropped": 2, "truncated": True}
    assert records.inspect_record(root)["trace"]["dropped_rows"] == 2
    assert records.verify_record(root)["ok"]


def test_checkpoint_retention_and_crash_orphan_recovery(tmp_path, small_packs):
    root = records.begin_record(RunSpec(seed=11), tmp_path / "record")
    first = records.write_checkpoint(root, {"step": 1}, arrays={"population": np.array([1])})
    second = records.write_checkpoint(root, {"step": 2}, arrays={"population": np.array([2])})
    third = records.write_checkpoint(root, {"step": 3}, arrays={"population": np.array([3])})
    assert not first.exists()
    assert second.exists() and third.exists()
    orphan = root / "checkpoints" / "uncommitted" / "population.npy"
    orphan.parent.mkdir(parents=True)
    np.save(orphan, np.array([999]))
    (orphan.parent / "checkpoint.json").write_text("{}", encoding="utf-8")
    checked = records.verify_record(root)
    assert checked["ok"]
    assert checked["uncommitted_checkpoint_revisions"] == ["uncommitted"]
    assert records.read_checkpoint(root)["state"]["step"] == 3
    with pytest.raises(ValueError, match="uncommitted checkpoint"):
        records.read_checkpoint(orphan.parent / "checkpoint.json")
    archive = records.export_record(root, tmp_path / "portable.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert all("uncommitted" not in name for name in bundle.namelist())
    records.annotate_record(root, {"parent": {"scientific_id": "parent-id"}})
    assert records.inspect_record(root)["parent"]["scientific_id"] == "parent-id"
    assert records.verify_record(root)["ok"]


def test_checkpoint_commit_reuses_immutable_hashes_but_verify_rehashes(
    tmp_path, small_packs, monkeypatch,
):
    root = records.begin_record(RunSpec(seed=13), tmp_path / "record")
    original = records._hash
    seen = []

    def counted(path):
        seen.append(Path(path).relative_to(root).as_posix())
        return original(path)

    monkeypatch.setattr(records, "_hash", counted)
    records.write_checkpoint(root, {"step": 1})
    assert "source/tables.zip" not in seen
    seen.clear()
    assert records.verify_record(root)["ok"]
    assert "source/tables.zip" in seen
    (root / "source" / "tables.zip").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="committed record file changed"):
        records.write_checkpoint(root, {"step": 2})


def test_source_archive_has_stable_paths_in_checkout_and_installed_layout(tmp_path):
    checkout = tmp_path / "checkout"
    package = checkout / "src" / "toylife2"
    package.mkdir(parents=True)
    (package / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (checkout / "pyproject.toml").write_text("[project]\nname='toylife2'\n", encoding="utf-8")
    (checkout / "docs").mkdir()
    (checkout / "docs" / "decisions.md").write_text("canonical\n", encoding="utf-8")
    author = checkout / "upstream" / "toylife"
    author.mkdir(parents=True)
    (author / "toy_plugin.cpp").write_text("// original\n", encoding="utf-8")
    (author / "toyLIFE.pdf").write_bytes(b"%PDF-1.4 test")
    (author / "patterns" / "results").mkdir(parents=True)
    (author / "patterns" / "results" / "historical.pdf").write_bytes(b"historical")
    first = tmp_path / "checkout.zip"
    records._source_archive(first, package)

    installed = tmp_path / "site-packages" / "toylife2"
    installed.mkdir(parents=True)
    (installed / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    provenance = installed / "_provenance"
    provenance.mkdir()
    vendor = installed / "_vendor" / "toylife"
    vendor.mkdir(parents=True)
    (provenance / "pyproject.toml").write_text("[project]\nname='toylife2'\n",
                                                    encoding="utf-8")
    (provenance / "decisions.md").write_text("canonical\n", encoding="utf-8")
    (vendor / "toy_plugin.cpp").write_text("// original\n", encoding="utf-8")
    (vendor / "toyLIFE.pdf").write_bytes(b"%PDF-1.4 test")
    second = tmp_path / "installed.zip"
    records._source_archive(second, installed)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(second) as bundle:
        assert "docs/decisions.md" in bundle.namelist()
        assert "upstream/toylife/toyLIFE.pdf" in bundle.namelist()
        assert not any("results/" in name for name in bundle.namelist())


def test_boundary_trace_and_checkpoint_commit_together(tmp_path, small_packs, monkeypatch):
    spec = RunSpec(seed=5, recording=RecordingSpec(mode="full", trace_budget_mib=1))
    root = records.begin_record(spec, tmp_path / "record")
    initial = records.write_checkpoint(root, {"step": 0}, arrays={"population": np.array([0])})
    original_commit = records._commit

    def interrupted(record, manifest, changed=None):
        raise RuntimeError("simulated crash before manifest commit")

    monkeypatch.setattr(records, "_commit", interrupted)
    with pytest.raises(RuntimeError, match="simulated crash"):
        records.write_boundary(root, {"step": 1}, arrays={"population": np.array([1])},
                               events=[{"step": 1, "before": [0], "after": [1]}])
    monkeypatch.setattr(records, "_commit", original_commit)
    assert records.read_checkpoint(root)["state"]["step"] == 0
    checked = records.verify_record(root)
    assert checked["ok"]
    assert checked["uncommitted_checkpoint_revisions"]
    assert checked["uncommitted_trace_chunks"]
    assert initial.exists()

    committed, accepted = records.write_boundary(
        root, {"step": 1}, arrays={"population": np.array([1])},
        events=[{"step": 1, "before": [0], "after": [1]}],
    )
    assert accepted and committed.exists()
    manifest = records.inspect_record(root)
    assert manifest["trace"]["rows"] == 1
    assert len(manifest["trace"]["chunks"]) == 1
    assert records.read_checkpoint(root)["state"]["step"] == 1
    assert records.verify_record(root)["ok"]
    archive = records.export_record(root, tmp_path / "portable.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert len([name for name in bundle.namelist() if "/traces/" in name]) == 1


def test_boundary_budget_rejection_keeps_pending_event(tmp_path, small_packs):
    spec = RunSpec(seed=5, recording=RecordingSpec(mode="full", trace_budget_mib=1))
    root = records.begin_record(spec, tmp_path / "record")
    event = {"payload": "x" * (2 * 1024 * 1024)}
    checkpoint, accepted = records.write_boundary(root, {"step": 1}, events=[event])
    assert not accepted
    assert checkpoint.exists()
    state = records.read_checkpoint(root)["state"]
    assert state["pending_trace"] == event
    assert state["recording_limit"] == "trace_budget"
    manifest = records.inspect_record(root)
    assert manifest["trace"]["rows"] == 0
    assert manifest["trace"]["pending_rows"] == 1
    assert records.verify_record(root)["ok"]
