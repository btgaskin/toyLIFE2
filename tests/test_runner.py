from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from toylife2 import Model, records, runner
from toylife2.records import inspect_record, verify_record
from toylife2.specs import CellSpec, EvolutionSpec, ExecutionSpec, ModelSpec, RunSpec, SpatialSpec


@pytest.fixture(scope="module")
def tables(canonical_tables):
    return canonical_tables


@pytest.fixture
def small_record_packs(monkeypatch, tables):
    # Record-format/pack integrity is tested independently with real snapshots.
    monkeypatch.setattr(
        records, "_source_archive", lambda path, source=None: records._write_bytes(path, b"source")
    )

    def pack(root, tables=None):
        path = root / "source" / "tables.zip"
        records._write_bytes(path, b"tables")
        return path

    monkeypatch.setattr(records, "_table_archive", pack)
    monkeypatch.setattr(records, "_archive_table_digest", lambda path: tables.digest)
    monkeypatch.setattr(
        runner, "Model", lambda execution=None, tables=tables: Model(tables, execution)
    )


def _array(record, name):
    description = inspect_record(record)["artifacts"]["arrays"][name]
    return np.load(record / description["path"], allow_pickle=False)


def test_real_cpu_wf_record_resume_and_reproduction(tmp_path, small_record_packs):
    target = tmp_path / "target.npy"
    np.save(target, np.zeros((3, 3), np.uint8))
    spec = RunSpec(
        operation="wright_fisher",
        seed=912,
        spatial=SpatialSpec(cells=3, steps=3, target=str(target)),
        evolution=EvolutionSpec(trials=2, population_size=5, generations=3, mutation_probability=1),
    )
    full = runner.run(spec, destination=tmp_path / "full")
    limited = dataclasses.replace(spec, execution=ExecutionSpec(max_steps=1))
    partial = runner.run(limited, destination=tmp_path / "partial")
    assert inspect_record(partial)["status"] == "partial"
    continued = runner.resume(
        partial,
        destination=tmp_path / "continued",
        execution=ExecutionSpec(max_steps=100, batch_size=3),
    )
    for name in ("populations", "history"):
        np.testing.assert_array_equal(_array(full, name), _array(continued, name))
    assert (
        inspect_record(full)["scientific_identity"]
        == (inspect_record(continued)["scientific_identity"])
    )
    assert "checkpoint_sha256" in inspect_record(continued)["parent"]
    repeated = runner.reproduce(full, tmp_path / "repeated")
    assert verify_record(repeated)["ok"]


def test_cell_record_supports_33_genes_and_step_resume(tmp_path, small_record_packs):
    sequence = "0101" + "0" * 16 + "0" * (20 * 31) + "0101" + "0" * 16
    spec = RunSpec(
        operation="cell",
        genotypes=(sequence,),
        model=ModelSpec(genes=33),
        cell=CellSpec(steps=3),
        execution=ExecutionSpec(max_steps=1),
    )
    partial = runner.run(spec, destination=tmp_path / "cell_partial")
    assert inspect_record(partial)["status"] == "partial"
    complete = runner.resume(
        partial, destination=tmp_path / "cell_complete", execution=ExecutionSpec(max_steps=10)
    )
    expression = _array(complete, "expression")
    assert expression.shape == (1, 3, 33)
    assert np.all(expression[:, :, [0, 32]] == 1)
    assert np.all(expression[:, :, 1:32] == 0)
    assert verify_record(complete)["ok"]
    source = runner.run(
        dataclasses.replace(spec, execution=ExecutionSpec(max_steps=10)),
        destination=tmp_path / "cell_source",
    )
    recreated = runner.reproduce(source, tmp_path / "cell_recreated")
    assert (
        json.loads((source / "summary.json").read_text())["trajectories"]
        == (json.loads((recreated / "summary.json").read_text())["trajectories"])
    )
