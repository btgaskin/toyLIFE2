"""Recorded execution uses captured tables and explicit stochastic coordinates."""

import dataclasses
import shutil
import zipfile

import numpy as np
import pytest

from toylife2 import BatchRunner
from toylife2.protocols import EpisodeSpec, TaskSpec, WorkLimits
from toylife2.records import inspect_record, load_session_inputs, verify_record
from toylife2.runner import reproduce, resume, run
from toylife2.specs import AssaySpec, ExecutionSpec, RunSpec, SpatialSpec
from toylife2.tables import DEFAULT_SOURCE, Tables


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    return ExecutionSpec(cache_dir=str(tmp_path_factory.mktemp("record-input-cache")))


@pytest.mark.integration
def test_alternate_table_pack_is_captured_and_replayed_without_original(execution, tmp_path):
    source = tmp_path / "alternate"
    shutil.copytree(DEFAULT_SOURCE, source)
    path = source / "data" / "protein_gene.txt"
    lines = path.read_text().splitlines()
    lines[0] = "1"
    path.write_text("\n".join(lines) + "\n")
    tables = Tables.load(source, execution.cache_dir)
    spec = RunSpec(schema_version=2, operation="chemistry", seed=3, samples=1, execution=execution)
    record = run(spec, destination=tmp_path / "alternate-run", tables=tables)
    assert inspect_record(record)["table_source_digest"] == tables.digest
    assert verify_record(record)["ok"]
    with zipfile.ZipFile(record / "source" / "tables.zip") as archive:
        assert archive.read("data/protein_gene.txt").splitlines()[0] == b"1"
    shutil.rmtree(source)
    replay = reproduce(record, tmp_path / "replay")
    assert verify_record(replay)["ok"]
    assert (
        inspect_record(replay)["scientific_identity"]
        == inspect_record(record)["scientific_identity"]
    )


def test_recording_rejects_in_memory_tables_without_corresponding_source(execution, tmp_path):
    tables = Tables.load(cache_dir=execution.cache_dir)
    genes = tables.gene_to_protein.copy()
    genes[0] = 1
    synthetic = dataclasses.replace(tables, gene_to_protein=genes)
    spec = RunSpec(seed=4, samples=1, execution=execution)
    with pytest.raises(ValueError, match="executing table arrays"):
        run(spec, destination=tmp_path / "invalid", tables=synthetic)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.integration
def test_explicit_panel_and_numeric_identities_survive_resume_and_replay(execution, tmp_path):
    spec = RunSpec(
        schema_version=2,
        operation="rollout",
        seed=123,
        samples=2,
        execution=execution,
        task=TaskSpec(kind="chemotaxis"),
        episode=EpisodeSpec(horizon=2),
        limits=WorkLimits(max_world_ticks=2),
    )
    genes = np.zeros((2, 2), np.uint32)
    spec = dataclasses.replace(spec, genotypes=tuple("0" * 40 for _ in genes))
    identities = {"candidates": np.array([91, 17], np.int64), "coordinates": (3, 8)}
    with BatchRunner(execution) as runner:
        session = runner.rollout(
            genes, spec=spec, panel=7, identities=identities, destination=tmp_path / "partial"
        )
        session.run()
    record = tmp_path / "partial"
    assert inspect_record(record)["status"] == "partial"
    saved = load_session_inputs(record)
    assert saved["panel"] == 7
    assert saved["identities"]["candidates"].dtype == np.int64
    assert saved["identities"]["coordinates"] == (3, 8)
    child = resume(record, destination=tmp_path / "continued", limits=WorkLimits())
    replay = reproduce(child, tmp_path / "replay")
    assert verify_record(replay)["ok"]
    assert load_session_inputs(replay)["panel"] == 7
    default = run(dataclasses.replace(spec, limits=WorkLimits()), destination=tmp_path / "default")
    assert (
        inspect_record(default)["scientific_identity"]
        != inspect_record(child)["scientific_identity"]
    )


@pytest.mark.parametrize(
    "operation,phenotype",
    [
        ("regulatory", "regulatory"),
        ("metabolic", "metabolic"),
        ("neighbourhood", "regulatory"),
        ("neighbourhood", "metabolic"),
        ("patterns", "regulatory"),
    ],
)
@pytest.mark.qualification
def test_every_coalesced_assay_child_has_standalone_artifacts_and_replays(
    execution, tmp_path, operation, phenotype
):
    spec = RunSpec(
        schema_version=2,
        operation=operation,
        seed=13,
        samples=1,
        execution=execution,
        assay=AssaySpec(phenotype=phenotype, environment=("0000",)),
        spatial=SpatialSpec(cells=3, steps=2),
    )
    with BatchRunner(execution) as runner:
        records = runner.run_batch(
            {"a": spec, "b": dataclasses.replace(spec, seed=14)}, destination=tmp_path / "batch"
        )
    for name, record in records.items():
        replay = reproduce(record, tmp_path / f"replay-{name}")
        assert verify_record(replay)["ok"]
        if operation == "neighbourhood":
            assert {"robustness", "neutral_mutations"} <= inspect_record(record)["artifacts"][
                "arrays"
            ].keys()


@pytest.mark.integration
def test_batch_recovery_rejects_changed_table_pack_before_new_children(execution, tmp_path):
    tables = Tables.load(cache_dir=execution.cache_dir)
    spec = RunSpec(
        schema_version=2, operation="regulatory", seed=29, samples=1, execution=execution
    )
    with BatchRunner(execution, tables=tables) as runner:
        runner.run_batch({"assay": spec}, destination=tmp_path / "batch")
    alternate = dataclasses.replace(tables, digest="alternate-model")
    with BatchRunner(execution, tables=alternate) as runner:
        with pytest.raises(ValueError, match="molecular tables differ"):
            runner.run_batch({"assay": spec}, destination=tmp_path / "batch", resume=True)
    assert not (tmp_path / "batch" / "assay-continuation-1").exists()


def test_recorded_neighbourhood_summary_admits_retained_result_and_scratch(execution):
    from toylife2 import Model
    from toylife2.runner import _fixed_artifacts

    physical = dataclasses.replace(execution, memory_budget_mib=5)
    model = Model(execution=physical)
    model.regulatory(np.zeros((1, 2), np.uint32))
    genes = np.zeros((170, 2), np.uint32)
    adjacent = model.neighbourhood(genes, phenotype="regulatory")
    spec = RunSpec(
        schema_version=2,
        seed=31,
        operation="neighbourhood",
        execution=physical,
        assay=AssaySpec(phenotype="regulatory"),
    )
    with pytest.raises(MemoryError, match="neighbourhood summary"):
        _fixed_artifacts(model, spec, genes, adjacent)
    assert model.engine._reserved_assay_bytes == 0
    assert model.engine.execution == physical
