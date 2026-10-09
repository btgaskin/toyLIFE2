import dataclasses

import numpy as np
import pytest

from toylife2.batch import BatchRunner
from toylife2.protocols import EpisodeSpec, TaskSpec, WorkLimits
from toylife2.records import inspect_record, read_checkpoint, verify_record
from toylife2.runner import resume, run
from toylife2.specs import ExecutionSpec, RunSpec


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    return ExecutionSpec(cache_dir=str(tmp_path_factory.mktemp("batch-cache")))


def test_shared_runner_preserves_nested_axes_and_resume(execution):
    with BatchRunner(execution) as runner:
        genes = np.zeros((2, 3, 2), dtype=np.int32)
        episode = EpisodeSpec(horizon=2, episodes=2, chemical_updates=2)
        session = runner.rollout(
            genes, protocol=episode, limits=WorkLimits(max_chemical_updates=12)
        )
        session.run()
        assert session.tick == 0
        continued = runner.resume(session.checkpoint())
        continued.run()
        assert continued.scores.shape == (2, 3, 2)
        assert continued.tick == 2
        assert runner.model.engine is continued.model.engine
        assert continued.counters["total"]["chemical_updates"] == 48
    with pytest.raises(RuntimeError, match="closed"):
        runner.chemistry([0, 0])


def test_recorded_v2_partial_resume_and_reproduce(execution, tmp_path):
    spec = RunSpec(
        schema_version=2,
        operation="rollout",
        seed=61,
        samples=3,
        execution=execution,
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=3, episodes=2),
        limits=WorkLimits(max_world_ticks=6),
    )
    original = run(spec, destination=tmp_path / "partial")
    assert inspect_record(original)["status"] == "partial"
    assert verify_record(original)["ok"]
    child = resume(original, destination=tmp_path / "child", limits=WorkLimits())
    assert inspect_record(child)["status"] == "complete"
    assert verify_record(child)["ok"]
    direct = run(dataclasses.replace(spec, limits=WorkLimits()), destination=tmp_path / "direct")
    for name, artifact in inspect_record(child)["artifacts"]["arrays"].items():
        expected = inspect_record(direct)["artifacts"]["arrays"][name]
        np.testing.assert_array_equal(
            np.load(child / artifact["path"]), np.load(direct / expected["path"])
        )
    state = read_checkpoint(child)["state"]
    assert state["session_state"]["base"]["pending_trace"] == []
    from toylife2.runner import reproduce

    repeated = reproduce(child, tmp_path / "reproduced")
    assert verify_record(repeated)["ok"]


def test_runner_v2_chemistry_and_tissue(execution, tmp_path):
    for operation in ("chemistry", "tissue"):
        spec = RunSpec(
            schema_version=2, operation=operation, seed=1, samples=2, execution=execution
        )
        record = run(spec, destination=tmp_path / operation)
        assert inspect_record(record)["status"] == "complete"
        assert verify_record(record)["ok"]


def test_named_batch_index_and_idempotent_recovery(execution, tmp_path):
    import json

    specs = {
        name: RunSpec(
            schema_version=2,
            operation="rollout",
            seed=i,
            execution=execution,
            episode=EpisodeSpec(horizon=2),
            task=TaskSpec(kind="temporal_probe"),
        )
        for i, name in enumerate(("first", "second"))
    }
    with BatchRunner(execution) as batch:
        outputs = batch.run_batch(specs, destination=tmp_path / "batch")
        assert set(outputs) == set(specs)
        assert all(inspect_record(path)["status"] == "complete" for path in outputs.values())
        again = batch.run_batch(specs, destination=tmp_path / "batch", resume=True)
        assert again == outputs
    index = json.loads((tmp_path / "batch" / "batch.json").read_text())
    assert set(index["children"]) == set(specs)
    assert all(value["status"] == "complete" for value in index["children"].values())


def test_full_trace_rejected_checkpoint_retries_without_dropping_rows(execution, tmp_path):
    from toylife2.records import begin_record, write_boundary
    from toylife2.sessions import Checkpoint
    from toylife2.specs import RecordingSpec

    spec = RunSpec(
        schema_version=2,
        operation="rollout",
        seed=1,
        execution=execution,
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=1),
        recording=RecordingSpec(mode="full", trace_budget_mib=1),
    )
    record = begin_record(spec, tmp_path / "trace")
    with BatchRunner(execution) as runner:
        original = runner.from_spec(spec)
        original.run()
        rows = original.drain_trace()
        assert len(rows) >= 2
        # A single indivisible boundary larger than the record's budget must
        # retain its complete trace alongside the staged checkpoint.
        rows[0]["large"] = "x" * (2 * 1024 * 1024)
        cp = original.checkpoint()
        _, accepted = write_boundary(record, cp.record_state(), arrays=cp.arrays, events=rows)
        assert not accepted
        loaded = Checkpoint.from_record(read_checkpoint(record))
        assert loaded.payload["base"]["pending_trace"][0]["large"]
        resumed = runner.resume(loaded)
        seen = []
        resumed._record_callback = lambda checkpoint, events: seen.extend(events) or True
        resumed.run()
        assert resumed.tick == 1
        assert len(seen) == len(rows)
        assert not resumed._pending_trace


def test_fixed_assays_coalesce_across_named_runs(execution, tmp_path, monkeypatch):
    from toylife2 import runner as runner_module

    calls = []
    evaluate = runner_module._evaluate

    def tracked(model, spec, genes):
        calls.append(len(genes))
        return evaluate(model, spec, genes)

    monkeypatch.setattr(runner_module, "_evaluate", tracked)
    specs = {
        name: RunSpec(
            schema_version=2, operation="regulatory", seed=i, samples=3, execution=execution
        )
        for i, name in enumerate(("a", "b", "c"))
    }
    with BatchRunner(execution) as batch:
        outputs = batch.run_batch(specs, destination=tmp_path / "assays")
    assert calls == [9]
    assert all(verify_record(path)["ok"] for path in outputs.values())


def test_explicit_v1_record_conversion_into_linked_v2(execution, tmp_path):
    from toylife2.runner import convert_checkpoint
    from toylife2.specs import AssaySpec, EvolutionSpec

    v1 = RunSpec(
        operation="neutral_walk",
        seed=3,
        execution=dataclasses.replace(execution, max_steps=1),
        evolution=EvolutionSpec(trials=2, generations=2),
        assay=AssaySpec(phenotype="regulatory"),
    )
    record = run(v1, destination=tmp_path / "legacy")
    assert inspect_record(record)["status"] == "partial"
    child = convert_checkpoint(record, tmp_path / "converted", limits=WorkLimits())
    assert inspect_record(child)["status"] == "complete"
    assert inspect_record(child)["parent"]["relationship"] == "convert"
    assert verify_record(child)["ok"]


def test_numerical_checkpoint_reserved_keys_are_unambiguous():
    from toylife2.sessions import Checkpoint, pack

    values = {
        "world": {"__array__": "scientific-label"},
        "tuple": {"__tuple__": [1, 2]},
        "mapping": {"__mapping__": {"label": 3}},
        "actual": np.array([1, 2], np.int64),
    }
    arrays = {}
    cp = Checkpoint("test", "identity", pack(values, arrays), arrays)
    restored = cp.payload
    assert restored["world"] == values["world"]
    assert restored["tuple"] == values["tuple"]
    assert restored["mapping"] == values["mapping"]
    np.testing.assert_array_equal(restored["actual"], values["actual"])


def test_batch_partial_fixed_assay_restarts_with_larger_physical_cap(execution, tmp_path):
    from toylife2.specs import AssaySpec

    capped = RunSpec(
        schema_version=2,
        operation="metabolic",
        genotypes=("0" * 40,),
        assay=AssaySpec(environment=("0000",)),
        execution=execution,
        limits=WorkLimits(max_assay_steps=1),
    )
    with BatchRunner(execution) as batch:
        partial = batch.run_batch({"assay": capped}, destination=tmp_path / "batch")
        assert inspect_record(partial["assay"])["status"] == "partial"
        revised = dataclasses.replace(capped, limits=WorkLimits())
        continued = batch.run_batch({"assay": revised}, destination=tmp_path / "batch", resume=True)
        assert inspect_record(continued["assay"])["status"] == "complete"
        assert inspect_record(continued["assay"])["parent"]["relationship"] == "resume"
        assert verify_record(continued["assay"])["ok"]
