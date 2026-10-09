from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from toylife2.conversion import convert_legacy_state, prepare_conversion
from toylife2.evolution import neutral_walk, sswm, wright_fisher
from toylife2.population import PopulationSession
from toylife2.records import begin_record, inspect_record, write_checkpoint
from toylife2.results import ArrayResult
from toylife2.specs import AssaySpec, EvolutionSpec, ExecutionSpec, RunSpec, SpatialSpec


class ConstantModel:
    def patterns(self, genes, spec):
        return ArrayResult(
            np.zeros((*genes.shape[:-1], spec.steps, spec.cells), np.uint8),
            np.full(genes.shape[:-1], "complete"),
        )

    def regulatory(self, genes):
        return ArrayResult(
            np.zeros((*genes.shape[:-1], 4), np.uint8), np.full(genes.shape[:-1], "complete")
        )


def spec(operation="wright_fisher"):
    return RunSpec(
        operation=operation,
        seed=321,
        genotypes=("0" * 40,),
        spatial=SpatialSpec(steps=2, cells=3),
        assay=AssaySpec(phenotype="regulatory"),
        evolution=EvolutionSpec(
            trials=3,
            trial_start=7,
            population_size=7,
            generations=4,
            mutation_probability=1,
            time_horizon=4,
        ),
    )


def legacy(model, run):
    function = {"wright_fisher": wright_fisher, "sswm": sswm, "neutral_walk": neutral_walk}[
        run.operation
    ]
    kwargs = {} if run.operation == "neutral_walk" else {"target": np.ones((2, 3), np.uint8)}
    return function(model, run, **kwargs)


@pytest.mark.parametrize("operation", ["wright_fisher", "sswm", "neutral_walk"])
def test_explicit_conversion_preserves_populations_rng_pending_and_retained_history(operation):
    model = ConstantModel()
    run = spec(operation)
    full = legacy(model, run)
    limited = dataclasses.replace(run, execution=ExecutionSpec(max_steps=1, max_events=1))
    partial = legacy(model, limited)
    loaded = {"state": partial["state"], "loaded_arrays": partial["arrays"]}
    converted, cp = convert_legacy_state(model, limited, loaded, target=np.ones((2, 3), np.uint8))
    assert converted.schema_version == 2
    payload = cp.payload["payload"]
    np.testing.assert_array_equal(payload["replicates"], [7, 8, 9])
    assert [row for key, row in payload["history"]] == partial["state"]["history"]
    assert payload["observation_count"] == len(partial["state"]["history"])
    assert cp.payload["base"]["total"]["evolution_updates"] == 1
    resumed = PopulationSession(model, converted, checkpoint=cp, target=np.ones((2, 3), np.uint8))
    # No quota supplied by the legacy spec; v2 continuation reaches its horizon.
    resumed.run()
    key = "populations" if operation == "wright_fisher" else "founders"
    np.testing.assert_array_equal(resumed.snapshot()["arrays"][key], full["arrays"][key])
    assert resumed.arrivals == full["state"]["arrivals"]
    if operation == "sswm":
        assert resumed.pending == full["state"]["pending_events"]


def test_finished_lanes_and_pending_overshoot_convert_without_consuming_rng():
    model = ConstantModel()
    run = dataclasses.replace(
        spec("sswm"), evolution=dataclasses.replace(spec("sswm").evolution, time_horizon=1e-100)
    )
    finished = legacy(model, run)
    converted, cp = convert_legacy_state(
        model,
        run,
        {"state": finished["state"], "loaded_arrays": finished["arrays"]},
        target=np.ones((2, 3), np.uint8),
    )
    payload = cp.payload["payload"]
    assert payload["lane_status"].tolist() == ["complete"] * 3
    assert payload["pending"] == finished["state"]["pending_events"]
    assert cp.payload["base"]["status"] == "complete"
    np.testing.assert_array_equal(payload["cursors"], [0, 0, 0])
    assert converted.seed == run.seed


def test_invalid_legacy_arrays_rng_and_erased_sswm_cursor_refused():
    model = ConstantModel()
    run = spec()
    result = legacy(model, dataclasses.replace(run, execution=ExecutionSpec(max_steps=1)))
    loaded = {"state": dict(result["state"]), "loaded_arrays": result["arrays"]}
    loaded["state"]["rng"] = "unknown-rng"
    with pytest.raises(ValueError, match="RNG"):
        convert_legacy_state(model, run, loaded, target=np.ones((2, 3), np.uint8))
    loaded["state"] = result["state"]
    loaded["loaded_arrays"] = {"populations": np.zeros((1, 7, 2), np.uint32)}
    with pytest.raises(ValueError, match="population arrays"):
        convert_legacy_state(model, run, loaded, target=np.ones((2, 3), np.uint8))
    ss = spec("sswm")
    completed = legacy(model, ss)
    completed["state"]["history"] = []
    with pytest.raises(ValueError, match="erased"):
        convert_legacy_state(
            model,
            ss,
            {
                "state": completed["state"],
                "loaded_arrays": {"founders": completed["arrays"]["founders"]},
            },
            target=np.ones((2, 3), np.uint8),
        )


def test_recorded_prepare_links_original_manifest_checkpoint_and_source(tmp_path):
    model = ConstantModel()
    run = dataclasses.replace(spec("neutral_walk"), execution=ExecutionSpec(max_steps=1))
    result = legacy(model, run)
    record = begin_record(run, tmp_path / "legacy")
    original = write_checkpoint(record, result["state"], arrays=result["arrays"])
    prepared = prepare_conversion(original, model)
    assert prepared.spec.schema_version == 2
    assert prepared.checkpoint.kind == "population"
    assert prepared.parent["relationship"] == "convert"
    assert prepared.parent["manifest_sha256"] and prepared.parent["checkpoint_sha256"]
    assert prepared.parent["scientific_identity"] == inspect_record(record)["scientific_identity"]
    assert prepared.compatibility in {"identical_source", "identical_v1_numerical_core"}


def test_source_mismatch_never_executes_captured_code_and_refuses_changed_core(
    tmp_path, monkeypatch
):
    import zipfile

    from toylife2 import runner
    from toylife2.conversion import _qualify_captured_core

    run = dataclasses.replace(spec("neutral_walk"), execution=ExecutionSpec(max_steps=1))
    result = legacy(ConstantModel(), run)
    record = begin_record(run, tmp_path / "legacy")
    original = write_checkpoint(record, result["state"], arrays=result["arrays"])
    monkeypatch.setattr(
        runner,
        "_check_source",
        lambda path: (_ for _ in ()).throw(
            ValueError("executed source differs from record; restore captured source first")
        ),
    )
    assert (
        prepare_conversion(original, ConstantModel()).compatibility == "identical_v1_numerical_core"
    )
    source = record / "source" / "source.zip"
    with zipfile.ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["src/toylife2/random.py"] = b"raise RuntimeError('must never execute')\n"
    with zipfile.ZipFile(source, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="numerical source differs: random.py"):
        _qualify_captured_core(record)
