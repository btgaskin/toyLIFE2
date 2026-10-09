import dataclasses

import numpy as np
import pytest

from toylife2.evolution import neutral_walk, sswm, wright_fisher
from toylife2.population import PopulationSession, wright_fisher_step
from toylife2.protocols import ObjectiveSpec, WorkLimits
from toylife2.random import Stream
from toylife2.results import ArrayResult
from toylife2.specs import AssaySpec, EvolutionSpec, ExecutionSpec, RunSpec, SpatialSpec


class ConstantModel:
    def __init__(self):
        self.fail_mutants = False

    def patterns(self, genes, spec):
        status = np.full(genes.shape[:-1], "complete", dtype="U12")
        if self.fail_mutants:
            status[np.any(genes != 0, axis=-1)] = "incomplete"
        return ArrayResult(np.zeros((*genes.shape[:-1], spec.steps, spec.cells), np.uint8), status)

    def regulatory(self, genes):
        return ArrayResult(
            np.zeros((*genes.shape[:-1], 4), np.uint8), np.full(genes.shape[:-1], "complete")
        )


def spec(operation="wright_fisher", **kwargs):
    return RunSpec(
        operation=operation,
        seed=321,
        genotypes=("0" * 40,),
        spatial=SpatialSpec(steps=2, cells=3),
        assay=AssaySpec(phenotype="regulatory"),
        evolution=EvolutionSpec(
            trials=3,
            population_size=7,
            generations=4,
            mutation_probability=1,
            time_horizon=4,
            **kwargs,
        ),
    )


@pytest.mark.parametrize("operation", ["wright_fisher", "sswm", "neutral_walk"])
def test_actual_legacy_trajectories_histories_arrivals_and_pending_equal(operation):
    run = spec(operation)
    target = np.ones((2, 3), np.uint8)
    model = ConstantModel()
    legacy = (
        neutral_walk(model, run)
        if operation == "neutral_walk"
        else {"wright_fisher": wright_fisher, "sswm": sswm}[operation](model, run, target)
    )
    session = PopulationSession(model, run, target=target)
    assert session.run().complete
    result = session.snapshot()
    key = "populations" if operation == "wright_fisher" else "founders"
    np.testing.assert_array_equal(result["arrays"][key], legacy["arrays"][key])
    np.testing.assert_array_equal(result["arrays"]["history"], legacy["arrays"]["history"])
    assert result["summary"]["arrivals"] == legacy["summary"]["arrivals"]
    assert result["summary"]["observed_horizons"] == legacy["summary"]["observed_horizons"]
    if operation == "sswm":
        assert result["summary"]["pending_events"] == legacy["summary"]["pending_events"]


@pytest.mark.parametrize("operation", ["wright_fisher", "sswm", "neutral_walk"])
def test_small_invocation_resume_preserves_frontier_and_pinned_history(operation):
    run = spec(operation)
    target = np.ones((2, 3), np.uint8)
    model = ConstantModel()
    full = PopulationSession(model, run, target=target)
    full.run()
    partial = PopulationSession(
        model, run, target=target, limits=WorkLimits(max_evolution_updates=1)
    )
    assert partial.run().status == "partial"
    count = partial.counters["invocation"]["evolution_updates"]
    partial.observe()
    assert partial.counters["invocation"]["evolution_updates"] == count
    cp = partial.checkpoint()
    # Each construction represents a new authorized invocation quota.
    for _ in range(60):
        resumed = PopulationSession(
            model, run, target=target, checkpoint=cp, limits=WorkLimits(max_evolution_updates=1)
        )
        report = resumed.run()
        cp = resumed.checkpoint()
        if report.complete:
            break
    assert report.complete
    for key in ("populations", "history"):
        np.testing.assert_array_equal(
            full.snapshot()["arrays"][key], resumed.snapshot()["arrays"][key]
        )
    assert full.arrivals == resumed.arrivals


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_single_member_all_zero_weights_and_mutation_extremes(probability):
    genes = np.zeros((1, 2), np.uint32)
    out, parents, bits, gate = wright_fisher_step(
        genes,
        np.asarray([0]),
        stream=Stream(9),
        replicate=55,
        generation=2,
        mutation_probability=probability,
    )
    assert int(parents[0]) == 0
    assert bool(gate[0]) == bool(probability)
    assert sum(int(v).bit_count() for v in out[0]) == int(probability)


def test_wf_prefix_admission_and_right_search_do_not_select_zero_weight():
    genes = np.asarray([[0, 0], [1, 0], [2, 0]], np.uint32)
    _, parents, _, _ = wright_fisher_step(
        genes,
        np.asarray([0, 5, 0]),
        stream=Stream(1),
        replicate=0,
        generation=0,
        mutation_probability=0,
    )
    np.testing.assert_array_equal(parents, [1, 1, 1])
    with pytest.raises(OverflowError):
        wright_fisher_step(
            genes,
            np.full(3, (1 << 62), np.int64),
            stream=Stream(1),
            replicate=0,
            generation=0,
            mutation_probability=0,
        )


def test_failed_rescore_retries_advanced_population_without_new_transition():
    run = spec()
    target = np.ones((2, 3), np.uint8)
    model = ConstantModel()
    session = PopulationSession(model, run, target=target)
    session.observe()
    model.fail_mutants = True
    assert session.step().status == "partial"
    np.testing.assert_array_equal(session.cursors, [1, 1, 1])
    np.testing.assert_array_equal(session.observed, [0, 0, 0])
    saved = session.populations.copy()
    cp = session.checkpoint()
    model.fail_mutants = False
    resumed = PopulationSession(
        model, run, target=target, checkpoint=cp, limits=WorkLimits(max_evolution_updates=0)
    )
    assert resumed.run().status == "partial"
    np.testing.assert_array_equal(resumed.populations, saved)
    np.testing.assert_array_equal(resumed.observed, [1, 1, 1])
    assert resumed.counters["invocation"]["evolution_updates"] == 0


def test_zero_generations_observes_initial_even_with_zero_transition_quota():
    run = dataclasses.replace(
        spec(),
        schema_version=2,
        evolution=EvolutionSpec(trials=2, population_size=1, generations=0),
        objective=ObjectiveSpec(kind="pattern_match"),
    )
    session = PopulationSession(
        ConstantModel(),
        run,
        target=np.zeros((2, 3), np.uint8),
        limits=WorkLimits(max_evolution_updates=0),
    )
    assert session.run().complete
    assert session.counters["total"]["evolution_updates"] == 0
    assert session.arrivals == [0, 0]
    assert len(session.history) == 2


def test_assay_chunk_sizes_and_active_cohorts_preserve_completed_output():
    base = dataclasses.replace(
        spec(), schema_version=2, objective=ObjectiveSpec(kind="pattern_match")
    )
    target = np.zeros((2, 3), np.uint8)
    outputs = []
    for chunk, active in ((1, 1), (2, 2), (100, 64)):
        run = dataclasses.replace(
            base, execution=ExecutionSpec(batch_size=chunk, max_active_lanes=active)
        )
        session = PopulationSession(ConstantModel(), run, target=target)
        assert session.run().complete
        outputs.append(session.snapshot())
    for output in outputs[1:]:
        for key in ("populations", "history"):
            np.testing.assert_array_equal(output["arrays"][key], outputs[0]["arrays"][key])
        assert output["summary"]["arrivals"] == outputs[0]["summary"]["arrivals"]


def test_pinned_canonical_last_history_rows_ignore_observation_arrival_order():
    base = dataclasses.replace(
        spec(), schema_version=2, objective=ObjectiveSpec(kind="pattern_match")
    )
    outputs = []
    for active in (1, 3):
        run = dataclasses.replace(base, execution=ExecutionSpec(max_active_lanes=active))
        session = PopulationSession(ConstantModel(), run, target=np.zeros((2, 3), np.uint8))
        session.history_capacity = 2
        assert session.run().complete
        assert session.observation_count == 15
        assert list(session.history) == [(2, 3), (2, 4)]
        assert session.snapshot()["summary"]["history_dropped"] == 13
        outputs.append(session.snapshot()["arrays"]["history"])
    np.testing.assert_array_equal(*outputs)
