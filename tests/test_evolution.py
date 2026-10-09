import dataclasses
import math

import numpy as np
import pytest

from toylife2.evolution import Fitness, log_acceptance, neutral_walk, sswm, wright_fisher
from toylife2.results import ArrayResult
from toylife2.specs import (
    AssaySpec,
    EvolutionSpec,
    ExecutionSpec,
    RecordingSpec,
    RunSpec,
    SpatialSpec,
)


class ConstantModel:
    def patterns(self, genes, spec):
        return ArrayResult(np.zeros((*genes.shape[:-1], spec.steps, spec.cells), np.uint8),
                           np.full(genes.shape[:-1], "complete"))

    def regulatory(self, genes):
        return ArrayResult(np.zeros((*genes.shape[:-1], 4), np.uint8),
                           np.full(genes.shape[:-1], "complete"))


def specification(operation, **changes):
    return RunSpec(operation=operation, seed=321, genotypes=("0" * 40,),
                   spatial=SpatialSpec(steps=2, cells=3),
                   assay=AssaySpec(phenotype="regulatory"),
                   evolution=EvolutionSpec(trials=2, population_size=17, generations=5,
                                           mutation_probability=1, **changes))


def test_fixation_rate_neutral_and_extreme_limits():
    assert log_acceptance(1, 1, 10000) == 0
    assert log_acceptance(1, 0, 100) == -math.inf
    assert log_acceptance(0, 1, 100) == math.log(100)
    assert math.exp(log_acceptance(.5, 1, 10)) == pytest.approx(10 * -.5 / (.5**10 - 1))
    assert log_acceptance(1, 1e-8, 10000) < -100000
    assert math.isfinite(log_acceptance(1, 1e-320, 10000))
    assert log_acceptance(1e300, 1e-300, 1) == 0
    assert math.exp(log_acceptance(1, 1 + 1e-12, 10000)) == pytest.approx(1, rel=1e-7)


def test_zero_fitness_uniform_parent_selection_still_mutates_and_resumes():
    spec = specification("wright_fisher", fitness_floor=0)
    target = np.ones((2, 3), np.uint8)
    model = ConstantModel()
    full = wright_fisher(model, spec, target)
    partial = wright_fisher(model, dataclasses.replace(spec, execution=ExecutionSpec(max_steps=1)),
                            target)
    assert partial["status"] == "partial"
    assert np.all(np.count_nonzero(partial["arrays"]["populations"][0], axis=-1) == 1)
    assert not np.any(partial["arrays"]["populations"][1])
    checkpoint = {"state": partial["state"], "loaded_arrays": partial["arrays"]}
    resumed = wright_fisher(model, spec, target, checkpoint=checkpoint)
    np.testing.assert_array_equal(full["arrays"]["populations"], resumed["arrays"]["populations"])
    np.testing.assert_array_equal(full["arrays"]["history"], resumed["arrays"]["history"])
    assert full["summary"]["mean_observed_arrival"] is None


def test_sswm_time_zero_and_overshoot_checkpoint():
    spec = specification("sswm", time_horizon=0)
    result = sswm(ConstantModel(), spec, np.zeros((2, 3), np.uint8))
    assert result["summary"]["arrivals"] == [0, 0]
    np.testing.assert_array_equal(result["arrays"]["founders"], np.zeros((2, 2)))
    spec = dataclasses.replace(spec, evolution=dataclasses.replace(spec.evolution,
                                                                  time_horizon=1e-100))
    checkpoints = []
    result = sswm(ConstantModel(), spec, np.ones((2, 3), np.uint8),
                  save=lambda s, a: checkpoints.append((dict(s), a)))
    assert len(result["arrays"]["history"]) == 2  # no overshoot mutation
    np.testing.assert_array_equal(result["arrays"]["founders"], np.zeros((2, 2)))


def test_sswm_and_neutral_walk_resume_are_exact():
    for operation in ("sswm", "neutral_walk"):
        spec = specification(operation, time_horizon=4)
        function = sswm if operation == "sswm" else neutral_walk
        kwargs = {"target": np.ones((2, 3), np.uint8)} if operation == "sswm" else {}
        full = function(ConstantModel(), spec, **kwargs)
        limited = dataclasses.replace(spec, execution=ExecutionSpec(max_steps=1, max_events=1))
        partial = function(ConstantModel(), limited, **kwargs)
        assert partial["status"] == "partial"
        resumed = function(ConstantModel(), spec,
                           checkpoint={"state": partial["state"],
                                       "loaded_arrays": partial["arrays"]}, **kwargs)
        np.testing.assert_array_equal(full["arrays"]["founders"], resumed["arrays"]["founders"])
        np.testing.assert_array_equal(full["arrays"]["history"], resumed["arrays"]["history"])


def test_fitness_is_exact_rational_and_floor():
    spec = specification("wright_fisher")
    fitness = Fitness(ConstantModel(), spec, np.ones((2, 3), np.uint8))
    assert fitness.denominator == 300000000
    np.testing.assert_array_equal(fitness.weights(np.zeros((2, 2), np.uint32)), [3, 3])


def test_every_terminal_save_rejection_stops_full_recording():
    for operation in ("wright_fisher", "sswm", "neutral_walk"):
        spec = specification(operation, time_horizon=0)
        spec = dataclasses.replace(spec, recording=RecordingSpec(mode="full"),
                                   evolution=dataclasses.replace(spec.evolution,
                                                                 trials=1, generations=1))
        function = {"wright_fisher": wright_fisher, "sswm": sswm,
                    "neutral_walk": neutral_walk}[operation]
        kwargs = {"target": np.zeros((2, 3), np.uint8)} if operation != "neutral_walk" else {}
        result = function(ConstantModel(), spec, **kwargs,
                          save=lambda state, arrays: state["trial"] == 0)
        assert result["status"] == "partial"
        assert result["state"]["trace_event"]["type"] == "trial_complete"
        assert result["state"]["trace_event"]["trial"] == 0


def test_full_wf_trace_matches_population_transition():
    spec = specification("wright_fisher")
    spec = dataclasses.replace(spec, recording=RecordingSpec(mode="full"))
    transitions = []

    def save(state, arrays):
        if "trace_event" in state:
            transitions.append(state.pop("trace_event"))
        return True

    wright_fisher(ConstantModel(), spec, np.zeros((2, 3), np.uint8), save=save)
    for event in transitions:
        if event["type"] != "generation":
            continue
        before = np.asarray(event["before"], np.uint32)
        expected = before[np.asarray(event["parents"], np.intp)].copy()
        for individual, position in event["mutations"]:
            expected[individual, position // 20] ^= np.uint32(1 << (19 - position % 20))
        np.testing.assert_array_equal(expected, event["after"])
        assert event["to_generation"] == event["from_generation"] + 1


def test_trial_shards_retain_logical_rng_identity():
    spec = specification("wright_fisher")
    target = np.ones((2, 3), np.uint8)
    all_trials = wright_fisher(ConstantModel(), spec, target)
    pieces = []
    for trial in range(2):
        shard = dataclasses.replace(spec, evolution=dataclasses.replace(spec.evolution,
                                                                       trials=1, trial_start=trial))
        pieces.append(wright_fisher(ConstantModel(), shard, target)["arrays"]["populations"])
    np.testing.assert_array_equal(all_trials["arrays"]["populations"], np.concatenate(pieces))


def test_terminal_quota_does_not_start_next_trial():
    spec = specification("wright_fisher")
    spec = dataclasses.replace(spec, recording=RecordingSpec(mode="full"))
    result = wright_fisher(ConstantModel(), spec, np.ones((2, 3), np.uint8),
                           save=lambda state, arrays: state["trial"] != 1)
    assert result["summary"]["completed_trials"] == 1
    assert result["summary"]["started_trials"] == 1
    assert result["summary"]["trial_status"] == ["complete", "unstarted"]
