"""Clonal inheritance preserves the existing two-cell information route."""

from dataclasses import replace

import numpy as np
import pytest

from toylife2 import BatchRunner, GeneCopySpec, GenomeBatch, Model
from toylife2.molecular_experiments import profile_spec
from toylife2.protocols import WorkLimits
from toylife2.rollout import RolloutSession
from toylife2.specs import plain


@pytest.mark.device
def test_clonal_matches_explicit_pairs_and_mixed_length_resume(canonical_tables):
    spec = profile_spec("signalling_game", genes=2, response_updates=2)
    model = Model(tables=canonical_tables)
    spec = replace(
        spec,
        execution=model.execution,
        episode=replace(spec.episode, horizon=4),
        task=replace(spec.task, config={**spec.task.config, "rounds": 2}),
        evolution=replace(spec.evolution, gene_copy=GeneCopySpec(0.1, 0.1)),
    )
    genomes = GenomeBatch(np.array([[0, 0], [335699, 583833]], np.uint32), [1, 2])
    with BatchRunner(model=model) as runner:
        evaluator = runner.evaluator(spec)
        expected = evaluator.evaluate(genomes)
        for i, count in enumerate(genomes.gene_counts):
            genes = genomes.genes[i : i + 1, :count]
            explicit = RolloutSession(
                model,
                np.repeat(genes[:, None], 2, axis=1),
                replace(
                    spec.task,
                    config={k: v for k, v in spec.task.config.items() if k != "pairing"},
                ),
                spec.episode,
                spec.controller,
                seed=spec.seed,
                agent_axis=-2,
            )
            explicit.run()
            np.testing.assert_array_equal(expected.scores[i], explicit.aggregate_scores()[0])
        partial = runner.evaluator(spec)
        partial.set_work_limits(WorkLimits(max_chemical_updates=3))
        assert not partial.evaluate(genomes).complete
        saved = partial.checkpoint_payload()
        partial.close()
        resumed = runner.evaluator(spec)
        resumed.restore_payload(saved)
        actual = resumed.evaluate(genomes)
        assert actual.complete
        np.testing.assert_array_equal(actual.scores, expected.scores)
        np.testing.assert_array_equal(actual.weights, expected.weights)


@pytest.mark.device
def test_sender_private_cue_and_delivered_message_use_separate_cells(canonical_tables):
    spec = profile_spec("signalling_game")
    model = Model(tables=canonical_tables)
    rollout = RolloutSession(
        model,
        np.array([[0, 1]], np.uint32),
        spec.task,
        spec.episode,
        spec.controller,
        seed=spec.seed,
    )
    state = {
        "tick": 0,
        "private": np.ones(rollout.world_shape, np.int32),
        "delivered": np.zeros(rollout.world_shape, np.int32),
    }
    encoded, _ = rollout._encode(rollout.world.observe(state), {})
    cue = spec.controller.interface.inputs["cue"]
    active = spec.controller.interface.inputs["active"]
    np.testing.assert_array_equal(encoded[cue][..., 0], 8)
    np.testing.assert_array_equal(encoded[cue][..., 1], 0)
    np.testing.assert_array_equal(encoded[active][..., 1], 0)
    state, _, _ = rollout.world.advance(state, np.zeros(rollout.controller_shape, np.int32))
    encoded, _ = rollout._encode(rollout.world.observe(state), {})
    np.testing.assert_array_equal(encoded[cue], 0)  # Receiver sees the delivered zero.
    np.testing.assert_array_equal(encoded[active][..., 0], 0)
    np.testing.assert_array_equal(encoded[active][..., 1], 8)
    payload = rollout._checkpoint_payload()
    assert payload["agent_axis"] is None
    assert np.array_equal(payload["genotypes"], [[0, 1]])
    assert plain(spec.task)["config"]["pairing"] == "clonal"


def test_factorial_analysis_pairs_trials_and_separates_main_effects():
    import itertools
    import runpy
    from pathlib import Path

    analysis = runpy.run_path(str(Path(__file__).parents[1] / "examples/signalling_factorial.py"))
    rows = [
        {
            "switches": switches,
            "accuracy": [0.4 + 0.2 * switches[0] + 0.01 * trial for trial in range(8)],
        }
        for switches in itertools.product((0, 1), repeat=3)
    ]
    effects = analysis["contrast"](rows, "accuracy")
    np.testing.assert_allclose(effects["mutation"]["paired_replicates"], 0.2)
    for factor, effect in effects.items():
        if factor != "mutation":
            np.testing.assert_allclose(effect["paired_replicates"], 0, atol=1e-15)
