"""Standalone CPU/Metal session qualification; run each backend in a fresh process.

Tests exact same-backend continuation, batched signalling and evolutionary state.
Integer scalar chemistry parity is covered by tests/test_chemistry.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import quadrants as qd

from toylife2 import BatchRunner, EpisodeSpec, ExecutionSpec, RunSpec, TaskSpec, WorkLimits
from toylife2.protocols import ObjectiveSpec
from toylife2.specs import EvolutionSpec

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
execution = ExecutionSpec(
    backend=args.backend,
    cache_dir=f".cache/toylife2-session-{args.backend}",
    batch_size=4,
    max_active_lanes=2,
)
with BatchRunner(execution) as runner:
    assert qd.lang.impl.current_cfg().arch == (qd.metal if args.backend == "metal" else qd.cpu)
    genes = np.array([[907050, 1048465], [1048465, 907050], [0, 0]], np.int32)
    for task in (TaskSpec(kind="temporal_probe"), TaskSpec(kind="chemotaxis")):
        episode = EpisodeSpec(horizon=4, episodes=2, chemical_updates=2)
        direct = runner.rollout(genes, task, episode, seed=101)
        direct.run()
        partial = runner.rollout(
            genes, task, episode, seed=101, limits=WorkLimits(max_chemical_updates=6)
        )
        partial.run()
        restored = runner.resume(partial.checkpoint())
        restored.run()
        assert restored.tick == 4
        np.testing.assert_array_equal(restored.scores, direct.scores)
        if task.kind == "chemotaxis":
            np.testing.assert_array_equal(
                restored.snapshot()["world"]["position"], direct.snapshot()["world"]["position"]
            )
    game = runner.rollout(
        np.stack([genes, genes], axis=1),
        TaskSpec(kind="signalling_game", config={"rounds": 2}),
        EpisodeSpec(horizon=4, episodes=2),
        agent_axis=-2,
        seed=23,
    )
    game.run()
    assert np.all(game.status == "complete")
    spec = RunSpec(
        schema_version=2,
        operation="wright_fisher",
        seed=123,
        execution=execution,
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=3, episodes=2),
        objective=ObjectiveSpec(kind="bounded_reward"),
        evolution=EvolutionSpec(
            trials=2, population_size=4, generations=2, mutation_probability=0.2
        ),
    )
    full = runner.evolution(spec)
    assert full.run().complete
    capped = runner.evolution(spec, limits=WorkLimits(max_evolution_updates=1))
    assert capped.run().status == "partial"
    continued = runner.resume(capped.checkpoint())
    assert continued.run().complete
    np.testing.assert_array_equal(continued.populations, full.populations)
    np.testing.assert_array_equal(
        continued.snapshot()["arrays"]["history"], full.snapshot()["arrays"]["history"]
    )
    result = {
        "backend": args.backend,
        "fallback": False,
        "checks": [
            "temporal",
            "chemotaxis",
            "signalling",
            "population",
            "pending_tick_replay",
            "population_replay",
        ],
        "population_counts": full.counters,
        "runtime": runner.metrics,
    }
    if args.output:
        from toylife2.records import _write_json

        _write_json(args.output, result)
    print(json.dumps(result))
