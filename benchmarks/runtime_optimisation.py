"""Synchronized chemistry and complete rollout timings in a fresh backend process."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from toylife2 import Model
from toylife2.chemistry import ChemistrySession
from toylife2.protocols import ControllerSpec, EpisodeSpec, PortSpec, ReadoutSpec, TaskSpec
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--genes", type=int, default=8)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--chemical-updates", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--duplicate", action="store_true")
    parser.add_argument("--metadata-layout", choices=("shared", "replicated"), default="shared")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=0)
    parser.add_argument(
        "--buckets", action="store_true", help="Measure separate graph-size cohorts"
    )
    args = parser.parse_args()
    model = Model(execution=ExecutionSpec(backend=args.backend, cpu_threads=args.threads))
    model.engine._chemistry_metadata_layout = args.metadata_layout
    model.engine._chemistry_block_size = args.block_size
    genes = np.random.default_rng(483).integers(
        0, 1 << 20, (args.population, args.genes), dtype=np.int32
    )
    if args.duplicate:
        genes[:] = genes[0]
    batch = np.repeat(genes[:, None, :], args.episodes, axis=1)
    spec = ControllerSpec(
        ports=PortSpec(
            metabolites=("0001", "0111"),
            max_dose=1,
            input_mode="replenish",
            readouts=(
                ReadoutSpec(name="motor", kind="bound", metabolite="0111", stage="after_output"),
            ),
        )
    )
    start = time.perf_counter()
    chemistry = ChemistrySession(model, batch, spec)
    cohorts = [chemistry]
    if args.buckets:
        from collections import defaultdict

        groups = defaultdict(list)
        for candidate in range(args.population):
            graph = chemistry._batch.graphs[candidate * args.episodes]
            groups[(max(1, len(graph[0])).bit_length(), max(1, len(graph[2])).bit_length())].append(
                candidate
            )
        cohorts = [ChemistrySession(model, batch[indices], spec) for indices in groups.values()]
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for cohort in cohorts:
        cohort.step({"0001": 0, "0111": 1})
    first = time.perf_counter() - start
    chemistry_samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        for tick in range(args.updates):
            for cohort in cohorts:
                cohort.step({"0001": tick % 2, "0111": 1})
        model.engine.qd.sync()
        chemistry_samples.append(time.perf_counter() - start)
    rollout_samples = []
    rollout_setup = []
    for _ in range(args.repeats + 1):
        construct = time.perf_counter()
        rollout = RolloutSession(
            model,
            genes,
            TaskSpec(kind="run_tumble"),
            EpisodeSpec(
                horizon=args.updates, episodes=args.episodes, chemical_updates=args.chemical_updates
            ),
            seed=484,
        )
        start = time.perf_counter()
        rollout_setup.append(start - construct)
        report = rollout.run()
        model.engine.qd.sync()
        assert report.status == "complete"
        rollout_samples.append(time.perf_counter() - start)
    rollout_digest = hashlib.sha256(
        rollout.scores.tobytes()
        + rollout.snapshot(fields=("world",))["world"]["position"].tobytes()
    ).hexdigest()
    generation_samples = []
    generation_digests = []
    if args.generations:
        from toylife2 import BatchRunner
        from toylife2.protocols import ObjectiveSpec
        from toylife2.specs import EvolutionSpec, ModelSpec, RunSpec

        with BatchRunner(model=model) as runner:
            for repeat in range(args.repeats):
                campaign = RunSpec(
                    schema_version=2,
                    operation="wright_fisher",
                    seed=484 + repeat,
                    execution=model.execution,
                    model=ModelSpec(genes=args.genes),
                    task=TaskSpec(kind="run_tumble"),
                    episode=EpisodeSpec(
                        horizon=args.updates,
                        episodes=args.episodes,
                        chemical_updates=args.chemical_updates,
                    ),
                    objective=ObjectiveSpec(kind="bounded_reward"),
                    evolution=EvolutionSpec(
                        trials=2,
                        population_size=args.population,
                        generations=args.generations,
                        mutation_probability=0.2,
                        initial="monomorphic_random" if args.duplicate else "independent_random",
                    ),
                )
                start = time.perf_counter()
                population = runner.evolution(campaign)
                assert population.run().complete
                model.engine.qd.sync()
                generation_samples.append(time.perf_counter() - start)
                generation_digests.append(
                    hashlib.sha256(population.populations.tobytes()).hexdigest()
                )
                runner.release(population)
    import toylife2

    source_root = Path(toylife2.__file__).parent
    fingerprint = hashlib.sha256()
    for source in sorted(source_root.glob("*.py")):
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "source_sha256": fingerprint.hexdigest(),
                "parameters": vars(args),
                "chemistry_setup_s": setup,
                "chemistry_first_step_s": first,
                "chemistry_samples_s": chemistry_samples,
                "rollout_samples_s": rollout_samples[1:],
                "rollout_setup_samples_s": rollout_setup,
                "generation_samples_s": generation_samples,
                "generation_digests": generation_digests,
                "rollout_digest": rollout_digest,
                "chemistry_buffer_bytes": dict(chemistry.buffer_memory),
                "lane_updates_per_sample": args.population * args.episodes * args.updates,
                "materialized_kernels": (
                    model.engine.qd.lang.impl.get_runtime().get_num_compiled_functions()
                ),
                "cohorts": len(cohorts),
                "cohort_buffer_bytes": sum(cohort.buffer_bytes for cohort in cohorts),
                "graph_cache": {
                    name: getattr(model.engine._chemistry_cache, name, None)
                    for name in ("hits", "misses", "bytes")
                }
                if hasattr(model.engine, "_chemistry_cache")
                else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
