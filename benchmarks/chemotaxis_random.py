"""Completed chemotaxis rollouts for host/direct RNG comparison; no evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from toylife2 import Model
from toylife2.protocols import (
    SINGLE_CELL_CONTRACT,
    BinaryOutputSpec,
    EpisodeSpec,
    StructuralInterface,
    TaskSpec,
)
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec
from toylife2.tables import Tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--provider", choices=("baseline", "host", "direct"), default="host")
    parser.add_argument("--genomes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--decisions", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--reuse-panel", action="store_true")
    parser.add_argument("--field", choices=("static", "source", "dynamic"), default="static")
    parser.add_argument("--memory-mib", type=int, default=1024)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.genomes, args.episodes, args.decisions, args.repeats) < 1:
        parser.error("workload dimensions must be positive")
    started = time.perf_counter()
    execution = ExecutionSpec(
        backend=args.backend, cpu_threads=1, memory_budget_mib=args.memory_mib
    )
    tables = Tables.load(
        source=Path(__file__).resolve().parents[1] / "upstream/toylife", cache_dir=".cache/toylife2"
    )
    model = Model(tables=tables, execution=execution)
    model.engine._world_random_provider = args.provider
    model.engine._world_block_size = args.block_size
    construction = time.perf_counter() - started
    config = dict(
        experiment_contract=SINGLE_CELL_CONTRACT,
        world_engine="quadrants",
        world_precision="float32",
        field="source" if args.field == "source" else "fluctuating",
        static=args.field != "dynamic",
    )
    if args.provider != "baseline":
        config["world_random_math"] = "box_muller_f32_v1"
    task = TaskSpec(kind="chemotaxis", config=config)
    controller = StructuralInterface(
        inputs={"cue": "00101110"}, targets={"motor": "1011"}
    ).controller(output=BinaryOutputSpec(markers=("motor",)))
    results = []
    for case, count in enumerate(args.genomes):
        genes = np.random.default_rng(123).integers(0, 1 << 20, (count, 2), dtype=np.uint32)
        samples = []
        for repeat in range(args.repeats + 1):
            panel = 100 if args.reuse_panel else 100 + repeat + case * 1000
            model.engine.qd.sync()
            start = time.perf_counter()
            rollout = RolloutSession(
                model,
                genes,
                task,
                EpisodeSpec(horizon=args.decisions, episodes=args.episodes, panel="fixed"),
                controller,
                seed=123,
                panel=panel,
            )
            report = rollout.run()
            model.engine.qd.sync()
            elapsed = time.perf_counter() - start
            assert report.status == "complete", report.status
            buffers = rollout._resident.world
            row = dict(
                seconds=elapsed,
                scores=rollout.scores.tolist(),
                implementation=getattr(buffers, "random_implementation", "baseline"),
                uploads=buffers.uploads,
                readbacks=buffers.readbacks,
                cache_hits=model.engine._world_random_cache.hits,
            )
            samples.append(row)
            del rollout
        results.append(
            dict(
                genomes=count,
                first_seconds=samples[0]["seconds"],
                samples=samples[1:],
                median_seconds=float(np.median([s["seconds"] for s in samples[1:]])),
            )
        )
        print(
            json.dumps(
                {key: results[-1][key] for key in ("genomes", "first_seconds", "median_seconds")}
            ),
            flush=True,
        )
    import quadrants as qd

    import toylife2

    source = Path(toylife2.__file__).parent
    receipt = dict(
        backend=execution.backend,
        provider=args.provider,
        platform=platform.platform(),
        quadrants=qd.__version__,
        module=str(source),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
        },
        setup_seconds=construction,
        parameters=vars(args) | {"destination": str(args.destination)},
        results=results,
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
