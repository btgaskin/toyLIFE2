"""Prepare a matched single-cell operator campaign; execution requires --run.

The default matrix has 2 founder constructions × 8 operator settings × 5 sizes.
Each interface seed is a separate block. Output rules are explicit campaign choices.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from toylife2 import BatchRunner, GeneCopySpec, Model
from toylife2.config import write_spec
from toylife2.molecular_experiments import profile_spec
from toylife2.specs import plain, scientific_identity

FACTORS = ("mutation", "deletion", "duplication")


def conditions(args):
    for interface_seed in args.interface_seeds:
        base = profile_spec(
            args.profile,
            seed=args.seed,
            backend=args.backend,
            interface_seed=interface_seed,
            cue_length=args.cue_length,
            marker_length=args.marker_length,
            output_representation=args.output_representation,
            output_rule=args.output_rule,
            marker_copies=args.marker_copies,
            response_updates=args.response_updates,
        )
        for initial, genes, switches in itertools.product(
            args.initial, args.genes, itertools.product((0, 1), repeat=3)
        ):
            spec = replace(
                base,
                model=replace(base.model, genes=genes),
                evolution=replace(
                    base.evolution,
                    initial=initial,
                    population_size=args.population,
                    generations=args.generations,
                    trials=args.trials,
                    mutation_probability=args.mutation if switches[0] else 0,
                    gene_copy=GeneCopySpec(
                        deletion_probability=args.deletion if switches[1] else 0,
                        duplication_probability=args.duplication if switches[2] else 0,
                    ),
                ),
            )
            name = f"i{interface_seed}-{initial}-g{genes}-{''.join(map(str, switches))}"
            yield name, switches, spec


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_condition(model, spec, root):
    """Sampled reward selects; expected accuracy defines diagnostic solution events."""
    with BatchRunner(model=model) as runner:
        session = runner.evolution(spec, destination=root / "evolution")
        session.observe()
        founders = [set() for _ in session.replicates]
        first_solution = [None] * len(founders)
        first_novel_solution = [None] * len(founders)
        initial_unique_solutions = [0] * len(founders)
        rows = []
        np.savez_compressed(
            root / "initial_population.npz",
            genes=session.populations,
            gene_counts=session.gene_counts,
        )
        founder_hash = hashlib.sha256(
            session.populations.tobytes() + session.gene_counts.tobytes()
        ).hexdigest()
        last_observed = None
        started = time.perf_counter()

        def observe():
            nonlocal last_observed
            generations = tuple(map(int, session.observed))
            if generations == last_observed:
                return
            last_observed = generations
            for trial, value in enumerate(session.evaluations):
                if value is None:
                    continue
                generation = generations[trial]
                keys = [
                    tuple(map(int, genome[:count]))
                    for genome, count in zip(
                        session.populations[trial], session.gene_counts[trial], strict=True
                    )
                ]
                expected = value.metrics["expected_accuracy"]
                if generation == 0:
                    founders[trial] = set(keys)
                    initial_unique_solutions[trial] = len(
                        {key for key, score in zip(keys, expected, strict=True) if score == 1}
                    )
                perfect = np.flatnonzero(expected == 1)
                if len(perfect) and first_solution[trial] is None:
                    first_solution[trial] = generation
                if first_novel_solution[trial] is None and any(
                    keys[i] not in founders[trial] for i in perfect
                ):
                    first_novel_solution[trial] = generation
                row = dict(
                    trial=int(session.replicates[trial]),
                    generation=generation,
                    sampled_mean=float(value.scores.mean()),
                    sampled_best=float(value.scores.max()),
                    expected_mean=float(expected.mean()),
                    expected_best=float(expected.max()),
                    expected_perfect_fraction=float(np.mean(expected == 1)),
                    mean_genes=float(session.gene_counts[trial].mean()),
                    production={
                        k: float(np.mean(v))
                        for k, v in value.metrics.items()
                        if k.startswith(("translation_events_", "chemical_updates_"))
                    },
                )
                rows.append(row)
            progress = dict(
                generations=generations,
                elapsed_seconds=time.perf_counter() - started,
                evaluator=dict(session.evaluator.metrics),
                latest=rows[-len(founders) :],
            )
            write_json(root / "progress.json", progress)
            print(json.dumps(dict(condition=root.name, **progress)), flush=True)

        observe()
        while session.state["status"] not in {"complete", "partial", "failed"}:
            session.step()
            observe()
        if session.state["status"] != "complete":
            runner.finish(session)
            raise RuntimeError(f"incomplete condition: {session.state}")
        result = dict(
            founder_sha256=founder_hash,
            initial_unique_solutions=initial_unique_solutions,
            first_solution=first_solution,
            first_novel_solution=first_novel_solution,
            rows=rows,
            operator_counts={k: v.tolist() for k, v in session.copy_statistics.items()},
            operator_columns=["none", "mutation", "duplication", "deletion"],
            backend=model.execution.backend,
            elapsed_seconds=time.perf_counter() - started,
        )
        np.savez_compressed(
            root / "final_population.npz",
            genes=session.populations,
            gene_counts=session.gene_counts,
        )
        runner.finish(session)
        write_json(root / "analysis.json", result)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--run", action="store_true", help="execute the prepared campaign")
    p.add_argument("--profile", choices=("basic", "cue", "cue_alternatives"), default="cue")
    p.add_argument(
        "--initial",
        nargs="+",
        choices=("independent_random", "monomorphic_random"),
        default=["independent_random", "monomorphic_random"],
    )
    p.add_argument("--genes", nargs="+", type=int, choices=range(1, 21), default=[1, 2, 3, 4, 5])
    p.add_argument("--interface-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--response-updates", type=int, default=8)
    p.add_argument("--mutation", type=float, default=0.1)
    p.add_argument("--deletion", type=float, default=0.01)
    p.add_argument("--duplication", type=float, default=0.01)
    p.add_argument("--cue-length", type=int, choices=range(4, 9), default=8)
    p.add_argument("--marker-length", type=int, choices=range(4, 9), default=4)
    p.add_argument("--marker-copies", type=int, default=1)
    p.add_argument(
        "--output-representation", choices=("bound_free", "bound_pair"), default="bound_free"
    )
    p.add_argument("--output-rule", choices=("proportional", "majority"), default="proportional")
    p.add_argument("--backend", choices=("auto", "cpu", "metal", "cuda"), default="auto")
    return p


def main():
    args = parser().parse_args()
    matrix = list(conditions(args))
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    for name, _, spec in matrix:
        write_spec(spec, root / name / "spec.toml")
    runner_source = Path(__file__).read_bytes()
    (root / "runner.py").write_bytes(runner_source)
    workload = sum(
        (s.evolution.generations + 1)
        * s.evolution.trials
        * s.evolution.population_size
        * s.episode.episodes
        * s.episode.horizon
        * s.episode.chemical_updates
        for _, _, s in matrix
    )
    manifest = dict(
        status="prepared",
        runner_sha256=hashlib.sha256(runner_source).hexdigest(),
        specification_identities={name: scientific_identity(spec) for name, _, spec in matrix},
        factors=FACTORS,
        conditions=[name for name, _, _ in matrix],
        trials_per_condition=args.trials,
        worst_case_chemical_updates_without_cache=workload,
        convention=(
            "Operator availability changes total variation input; "
            "point probability is per offspring."
        ),
        solution_criterion="expected accuracy == 1 on the declared exhaustive cue panel",
        realization="one viable cell background; fixed interface, output assignment and panels",
    )
    write_json(root / "campaign.json", manifest)
    print(json.dumps(manifest), flush=True)
    if args.run:
        manifest["status"] = "running"
        write_json(root / "campaign.json", manifest)
        started = time.perf_counter()
        try:
            model = Model(execution=matrix[0][2].execution)
            manifest["backend"] = model.execution.backend
            paired = {}
            for name, _, spec in matrix:
                result = run_condition(model, spec, root / name)
                block = (
                    spec.task.config["port_sampling"]["seed"],
                    spec.evolution.initial,
                    spec.model.genes,
                )
                previous = paired.setdefault(block, result["founder_sha256"])
                if previous != result["founder_sha256"]:
                    raise RuntimeError("matched operator arms have different founders")
            manifest["status"] = "complete"
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = repr(error)
            raise
        finally:
            manifest["elapsed_seconds"] = time.perf_counter() - started
            write_json(root / "campaign.json", manifest)


if __name__ == "__main__":
    main()
