"""Matched timing screens and evolutionary pilots for fixed molecular ports.

Run with: python examples/calibrate_molecular.py --destination results/interface-calibration
"""

import argparse
import hashlib
import html
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from toylife2 import Model
from toylife2.evaluation import Evaluator
from toylife2.molecular_experiments import PORT_POOLS, profile_spec, run_experiment
from toylife2.random import Stream
from toylife2.specs import ExecutionSpec, plain


def save_receipt(root, receipt):
    """Report exact addressed mutation counts and measured evaluation work."""
    rows = []
    for pilot in receipt["pilots"]:
        run = root / "runs" / pilot["name"]
        from toylife2.config import load_spec

        spec = load_spec(run / "profile.toml")
        summary = json.loads((run / "experiment.json").read_text())
        stream = Stream(spec.seed)
        ids = np.arange(spec.evolution.population_size, dtype=np.uint64)
        pilot["mutation_events"] = sum(
            int(
                np.count_nonzero(
                    stream.uniform(ids, role="mutation_gate", trial=trial, event=generation)
                    < spec.evolution.mutation_probability
                )
            )
            for trial in range(
                spec.evolution.trial_start, spec.evolution.trial_start + spec.evolution.trials
            )
            for generation in range(spec.evolution.generations)
        )
        pilot["unique_genotypes_evaluated"] = pilot["counters"]["total"]["chemical_updates"] // (
            spec.episode.episodes * spec.episode.horizon * spec.episode.chemical_updates
        )
        pilot["initial_scores"] = summary["initial_mean_scores"]
        interface = summary["resolved_interface"]
        pilot["interface"] = interface
        name = html.escape(pilot["name"])
        cue = html.escape(str(interface["inputs"]))
        target = html.escape(str(interface["targets"]))
        rows.append(
            f'<tr><td><a href="runs/{name}/report.html">{name}</a></td>'
            f"<td>{cue}</td><td>{target}</td>"
            f"<td>{pilot['scores']}</td><td>{pilot['mutation_events']}</td>"
            f"<td>{pilot['unique_genotypes_evaluated']}</td><td>{pilot['elapsed_seconds']:.2f}</td></tr>"
        )
    (root / "calibration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (root / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Molecular calibration</title>'
        "<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse}"
        "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>"
        "<h1>Molecular calibration</h1><p>Fixed ports; eight repeated binary observations; "
        "all successful and unsuccessful conditions retained.</p>"
        '<p><a href="calibration.json">Timing screens, protocol and numerical receipt</a></p>'
        "<table><tr><th>Condition</th><th>Cue</th><th>Target</th><th>Final replicate scores</th>"
        "<th>Mutation events</th><th>Fresh genome evaluations</th><th>Seconds</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--genes", type=int, default=10)
    args = parser.parse_args()
    if min(args.pairs, args.population, args.generations, args.trials, args.genes) < 1:
        parser.error("calibration dimensions must be positive")
    root = args.destination
    root.mkdir(parents=True, exist_ok=False)
    (root / "calibration_runner.py").write_bytes(Path(__file__).read_bytes())
    receipt = {
        "protocol": vars(args) | {"destination": str(root)},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "screens": [],
        "pilots": [],
    }
    model = Model(execution=ExecutionSpec(backend="cpu", cpu_threads=1))
    for pool in PORT_POOLS:
        for pair in range(args.pairs):
            spec = profile_spec("basic", genes=args.genes, port_pool=pool, interface_seed=pair)
            genes = np.random.default_rng(812).integers(
                0, 1 << 20, (128, args.genes), dtype=np.int32
            )
            screen = {
                "pool": pool,
                "interface_seed": pair,
                "interface": plain(spec.controller.interface),
                "windows": {},
            }
            for window in (4, 8, 16):
                timed = replace(spec, episode=replace(spec.episode, horizon=window))
                evaluation = Evaluator(model, timed).evaluate(genes)
                if not evaluation.complete:
                    raise RuntimeError("timing screen did not complete")
                values, counts = np.unique(evaluation.scores, return_counts=True)
                screen["windows"][window] = {
                    "scores": values.tolist(),
                    "counts": counts.tolist(),
                    "fraction_above_constant": float(np.mean(evaluation.scores > 0.5)),
                }
            receipt["screens"].append(screen)
    for pool in PORT_POOLS:
        for pair in range(args.pairs):
            for mutation in (0.001, 0.01, 0.1):
                spec = profile_spec("basic", genes=args.genes, port_pool=pool, interface_seed=pair)
                spec = replace(
                    spec,
                    evolution=replace(
                        spec.evolution,
                        population_size=args.population,
                        generations=args.generations,
                        trials=args.trials,
                        mutation_probability=mutation,
                    ),
                )
                name = f"{pool}-pair{pair}-mu{mutation}"
                name = name.replace(".", "_")
                run = run_experiment(spec, workspace=root, name=name)
                summary = json.loads((run / "experiment.json").read_text())
                history = np.load(run / "fitness_history.npy")
                receipt["pilots"].append(
                    {
                        "name": name,
                        "pool": pool,
                        "interface_seed": pair,
                        "mutation_probability": mutation,
                        "scores": summary["mean_scores"],
                        "best_seen": float(history[:, 3].max()),
                        "counters": summary["counters"],
                        "elapsed_seconds": summary["elapsed_seconds"],
                    }
                )
                save_receipt(root, receipt)
                print(name, summary["mean_scores"], flush=True)


if __name__ == "__main__":
    main()
