"""Matched 2^3 operator pilots for inverted cue reading and clonal signalling.

python examples/signalling_factorial.py --destination results/signalling-factorial
Every condition uses the same founders, categorical offspring operator, bounds,
selection floor, interface and evaluation panel within each task.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from toylife2 import BatchRunner, GeneCopySpec, Model
from toylife2.config import write_spec
from toylife2.molecular_experiments import profile_spec
from toylife2.specs import ExecutionSpec, plain

FACTORS = ("mutation", "deletion", "duplication")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_spec(game, switches, args, execution):
    spec = profile_spec(
        game,
        genes=args.genes,
        seed=args.seed,
        response_updates=args.response_updates,
        cue_length=args.cue_length,
    )
    if game == "signalling_game":
        spec = replace(
            spec,
            task=replace(
                spec.task, config={**spec.task.config, "rounds": args.rounds, "profile": game}
            ),
            episode=replace(spec.episode, horizon=2 * args.rounds),
        )
    return replace(
        spec,
        execution=execution,
        objective=replace(spec.objective, goal=1.0),
        evolution=replace(
            spec.evolution,
            population_size=args.population,
            generations=args.generations,
            trials=args.trials,
            mutation_probability=args.mutation if switches[0] else 0,
            gene_copy=GeneCopySpec(
                duplication_probability=args.duplication if switches[2] else 0,
                deletion_probability=args.deletion if switches[1] else 0,
                min_genes=1,
                max_genes=20,
            ),
        ),
    )


def contrast(rows, outcome):
    """Paired factorial effects: average high-minus-low, including interactions."""
    results = {}
    for degree in range(1, 4):
        for selected in itertools.combinations(range(3), degree):
            samples = (
                sum(
                    np.prod([2 * row["switches"][j] - 1 for j in selected])
                    * np.asarray(row[outcome])
                    for row in rows
                )
                / 4
            )
            results[" × ".join(FACTORS[j] for j in selected)] = {
                "mean": float(samples.mean()),
                "paired_replicates": samples.tolist(),
                "standard_error": float(samples.std(ddof=1) / np.sqrt(len(samples)))
                if len(samples) > 1
                else None,
            }
    return results


def analyse(root, receipt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for row in receipt["conditions"]:
        history = np.load(
            root / row["name"] / "evolution/arrays/arrays_history.npy", allow_pickle=False
        )
        trials = len(row["final_mean"])
        history = history.reshape(trials, -1, 5)
        best = history[:, :, 3]
        row["best_curves"] = best.tolist()
        row["best_ever"] = best.max(axis=1).tolist()
        row["first_above_chance_generation"] = [
            int(indices[0]) if len(indices := np.flatnonzero(curve > 0.5)) else None
            for curve in best
        ]
        counts = np.load(
            root / row["name"] / "evolution/arrays/arrays_gene_count_history.npy",
            allow_pickle=False,
        )
        means = counts @ np.arange(counts.shape[-1]) / counts.sum(axis=-1)
        row["gene_mean_curves"] = means.reshape(trials, -1).tolist()
    games = list(dict.fromkeys(row["game"] for row in receipt["conditions"]))
    fig, axes = plt.subplots(3, len(games), figsize=(7 * len(games), 11), squeeze=False)
    analysis = {}
    for column, game in enumerate(games):
        rows = [row for row in receipt["conditions"] if row["game"] == game]
        analysis[game] = {
            outcome: contrast(rows, outcome) for outcome in ("final_mean", "mean_auc", "best_ever")
        }
        for row in rows:
            for ax, curves in zip(
                axes[:, column],
                (
                    np.asarray(row["mean_curves"]),
                    np.maximum.accumulate(row["best_curves"], axis=1),
                    np.asarray(row["gene_mean_curves"]),
                ),
                strict=True,
            ):
                (line,) = ax.plot(curves.mean(axis=0), label=row["label"], linewidth=1.5)
                sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else 0
                ax.fill_between(
                    np.arange(curves.shape[1]),
                    curves.mean(axis=0) - sem,
                    curves.mean(axis=0) + sem,
                    color=line.get_color(),
                    alpha=0.1,
                )
        axes[0, column].set(
            title=game + (" (clonal pairs)" if game == "signalling_game" else ""),
            ylabel="Population mean accuracy",
            ylim=(0.45, 1.01),
        )
        axes[1, column].set(ylabel="Best accuracy found (running maximum)", ylim=(0.45, 1.01))
        axes[2, column].set(ylabel="Mean active genes", xlabel="Generation", ylim=(1, 20))
        for ax in axes[:2, column]:
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axes[0, column].legend(fontsize=8, ncols=2)
    fig.suptitle("Operator factorial — mean ± SE across evolutionary replicates")
    fig.tight_layout()
    fig.savefig(root / "accuracy.png", dpi=180)
    fig.savefig(root / "accuracy.svg")
    plt.close(fig)
    write_json(root / "analysis.json", analysis)
    cells = []
    for row in receipt["conditions"]:
        samples = np.asarray(row["final_mean"])
        reached = sum(score > 0.5 for score in row["best_ever"])
        retained = sum(score > 0.5 for score in row["final_best"])
        cells.append(
            f"| {row['game']} | {row['label']} | {samples.mean():.4f} | "
            f"{max(row['best_ever']):.4f} | {reached}/{len(samples)} | "
            f"{retained}/{len(samples)} | "
            f"{np.mean(row['final_mean_genes']):.2f} | {row['elapsed_seconds']:.1f} |"
        )
    (root / "report.md").write_text(
        "# Signalling operator factorial\n\n"
        "A matched pilot with independent evolutionary replicates. Two-cell candidates are "
        "clonal pairs with independent chemistry and persistent cell state. The receiver sees "
        "the sender's message. One fixed molecular interface and fixed scenario panel per game. "
        "All-off includes selection and drift on standing variation. Enabled probabilities "
        "stay fixed; the factorial includes changes in the total event probability.\n\n"
        "![Accuracy, discoveries and gene counts](accuracy.png)\n\n"
        "| Task | Operators | Final mean | Best ever | Trials >50% ever | "
        "Retained >50% | Mean genes | Seconds |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(cells) + "\n\n"
        "Best-ever accuracy is the highest individual score over all generations; reached and "
        "retained counts refer to evolutionary trials with an individual above 50%. Curves show "
        "mean ± standard error across trials. Full replicate values, operator counts, gene "
        "histograms and timings are in `experiment.json`; paired factorial effects are in "
        "`analysis.json`. Each effect is the signed sum over eight cells divided by four. "
        "The two-cell private-state schedule pairs complementary bits across adjacent rounds; "
        "a communication claim needs a message-clamping control. These are discovery scores "
        "on a fixed panel. A new port pair or scenario panel requires a separate experiment.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--genes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--response-updates", type=int, default=8)
    parser.add_argument("--cue-length", type=int, choices=range(4, 9), default=8)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--mutation", type=float, default=0.1)
    parser.add_argument("--deletion", type=float, default=0.01)
    parser.add_argument("--duplication", type=float, default=0.01)
    parser.add_argument("--backend", choices=("auto", "metal", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--games", nargs="+", choices=("cue", "signalling_game"), default=["cue", "signalling_game"]
    )
    args = parser.parse_args()
    if min(args.population, args.generations, args.trials, args.response_updates, args.rounds) < 1:
        parser.error("pilot dimensions must be positive")
    if args.rounds % 2:
        parser.error("use an even number of rounds to balance private states")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "runner.py")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary"], text=True)
    (root / "source.patch").write_text(diff)
    receipt = dict(
        status="running",
        started=datetime.now(UTC).isoformat(),
        protocol=vars(args) | {"destination": str(root)},
        revision=revision,
        source_patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        conditions=[],
    )
    execution = ExecutionSpec(backend=args.backend, cpu_threads=1, memory_budget_mib=2048)
    model = Model(execution=execution)
    receipt["backend"] = execution.backend
    receipt["actual_arch"] = str(model.engine.qd.lang.impl.current_cfg().arch)
    receipt["table_digest"] = model.tables.digest
    founders = None
    started = time.perf_counter()
    write_json(root / "experiment.json", receipt)
    try:
        for game in args.games:
            for switches in itertools.product((0, 1), repeat=3):
                label = "+".join(name for name, on in zip(FACTORS, switches, strict=True) if on)
                label = label or "all-off"
                name = game + "-" + "".join(map(str, switches))
                spec = make_spec(game, switches, args, execution)
                condition_root = root / name
                condition_root.mkdir()
                write_spec(spec, condition_root / "spec.toml")
                condition_started = time.perf_counter()
                print(
                    json.dumps(
                        dict(
                            event="starting",
                            condition=name,
                            operators=label,
                            backend=execution.backend,
                        )
                    ),
                    flush=True,
                )
                with BatchRunner(model=model) as runner:
                    session = runner.evolution(spec, destination=condition_root / "evolution")
                    found_hash = hashlib.sha256(
                        session.populations.tobytes() + session.gene_counts.tobytes()
                    ).hexdigest()
                    if founders is None:
                        founders = found_hash
                    if founders != found_hash:
                        raise RuntimeError("factorial founders differ")
                    session.observe()
                    initial_scores = [out.scores.tolist() for out in session.evaluations]
                    last_generation, last_time = 0, time.perf_counter()
                    while session._status not in {"complete", "partial", "failed"}:
                        session.step()
                        generation = int(session.cursors.min())
                        if generation >= last_generation + 10 or session._status == "complete":
                            now = time.perf_counter()
                            progress = dict(
                                event="progress",
                                condition=name,
                                generation=generation,
                                generations=args.generations,
                                offspring_per_second=(generation - last_generation)
                                * args.population
                                * args.trials
                                / max(now - last_time, 1e-9),
                                elapsed_seconds=now - condition_started,
                                mean_accuracy=float(
                                    np.mean([out.scores.mean() for out in session.evaluations])
                                ),
                                evaluator_metrics=dict(session.evaluator.metrics),
                            )
                            write_json(root / "progress.json", progress)
                            print(json.dumps(progress), flush=True)
                            last_generation, last_time = generation, now
                    if session._status != "complete":
                        raise RuntimeError(f"{name}: {session._status}, {session._reason}")
                    curves = np.asarray(
                        [
                            [session.history[(int(rep), g)][2] for g in range(args.generations + 1)]
                            for rep in session.replicates
                        ]
                    )
                    row = dict(
                        game=game,
                        switches=switches,
                        name=name,
                        label=label,
                        founder_sha256=found_hash,
                        interface=plain(spec.controller.interface),
                        initial_scores=initial_scores,
                        mean_curves=curves.tolist(),
                        final_mean=[float(out.scores.mean()) for out in session.evaluations],
                        final_best=[float(out.scores.max()) for out in session.evaluations],
                        final_fraction_perfect=[
                            float(np.mean(out.scores == 1)) for out in session.evaluations
                        ],
                        first_perfect_generation=session.arrivals,
                        mean_auc=(np.trapezoid(curves, axis=1) / args.generations).tolist(),
                        final_mean_genes=session.gene_counts.mean(axis=1).tolist(),
                        final_gene_histograms=[
                            np.bincount(counts, minlength=21).tolist()
                            for counts in session.gene_counts
                        ],
                        operator_counts={k: v.tolist() for k, v in session.copy_statistics.items()},
                        operator_columns=["none", "mutation", "duplication", "deletion"],
                        evaluator_metrics=dict(session.evaluator.metrics),
                        counters=plain(session.counters),
                    )
                    np.savez_compressed(
                        condition_root / "final.npz",
                        genes=session.populations,
                        gene_counts=session.gene_counts,
                        scores=np.stack([out.scores for out in session.evaluations]),
                    )
                    runner.finish(session)
                row["elapsed_seconds"] = time.perf_counter() - condition_started
                receipt["conditions"].append(row)
                receipt["elapsed_seconds"] = time.perf_counter() - started
                write_json(root / "experiment.json", receipt)
        analyse(root, receipt)
        receipt["status"] = "complete"
        receipt["completed"] = datetime.now(UTC).isoformat()
    except BaseException as error:
        receipt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_json(root / "experiment.json", receipt)
    print(
        json.dumps(
            dict(
                event="complete", destination=str(root), elapsed_seconds=receipt["elapsed_seconds"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
