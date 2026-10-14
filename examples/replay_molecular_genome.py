"""Inspect the best saved selected genome under the current chemotaxis dynamics.

Usage: python examples/replay_molecular_genome.py PARENT DESTINATION
Selection uses the parent's mean held-out score; this is a behavioural inspection.
The genotype is held fixed. A new recorded rollout captures the current physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from toylife2 import BatchRunner
from toylife2.config import load_spec, write_spec
from toylife2.genotypes import GenomeBatch, decode, load_selected
from toylife2.molecular_animation import render_molecular_animation
from toylife2.specs import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    parent, root = args.parent.resolve(), args.destination.resolve()
    original = json.loads((parent / "experiment.json").read_text())
    index = int(np.argmax(original["mean_scores"]))
    saved = load_selected(parent)
    selected = (
        saved.active(index)[None, :] if isinstance(saved, GenomeBatch) else saved[index : index + 1]
    )
    spec = load_spec(parent / "profile.toml")
    spec = replace(
        spec,
        operation="rollout",
        model=replace(spec.model, genes=selected.shape[-1]),
        evolution=replace(spec.evolution, gene_copy=None),
        genotypes=tuple(np.asarray(decode(selected)).tolist()),
        seed=original["heldout_seed"],
        episode=replace(spec.episode, episodes=32, panel="fixed"),
        recording=replace(spec.recording, checkpoint_interval=25),
    )
    root.mkdir(parents=True, exist_ok=False)
    write_spec(spec, root / "profile.toml")
    np.save(root / "selected_genomes.npy", selected, allow_pickle=False)
    provenance = dict(
        relationship="fixed_genome_new_motion_inspection",
        path=str(parent),
        selected_trial=index + 1,
        selection="highest parent mean held-out score",
        original_score=original["mean_scores"][index],
        original_summary_sha256=hashlib.sha256(
            (parent / "experiment.json").read_bytes()
        ).hexdigest(),
    )
    (root / "parent.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.perf_counter()
    rows = {}
    with BatchRunner(spec.execution) as runner:
        rollout = runner.rollout(
            selected, spec=spec, panel=original["heldout_panel"], destination=root / "rollout"
        )
        initial = rollout.snapshot(fields=["world"])["world"]
        for name in ("position", "theta", "phases", "source", "orientation", "tumble_remaining"):
            rows["initial_" + name] = [initial[name]]
        while rollout.tick < spec.episode.horizon:
            report = rollout.step()
            transition = report.transition
            if transition is None or "actions" not in transition:
                raise RuntimeError("genome inspection interrupted before its horizon")
            values = dict(
                actions=transition["actions"],
                reward=transition["reward"],
                sensed_exposure=transition["observation"]["exposure"],
                expression=rollout.chemistry.state["expression"],
                **transition["channels"],
            )
            values.update({"input_" + k: v for k, v in transition["observation"]["inputs"].items()})
            values.update({"readout_" + k: v for k, v in rollout.chemistry.last_readouts.items()})
            for name, value in values.items():
                rows.setdefault(name, []).append(np.asarray(value).copy())
            if rollout.tick % 25 == 0:
                print(
                    f"World steps {rollout.tick}/{spec.episode.horizon}; "
                    f"elapsed {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        arrays = {name: np.stack(values) for name, values in rows.items()}
        arrays["scores"] = rollout.scores.copy()
        runner.finish(rollout)
    np.savez_compressed(root / "behaviour.npz", **arrays)
    summary = dict(
        profile="fixed_genome_inspection",
        field=spec.task.config["field"],
        behaviour_status="complete",
        parent=provenance,
        task=plain(spec.task),
        episode=plain(spec.episode),
        selected_genomes=selected.tolist(),
        mean_scores=arrays["scores"].mean(axis=-1).tolist(),
        elapsed_seconds=time.perf_counter() - started,
    )
    summary["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.iterdir()
        if p.suffix in {".npz", ".npy", ".toml"}
    }
    (root / "experiment.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(render_molecular_animation(root, frame_stride=1), flush=True)
    print(
        json.dumps(
            {
                "selected_trial": index + 1,
                "mean_score": summary["mean_scores"][0],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
