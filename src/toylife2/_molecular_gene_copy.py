"""Grouped fixed-width inspection and count-aware molecular result artifacts."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import replace

import numpy as np

from .config import write_spec
from .genotypes import save_selected
from .specs import plain
from .variation import PROTOCOL


def _inspection_merge_bytes(root, groups):
    # Uncompressed NPY sizes include every selected lane and the fixed 32x4
    # history probes. Three copies cover merged arrays, a child and save/render
    # scratch; repeated global fields make this a conservative admission bound.
    total = 0
    for group in groups:
        for name in ("behaviour.npz", "initial_behaviour.npz", "history_probes.npz"):
            path = root / group["path"] / name
            if path.exists():
                with zipfile.ZipFile(path) as archive:
                    total += sum(info.file_size for info in archive.infolist())
    return 3 * total + 64 * 1024


def _merge_arrays(root, groups, name, size, counts):
    merged = {}
    for group in groups:
        path = root / group["path"] / name
        if not path.exists():
            continue
        indices = np.asarray(group["selection_ids"])
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if key == "allow_pickle":
                    continue
                value = saved[key]
                if key in {"field_initial", "field_final", "inputs"} or value.ndim < 2:
                    merged.setdefault(key, value.copy())
                    continue
                axis = 0 if key == "scores" else 1
                # Expression is an active-genome bit mask, not a padded gene axis.
                shape = list(value.shape)
                shape[axis] = size
                if key not in merged:
                    merged[key] = np.empty(shape, value.dtype)
                target = [slice(None)] * value.ndim
                target[axis] = indices
                merged[key][tuple(target)] = value
    if merged:
        merged["gene_counts"] = counts
        np.savez_compressed(root / name, **merged)


def inspect_variable_selected(
    runner, spec, selected, initial_selected, root, *, history, **arguments
):
    from .molecular_experiments import _inspect_selected

    groups = []
    summaries = []
    for count in sorted(set(map(int, selected.gene_counts))):
        indices = np.flatnonzero(selected.gene_counts == count)
        path = root / "inspection" / f"genes-{count}"
        path.mkdir(parents=True, exist_ok=False)
        child_spec = replace(
            spec,
            model=replace(spec.model, genes=count),
            genotypes=(),
            evolution=replace(spec.evolution, gene_copy=None),
        )
        child = selected.genes[indices, :count]
        initial = None
        if initial_selected is not None:
            initial_counts = initial_selected.gene_counts[indices]
            if len(np.unique(initial_counts)) != 1:
                raise ValueError("founder selections must share their declared founder length")
            initial = initial_selected.genes[indices, : int(initial_counts[0])]
            save_selected(path, initial, initial=True)
        save_selected(path, child)
        write_spec(child_spec, path / "profile.toml")
        rows = history[np.isin(history[:, 0], indices)].copy()
        for local, original in enumerate(indices):
            rows[history[np.isin(history[:, 0], indices), 0] == original, 0] = local
        np.save(path / "fitness_history.npy", rows, allow_pickle=False)
        _inspect_selected(runner, child_spec, child, initial, path, **arguments)
        groups.append(
            dict(gene_count=count, selection_ids=indices.tolist(), path=str(path.relative_to(root)))
        )
        summaries.append(json.loads((path / "experiment.json").read_text()))
    inspection_bytes = _inspection_merge_bytes(root, groups)
    with runner.model.engine._temporary_bytes(
        inspection_bytes, message="grouped inspection arrays exceed memory budget"
    ):
        _merge_arrays(
            root,
            groups,
            "behaviour.npz",
            len(selected),
            selected.gene_counts,
        )
        _merge_arrays(
            root,
            groups,
            "history_probes.npz",
            len(selected),
            selected.gene_counts,
        )
        if initial_selected is not None:
            _merge_arrays(
                root,
                groups,
                "initial_behaviour.npz",
                len(selected),
                initial_selected.gene_counts,
            )
        summary = summaries[0].copy()
        for key in ("mean_scores", "initial_mean_scores", "reset_mean_scores"):
            if key in summary:
                values = [None] * len(selected)
                for group, child in zip(groups, summaries, strict=True):
                    for index, value in zip(group["selection_ids"], child[key], strict=True):
                        values[index] = value
                summary[key] = values
        summary.update(
            task=plain(spec.task),
            parent=arguments["parent"],
            selected_genomes=[list(row) for row in selected.keys()],
            initial_selected_genomes=None
            if initial_selected is None
            else [list(row) for row in initial_selected.keys()],
            selected_gene_counts=selected.gene_counts.tolist(),
            initial_selected_gene_counts=None
            if initial_selected is None
            else initial_selected.gene_counts.tolist(),
            gene_copy_protocol=PROTOCOL,
            gene_copy=plain(spec.evolution.gene_copy),
            genome_format="gene-counts-v1",
            inspection_groups=groups,
            elapsed_seconds=time.perf_counter() - arguments["started"],
        )
        for key in ("target_fission_totals",):
            if key in summary:
                combined = {name: [None] * len(selected) for name in summary[key]}
                for group, child in zip(groups, summaries, strict=True):
                    for name in combined:
                        for index, value in zip(
                            group["selection_ids"], child[key][name], strict=True
                        ):
                            combined[name][index] = value
                summary[key] = combined
        summary["artifacts"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.iterdir()
            if p.suffix in {".npy", ".npz", ".toml", ".json"}
        }
        (root / "experiment.json").write_text(json.dumps(summary, indent=2) + "\n")
        from .reports import render_molecular_report

        if arguments["animate"] and spec.task.kind == "chemotaxis":
            from .molecular_animation import render_molecular_animation

            render_molecular_animation(root, frame_stride=arguments["frame_stride"])
        render_molecular_report(root)
    return root
