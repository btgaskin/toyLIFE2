"""Static, dependency-light HTML reports built solely from saved records."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .records import inspect_record, verify_record


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        rows.append(f"<tr><th>{_esc(key)}</th><td>{_esc(value)}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _heatmap(values: np.ndarray) -> str:
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        return ""
    row_step = max(1, int(np.ceil(values.shape[0] / 80)))
    col_step = max(1, int(np.ceil(values.shape[1] / 100)))
    sample = values[::row_step, ::col_step]
    if sample.size == 0:
        return ""
    finite = sample[np.isfinite(sample)]
    if not finite.size:
        return ""
    low, high = float(np.min(finite)), float(np.max(finite))
    scale = high - low or 1.0
    cells = []
    for row in range(sample.shape[0]):
        for column in range(sample.shape[1]):
            value = float(sample[row, column])
            intensity = 0 if not np.isfinite(value) else int(255 * (value - low) / scale)
            red, green, blue = intensity // 3, 65 + intensity // 2, 130 + intensity // 3
            cells.append(
                f'<rect x="{column * 5}" y="{row * 5}" width="5" height="5" '
                f'fill="rgb({red},{green},{blue})"/>'
            )
    return (
        f'<svg role="img" aria-label="Saved array heatmap" viewBox="0 0 '
        f'{sample.shape[1] * 5} {sample.shape[0] * 5}" '
        'preserveAspectRatio="none">' + "".join(cells) + "</svg>"
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{_esc(title)}</title>
<style>
:root {{color-scheme:light; font-family:system-ui,-apple-system,sans-serif;
color:#182b38; background:#f5f7f7}}
body {{max-width:1100px; margin:2rem auto; padding:0 1rem; line-height:1.5}}
h1,h2 {{letter-spacing:-.025em}} h2 {{margin-top:2.5rem}} .lead {{font-size:1.15rem;color:#48616d}}
.card {{background:white;border:1px solid #d8e1e3;border-radius:12px;padding:1.2rem;margin:1rem 0}}
table {{border-collapse:collapse;width:100%;overflow-wrap:anywhere}}
th,td {{border-bottom:1px solid #e5ebec;text-align:left;padding:.5rem;vertical-align:top}}
th {{width:18rem;color:#48616d;font-weight:600}}
code {{overflow-wrap:anywhere}}
svg {{display:block;width:100%;max-height:450px;image-rendering:pixelated}}
.ok {{color:#176044}} .bad {{color:#9c3030}} .muted {{color:#61747c}}
</style></head><body>{body}</body></html>"""


def render_report(record: str | Path, destination: str | Path | None = None) -> Path:
    """Render a record without recomputing any phenotype or simulation."""
    root = Path(record)
    if (root / "experiment.json").is_file():
        rendered = render_molecular_report(root)
        if destination is not None:
            # The report's plots remain relative to its experiment directory.
            target = Path(destination)
            import re

            body = rendered.read_text()
            body = re.sub(
                r"src='([^']+\.svg)'",
                lambda match: "src='" + (root / match[1]).resolve().as_uri() + "'",
                body,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
            return target
        return rendered
    manifest = inspect_record(root)
    integrity = verify_record(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    body = [
        f"<h1>toyLIFE run: {_esc(manifest['operation'])}</h1>",
        f"<p class=lead>Status: <strong>{_esc(manifest['status'])}</strong> · "
        f"Seed: {_esc(manifest['seed'])} · Rules: {_esc(manifest['ruleset'])}</p>",
        "<section class=card><h2>Summary</h2>" + _table(summary) + "</section>",
        "<section class=card><h2>Provenance</h2>"
        + _table(
            {
                "Scientific identity": manifest["scientific_identity"],
                "Record schema": manifest["record_schema"],
                "Created": manifest["created_at"],
                "Updated": manifest["updated_at"],
                "Source SHA-256": manifest["source_sha256"],
                "Requested backend": manifest.get("execution", {}).get("requested_backend"),
                "Actual backend": manifest.get("execution", {}).get("actual_backend"),
                "File integrity": "verified" if integrity["ok"] else "; ".join(integrity["errors"]),
            }
        )
        + "</section>",
    ]
    for name, item in manifest.get("artifacts", {}).get("arrays", {}).items():
        path = root / item["path"]
        if not path.is_file():
            continue
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        body.append(
            f"<section class=card><h2>{_esc(name)}</h2><p class=muted>"
            f"Shape {_esc(item['shape'])}; dtype {_esc(item['dtype'])}</p>"
            + _heatmap(array)
            + "</section>"
        )
    target = Path(destination) if destination else root.with_name(root.name + "-report.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_page(f"toyLIFE {manifest['operation']}", "\n".join(body)), encoding="utf-8")
    return target


def render_comparison(records: Sequence[str | Path], destination: str | Path) -> Path:
    rows = []
    for record in records:
        manifest = inspect_record(record)
        rows.append(
            {
                "Record": Path(record).name,
                "Operation": manifest["operation"],
                "Status": manifest["status"],
                "Scientific identity": manifest["scientific_identity"],
                "Seed": manifest["seed"],
            }
        )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(f"<section class=card>{_table(row)}</section>" for row in rows)
    target.write_text(
        _page("toyLIFE run comparison", "<h1>Saved run comparison</h1>" + cards), encoding="utf-8"
    )
    return target


def render_molecular_report(record: str | Path) -> Path:
    """Render selected-cell responses and trajectories from saved experiment arrays."""
    root = Path(record)
    summary = json.loads((root / "experiment.json").read_text())
    body = [
        f"<h1>Molecular cells: {_esc(summary['profile'])}</h1>",
        _table(
            {
                key: summary[key]
                for key in (
                    "field",
                    "evolution_status",
                    "behaviour_status",
                    "mean_scores",
                    "selected_genomes",
                    "elapsed_seconds",
                )
            }
        ),
    ]
    if "selected_gene_counts" in summary:
        body.extend(
            [
                "<h2>Gene-copy evolution</h2>",
                _table({"Selected active genes": summary["selected_gene_counts"]}),
            ]
        )
        statistics = root / "gene_copy_statistics.json"
        if statistics.is_file():
            details = json.loads(statistics.read_text())
            body.append(
                _table(
                    {
                        "Protocol": details["gene_copy_protocol"],
                        "Operation order": ["unchanged", "point", "duplication", "deletion"],
                        **details["gene_copy_statistics"],
                    }
                )
            )
    if summary.get("output"):
        body.extend(
            [
                "<h2>Assumed output apparatus</h2>",
                _table(summary["output"]),
                "<h2>Production measurements</h2>",
                _table(summary["production"]),
            ]
        )
        if "expected_accuracy" in summary:
            body.append(
                _table(
                    {
                        "Expected accuracy": summary["expected_accuracy"],
                        "Sampled accuracy": summary["mean_scores"],
                    }
                )
            )
    if "baselines" in summary:
        body.extend(
            [
                "<h2>Matched motor controls</h2>",
                _table(
                    {name: float(np.mean(scores)) for name, scores in summary["baselines"].items()}
                ),
                _table({"Chemistry reset each decision": summary["reset_mean_scores"]}),
            ]
        )
    with np.load(root / "behaviour.npz", allow_pickle=False) as saved:
        data = dict(saved)
    if summary["field"] is None:
        body.append("<h2>Exhaustive cue responses</h2>")
        stages = [("Final selected cell", data)]
        if (root / "initial_behaviour.npz").is_file():
            with np.load(root / "initial_behaviour.npz", allow_pickle=False) as saved:
                stages.insert(0, ("Initial selected cell", dict(saved)))
        for label, stage in stages:
            body.append(f"<h3>{label}</h3>")
            for cell in range(stage["actions"].shape[1]):
                body.append(
                    _table(
                        {
                            "Cell": cell,
                            "Binary action trajectory": stage["actions"][:, cell].tolist(),
                            "Correctness trajectory": stage["reward"][:, cell].tolist(),
                            "Judged observations": stage.get(
                                "judged", np.ones(len(stage["actions"]), dtype=bool)
                            ).tolist(),
                            **{
                                name: stage[name][-1, cell].tolist()
                                for name in stage
                                if name in {"actions", "reward"}
                                or name.startswith(("input_", "readout_"))
                            },
                        }
                    )
                )
    if (root / "animation.html").is_file():
        body.append("<p><a href='animation.html'>Watch initial and final cell playback</a></p>")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        body.append("<p>Install toyLIFE2[reports] to render the saved plot arrays.</p>")
    else:

        def save(figure, name, caption):
            figure.tight_layout()
            figure.savefig(root / name, format="svg")
            plt.close(figure)
            body.append(
                f"<figure><img style='width:100%' src='{name}' alt='{_esc(caption)}'>"
                f"<figcaption>{_esc(caption)}</figcaption></figure>"
            )

        history = np.load(root / "fitness_history.npy", allow_pickle=False)
        fig, ax = plt.subplots(figsize=(9, 3))
        for cell in np.unique(history[:, 0]):
            rows = history[history[:, 0] == cell]
            ax.plot(rows[:, 1], rows[:, 2], label=f"Population {int(cell)} mean")
            ax.plot(rows[:, 1], rows[:, 3], linestyle=":", label=f"Population {int(cell)} best")
        ax.set(xlabel="Generation", ylabel="Task score")
        ax.legend()
        save(fig, "fitness.svg", "Independent evolutionary replicates")
        with np.load(root / "history_probes.npz", allow_pickle=False) as probes:
            fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            for lane in range(4):
                axes[0].plot(probes["inputs"][:, lane], label=f"History {lane}")
                target = next(iter(summary["resolved_interface"]["targets"]))
                axes[1].plot(probes[target][:, 0, lane], label=f"History {lane}")
            axes[0].set(ylabel="Supplied cue")
            axes[1].set(xlabel="Chemical update", ylabel="Intact target occupancy")
            axes[0].legend()
            save(
                fig,
                "history.svg",
                "Persistent chemistry: four independent input histories, first selected cell",
            )
        if summary["field"] is not None:
            domain = summary["task"]["config"]["domain"]
            fig, ax = plt.subplots(figsize=(6, 6))
            image = ax.imshow(
                data["field_initial"][0], origin="lower", extent=(0, domain, 0, domain)
            )
            fig.colorbar(image, ax=ax, label="Initial concentration")
            path = np.concatenate(
                [data["initial_position"][0, 0, 0][None], data["position"][:, 0, 0]], axis=0
            )
            # Separate wrapped segments so a torus crossing draws no long chord.
            jumps = np.any(np.abs(np.diff(path, axis=0)) > domain / 2, axis=-1)
            plotted = path.copy()
            plotted[1:][jumps] = np.nan
            ax.plot(plotted[:, 0], plotted[:, 1], color="white", linewidth=1.5)
            ax.set(xlabel="x", ylabel="y")
            save(
                fig,
                "trajectory.svg",
                "First held-out trajectory over its initial field; phases change during motion",
            )
            fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
            time = np.arange(len(data["actions"])) * summary["episode"]["dt"]
            config = summary["task"]["config"]
            names = (
                "sensed_exposure",
                "input_" + config["input_channel"],
                "readout_"
                + (
                    summary["output"]["markers"][0]
                    if summary.get("output")
                    else config["action_outputs"][0]
                ),
                "tumbling",
            )
            for ax, name in zip(axes, names, strict=True):
                ax.plot(time, data[name][:, 0, 0].reshape(len(time), -1)[:, 0])
                ax.set(ylabel=name)
            axes[-1].set(xlabel="Time (seconds)")
            save(
                fig,
                "sensing.svg",
                "Pre-move concentration, sampled molecular cue, occupancy and tumble events",
            )
    body.extend(
        [
            "<h2>Resolved molecular interface</h2>",
            _table(summary["resolved_interface"]),
            "<h2>Endpoint contact profiles</h2>",
            _table(summary.get("endpoint_information", {})),
            "<h2>Task and timing</h2>",
            _table(summary["task"]),
            _table(summary["episode"]),
            "<h2>Further experiments</h2>",
            f"<p>{_esc(summary['future_work'])}</p>",
        ]
    )
    target = root / "report.html"
    target.write_text(_page("Molecular cell behaviours", "\n".join(body)), encoding="utf-8")
    return target
