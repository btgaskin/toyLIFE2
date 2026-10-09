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
            cells.append(f'<rect x="{column * 5}" y="{row * 5}" width="5" height="5" '
                         f'fill="rgb({red},{green},{blue})"/>')
    return (f'<svg role="img" aria-label="Saved array heatmap" viewBox="0 0 '
            f'{sample.shape[1] * 5} {sample.shape[0] * 5}" '
            'preserveAspectRatio="none">' + "".join(cells) + "</svg>")


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
    manifest = inspect_record(root)
    integrity = verify_record(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    body = [f"<h1>toyLIFE run: {_esc(manifest['operation'])}</h1>",
            f"<p class=lead>Status: <strong>{_esc(manifest['status'])}</strong> · "
            f"Seed: {_esc(manifest['seed'])} · Rules: {_esc(manifest['ruleset'])}</p>",
            "<section class=card><h2>Summary</h2>" + _table(summary) + "</section>",
            "<section class=card><h2>Provenance</h2>" + _table({
                "Scientific identity": manifest["scientific_identity"],
                "Record schema": manifest["record_schema"],
                "Created": manifest["created_at"], "Updated": manifest["updated_at"],
                "Source SHA-256": manifest["source_sha256"],
                "Requested backend": manifest.get("execution", {}).get("requested_backend"),
                "Actual backend": manifest.get("execution", {}).get("actual_backend"),
                "File integrity": "verified" if integrity["ok"] else "; ".join(integrity["errors"]),
            }) + "</section>"]
    for name, item in manifest.get("artifacts", {}).get("arrays", {}).items():
        path = root / item["path"]
        if not path.is_file():
            continue
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        body.append(f"<section class=card><h2>{_esc(name)}</h2><p class=muted>"
                    f"Shape {_esc(item['shape'])}; dtype {_esc(item['dtype'])}</p>"
                    + _heatmap(array) + "</section>")
    target = Path(destination) if destination else root.with_name(root.name + "-report.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_page(f"toyLIFE {manifest['operation']}", "\n".join(body)), encoding="utf-8")
    return target


def render_comparison(records: Sequence[str | Path], destination: str | Path) -> Path:
    rows = []
    for record in records:
        manifest = inspect_record(record)
        rows.append({"Record": Path(record).name, "Operation": manifest["operation"],
                     "Status": manifest["status"], "Scientific identity":
                     manifest["scientific_identity"], "Seed": manifest["seed"]})
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    cards = "".join(f"<section class=card>{_table(row)}</section>" for row in rows)
    target.write_text(_page("toyLIFE run comparison", "<h1>Saved run comparison</h1>" + cards),
                      encoding="utf-8")
    return target
