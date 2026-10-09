"""Explicit molecular inventories and complete product traces for cell protocols."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .reference import CellState, ReferenceModel
from .results import ArrayResult
from .specs import RunSpec
from .tables import canonical_metabolite


def _state(key) -> CellState:
    return CellState(Counter({int(k): int(v) for k, v in key[0]}),
                     Counter({tuple(k): int(v) for k, v in key[1]}),
                     Counter({str(k): int(v) for k, v in key[2]}),
                     Counter({tuple(k): int(v) for k, v in key[3]}))


def simulate(tables, genes: np.ndarray, spec: RunSpec, *, checkpoint=None, save=None) -> dict:
    """Scalar oracle simulation with admitted output and resumable step boundaries."""
    oracle = ReferenceModel(tables)
    budget = min(spec.execution.memory_budget_mib * 1024 * 1024 // 4,
                 spec.recording.trace_budget_mib * 1024 * 1024)
    output_bytes = len(genes) * spec.cell.steps * (2 * spec.model.genes + 160) + genes.nbytes
    if output_bytes > budget:
        raise MemoryError("cell expression output exceeds the admitted host budget")
    expression = (checkpoint["loaded_arrays"]["expression"].copy() if checkpoint else
                  np.full((len(genes), spec.cell.steps, spec.model.genes), 2, np.uint8))
    cursor = dict(checkpoint["state"]) if checkpoint else {
        "algorithm": "cell", "sample": 0, "tick": 0, "state": None, "trajectories": [],
        "trace_bytes": 0}
    if cursor["algorithm"] != "cell":
        raise ValueError("cell checkpoint algorithm mismatch")
    consumed = 0

    def result(status):
        array = ArrayResult(expression,
                            np.where(np.all(expression < 2, axis=-1), "complete", "incomplete"),
                            {"representation": "gene-bits-msb-first",
                             "backend": "python_reference"})
        return {"arrays": {"genotypes": genes, "expression": array}, "status": status,
                "summary": {"trajectories": cursor["trajectories"],
                            "cell_steps": spec.cell.steps, "next_sample": cursor["sample"],
                            "next_tick": cursor["tick"],
                            "recording_limit": cursor.get("recording_limit")}}

    for sample in range(cursor["sample"], len(genes)):
        genotype = genes[sample]
        tick = cursor["tick"]
        state = (_state(cursor["state"]) if cursor["state"] is not None else
                 CellState(proteins=oracle.translated(genotype, spec.cell.initial_expression),
                           metabolites=Counter(canonical_metabolite(x)
                                               for x in spec.cell.metabolites)))
        if len(cursor["trajectories"]) <= sample:
            cursor["trajectories"].append([])
        while tick < spec.cell.steps:
            if consumed >= spec.execution.max_steps:
                if save:
                    save(cursor, {"genotypes": genes, "expression": expression})
                return result("partial")
            step = oracle.step(genotype, state)
            event = {"sample": sample, "tick": tick, "expression": step.expression,
                     "fissions": step.fissions, "before": state.key(), "after": step.state.key()}
            event_bytes = len(json.dumps(event)) * 4  # include Python object/JSON overhead
            if cursor["trace_bytes"] + event_bytes + output_bytes > budget:
                cursor["recording_limit"] = "cell_output_budget"
                if save:
                    save(cursor, {"genotypes": genes, "expression": expression})
                return result("partial")
            expression[sample, tick] = [
                (step.expression >> (spec.model.genes - g - 1)) & 1
                for g in range(spec.model.genes)]
            cursor.pop("recording_limit", None)
            cursor["trajectories"][sample].append(event)
            cursor["trace_bytes"] += event_bytes
            state = step.state
            tick += 1
            consumed += 1
            cursor.update(sample=sample, tick=tick, state=state.key())
            if spec.recording.mode == "full":
                cursor["trace_event"] = event
            if save and (spec.recording.mode == "full" or
                         tick % spec.recording.checkpoint_interval == 0):
                if save(cursor, {"genotypes": genes, "expression": expression}) is False:
                    return result("partial")
        cursor.update(sample=sample + 1, tick=0, state=None)
        if save and save(cursor, {"genotypes": genes, "expression": expression}) is False:
            return result("partial")
    return result("complete")
