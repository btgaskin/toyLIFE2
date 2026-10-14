"""Length-grouped molecular evaluation with aggregate budgets and continuation."""

from __future__ import annotations

import dataclasses
import sys
from collections import OrderedDict

import numpy as np

from .evaluation import EvaluationResult, Evaluator, _plain
from .genotypes import GenomeBatch
from .specs import digest


def _bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes + 128
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_bytes(k) + _bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(map(_bytes, value))
    return sys.getsizeof(value)


class _SharedCache(OrderedDict):
    def __init__(self, engine, capacity):
        super().__init__()
        self.engine, self.capacity, self.bytes = engine, capacity, 0
        self.sizes = {}

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        size = _bytes(key) + _bytes(value) + 128
        if size > self.capacity:
            return
        while self and self.bytes + size > self.capacity:
            self.popitem(last=False)
        try:
            self.engine._check_memory(
                size, message="gene-count evaluation cache exceeds memory budget"
            )
        except MemoryError:
            self.clear()
            return  # A cache admission failure never invalidates a completed score.
        super().__setitem__(key, value)
        self.sizes[key] = size
        self.bytes += size
        self.engine._reserved_assay_bytes += size

    def __delitem__(self, key):
        super().__delitem__(key)
        size = self.sizes.pop(key)
        self.bytes -= size
        self.engine._reserved_assay_bytes -= size

    def popitem(self, last=True):
        key = next(reversed(self)) if last else next(iter(self))
        value = self[key]
        del self[key]
        return key, value

    def clear(self):
        while self:
            self.popitem()


class VariableGeneEvaluator:
    def __init__(self, model, spec, *, target=None, rollout_factory=None, runner=None):
        base = Evaluator(model, spec, target=target, rollout_factory=rollout_factory, runner=runner)
        self.model, self.spec = model, spec
        self.identity_spec = base.identity_spec
        self.context = digest(["gene-counts-v1", base.context])
        self.stochastic, self.kind = base.stochastic, base.kind
        self.metrics = base.metrics
        self.work_limits = base.work_limits
        self.children = {}
        self.cache = _SharedCache(model.engine, spec.execution.memory_budget_mib * 1024 * 1024 // 8)
        self.rollout_factory, self.runner, self.target = rollout_factory, runner, target
        self._shared_cache = False

    def _child(self, count):
        count = int(count)
        if count not in self.children:
            spec = dataclasses.replace(
                self.spec,
                model=dataclasses.replace(self.spec.model, genes=count),
                evolution=dataclasses.replace(self.spec.evolution, gene_copy=None),
                genotypes=(),
            )
            child = Evaluator(
                self.model,
                spec,
                target=self.target,
                rollout_factory=self.rollout_factory,
                runner=self.runner,
            )
            child.cache = self.cache
            child.cache_capacity = sys.maxsize
            child.metrics = self.metrics
            self.children[count] = child
        return self.children[count]

    def set_work_limits(self, limits):
        self.work_limits = limits

    def retain_cache_panels(self, panels):
        keep = {digest(_plain(panel)) for panel in panels}
        keep.update(
            digest(_plain(arguments["panel"]))
            for child in self.children.values()
            for arguments in child.pending_arguments.values()
        )
        for key in list(self.cache):
            if key[1] not in keep:
                del self.cache[key]

    def evaluate(self, genomes, candidate_ids=None, *, panel=0, identities=None, agent_axis=None):
        if not isinstance(genomes, GenomeBatch) or identities is not None or agent_axis is not None:
            raise ValueError(
                "variable-gene evaluation requires GenomeBatch and independent candidates"
            )
        config = self.spec.evolution.gene_copy
        genomes = GenomeBatch(
            genomes.genes,
            genomes.gene_counts,
            min_genes=config.min_genes,
            max_genes=config.max_genes,
        )
        shape = genomes.gene_counts.shape
        ids = (
            np.arange(genomes.gene_counts.size, dtype=np.uint64).reshape(shape)
            if candidate_ids is None
            else np.asarray(candidate_ids)
        )
        if ids.shape != shape:
            raise ValueError("candidate IDs must match genome batch axes")
        flat = genomes.genes.reshape(-1, genomes.genes.shape[-1])
        counts = genomes.gene_counts.reshape(-1)
        before = dict(self.metrics)
        chunks = []
        with self.model.engine._temporary_bytes(
            flat.nbytes * 2 + counts.nbytes + len(flat) * 512,
            message="mixed-length evaluation scratch exceeds memory budget",
        ):
            for count in sorted(set(map(int, counts))):
                indices = np.flatnonzero(counts == count)
                child = self._child(count)
                remaining = {}
                for resource in ("world_ticks", "chemical_updates"):
                    cap = getattr(self.work_limits, "max_" + resource)
                    spent = self.metrics[resource] - before[resource]
                    remaining["max_" + resource] = None if cap is None else max(0, cap - spent)
                child.set_work_limits(dataclasses.replace(self.work_limits, **remaining))
                out = child.evaluate(flat[indices, :count], ids.reshape(-1)[indices], panel=panel)
                chunks.append((indices, out))
            values = {}
            for name in (
                "scores",
                "status",
                "weights",
                "selection_fitness",
                "success",
                "descriptor",
            ):
                first = next(
                    (getattr(out, name) for _, out in chunks if getattr(out, name) is not None),
                    None,
                )
                if name == "descriptor" and any(not out.complete for _, out in chunks):
                    first = None
                if first is None:
                    values[name] = None
                    continue
                merged = np.zeros((len(flat), *first.shape[1:]), first.dtype)
                for indices, out in chunks:
                    part = getattr(out, name)
                    if part is not None:
                        merged[indices] = part
                values[name] = merged.reshape((*shape, *first.shape[1:]))
            denominators = {out.denominator for _, out in chunks if out.weights is not None}
            if len(denominators) > 1:
                raise ValueError("gene-count groups require one exact selection denominator")
            if denominators and any(
                out.weights is None and np.any(out.status == "complete") for _, out in chunks
            ):
                raise ValueError("complete candidates require consistent exact selection weights")
            metrics = {}
            for name in dict.fromkeys(name for _, out in chunks for name in out.metrics):
                indices, out = next((i, o) for i, o in chunks if name in o.metrics)
                first = out.metrics[name]
                if isinstance(first, np.ndarray) and first.shape[:1] == (len(indices),):
                    merged = np.zeros((len(flat), *first.shape[1:]), first.dtype)
                    for indices, out in chunks:
                        if name in out.metrics:
                            merged[indices] = out.metrics[name]
                    metrics[name] = merged.reshape((*shape, *first.shape[1:]))
                else:
                    metrics[name] = first
            metrics["gene_count_groups"] = {
                str(int(counts[indices[0]])): len(indices) for indices, _ in chunks
            }
            return EvaluationResult(
                **values,
                candidate_ids=ids,
                direction=chunks[0][1].direction,
                denominator=denominators.pop() if denominators else None,
                metrics=metrics,
            )

    def take_events(self):
        return [
            event for count in sorted(self.children) for event in self.children[count].take_events()
        ]

    def checkpoint_payload(self):
        children = {}
        for n, child in self.children.items():
            state = child.checkpoint_payload()
            state["cache"] = []
            children[str(n)] = state
        return dict(context=self.context, children=children, cache=list(self.cache.items()))

    def restore_payload(self, data):
        if data["context"] != self.context:
            raise ValueError("variable-gene evaluator context mismatch")
        self.close()
        self.children.clear()
        for count, state in sorted(data["children"].items(), key=lambda row: int(row[0])):
            self._child(int(count)).restore_payload(state)
        for key, value in data.get("cache", []):
            self.cache[(key[0], key[1], tuple(key[2]))] = value

    def close(self):
        for child in self.children.values():
            if self.runner is not None:
                for session in list(child.pending.values()):
                    self.runner.release(session)
            child.pending.clear()
            child.pending_arguments.clear()
        self.cache.clear()
