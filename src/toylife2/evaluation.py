"""Reusable complete-result objective evaluation with explicit selection mappings."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .genotypes import validate
from .protocols import ObjectiveSpec, WorkLimits, freeze
from .specs import digest


def _owned(value: Any) -> Any:
    if value is None:
        return None
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    return value


@dataclass(frozen=True)
class ObjectiveOutput:
    scores: np.ndarray
    metrics: Mapping[str, Any] = field(default_factory=dict)
    weights: np.ndarray | None = None
    denominator: int | None = None
    selection_fitness: np.ndarray | None = None
    success: np.ndarray | None = None
    descriptor: np.ndarray | None = None

    def __post_init__(self):
        for name in ("scores", "weights", "selection_fitness", "success", "descriptor"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(
                {
                    k: _owned(v)
                    if isinstance(v, np.ndarray)
                    else freeze(v.item() if isinstance(v, np.generic) else v)
                    for k, v in self.metrics.items()
                }
            ),
        )

    @property
    def exact_weights(self):
        return self.weights


@dataclass(frozen=True)
class EvaluationResult(ObjectiveOutput):
    direction: str = "maximize"
    status: np.ndarray = field(default_factory=lambda: np.asarray("complete"))
    candidate_ids: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "status", _owned(self.status))
        object.__setattr__(self, "candidate_ids", _owned(self.candidate_ids))
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("invalid objective direction")
        if self.status.shape != self.scores.shape:
            raise ValueError("evaluation status must match candidate axes")
        if np.any(~np.isfinite(self.scores[self.status == "complete"])):
            raise ValueError("complete objective scores must be finite")
        for name in ("weights", "selection_fitness", "success"):
            value = getattr(self, name)
            if value is not None and value.shape != self.scores.shape:
                raise ValueError(f"{name} must match candidate axes")
        if self.weights is not None:
            if self.weights.dtype.kind not in "iu" or np.any(self.weights < 0):
                raise ValueError("exact selection weights must be nonnegative integers")
            if type(self.denominator) is not int or not 0 < self.denominator < 1 << 63:
                raise ValueError("exact selection weights require a positive int64 denominator")
            if np.any(self.weights >= 1 << 63):
                raise OverflowError("selection weights exceed int64 admission")
        if self.selection_fitness is not None and np.any(
            (self.status == "complete")
            & (~np.isfinite(self.selection_fitness) | (self.selection_fitness < 0))
        ):
            raise ValueError("selection fitness must be finite and nonnegative")

    @property
    def complete(self):
        return bool(np.all(self.status == "complete"))

    def require_complete(self):
        if not self.complete:
            raise RuntimeError("objective evaluation is incomplete")
        return self


def _entrypoint(name):
    module, sep, attr = name.partition(":")
    if not sep:
        module, _, attr = name.rpartition(".")
    if not module or not attr:
        raise ValueError("entrypoint must be module:callable")
    result = getattr(importlib.import_module(module), attr)
    if not callable(result):
        raise ValueError("objective entrypoint must be callable")
    return result


class Evaluator:
    """Score ordered genomes without changing their identity or batch axes.

    Pure candidate objectives share complete scores across duplicate requests.
    Population objectives evaluate the supplied roster as a single barrier.
    """

    def __init__(self, model, spec, *, target=None, rollout_factory=None, runner=None):
        self.model, self.spec = model, spec
        self.objective = getattr(spec, "objective", ObjectiveSpec())
        self.target = _owned(target) if target is not None else None
        if self.target is None and self.objective.kind in {"auto", "pattern_match"}:
            path = spec.spatial.target
            if path:
                self.target = _owned(np.load(path, allow_pickle=False))
        if self.target is not None:
            if self.target.shape != (spec.spatial.steps, spec.spatial.cells):
                raise ValueError("target must match spatial steps and cells")
            if self.target.dtype.kind not in "iu" or np.any((self.target < 0) | (self.target > 3)):
                raise ValueError("target requires integer tissue states 0..3")
        self.rollout_factory, self.runner = rollout_factory, runner
        self.work_limits = getattr(spec, "limits", WorkLimits())
        self._request_used = dict(world_ticks=0, chemical_updates=0)
        self.cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
        self.pending = {}
        self.pending_arguments = {}
        self.events = []
        self._custom_denominator = None
        self.cache_capacity = max(
            1, spec.execution.memory_budget_mib * 1024 * 1024 // (8 * (160 + 40 * spec.model.genes))
        )
        scientific_task = _plain(getattr(spec, "task", None))
        scientific_objective = _plain(self.objective)
        for component in (scientific_task, scientific_objective):
            if component is not None and "inputs" in component["config"]:
                component["config"]["inputs"] = {
                    name: {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for name, path in component["config"]["inputs"].items()
                }
        spatial = _plain(spec.spatial)
        if spatial["target"] is not None:
            spatial["target"] = "target-content"
        from .components import capture

        components = capture(spec)
        self.identity_spec = dataclasses.replace(
            spec,
            task=dataclasses.replace(spec.task, config=scientific_task["config"]),
            objective=dataclasses.replace(self.objective, config=scientific_objective["config"]),
        )
        self.context = digest(
            _plain(
                {
                    "rules": spec.model,
                    "assay": spec.assay,
                    "spatial": spatial,
                    "target": self.target,
                    "objective": scientific_objective,
                    "task": scientific_task,
                    "components": None if components is None else components[1]["identity"],
                    "episode": getattr(spec, "episode", None),
                    "controller": (
                        {k: v for k, v in _plain(spec.controller).items() if k != "capacity"}
                        if hasattr(spec, "controller")
                        else None
                    ),
                    "table_digest": getattr(
                        getattr(model, "tables", None),
                        "digest",
                        getattr(model, "table_digest", None),
                    ),
                }
            )
        )
        self.metrics = dict(
            score_requests=0,
            duplicate_requests=0,
            cache_hits=0,
            new_genotype_evaluations=0,
            chunks=0,
            world_ticks=0,
            chemical_updates=0,
        )

    @property
    def stochastic(self):
        task = self.spec.task
        return task.kind in {"signalling_game", "run_tumble"} or (
            task.kind == "python" and bool(task.config.get("stochastic", True))
        )

    @property
    def kind(self):
        if self.objective.kind != "auto":
            return self.objective.kind
        if getattr(getattr(self.spec, "task", None), "kind", "assay") != "assay":
            return "bounded_reward"
        if self.spec.operation in {"wright_fisher", "sswm", "patterns"}:
            return "pattern_match"
        if self.spec.operation == "neutral_walk":
            return "descriptor"
        return "metabolic_coverage" if self.spec.assay.phenotype == "metabolic" else "descriptor"

    def _floor(self):
        if self.kind == "pattern_match" and self.objective.kind == "auto":
            return Fraction(str(self.spec.evolution.fitness_floor))
        return Fraction(self.objective.floor_numerator, self.objective.floor_denominator)

    def _rational(self, counts, total):
        floor = self._floor()
        denominator = math.lcm(int(total), floor.denominator)
        if denominator >= 1 << 63:
            raise OverflowError("objective denominator exceeds int64 admission")
        weights = np.maximum(
            np.asarray(counts, np.int64) * (denominator // total),
            floor.numerator * (denominator // floor.denominator),
        )
        scores = np.asarray(counts, float) / total
        # Selection fitness retains the exact floor used by the legacy chain.
        return scores, weights, denominator, np.maximum(scores, float(floor))

    def _assay(self, genes):
        model = self.model
        if hasattr(model, "execution"):
            execution = dataclasses.replace(
                model.execution,
                max_steps=min(self.work_limits.max_assay_steps, model.execution.max_steps),
            )
            model = copy.copy(model)
            model.execution = execution
            if hasattr(model, "engine"):
                model.engine = copy.copy(model.engine)
                model.engine.execution = execution
        if self.kind == "pattern_match":
            if self.target is None:
                raise ValueError("pattern_match requires a target")
            result = model.patterns(genes, self.spec.spatial)
        elif self.spec.assay.phenotype == "regulatory" and self.kind in {"descriptor", "python"}:
            result = model.regulatory(genes)
        else:
            result = model.metabolic(genes, self.spec.assay.environment)
        status = np.asarray(result.status)
        leading = genes.shape[:-1]
        if status.shape != leading:
            status = np.where(
                np.all(status.reshape((*leading, -1)) == "complete", axis=-1),
                "complete",
                "incomplete",
            )
        return result, status

    def _score(self, genes, *, panel, identities, agent_axis):
        task: Any = getattr(self.spec, "task", None)
        if task is not None and self.kind == "descriptor" and task.kind != "assay":
            task = dataclasses.replace(task, config={**task.config, "record_descriptor": True})
        result: Any
        episodic = task is not None and task.kind != "assay"
        if episodic:
            from .rollout import RolloutSession

            factory = self.rollout_factory or RolloutSession
            key = digest(_plain([self.context, panel, genes, identities, agent_axis]))
            limits = self._remaining_limits()
            session = self.pending.get(key)
            if session is None:
                session = factory(
                    self.model,
                    genes,
                    task,
                    self.spec.episode,
                    self.spec.controller,
                    limits=limits,
                    recording=self.spec.recording,
                    seed=self.spec.seed,
                    panel=panel,
                    agent_axis=agent_axis,
                    identities=identities,
                )
                self.pending[key] = session
                self.pending_arguments[key] = dict(
                    genes=np.array(genes, copy=True),
                    panel=panel,
                    identities=identities,
                    agent_axis=agent_axis,
                )
            if hasattr(session, "counters"):
                before = session.counters["invocation"]
                adjusted = {
                    name: (
                        None
                        if getattr(limits, "max_" + name) is None
                        else before[name] + getattr(limits, "max_" + name)
                    )
                    for name in self._request_used
                }
                session.limits = dataclasses.replace(
                    limits, **{"max_" + k: v for k, v in adjusted.items()}
                )
            else:
                before = {}
            while True:
                report = session.run()
                rows = []
                if self.spec.recording.mode == "full" and hasattr(session, "drain_trace"):
                    rows = session.drain_trace()
                    for event in rows:
                        self.events.append(
                            dict(
                                type="rollout_transition",
                                context=self.context,
                                panel=panel,
                                genotypes=genes.tolist(),
                                event=_plain(event),
                            )
                        )
                if not rows or getattr(report, "reason", None) != "trace_budget":
                    break
            if before:
                after = session.counters["invocation"]
                for resource in self._request_used:
                    spent = after[resource] - before[resource]
                    self._request_used[resource] += spent
                    self.metrics[resource] += spent
            scores = np.asarray(session.aggregate_scores(), float)
            status = np.asarray(session.status)
            if status.shape != scores.shape:
                status = np.where(np.all(status == "complete", axis=-1), "complete", "incomplete")
            result = session
            descriptor = None
            if self.kind in {"descriptor", "python"} and getattr(session, "descriptor", ()):
                # Actions and binary messages form the declared discrete trace.
                parts = []
                for entry in session.descriptor:
                    for name in ("actions", "message"):
                        value = np.asarray(entry[name])
                        if value.size:
                            if value.dtype.kind not in "biuUS":
                                raise ValueError(
                                    "neutral task traces must use discrete actions/messages"
                                )
                            parts.append(value.reshape((*scores.shape, -1)))
                descriptor = (
                    np.concatenate(parts, axis=-1)
                    if parts
                    else np.empty((*scores.shape, 0), np.int64)
                )
        else:
            result, status = self._assay(genes)
            descriptor = np.asarray(result.values)
            scores = None
        complete = status == "complete"
        weights = selection = denominator = None
        metrics = {}
        if self.kind == "python" and not np.all(complete):
            return EvaluationResult(
                np.full(status.shape, np.nan), direction=self.objective.direction, status=status
            )
        if self.kind == "python":
            readonly = _owned(genes)
            output = _entrypoint(self.objective.entrypoint)(readonly, result, self.objective.config)
            if not isinstance(output, ObjectiveOutput):
                raise TypeError("custom objectives must return ObjectiveOutput")
            if output.scores.shape != status.shape:
                raise ValueError("custom objective must preserve candidate axes")
            scores, weights, denominator = output.scores, output.weights, output.denominator
            if weights is not None:
                if self._custom_denominator is not None and denominator != self._custom_denominator:
                    raise ValueError("custom exact selection mappings require a fixed denominator")
                self._custom_denominator = denominator
            selection, descriptor, metrics = (
                output.selection_fitness,
                output.descriptor,
                output.metrics,
            )
            success = output.success
        else:
            success = None
            if self.kind == "pattern_match":
                if self.target is None:
                    raise ValueError("pattern_match requires a target")
                counts = np.count_nonzero(result.values == self.target, axis=(-2, -1))
                scores, weights, denominator, selection = self._rational(counts, self.target.size)
                if self.objective.kind == "auto" and self.spec.schema_version == 1:
                    scores = selection.copy()
            elif self.kind == "metabolic_coverage":
                supplied = self.spec.assay.environment
                total = len(supplied) if supplied is not None else result.values.shape[-1]
                if total < 1 or result.values.shape[-1] != total:
                    raise ValueError("metabolic coverage requires all supplied environment entries")
                counts = np.count_nonzero(result.values, axis=-1)
                scores, weights, denominator, selection = self._rational(counts, total)
            elif self.kind == "bounded_reward":
                if scores is None:
                    raise ValueError("bounded_reward requires episodic task rewards")
                if np.any(complete & ((scores < 0) | (scores > 1) | ~np.isfinite(scores))):
                    raise ValueError("bounded_reward requires rewards in [0,1]")
                counts = np.floor(
                    np.where(complete, scores, 0) * self.objective.quantization + 0.5
                ).astype(np.int64)
                _, weights, denominator, _ = self._rational(counts, self.objective.quantization)
                if task is not None and task.kind == "signalling_game" and np.all(complete):
                    totals = np.sum(result.reward_count, axis=-1)
                    exact_counts = np.sum(result.reward_sum, axis=-1)
                    if np.all(totals == 0):
                        scores, weights, denominator, _ = self._rational(
                            np.zeros(totals.shape, np.int64), 1
                        )
                        metrics = {
                            "empty": True,
                            "reward_numerator": np.zeros(totals.shape, np.int64),
                            "reward_denominator": 1,
                        }
                        success = np.zeros(totals.shape, bool)
                    else:
                        if np.any(totals <= 0) or np.any(totals != totals.flat[0]):
                            raise ValueError(
                                "signalling exact weights require a fixed judged denominator"
                            )
                        scores, weights, denominator, _ = self._rational(
                            exact_counts.astype(np.int64), int(totals.flat[0])
                        )
                        metrics = {
                            "reward_numerator": exact_counts.astype(np.int64),
                            "reward_denominator": int(totals.flat[0]),
                        }
                selection = np.maximum(scores, float(self._floor()))
            else:
                if descriptor is None or np.asarray(descriptor).dtype.kind not in "biuUS":
                    raise ValueError("neutral objectives require an explicit discrete descriptor")
                scores = np.zeros(genes.shape[:-1], float) if scores is None else scores
        if self.kind != "python" and self.objective.direction == "minimize":
            weights = selection = denominator = None
        if success is None:
            goal = self.objective.goal
            if goal is None and self.kind in {"pattern_match", "metabolic_coverage"}:
                goal = 1.0
            if goal is not None:
                operator = self.objective.goal_operator
                success = {"ge": np.greater_equal, "le": np.less_equal, "eq": np.equal}[operator](
                    scores, goal
                )
        scores = np.where(complete | (status == "empty"), scores, np.nan)
        if success is not None:
            success = np.asarray(success) & complete
        return EvaluationResult(
            scores,
            metrics,
            weights,
            denominator,
            selection,
            success,
            descriptor,
            self.objective.direction,
            status,
        )

    def take_events(self):
        result, self.events = self.events, []
        return result

    def _trace_request(self, result, genes, ids, panel):
        if self.spec.recording.mode == "full":
            valid_scores = np.array(result.scores, dtype=object, copy=True)
            valid_scores[result.status != "complete"] = None
            self.events.append(
                dict(
                    type="evaluation_request",
                    context=self.context,
                    panel=panel,
                    candidate_ids=ids.tolist(),
                    genotypes=genes.tolist(),
                    status=result.status.tolist(),
                    scores=valid_scores.tolist(),
                    validity=(result.status == "complete").tolist(),
                )
            )
        return result

    def _forget_completed(self, genes, panel, identities=None, agent_axis=None):
        key = digest(_plain([self.context, panel, genes, identities, agent_axis]))
        session = self.pending.get(key)
        if session is not None and np.all(session.status == "complete"):
            if self.runner is not None:
                self.runner.release(session)
            del self.pending[key]
            del self.pending_arguments[key]

    def set_work_limits(self, limits):
        self.work_limits = limits

    def _remaining_limits(self):
        values = {}
        for name, spent in self._request_used.items():
            cap = getattr(self.work_limits, "max_" + name)
            values["max_" + name] = None if cap is None else max(0, cap - spent)
        return dataclasses.replace(self.work_limits, **values)

    def checkpoint_payload(self):
        result = []
        for key, session in self.pending.items():
            cp = session.checkpoint()
            result.append(
                dict(
                    key=key,
                    arguments=self.pending_arguments[key],
                    kind=cp.kind,
                    identity=cp.identity,
                    payload=cp.payload,
                )
            )
        return dict(
            context=self.context,
            pending=result,
            events=self.events,
            custom_denominator=self._custom_denominator,
            cache=[[key, row] for key, row in self.cache.items()],
        )

    def restore_payload(self, data):
        from .rollout import RolloutSession
        from .sessions import Checkpoint, pack

        if data["context"] != self.context:
            raise ValueError("evaluator checkpoint context mismatch")
        self.events = list(data.get("events", []))
        self._custom_denominator = data.get("custom_denominator")
        for key, row in data.get("cache", []):
            key = (key[0], key[1], tuple(key[2]))
            self.cache[key] = row
            while len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)
        for item in data["pending"]:
            args = item["arguments"]
            arrays = {}
            state = pack(item["payload"], arrays)
            cp = Checkpoint(item["kind"], item["identity"], state, arrays)
            factory = self.rollout_factory or RolloutSession
            task = self.spec.task
            if self.kind == "descriptor" and task.kind != "assay":
                task = dataclasses.replace(task, config={**task.config, "record_descriptor": True})
            session = factory(
                self.model,
                args["genes"],
                task,
                self.spec.episode,
                self.spec.controller,
                limits=self.work_limits,
                recording=self.spec.recording,
                seed=self.spec.seed,
                panel=args["panel"],
                identities=args["identities"],
                agent_axis=args["agent_axis"],
                checkpoint=cp,
            )
            self.pending[item["key"]] = session
            self.pending_arguments[item["key"]] = args

    def evaluate_population(self, population, *, replicate, generation):
        """Freeze the roster and average balanced roles over four cyclic opponents."""
        genes = validate(population)
        if genes.ndim != 2:
            raise ValueError("contextual roster must be [individual, gene]")
        panel = (
            generation
            if self.spec.episode.panel == "generation"
            or (self.spec.episode.panel == "auto" and self.stochastic)
            else 0
        )
        if self.spec.task.kind != "signalling_game":
            ids = np.arange(len(genes), dtype=np.uint64)
            result = self.evaluate(genes, panel=panel, identities=ids)
            if result.complete:
                self._forget_completed(genes, panel, ids)
            return result
        from .random import Stream

        ids = np.arange(len(genes), dtype=np.uint64)
        draws = Stream(self.spec.seed).words(
            ids, role="opponent_order", trial=replicate, event=generation
        )
        order = np.asarray(np.lexsort((ids, draws[:, 0])), dtype=np.int64)
        ranks = np.empty(len(genes), np.int64)
        ranks[order] = np.arange(len(genes))
        offsets = (
            np.arange(1, min(4, len(genes) - 1) + 1, dtype=np.int64)
            if len(genes) > 1
            else np.asarray([0], dtype=np.int64)
        )
        matches = len(offsets)
        opponents = order[(ranks[:, None] + offsets) % len(genes)]
        pairs = np.stack(
            (
                np.broadcast_to(genes[:, None], (len(genes), matches, genes.shape[-1])),
                genes[opponents],
            ),
            axis=-2,
        )
        # The ordered focal/opponent IDs anchor worlds; genotype changes never change their panel.
        identities = np.stack(
            (np.broadcast_to(ids[:, None], (len(genes), matches)), opponents), axis=-1
        )
        left = self.evaluate(pairs, panel=panel, identities=identities, agent_axis=-2)
        original_limits = self.work_limits
        self.work_limits = self._remaining_limits()
        right = self.evaluate(
            pairs[..., ::-1, :], panel=panel, identities=identities[..., ::-1], agent_axis=-2
        )
        self.work_limits = original_limits
        status = np.where(
            np.all((left.status == "complete") & (right.status == "complete"), axis=-1),
            "complete",
            "incomplete",
        )
        scores = (left.scores.mean(axis=-1) + right.scores.mean(axis=-1)) / 2
        weights = selection = denominator = None
        if np.all(status == "complete") and left.weights is not None and right.weights is not None:
            if "reward_numerator" in left.metrics and "reward_numerator" in right.metrics:
                left_den = int(left.metrics["reward_denominator"])
                right_den = int(right.metrics["reward_denominator"])
                common = math.lcm(left_den, right_den, self._floor().denominator)
                denominator = common * (2 * matches)
                weights = left.metrics["reward_numerator"].sum(axis=-1) * (
                    common // left_den
                ) + right.metrics["reward_numerator"].sum(axis=-1) * (common // right_den)
                weights = np.maximum(
                    weights, self._floor().numerator * (denominator // self._floor().denominator)
                ).astype(np.int64)
            else:
                common = math.lcm(left.denominator, right.denominator)
                denominator = common * (2 * matches)
                weights = (
                    left.weights.sum(axis=-1) * (common // left.denominator)
                    + right.weights.sum(axis=-1) * (common // right.denominator)
                ).astype(np.int64)
            if denominator >= 1 << 63:
                raise OverflowError("contextual denominator exceeds int64 admission")
            selection = np.maximum(scores, float(self._floor()))
        goal = self.objective.goal
        success = (
            None
            if goal is None
            else {"ge": np.greater_equal, "le": np.less_equal, "eq": np.equal}[
                self.objective.goal_operator
            ](scores, goal)
            & (status == "complete")
        )
        if np.all(status == "complete"):
            self._forget_completed(pairs, panel, identities, -2)
            self._forget_completed(pairs[..., ::-1, :], panel, identities[..., ::-1], -2)
            for key, session in list(self.pending.items()):
                if np.all(session.status == "complete"):
                    if self.runner is not None:
                        self.runner.release(session)
                    del self.pending[key]
                    del self.pending_arguments[key]
        return EvaluationResult(
            scores,
            weights=weights,
            denominator=denominator,
            selection_fitness=selection,
            success=success,
            direction=self.objective.direction,
            status=status,
            candidate_ids=ids,
            metrics=dict(opponents=opponents, roster=genes),
        )

    def _score_cohort(self, genes, *, panel, identities, agent_axis, shape):
        # Frozen matchups remain the scoring context while worlds use bounded cohorts.
        arranged = np.moveaxis(genes, agent_axis % genes.ndim, -2)
        flat = arranged.reshape(-1, arranged.shape[-2], arranged.shape[-1])
        flat_ids = None if identities is None else np.asarray(identities).reshape(len(flat), -1)
        outputs = []
        cursor = 0
        while cursor < len(flat):
            width = min(self.spec.execution.batch_size, len(flat) - cursor)
            saved = None
            for args in self.pending_arguments.values():
                if args["agent_axis"] != -2 or args["panel"] != panel:
                    continue
                size = len(args["genes"])
                ids = None if flat_ids is None else flat_ids[cursor : cursor + size]
                same_ids = args["identities"] is None and ids is None
                if args["identities"] is not None and ids is not None:
                    same_ids = np.array_equal(args["identities"], ids)
                if np.array_equal(args["genes"], flat[cursor : cursor + size]) and same_ids:
                    saved = size
                    break
            if saved is not None:
                width = saved
            else:
                available = self._remaining_limits()
                for resource, lanes in (
                    ("world_ticks", self.spec.episode.episodes),
                    ("chemical_updates", self.spec.episode.episodes * arranged.shape[-2]),
                ):
                    cap = getattr(available, "max_" + resource)
                    if cap is not None:
                        width = min(width, max(1, cap // lanes))
            chunk_ids = None if flat_ids is None else flat_ids[cursor : cursor + width]
            self.metrics["chunks"] += 1
            outputs.append(
                self._score(
                    flat[cursor : cursor + width], panel=panel, identities=chunk_ids, agent_axis=-2
                )
            )
            cursor += width
        status = np.concatenate([out.status for out in outputs]).reshape(shape)
        scores = np.concatenate([out.scores for out in outputs]).reshape(shape)
        denominator = next(
            (out.denominator for out in outputs if out.complete), outputs[0].denominator
        )
        values: dict[str, Any] = dict(
            scores=scores,
            status=status,
            direction=self.objective.direction,
            denominator=denominator,
        )
        for name in ("weights", "selection_fitness", "success", "descriptor"):
            if any(getattr(out, name) is None for out in outputs):
                values[name] = None
                continue
            parts = []
            for out in outputs:
                value = getattr(out, name)
                if name == "weights" and out.denominator != denominator:
                    if out.complete:
                        raise ValueError("cohort exact weights require a fixed denominator")
                    value = np.zeros_like(value)
                parts.append(value)
            joined = np.concatenate(parts)
            values[name] = joined.reshape((*shape, *joined.shape[1:]))
        names = set.intersection(*(set(out.metrics) for out in outputs))
        metrics = {}
        for name in names:
            parts = [out.metrics[name] for out in outputs]
            metrics[name] = (
                np.concatenate(parts).reshape(shape)
                if isinstance(parts[0], np.ndarray)
                else parts[0]
            )
        values["metrics"] = metrics
        return EvaluationResult(**values)

    def evaluate(self, genotypes, candidate_ids=None, *, panel=0, identities=None, agent_axis=None):
        self._request_used = dict(world_ticks=0, chemical_updates=0)
        genes = validate(genotypes)
        shape = genes.shape[:-1]
        if agent_axis is not None:
            axis = agent_axis % genes.ndim
            if axis == genes.ndim - 1:
                raise ValueError("agent axis cannot be the gene axis")
            shape = shape[:axis] + shape[axis + 1 :]
        if genes.shape[-1] != self.spec.model.genes:
            raise ValueError("genotype gene axis differs from ModelSpec")
        ids = (
            np.arange(math.prod(shape), dtype=np.uint64).reshape(shape)
            if candidate_ids is None
            else np.asarray(candidate_ids)
        )
        if ids.shape != shape:
            raise ValueError("candidate_ids must match candidate axes")
        flat = genes.reshape(-1, genes.shape[-1])
        self.metrics["score_requests"] += len(flat)
        cacheable = (
            self.objective.pure and self.objective.scope == "candidate" and agent_axis is None
        )
        if not cacheable:
            self.metrics["new_genotype_evaluations"] += len(flat)
            if (
                agent_axis is not None
                and self.objective.scope == "population"
                and self.kind == "bounded_reward"
            ):
                out = self._score_cohort(
                    genes, panel=panel, identities=identities, agent_axis=agent_axis, shape=shape
                )
            else:
                self.metrics["chunks"] += 1
                out = self._score(genes, panel=panel, identities=identities, agent_axis=agent_axis)
            if out.complete and self.objective.scope != "population":
                self._forget_completed(genes, panel, identities, agent_axis)
            return self._trace_request(
                dataclasses.replace(out, candidate_ids=ids), genes, ids, panel
            )
        keys = [(self.context, _plain(panel), tuple(map(int, row))) for row in flat]
        # Hash panel labels even when a caller supplies a structured panel identity.
        keys = [(context, digest(panel_id), genome) for context, panel_id, genome in keys]
        unique = list(dict.fromkeys(keys))
        self.metrics["duplicate_requests"] += len(keys) - len(unique)
        local = {k: self.cache[k] for k in unique if k in self.cache}
        self.metrics["cache_hits"] += sum(k in self.cache for k in keys)
        missing = sorted(k for k in unique if k not in local)
        batches = []
        covered = set()
        episodic = getattr(getattr(self.spec, "task", None), "kind", "assay") != "assay"
        if episodic:
            for args in self.pending_arguments.values():
                if args["agent_axis"] is not None or digest(_plain(args["panel"])) != digest(
                    _plain(panel)
                ):
                    continue
                batch = [
                    (self.context, digest(_plain(panel)), tuple(map(int, row)))
                    for row in args["genes"].reshape(-1, genes.shape[-1])
                ]
                if any(key in missing for key in batch):
                    batches.append(batch)
                    covered.update(batch)
        missing = [key for key in missing if key not in covered]
        chunk_size = self.spec.execution.batch_size

        def admitted_batches():
            yield from batches
            cursor = 0
            while cursor < len(missing):
                width = chunk_size
                if episodic:
                    available = self._remaining_limits()
                    episodes = self.spec.episode.episodes
                    for resource in ("world_ticks", "chemical_updates"):
                        cap = getattr(available, "max_" + resource)
                        if cap is not None:
                            width = min(width, max(1, cap // episodes))
                yield missing[cursor : cursor + width]
                cursor += width

        for batch in admitted_batches():
            self.metrics["new_genotype_evaluations"] += len(batch)
            self.metrics["chunks"] += 1
            out = self._score(
                np.asarray([k[-1] for k in batch], np.uint32),
                panel=panel,
                identities=None,
                agent_axis=None,
            )
            for i, key in enumerate(batch):
                item: dict[str, Any] = dict(
                    scores=out.scores[i],
                    status=out.status[i],
                    direction=out.direction,
                    denominator=out.denominator,
                    metrics={
                        k: (
                            v[i]
                            if isinstance(v, np.ndarray) and v.shape[:1] == out.scores.shape[:1]
                            else v
                        )
                        for k, v in out.metrics.items()
                    },
                )
                for name in ("weights", "selection_fitness", "success", "descriptor"):
                    array = getattr(out, name)
                    item[name] = None if array is None else array[i]
                local[key] = item
                if item["status"] == "complete":
                    self.cache[key] = item
                    while len(self.cache) > self.cache_capacity:
                        self.cache.popitem(last=False)
            if out.complete:
                self._forget_completed(np.asarray([k[-1] for k in batch], np.uint32), panel)
        rows = [local[k] for k in keys]
        if not rows:
            return EvaluationResult(
                np.empty(shape), status=np.empty(shape, dtype="U10"), candidate_ids=ids
            )
        values: dict[str, Any] = dict(
            direction=rows[0]["direction"],
            denominator=rows[0]["denominator"],
            metrics={
                k: (
                    np.asarray([r["metrics"][k] for r in rows]).reshape(
                        (*shape, *np.asarray(v).shape)
                    )
                    if isinstance(v, np.ndarray) or np.isscalar(v)
                    else v
                )
                for k, v in rows[0]["metrics"].items()
            },
        )
        for name in ("scores", "status", "weights", "selection_fitness", "success", "descriptor"):
            values[name] = (
                None
                if rows[0][name] is None
                else np.asarray([r[name] for r in rows]).reshape(
                    (*shape, *np.asarray(rows[0][name]).shape)
                )
            )
        if rows[0]["weights"] is not None and any(
            row["denominator"] != rows[0]["denominator"] for row in rows
        ):
            raise ValueError("custom exact selection mappings require a fixed denominator")
        return self._trace_request(EvaluationResult(**values, candidate_ids=ids), genes, ids, panel)
