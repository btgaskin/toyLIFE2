"""Stepwise population evolution with stable replicate identities and observation barriers."""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from typing import Any, cast

import numpy as np

from .evaluation import EvaluationResult, Evaluator
from .evolution import _admit_population, _initial, _mutate, log_acceptance
from .genotypes import mutation, neighbours
from .protocols import WorkLimits
from .random import SCHEME, Stream
from .sessions import SessionBase, owned
from .specs import RunSpec, digest, plain, scientific_identity


def wright_fisher_step(population, weights, *, stream, replicate, generation, mutation_probability):
    """Exact integer parent selection; addresses match the published legacy runner."""
    pop = np.asarray(population, np.uint32)
    weights = np.asarray(weights)
    if weights.shape != pop.shape[:-1] or weights.dtype.kind not in "iu" or np.any(weights < 0):
        raise ValueError("one nonnegative integer weight is required per individual")
    if not len(pop):
        raise ValueError("population cannot be empty")
    if int(weights.max()) * len(weights) >= 1 << 63:
        raise OverflowError("population fitness sum exceeds int64 admission")
    prefix = np.cumsum(weights, dtype=np.int64)
    ids = np.arange(len(pop), dtype=np.uint64)
    if prefix[-1] == 0:
        parents = stream.integers(len(pop), ids, role="parents", trial=replicate, event=generation)
    else:
        draws = stream.integers(
            int(prefix[-1]), ids, role="parents", trial=replicate, event=generation
        )
        parents = np.searchsorted(prefix, draws, side="right")
    new = pop[parents.astype(np.intp)].copy()
    gates = (
        stream.uniform(ids, role="mutation_gate", trial=replicate, event=generation)
        < mutation_probability
    )
    bits = stream.integers(
        20 * pop.shape[-1], ids, role="mutation_bit", trial=replicate, event=generation
    )
    _mutate(new, bits, gates)
    return new, parents, bits, gates


def _result_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        f.name: dict(result.metrics) if f.name == "metrics" else getattr(result, f.name)
        for f in dataclasses.fields(result)
    }


class PopulationSession(SessionBase):
    """Observe initial and final populations, advancing one transition per ready lane."""

    def __init__(
        self,
        model,
        spec,
        evaluator=None,
        limits=None,
        recording=None,
        checkpoint=None,
        *,
        target=None,
    ):
        if spec.operation not in {"wright_fisher", "sswm", "neutral_walk"}:
            raise ValueError("PopulationSession requires an evolution operation")
        if spec.seed is None:
            raise ValueError("evolution requires an explicit seed")
        if (
            spec.operation == "sswm"
            and spec.task.kind != "assay"
            and spec.episode.panel == "generation"
        ):
            raise ValueError("SSWM requires a fixed episodic panel")
        self.model, self.spec = model, spec
        self.evaluator = evaluator or Evaluator(model, spec, target=target)
        self.algorithm = spec.operation
        self.stream = Stream(spec.seed)
        self.replicates = (
            np.arange(spec.evolution.trials, dtype=np.int64) + spec.evolution.trial_start
        )
        count = spec.evolution.population_size if self.algorithm == "wright_fisher" else 1
        _admit_population(spec, count)
        if limits is None:
            limits = getattr(spec, "limits", None)
            if limits is None or (
                getattr(spec, "schema_version", 1) == 1 and limits.max_evolution_updates is None
            ):
                cap = (
                    spec.execution.max_events
                    if self.algorithm == "sswm"
                    else spec.execution.max_steps
                )
                limits = WorkLimits(max_evolution_updates=cap)
        super().__init__(
            "population",
            digest(
                [
                    scientific_identity(
                        cast(RunSpec, getattr(self.evaluator, "identity_spec", spec))
                    ),
                    self.evaluator.context,
                ]
            ),
            limits,
            recording or spec.recording,
            checkpoint,
        )
        self.populations = np.stack(
            [_initial(spec, self.stream, t, count) for t in range(len(self.replicates))]
        )
        size = len(self.replicates)
        self.cursors = np.zeros(size, np.int64)
        self.times = np.zeros(size, float)
        self.observed = np.full(size, -1, np.int64)
        self.arrivals: list[int | float | None] = [None] * size
        self.lane_status = np.full(size, "ready", dtype="U12")
        self.pending: list[dict[str, Any] | None] = [None] * size
        self.evaluations: list[EvaluationResult | None] = [None] * size
        self.history_capacity = max(
            1,
            min(
                self.recording.trace_budget_mib * 1024 * 1024,
                spec.execution.memory_budget_mib * 1024 * 1024 // 8,
            )
            // 512,
        )
        self.history = OrderedDict()
        self.observation_count = 0
        self._evaluation_events = []
        self._conversion: dict[str, Any] | None = None
        self.round_robin = 0
        if checkpoint is not None:
            self._restore_payload(checkpoint.payload["payload"])

    def _restore_payload(self, data):
        if (
            data.get("version") != 2
            or data.get("rng") != SCHEME
            or data.get("algorithm") != self.algorithm
        ):
            raise ValueError("checkpoint algorithm or RNG mismatch")
        ids = np.asarray(data["replicates"])
        if ids.shape != self.replicates.shape or not np.array_equal(ids, self.replicates):
            raise ValueError("checkpoint replicate identities mismatch")
        pop = np.asarray(data["populations"])
        if (
            pop.shape != self.populations.shape
            or pop.dtype.kind not in "iu"
            or np.any(pop >= 1 << 20)
        ):
            raise ValueError("invalid checkpoint populations")
        self.populations[:] = pop
        for name in ("cursors", "times", "observed", "lane_status"):
            value = np.asarray(data[name])
            if value.shape != self.cursors.shape:
                raise ValueError("invalid checkpoint lane axes")
            setattr(self, name, value.copy())
        if (
            np.any(self.cursors < 0)
            or np.any(self.observed > self.cursors)
            or np.any(self.observed < -1)
        ):
            raise ValueError("invalid checkpoint observation cursors")
        self.arrivals = list(data["arrivals"])
        self.pending = list(data["pending"])
        self.evaluations = [
            None if x is None else EvaluationResult(**x) for x in data["evaluations"]
        ]
        self.history_capacity = int(data["history_capacity"])
        self.observation_count = int(data["observation_count"])
        if self.history_capacity < 1 or self.observation_count < 0:
            raise ValueError("invalid checkpoint history counters")
        self.history = OrderedDict((tuple(key), row) for key, row in data["history"])
        if len(self.history) > self.history_capacity:
            raise ValueError("checkpoint history exceeds pinned retention")
        self._conversion = data.get("conversion")
        self._evaluation_events = list(data.get("evaluation_events", []))
        self.round_robin = int(data["round_robin"])
        if not 0 <= self.round_robin < len(self.replicates):
            raise ValueError("invalid scheduler cursor")
        if "evaluator" in data and hasattr(self.evaluator, "restore_payload"):
            self.evaluator.restore_payload(data["evaluator"])

    def _checkpoint_payload(self):
        result = dict(
            spec=plain(self.spec),
            version=2,
            algorithm=self.algorithm,
            rng=SCHEME,
            replicates=self.replicates,
            populations=self.populations,
            cursors=self.cursors,
            times=self.times,
            observed=self.observed,
            arrivals=self.arrivals,
            lane_status=self.lane_status,
            pending=self.pending,
            evaluations=[None if x is None else _result_payload(x) for x in self.evaluations],
            evaluation_events=self._evaluation_events,
            conversion=self._conversion,
            history_capacity=self.history_capacity,
            observation_count=self.observation_count,
            history=[[list(k), v] for k, v in self.history.items()],
            round_robin=self.round_robin,
        )
        if hasattr(self.evaluator, "checkpoint_payload"):
            result["evaluator"] = self.evaluator.checkpoint_payload()
        return result

    def _evaluate(self, *args, population=False, **kwargs) -> EvaluationResult:
        if hasattr(self.evaluator, "set_work_limits"):
            current = dataclasses.replace(
                self.limits,
                max_world_ticks=self.remaining("world_ticks"),
                max_chemical_updates=self.remaining("chemical_updates"),
            )
            self.evaluator.set_work_limits(current)
        before = dict(getattr(self.evaluator, "metrics", {}))
        try:
            method = self.evaluator.evaluate_population if population else self.evaluator.evaluate
            return method(*args, **kwargs)
        finally:
            if hasattr(self.evaluator, "take_events"):
                self._evaluation_events.extend(self.evaluator.take_events())
            metrics = getattr(self.evaluator, "metrics", {})
            for resource in ("world_ticks", "chemical_updates"):
                spent = int(metrics.get(resource, 0) - before.get(resource, 0))
                if spent:
                    self._consume(resource, spent)

    def _take_evaluation_events(self):
        events, self._evaluation_events = self._evaluation_events, []
        return events

    def _panel(self, lane):
        episode = getattr(self.spec, "episode", None)
        return (
            int(self.cursors[lane])
            if episode is not None
            and (
                episode.panel == "generation"
                or (
                    episode.panel == "auto"
                    and self.algorithm == "wright_fisher"
                    and self.evaluator.stochastic
                )
            )
            else 0
        )

    def _retain(self, lane, row):
        key = (int(self.replicates[lane]), int(self.cursors[lane]))
        self.history[key] = row
        self.history = OrderedDict(sorted(self.history.items()))
        while len(self.history) > self.history_capacity:
            self.history.popitem(last=False)
        self.observation_count += 1

    def _terminal(self, lane):
        if self.algorithm == "sswm":
            return (
                self.times[lane] >= self.spec.evolution.time_horizon
                or self.lane_status[lane] == "complete"
            )
        return self.cursors[lane] >= self.spec.evolution.generations

    def observe(self):
        """Commit completed observations without consuming evolution updates."""
        if self._pending_trace and not self._flush_pending_trace():
            return self._report_state()
        self._reason = None
        events = []
        missing_scores = np.asarray([value is None for value in self.evaluations], dtype=bool)
        lanes = np.flatnonzero(
            ((self.observed < self.cursors) | missing_scores) & (self.lane_status != "complete")
        )
        groups = {}
        for lane in lanes:
            groups.setdefault(self._panel(lane), []).append(int(lane))
        incomplete = False
        for panel, members in groups.items():
            # A score request preserves every individual and the original order.
            genes = self.populations[members]
            ids = np.asarray(
                [
                    [int(self.replicates[i]) * genes.shape[1] + j for j in range(genes.shape[1])]
                    for i in members
                ],
                np.uint64,
            )
            if getattr(self.spec.objective, "scope", "candidate") == "population":
                member_results = [
                    self._evaluate(
                        self.populations[lane],
                        population=True,
                        replicate=int(self.replicates[lane]),
                        generation=int(self.cursors[lane]),
                    )
                    for lane in members
                ]
                evaluation = None
            else:
                evaluation = self._evaluate(genes, ids, panel=panel)
                member_results = None
            for offset, lane in enumerate(members):
                if member_results is not None:
                    fields = _result_payload(member_results[offset])
                else:
                    if evaluation is None:
                        raise RuntimeError("observation did not produce scores")
                    fields = _result_payload(evaluation)
                for name in (
                    "scores",
                    "status",
                    "candidate_ids",
                    "weights",
                    "selection_fitness",
                    "success",
                    "descriptor",
                ):
                    if member_results is None and fields[name] is not None:
                        fields[name] = fields[name][offset]
                result = EvaluationResult(**fields)
                if not result.complete:
                    self.lane_status[lane] = "partial"
                    incomplete = True
                    continue
                already_observed = self.observed[lane] == self.cursors[lane]
                self.evaluations[lane] = result
                self.observed[lane] = self.cursors[lane]
                self.lane_status[lane] = "complete" if self._terminal(lane) else "ready"
                if (
                    result.success is not None
                    and np.any(result.success)
                    and self.arrivals[lane] is None
                ):
                    self.arrivals[lane] = (
                        float(self.times[lane])
                        if self.algorithm == "sswm"
                        else int(self.cursors[lane])
                    )
                if self.algorithm == "wright_fisher":
                    if self.spec.schema_version == 1 and (
                        result.weights is None or result.denominator is None
                    ):
                        raise ValueError("legacy Wright–Fisher requires exact selection weights")
                    mean_score = (
                        float(result.weights.mean() / result.denominator)
                        if self.spec.schema_version == 1
                        and result.weights is not None
                        and result.denominator is not None
                        else float(result.scores.mean())
                    )
                    max_score = (
                        float(result.weights.max() / result.denominator)
                        if self.spec.schema_version == 1
                        and result.weights is not None
                        and result.denominator is not None
                        else float(result.scores.max())
                    )
                    row = [
                        int(lane),
                        int(self.cursors[lane]),
                        mean_score,
                        max_score,
                        len(np.unique(self.populations[lane], axis=0)),
                    ]
                else:
                    row = [
                        int(lane),
                        float(self.times[lane])
                        if self.algorithm == "sswm"
                        else int(self.cursors[lane]),
                        float(result.scores[0]),
                        int(self.cursors[lane]),
                    ]
                if self.algorithm != "neutral_walk" and not already_observed:
                    self._retain(lane, row)
                elif already_observed:
                    key = (int(self.replicates[lane]), int(self.cursors[lane]))
                    if key in self.history:
                        self.history[key] = row
                if already_observed:
                    continue
                events.append(
                    dict(
                        type="observation",
                        replicate=int(self.replicates[lane]),
                        cursor=int(self.cursors[lane]),
                        scores=result.scores.tolist(),
                        status=self.lane_status[lane],
                    )
                )
        if np.all(self.lane_status == "complete"):
            self._status = "complete"
        elif incomplete:
            self._status, self._reason = "partial", "incomplete_evaluation"
        else:
            self._status = "ready"
        events.extend(self._take_evaluation_events())
        self._boundary(events, force=self._status in {"complete", "partial"})
        return self._report_state()

    def _current(self, lane) -> EvaluationResult:
        result = self.evaluations[lane]
        if result is None:
            raise RuntimeError("current population has no valid scores")
        return result.require_complete()

    def _wf(self, lane):
        result = self._current(lane)
        if result.weights is None or result.denominator is None:
            raise ValueError("Wright–Fisher requires explicit exact weights and denominator")
        new, parents, bits, gates = wright_fisher_step(
            self.populations[lane],
            result.weights,
            stream=self.stream,
            replicate=int(self.replicates[lane]),
            generation=int(self.cursors[lane]),
            mutation_probability=self.spec.evolution.mutation_probability,
        )
        event = dict(
            type="generation",
            replicate=int(self.replicates[lane]),
            from_generation=int(self.cursors[lane]),
            to_generation=int(self.cursors[lane]) + 1,
            parents=parents.tolist(),
            mutations=[[int(i), int(bits[i])] for i in np.flatnonzero(gates)],
            before=self.populations[lane].tolist(),
            after=new.tolist(),
            parent_fitness_weights=result.weights.tolist(),
            fitness_denominator=result.denominator,
        )
        self.populations[lane] = new
        self.cursors[lane] += 1
        return event

    def _neutral(self, lane):
        current = self._current(lane)
        if current.descriptor is None or current.descriptor.dtype.kind not in "biuUS":
            raise ValueError("neutral walk requires an explicit discrete descriptor")
        position = int(
            self.stream.integers(
                20 * self.spec.model.genes,
                0,
                role="neutral_proposal",
                trial=int(self.replicates[lane]),
                event=int(self.cursors[lane]),
            )
        )
        candidate = mutation(self.populations[lane, 0], position)
        result = self._evaluate(candidate[None], panel=self._panel(lane))
        if not result.complete:
            self._status, self._reason = "partial", "incomplete_evaluation"
            return None
        if result.descriptor is None:
            raise ValueError("neutral candidate requires a discrete descriptor")
        accepted = np.array_equal(current.descriptor, result.descriptor)
        before = self.populations[lane, 0].copy()
        if accepted:
            self.populations[lane, 0] = candidate
        event = dict(
            type="neutral_proposal",
            replicate=int(self.replicates[lane]),
            event=int(self.cursors[lane]),
            position=position,
            accepted=accepted,
            before=before.tolist(),
            after=self.populations[lane, 0].tolist(),
        )
        self.cursors[lane] += 1
        self._retain(lane, [int(lane), int(self.cursors[lane]) - 1, position, int(accepted)])
        return event

    def _sswm(self, lane):
        current = self._current(lane)
        if current.selection_fitness is None:
            raise ValueError("SSWM requires explicit nonnegative selection fitness")
        if self.pending[lane] is None:
            mutants = neighbours(self.populations[lane, 0])
            result = self._evaluate(mutants, panel=self._panel(lane))
            if not result.complete:
                self._status, self._reason = "partial", "incomplete_evaluation"
                return None
            if result.selection_fitness is None:
                raise ValueError("SSWM requires explicit nonnegative selection fitness")
            value = float(current.selection_fitness[0])
            logs = np.asarray(
                [
                    log_acceptance(value, float(v), self.spec.evolution.population_size)
                    - math.log(len(mutants))
                    for v in result.selection_fitness
                ]
            )
            maximum = float(logs.max())
            if maximum == -math.inf:
                self.pending[lane] = dict(time=None, position=None)
            else:
                rates = np.exp(logs - maximum)
                log_total = maximum + math.log(float(rates.sum()))
                u = float(
                    self.stream.uniform(
                        0,
                        role="waiting_time",
                        trial=int(self.replicates[lane]),
                        event=int(self.cursors[lane]),
                    )
                )
                log_wait = math.log(-math.log(u)) - log_total
                wait = math.exp(log_wait) if log_wait < 709 else math.inf
                draw = float(
                    self.stream.uniform(
                        0,
                        role="transition",
                        trial=int(self.replicates[lane]),
                        event=int(self.cursors[lane]),
                    )
                ) * float(rates.sum())
                position = min(
                    int(np.searchsorted(np.cumsum(rates), draw, side="right")), len(mutants) - 1
                )
                time = self.times[lane] + wait
                self.pending[lane] = dict(
                    time=float(time) if math.isfinite(time) else None,
                    position=position,
                    log_wait=log_wait,
                )
        pending = self.pending[lane]
        if pending is None:
            raise RuntimeError("SSWM proposal did not establish a pending event")
        if pending["time"] is None or pending["time"] > self.spec.evolution.time_horizon:
            self.lane_status[lane] = "complete"
            return False
        before = self.populations[lane, 0].copy()
        self.populations[lane, 0] = mutation(before, pending["position"])
        event = dict(
            type="transition",
            replicate=int(self.replicates[lane]),
            event=int(self.cursors[lane]),
            from_time=float(self.times[lane]),
            to_time=pending["time"],
            position=pending["position"],
            before=before.tolist(),
            after=self.populations[lane, 0].tolist(),
            pending=dict(pending),
        )
        self.times[lane] = pending["time"]
        self.cursors[lane] += 1
        self.pending[lane] = None
        return event

    def step(self):
        if self._status == "complete":
            return self._report_state()
        self.observe()
        if self._status in {"complete", "partial"}:
            return self._report_state()
        ready = [i for i in range(len(self.replicates)) if self.lane_status[i] == "ready"]
        ordered = sorted(ready, key=lambda i: (i - self.round_robin) % len(self.replicates))
        max_active = getattr(self.spec.execution, "max_active_lanes", len(ordered))
        if getattr(self.spec.execution, "evolution_schedule", "wavefront") == "sequential":
            ordered = sorted(ready)[:1]
        else:
            ordered = ordered[:max_active]
        events = []
        for lane in ordered:
            if not self._allow("evolution_updates"):
                break
            event = {"wright_fisher": self._wf, "neutral_walk": self._neutral, "sswm": self._sswm}[
                self.algorithm
            ](lane)
            if event is None:
                break
            if event is not False:
                self._consume("evolution_updates")
                events.append(event)
            self.round_robin = (lane + 1) % len(self.replicates)
        quota_hit = self._reason == "max_evolution_updates"
        score_failure = self._reason == "incomplete_evaluation"
        events.extend(self._take_evaluation_events())
        if not self._boundary(events, force=quota_hit or score_failure):
            return self._report_state(transition=events)
        # Advanced-but-unobserved state is already durable before rescoring.
        self.observe()
        if quota_hit and self._status != "complete":
            self._status, self._reason = "partial", "max_evolution_updates"
            self._boundary(force=True)
        elif score_failure and self._status != "complete":
            self._status, self._reason = "partial", "incomplete_evaluation"
        return self._report_state(transition=events)

    def run(self):
        while True:
            report = self.step()
            if report.status in {"complete", "partial", "failed"}:
                return report

    def _report_state(self, transition=None):
        return self._report(
            dict(
                replicates=self.replicates,
                generations=self.cursors,
                events=self.cursors,
                times=self.times,
                observed=self.observed,
                lane_status=self.lane_status,
            ),
            dict(counters=self.counters, evaluator=getattr(self.evaluator, "metrics", {})),
            transition,
        )

    @property
    def state(self):
        return owned(
            dict(
                status=self._status,
                reason=self._reason,
                replicates=self.replicates,
                generations=self.cursors,
                times=self.times,
                observed=self.observed,
                score_valid=self.observed == self.cursors,
                lane_status=self.lane_status,
            )
        )

    def snapshot(self, fields=None, include_pending=False):
        history = np.asarray(list(self.history.values()))
        completed = int(np.count_nonzero(self.lane_status == "complete"))
        started = int(np.count_nonzero(self.observed >= 0))
        summary = dict(
            arrivals=list(self.arrivals),
            completed_trials=completed,
            started_trials=started,
            unstarted_trials=len(self.replicates) - started,
            rng=SCHEME,
            algorithm=self.algorithm,
            trial_status=self.lane_status.tolist(),
            observed_horizons=(
                [
                    self.spec.evolution.time_horizon
                    if self.lane_status[i] == "complete"
                    else float(self.times[i])
                    for i in range(len(self.replicates))
                ]
                if self.algorithm == "sswm"
                else self.observed.tolist()
            ),
            pending_events=owned(self.pending),
            terminal_times=self.times.tolist(),
            history_dropped=self.observation_count - len(self.history),
            arrival_probability=sum(x is not None for x in self.arrivals) / len(self.arrivals)
            if self._status == "complete"
            and (
                all(x is not None and x.success is not None for x in self.evaluations)
                or (self._conversion is not None and self.algorithm != "neutral_walk")
            )
            else None,
            mean_observed_arrival=np.mean([x for x in self.arrivals if x is not None]).item()
            if any(x is not None for x in self.arrivals)
            else None,
        )
        arrays = dict(
            populations=self.populations, founders=self.populations[:, 0], history=history
        )
        if fields is not None:
            names = [fields] if isinstance(fields, str) else list(fields)
            unknown = set(names) - arrays.keys()
            if unknown:
                raise ValueError(f"unknown population snapshot fields: {sorted(unknown)}")
            arrays = {name: arrays[name] for name in names}
        result = dict(status=self._status, state=self.state, arrays=arrays, summary=summary)
        if include_pending:
            result["pending"] = dict(
                transitions=self.pending, evaluations=self.evaluator.checkpoint_payload()
            )
        return owned(result)
