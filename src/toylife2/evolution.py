"""Exact host selection with batched Quadrants pattern fitness evaluation."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from .genotypes import mutation, neighbours, validate
from .random import SCHEME, Stream
from .specs import RunSpec


class Fitness:
    """Cache by ordered genotype; every distinct phenotype is evaluated in batches."""

    def __init__(self, model, spec: RunSpec, target: np.ndarray):
        self.model, self.spec = model, spec
        self.target = np.asarray(target)
        if self.target.shape != (spec.spatial.steps, spec.spatial.cells):
            raise ValueError("target must have shape [spatial.steps, spatial.cells]")
        if self.target.dtype.kind not in "iu" or np.any(self.target < 0) or np.any(self.target > 3):
            raise ValueError("target must contain integer tissue states 0 through 3")
        floor = Fraction(str(spec.evolution.fitness_floor))
        self.length = self.target.size
        self.denominator = math.lcm(self.length, floor.denominator)
        self.floor_weight = floor.numerator * (self.denominator // floor.denominator)
        self.cache: OrderedDict[tuple[int, ...], int] = OrderedDict()
        self.cache_capacity = max(
            1, spec.execution.memory_budget_mib * 1024 * 1024 // (8 * (160 + 40 * spec.model.genes))
        )

    def weights(self, genotypes: np.ndarray) -> np.ndarray:
        genes = validate(genotypes)
        flat = genes.reshape(-1, genes.shape[-1])
        keys = [tuple(map(int, row)) for row in flat]
        local = {key: self.cache[key] for key in keys if key in self.cache}
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            for start in range(0, len(missing), self.spec.execution.batch_size):
                batch = missing[start : start + self.spec.execution.batch_size]
                results = self.model.patterns(np.asarray(batch, np.uint32), self.spec.spatial)
                if not results.complete:
                    raise RuntimeError("fitness requires complete patterns")
                matches = np.count_nonzero(results.values == self.target, axis=(-2, -1))
                scale = self.denominator // self.length
                for key, match in zip(batch, matches, strict=True):
                    local[key] = max(int(match) * scale, self.floor_weight)
                    self.cache[key] = local[key]
                    self.cache.move_to_end(key)
                    while len(self.cache) > self.cache_capacity:
                        self.cache.popitem(last=False)
        maximum = max(self.denominator, self.floor_weight)
        if maximum >= 1 << 63:
            raise OverflowError("fitness rational denominator exceeds int64 admission")
        return np.asarray([local[key] for key in keys], np.int64).reshape(genes.shape[:-1])

    def values(self, genotypes: np.ndarray) -> np.ndarray:
        return self.weights(genotypes).astype(np.float64) / self.denominator


def log_acceptance(current: float, mutant: float, population: int) -> float:
    """Log N(r-1)/(r**N-1), r=current/mutant, with stable neutral limit."""
    if population < 1 or not math.isfinite(current) or not math.isfinite(mutant):
        raise ValueError("fitness must be finite and population positive")
    if current < 0 or mutant < 0:
        raise ValueError("fitness cannot be negative")
    if current == mutant:
        return 0.0
    if population == 1:
        return 0.0
    if mutant == 0:
        return -math.inf
    if current == 0:
        return math.log(population)
    delta = (current - mutant) / mutant
    log_r = (
        math.log1p(delta)
        if math.isfinite(delta) and -0.5 < delta < 0.5
        else math.log(current) - math.log(mutant)
    )
    power = population * log_r
    if log_r > 50:
        log_difference = log_r + math.log1p(-math.exp(-log_r))
    else:
        log_difference = math.log(abs(math.expm1(log_r)))
    log_numerator = math.log(population) + log_difference
    if power > 50:
        log_denominator = power + math.log1p(-math.exp(-power))
    else:
        log_denominator = math.log(abs(math.expm1(power)))
    return log_numerator - log_denominator


def _initial(spec: RunSpec, stream: Stream, trial: int, count: int) -> np.ndarray:
    from .genotypes import encode

    if spec.genotypes:
        genes = encode(spec.genotypes)
        if len(genes) not in (1, count):
            raise ValueError("initial genotypes must contain one founder or a full population")
        return np.broadcast_to(genes, (count, spec.model.genes)).copy()
    size = 1 if spec.evolution.initial == "monomorphic_random" else count
    return np.broadcast_to(
        stream.genotypes(size, spec.model.genes, trial=trial + spec.evolution.trial_start),
        (count, spec.model.genes),
    ).copy()


def _mutate(population: np.ndarray, positions: np.ndarray, selected: np.ndarray) -> None:
    individuals = np.flatnonzero(selected)
    bits = positions[individuals].astype(np.int64)
    population[individuals, bits // 20] ^= np.uint32(1) << (19 - bits % 20).astype(np.uint32)


def _history(state: dict, row: list, spec: RunSpec) -> None:
    # Bound Python object overhead as well as serialized trace storage.
    history_budget = min(
        spec.recording.trace_budget_mib * 1024 * 1024,
        spec.execution.memory_budget_mib * 1024 * 1024 // 8,
    )
    limit = max(1, history_budget // 512)
    state["history"].append(row)
    if len(state["history"]) > limit:
        del state["history"][0]
        state["history_dropped"] = state.get("history_dropped", 0) + 1


def _admit_population(spec: RunSpec, count: int) -> None:
    # Includes population copies, selection arrays, per-call keys, and outputs.
    width = (
        spec.evolution.gene_copy.max_genes
        if spec.evolution.gene_copy is not None
        else spec.model.genes
    )
    required = (
        spec.evolution.trials
        * count
        * (16 * width + (260 if spec.evolution.gene_copy is not None else 256))
    )
    budget = spec.execution.memory_budget_mib * 1024 * 1024
    if required > budget // 2:
        raise MemoryError(
            "trial/population host state exceeds half the memory budget; "
            "shard trials into independent runs with explicit trial IDs"
        )


def _trial_done(state: dict, arrays: dict, trial: int, spec: RunSpec, save) -> bool:
    """Store a matched terminal inventory before moving the algorithm cursor."""
    if spec.recording.mode == "full":
        state["trace_event"] = {
            "type": "trial_complete",
            "trial": trial + spec.evolution.trial_start,
            "generation": state.get("generation"),
            "event": state.get("event"),
            "time": state.get("time"),
            "inventory": {name: value[trial].tolist() for name, value in arrays.items()},
        }
    state["trial_status"].append("complete")
    state.update(trial=trial + 1, trial_active=False)
    if "generation" in state:
        state["generation"] = 0
    if "event" in state:
        state.update(event=0, time=0.0, pending=None)
    return save(state, arrays) is not False if save else True


def wright_fisher(
    model,
    spec: RunSpec,
    target: np.ndarray,
    *,
    checkpoint: dict | None = None,
    save: Callable | None = None,
) -> dict:
    """Generation-boundary checkpoints preserve individuals and logical RNG indices."""
    if spec.seed is None:
        raise ValueError("evolution requires a seed")
    stream = Stream(spec.seed)
    config = spec.evolution
    fitness = Fitness(model, spec, target)
    _admit_population(spec, config.population_size)
    state = (
        dict(checkpoint["state"])
        if checkpoint
        else {
            "algorithm": "wright_fisher",
            "rng": SCHEME,
            "trial": 0,
            "generation": 0,
            "arrivals": [None] * config.trials,
            "history": [],
            "trial_status": [],
        }
    )
    populations = (
        checkpoint["loaded_arrays"]["populations"].copy()
        if checkpoint
        else np.stack(
            [_initial(spec, stream, t, config.population_size) for t in range(config.trials)]
        )
    )
    if state.get("rng") != SCHEME or state.get("algorithm") != "wright_fisher":
        raise ValueError("checkpoint algorithm/RNG scheme mismatch")
    consumed = 0
    state.setdefault("observed_horizons", [None] * config.trials)
    for trial in range(state["trial"], config.trials):
        state["trial_active"] = True
        pop = populations[trial]
        start = state["generation"] if trial == state["trial"] else 0
        for generation in range(start, config.generations + 1):
            weights = fitness.weights(pop)
            state["observed_horizons"][trial] = generation
            if state["arrivals"][trial] is None and np.any(weights == fitness.denominator):
                state["arrivals"][trial] = generation
            if not state["history"] or state["history"][-1][:2] != [trial, generation]:
                _history(
                    state,
                    [
                        trial,
                        generation,
                        float(weights.mean() / fitness.denominator),
                        float(weights.max() / fitness.denominator),
                        len(np.unique(pop, axis=0)),
                    ],
                    spec,
                )
            state.update(trial=trial, generation=generation)
            if generation == config.generations:
                break
            if consumed >= spec.execution.max_steps:
                if save:
                    save(state, {"populations": populations})
                return _result(state, {"populations": populations}, "partial")
            maximum = int(weights.max())
            if maximum * len(weights) >= 1 << 63:
                raise OverflowError("population fitness sum exceeds int64 admission")
            prefix = np.cumsum(weights, dtype=np.int64)
            ids = np.arange(config.population_size, dtype=np.uint64)
            logical_trial = trial + config.trial_start
            if prefix[-1] == 0:
                parents = stream.integers(
                    len(pop), ids, role="parents", trial=logical_trial, event=generation
                )
            else:
                draws = stream.integers(
                    int(prefix[-1]), ids, role="parents", trial=logical_trial, event=generation
                )
                parents = np.searchsorted(prefix, draws, side="right")
            new = pop[parents.astype(np.intp)].copy()
            selected = (
                stream.uniform(ids, role="mutation_gate", trial=logical_trial, event=generation)
                < config.mutation_probability
            )
            positions = stream.integers(
                20 * spec.model.genes,
                ids,
                role="mutation_bit",
                trial=logical_trial,
                event=generation,
            )
            _mutate(new, positions, selected)
            if spec.recording.mode == "full":
                state["trace_event"] = {
                    "type": "generation",
                    "trial": logical_trial,
                    "from_generation": generation,
                    "to_generation": generation + 1,
                    "parents": parents.tolist(),
                    "mutations": [[int(i), int(positions[i])] for i in np.flatnonzero(selected)],
                    "before": pop.tolist(),
                    "after": new.tolist(),
                    "parent_fitness_weights": weights.tolist(),
                    "fitness_denominator": fitness.denominator,
                }
            pop[:] = new
            consumed += 1
            state["generation"] = generation + 1
            if save and (
                spec.recording.mode == "full"
                or (generation + 1) % spec.recording.checkpoint_interval == 0
            ):
                if save(state, {"populations": populations}) is False:
                    return _result(state, {"populations": populations}, "partial")
        if not _trial_done(state, {"populations": populations}, trial, spec, save):
            return _result(state, {"populations": populations}, "partial")
    return _result(state, {"populations": populations}, "complete")


def _result(state: dict, arrays: dict, status: str) -> dict:
    arrivals = state["arrivals"]
    observed = [value for value in arrivals if value is not None]
    completed = len(state["trial_status"])
    started = completed + int(state.get("trial_active", False) and completed < len(arrivals))
    trial_status = list(state["trial_status"]) + (["partial"] if started > completed else [])
    trial_status += ["unstarted"] * (len(arrivals) - started)
    return {
        "state": state,
        "arrays": {**arrays, "history": np.asarray(state["history"])},
        "status": status,
        "summary": {
            "arrivals": arrivals,
            "arrival_probability": len(observed) / len(arrivals) if status == "complete" else None,
            "completed_trials": completed,
            "started_trials": started,
            "unstarted_trials": len(arrivals) - started,
            "mean_observed_arrival": sum(observed) / len(observed) if observed else None,
            "censored_trials": sum(x is None for x in arrivals[:completed]),
            "rng": SCHEME,
            "history_dropped": state.get("history_dropped", 0),
            "recording_limit": state.get("recording_limit"),
            "terminal_times": state.get("terminal_times"),
            "observed_horizons": state.get("observed_horizons"),
            "pending_events": state.get("pending_events"),
            "algorithm": state["algorithm"],
            "trial_status": trial_status,
        },
    }


def sswm(
    model,
    spec: RunSpec,
    target: np.ndarray,
    *,
    checkpoint: dict | None = None,
    save: Callable | None = None,
) -> dict:
    """Continuous-time weak-mutation chain in mutation-time units.

    One-bit rates divide fixation acceptance by the number of genome bits.
    A pending sampled event beyond the horizon is retained in the checkpoint.
    """
    if spec.seed is None:
        raise ValueError("evolution requires a seed")
    stream = Stream(spec.seed)
    config = spec.evolution
    fitness = Fitness(model, spec, target)
    _admit_population(spec, 1)
    state = (
        dict(checkpoint["state"])
        if checkpoint
        else {
            "algorithm": "sswm",
            "rng": SCHEME,
            "trial": 0,
            "event": 0,
            "time": 0.0,
            "pending": None,
            "pending_events": [None] * config.trials,
            "terminal_times": [None] * config.trials,
            "arrivals": [None] * config.trials,
            "history": [],
            "trial_status": [],
        }
    )
    founders = (
        checkpoint["loaded_arrays"]["founders"].copy()
        if checkpoint
        else np.stack([_initial(spec, stream, t, 1)[0] for t in range(config.trials)])
    )
    if state.get("rng") != SCHEME or state.get("algorithm") != "sswm":
        raise ValueError("checkpoint algorithm/RNG scheme mismatch")
    consumed = 0
    state.setdefault("observed_horizons", [None] * config.trials)
    for trial in range(state["trial"], config.trials):
        state["trial_active"] = True
        genotype = founders[trial]
        while True:
            value = float(fitness.values(genotype))
            state["observed_horizons"][trial] = state["time"]
            if state["arrivals"][trial] is None and value == 1:
                state["arrivals"][trial] = state["time"]
            entry = [trial, state["time"], value, state["event"]]
            if not state["history"] or state["history"][-1] != entry:
                _history(state, entry, spec)
            if state["time"] >= config.time_horizon:
                break
            if consumed >= spec.execution.max_events:
                if save:
                    save(state, {"founders": founders})
                return _result(state, {"founders": founders}, "partial")
            if state["pending"] is None:
                mutants = neighbours(genotype)
                values = fitness.values(mutants)
                logs = np.asarray(
                    [
                        log_acceptance(value, float(v), config.population_size)
                        - math.log(len(mutants))
                        for v in values
                    ]
                )
                maximum = float(logs.max())
                if maximum == -math.inf:
                    state["pending"] = {"time": None, "position": None}
                else:
                    rates = np.exp(logs - maximum)
                    log_total = maximum + math.log(float(rates.sum()))
                    u = float(
                        stream.uniform(
                            0,
                            role="waiting_time",
                            trial=trial + config.trial_start,
                            event=state["event"],
                        )
                    )
                    log_wait = math.log(-math.log(u)) - log_total
                    wait = math.exp(log_wait) if log_wait < 709 else math.inf
                    choice = float(
                        stream.uniform(
                            0,
                            role="transition",
                            trial=trial + config.trial_start,
                            event=state["event"],
                        )
                    ) * float(rates.sum())
                    position = min(
                        int(np.searchsorted(np.cumsum(rates), choice, side="right")),
                        len(mutants) - 1,
                    )
                    event_time = state["time"] + wait
                    state["pending"] = {
                        "time": event_time if math.isfinite(event_time) else None,
                        "position": position,
                        "log_wait": log_wait,
                    }
            pending = state["pending"]
            event_time = pending["time"]
            if event_time is None or event_time > config.time_horizon:
                break
            position = pending["position"]
            if position is None:
                raise ValueError("finite pending event has no mutation position")
            candidate = mutation(genotype, position)
            if spec.recording.mode == "full":
                state["trace_event"] = {
                    "type": "transition",
                    "trial": trial + config.trial_start,
                    "event": state["event"],
                    "from_time": state["time"],
                    "to_time": event_time,
                    "position": position,
                    "before": genotype.tolist(),
                    "after": candidate.tolist(),
                    "pending": dict(pending),
                    "previous_fitness": value,
                }
            genotype[:] = candidate
            state.update(time=event_time, event=state["event"] + 1, pending=None)
            consumed += 1
            if save and (
                spec.recording.mode == "full"
                or state["event"] % spec.recording.checkpoint_interval == 0
            ):
                if save(state, {"founders": founders}) is False:
                    return _result(state, {"founders": founders}, "partial")
        state["pending_events"][trial] = state["pending"]
        state["terminal_times"][trial] = state["time"]
        state["observed_horizons"][trial] = config.time_horizon
        if not _trial_done(state, {"founders": founders}, trial, spec, save):
            return _result(state, {"founders": founders}, "partial")
    return _result(state, {"founders": founders}, "complete")


def neutral_walk(
    model, spec: RunSpec, *, checkpoint: dict | None = None, save: Callable | None = None
) -> dict:
    """Uniform proposed one-bit mutations, accepted iff the declared phenotype matches."""
    if spec.seed is None:
        raise ValueError("evolution requires a seed")
    stream = Stream(spec.seed)
    config = spec.evolution
    _admit_population(spec, 1)
    state = (
        dict(checkpoint["state"])
        if checkpoint
        else {
            "algorithm": "neutral_walk",
            "rng": SCHEME,
            "trial": 0,
            "generation": 0,
            "arrivals": [None] * config.trials,
            "history": [],
            "trial_status": [],
        }
    )
    founders = (
        checkpoint["loaded_arrays"]["founders"].copy()
        if checkpoint
        else np.stack([_initial(spec, stream, t, 1)[0] for t in range(config.trials)])
    )
    if state.get("rng") != SCHEME or state.get("algorithm") != "neutral_walk":
        raise ValueError("checkpoint algorithm/RNG scheme mismatch")

    def phenotype(genes):
        if spec.assay.phenotype == "regulatory":
            return model.regulatory(genes)
        return model.metabolic(genes, spec.assay.environment)

    consumed = 0
    state.setdefault("observed_horizons", [None] * config.trials)
    for trial in range(state["trial"], config.trials):
        state["trial_active"] = True
        genotype = founders[trial]
        base = phenotype(genotype)
        if not base.complete:
            raise RuntimeError("neutrality requires a complete phenotype")
        start = state["generation"] if trial == state["trial"] else 0
        for event in range(start, config.generations):
            if consumed >= spec.execution.max_steps:
                if save:
                    save(state, {"founders": founders})
                return _result(state, {"founders": founders}, "partial")
            position = int(
                stream.integers(
                    20 * spec.model.genes,
                    0,
                    role="neutral_proposal",
                    trial=trial + config.trial_start,
                    event=event,
                )
            )
            candidate = mutation(genotype, position)
            result = phenotype(candidate)
            if not result.complete:
                raise RuntimeError("neutrality requires a complete candidate phenotype")
            accepted = np.array_equal(base.values, result.values)
            if spec.recording.mode == "full":
                state["trace_event"] = {
                    "type": "neutral_proposal",
                    "trial": trial + config.trial_start,
                    "event": event,
                    "position": position,
                    "accepted": accepted,
                    "before": genotype.tolist(),
                    "after": candidate.tolist() if accepted else genotype.tolist(),
                    "base_phenotype": base.values.tolist(),
                    "candidate_phenotype": result.values.tolist(),
                }
            if accepted:
                genotype[:] = candidate
            _history(state, [trial, event, position, int(accepted)], spec)
            state.update(trial=trial, generation=event + 1)
            consumed += 1
            state["observed_horizons"][trial] = event + 1
            if save and (
                spec.recording.mode == "full"
                or (event + 1) % spec.recording.checkpoint_interval == 0
            ):
                if save(state, {"founders": founders}) is False:
                    return _result(state, {"founders": founders}, "partial")
        if not _trial_done(state, {"founders": founders}, trial, spec, save):
            return _result(state, {"founders": founders}, "partial")
    return _result(state, {"founders": founders}, "complete")
