"""Scientific specifications and physical execution settings.

Changes to execution placement never change a scientific identity. Canonical rule
interpretations are described in docs/decisions.md and identified by RULESET.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .protocols import (
    ControllerSpec,
    EpisodeSpec,
    ObjectiveSpec,
    PortSpec,
    TaskSpec,
    WorkLimits,
)

RULESET = "toylife-canonical-v1"
OPERATIONS = frozenset(
    {
        "regulatory",
        "metabolic",
        "neighbourhood",
        "patterns",
        "weighted_map",
        "wright_fisher",
        "sswm",
        "cell",
        "neutral_walk",
        "rollout",
        "chemistry",
        "tissue",
        "external",
    }
)


def positive(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class ModelSpec:
    rules: str = RULESET
    genes: int = 2

    def __post_init__(self) -> None:
        if self.rules != RULESET:
            raise ValueError(f"unsupported ruleset: {self.rules}")
        positive(self.genes, "genes")


@dataclass(frozen=True)
class AssaySpec:
    environment: tuple[str, ...] | None = None
    phenotype: str = "metabolic"

    def __post_init__(self) -> None:
        if self.phenotype not in ("metabolic", "regulatory"):
            raise ValueError("phenotype must be metabolic or regulatory")
        if self.environment is not None:
            for molecule in self.environment:
                if not isinstance(molecule, str) or not 4 <= len(molecule) <= 8:
                    raise ValueError("assay metabolites must have length 4 through 8")
                if set(molecule) - {"0", "1"}:
                    raise ValueError("metabolites use binary 0=P, 1=H sequences")


@dataclass(frozen=True)
class SpatialSpec:
    cells: int = 31
    steps: int = 100
    boundary: str = "open"
    diffusion: str = "A"
    morphogen: str = "central_A_pulse"
    target: str | None = None
    initial_state: tuple[int, ...] | None = None
    injections: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        positive(self.cells, "cells")
        positive(self.steps, "steps")
        if self.boundary not in ("open", "ring"):
            raise ValueError("boundary must be open or ring")
        if self.diffusion not in ("none", "A", "B", "both", "all"):
            raise ValueError("diffusion must be none, A, B, both or all")
        if self.morphogen not in (
            "none",
            "explicit",
            "central_A_pulse",
            "central_B_pulse",
            "central_A_continuous",
            "central_B_continuous",
        ):
            raise ValueError("unsupported morphogen schedule")
        if self.initial_state is not None:
            if len(self.initial_state) != self.cells or any(
                type(x) is not int or not 0 <= x <= 3 for x in self.initial_state
            ):
                raise ValueError("initial_state must contain cells integers in [0,3]")
        if self.injections and self.morphogen != "explicit":
            raise ValueError("injections require morphogen='explicit'")
        for injection in self.injections:
            if len(injection) != 3 or any(type(x) is not int for x in injection):
                raise ValueError("injections are [tick,cell,state] integer triples")
            tick, cell, state = injection
            if not 0 <= tick < self.steps or not 0 <= cell < self.cells or not 1 <= state <= 3:
                raise ValueError("injection outside trajectory or state not in [1,3]")


@dataclass(frozen=True)
class GeneCopySpec:
    duplication_probability: float = 0.0
    deletion_probability: float = 0.0
    min_genes: int = 1
    max_genes: int = 20

    def __post_init__(self) -> None:
        for name in ("duplication_probability", "deletion_probability"):
            finite(getattr(self, name), name)
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        positive(self.min_genes, "min_genes")
        positive(self.max_genes, "max_genes")
        if not self.min_genes <= self.max_genes <= 20:
            raise ValueError("gene-copy bounds must satisfy 1 <= min_genes <= max_genes <= 20")


@dataclass(frozen=True)
class EvolutionSpec:
    trials: int = 1
    population_size: int = 10000
    generations: int = 1000
    mutation_probability: float = 0.01
    mutation: str = "one_bit_per_mutating_offspring"
    fitness_floor: float = 1e-8
    time_horizon: float = 100.0
    initial: str = "monomorphic_random"
    trial_start: int = 0
    replicates: int | None = None
    replicate_start: int | None = None
    gene_copy: GeneCopySpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.gene_copy, Mapping):
            object.__setattr__(self, "gene_copy", GeneCopySpec(**self.gene_copy))
        if self.gene_copy is not None and not isinstance(self.gene_copy, GeneCopySpec):
            raise ValueError("gene_copy must be a GeneCopySpec or mapping")
        finite(self.mutation_probability, "mutation_probability")
        if self.gene_copy is not None and (
            self.mutation_probability
            + self.gene_copy.duplication_probability
            + self.gene_copy.deletion_probability
            > 1
        ):
            raise ValueError(
                "point, duplication and deletion probabilities must sum to at most one"
            )
        if self.replicates is not None:
            positive(self.replicates, "replicates")
            if self.trials != 1 and self.trials != self.replicates:
                raise ValueError("trials and replicates disagree")
            object.__setattr__(self, "trials", self.replicates)
        if self.replicate_start is not None:
            positive(self.replicate_start, "replicate_start", 0)
            if self.trial_start != 0 and self.trial_start != self.replicate_start:
                raise ValueError("trial_start and replicate_start disagree")
            object.__setattr__(self, "trial_start", self.replicate_start)
        for name in ("trials", "population_size"):
            positive(getattr(self, name), name)
        positive(self.generations, "generations", 0)
        positive(self.trial_start, "trial_start", 0)
        finite(self.mutation_probability, "mutation_probability")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be finite and in [0, 1]")
        if self.mutation != "one_bit_per_mutating_offspring":
            raise ValueError("unsupported mutation convention")
        finite(self.fitness_floor, "fitness_floor")
        if not 0 <= self.fitness_floor < 1:
            raise ValueError("fitness_floor must be finite and in [0, 1)")
        finite(self.time_horizon, "time_horizon")
        if self.time_horizon < 0:
            raise ValueError("time_horizon must be finite and nonnegative")
        if self.initial not in ("monomorphic_random", "independent_random"):
            raise ValueError("unsupported initial population")


def _preferred_backend(supported, system):
    """Choose an available native GPU, then the native CPU backend."""
    candidates = ("metal", "cuda") if system == "Darwin" else ("cuda", "metal")
    return next((name for name in candidates if supported(name)), "cpu")


def _default_backend():
    import quadrants as qd
    from quadrants.lang.misc import is_arch_supported

    return _preferred_backend(lambda name: is_arch_supported(getattr(qd, name)), platform.system())


@dataclass(frozen=True)
class ExecutionSpec:
    backend: str = "auto"
    batch_size: int = 256
    memory_budget_mib: int = 1024
    max_steps: int = 100000
    max_events: int = 100000
    max_truth_states: int = 1048576
    max_tiles: int = 1048576
    cache_dir: str = ".cache/toylife2"
    max_active_lanes: int = 64
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "cpu", "metal", "cuda"):
            raise ValueError("backend must be auto, cpu, metal or cuda")
        if self.backend == "auto":
            object.__setattr__(self, "backend", _default_backend())
        for name in (
            "batch_size",
            "memory_budget_mib",
            "max_steps",
            "max_events",
            "max_truth_states",
            "max_tiles",
            "max_active_lanes",
        ):
            positive(getattr(self, name), name)
        if self.cpu_threads is not None:
            positive(self.cpu_threads, "cpu_threads")


@dataclass(frozen=True)
class RecordingSpec:
    mode: str = "replayable"
    checkpoint_interval: int = 100
    trace_budget_mib: int = 128

    def __post_init__(self) -> None:
        if self.mode not in ("replayable", "full"):
            raise ValueError("recording mode must be replayable or full")
        positive(self.checkpoint_interval, "checkpoint_interval")
        positive(self.trace_budget_mib, "trace_budget_mib")


@dataclass(frozen=True)
class CellSpec:
    steps: int = 1
    initial_expression: int = 0
    metabolites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        positive(self.steps, "cell.steps")
        positive(self.initial_expression, "cell.initial_expression", 0)
        for sequence in self.metabolites:
            if not isinstance(sequence, str) or not sequence or set(sequence) - {"0", "1"}:
                raise ValueError("cell metabolites must be nonempty binary strings")


@dataclass(frozen=True)
class MapSpec:
    start_pair: int = 0
    stop_pair: int | None = None
    tile_pairs: int = 64

    def __post_init__(self) -> None:
        positive(self.start_pair, "map.start_pair", 0)
        positive(self.tile_pairs, "map.tile_pairs")
        if self.stop_pair is not None:
            positive(self.stop_pair, "map.stop_pair", 0)
            if self.stop_pair < self.start_pair:
                raise ValueError("map.stop_pair precedes start_pair")


@dataclass(frozen=True)
class RunSpec:
    operation: str = "regulatory"
    seed: int | None = None
    samples: int = 1
    genotypes: tuple[str, ...] = ()
    schema_version: int = 1
    model: ModelSpec = field(default_factory=ModelSpec)
    assay: AssaySpec = field(default_factory=AssaySpec)
    spatial: SpatialSpec = field(default_factory=SpatialSpec)
    evolution: EvolutionSpec = field(default_factory=EvolutionSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    recording: RecordingSpec = field(default_factory=RecordingSpec)
    cell: CellSpec = field(default_factory=CellSpec)
    map: MapSpec = field(default_factory=MapSpec)
    controller: ControllerSpec = field(default_factory=ControllerSpec)
    task: TaskSpec = field(default_factory=TaskSpec)
    episode: EpisodeSpec = field(default_factory=EpisodeSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    limits: WorkLimits = field(default_factory=WorkLimits)

    def __post_init__(self) -> None:
        if self.schema_version not in (1, 2):
            raise ValueError("unsupported schema_version")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported operation: {self.operation}")
        positive(self.samples, "samples")
        if self.schema_version == 1:
            if self.operation in {"rollout", "chemistry", "tissue", "external"}:
                raise ValueError("temporal operations require schema_version=2")
            positive(self.evolution.generations, "generations")
            defaults = {
                "controller": ControllerSpec(),
                "task": TaskSpec(),
                "episode": EpisodeSpec(),
                "objective": ObjectiveSpec(),
                "limits": WorkLimits(),
            }
            if any(getattr(self, key) != value for key, value in defaults.items()):
                raise ValueError("temporal protocols require schema_version=2")
            if self.evolution.replicates is not None or self.evolution.replicate_start is not None:
                raise ValueError("replicate aliases require schema_version=2")
            if self.execution.max_active_lanes != 64 or self.execution.cpu_threads is not None:
                raise ValueError("batch execution fields require schema_version=2")
        if self.cell.initial_expression >= 1 << self.model.genes:
            raise ValueError("cell.initial_expression exceeds gene count")
        if self.task.config.get("experiment_contract") is not None:
            from .single_cell import require_current

            require_current(self)
        copy = self.evolution.gene_copy
        if copy is not None:
            if not copy.min_genes <= self.model.genes <= copy.max_genes:
                raise ValueError("founder gene count must lie within gene-copy bounds")
            if (
                self.schema_version != 2
                or self.operation != "wright_fisher"
                or not (
                    self.task.kind in {"cue_reading", "chemotaxis"}
                    or (
                        self.task.kind == "signalling_game"
                        and self.task.config.get("pairing") == "clonal"
                    )
                )
                or self.controller.mode != "canonical_cell"
                or self.controller.interface is None
                or self.objective.scope != "candidate"
            ):
                raise ValueError(
                    "gene-copy evolution requires an interfaced molecular Wright–Fisher task"
                )
            if self.cell.initial_expression or self.controller.initial_expression:
                raise ValueError("gene-copy evolution requires zero initial expression masks")
            for readout in self.controller.ports.readouts:
                if readout.kind in {"expression", "functional_expression"} or (
                    readout.kind == "protein" and readout.protein is None
                ):
                    raise ValueError(
                        "gene-copy evolution requires gene-position-independent readouts"
                    )
        if self.seed is not None:
            positive(self.seed, "seed", 0)
            if self.seed >= 1 << 64:
                raise ValueError("seed must fit uint64")
        if self.seed is None and not self.genotypes and self.operation != "weighted_map":
            raise ValueError("random workflows require an explicit seed")
        if self.operation in ("wright_fisher", "sswm", "neutral_walk") and self.seed is None:
            raise ValueError("evolution requires an explicit seed")
        if (
            self.operation in ("patterns", "weighted_map", "tissue")
            or (self.operation in ("wright_fisher", "sswm") and self.schema_version == 1)
            or self.objective.kind == "pattern_match"
        ):
            if self.model.genes != 2:
                raise ValueError("the published tissue/evolution protocol requires two genes")
        for genotype in self.genotypes:
            if len(genotype) != 20 * self.model.genes or set(genotype) - {"0", "1"}:
                raise ValueError("genotype length must be 20*genes and use binary symbols")


def plain(spec: Any) -> Any:
    if dataclasses.is_dataclass(spec):
        data = {f.name: plain(getattr(spec, f.name)) for f in dataclasses.fields(spec)}
        if isinstance(spec, EvolutionSpec) and spec.gene_copy is None:
            data.pop("gene_copy")
        if isinstance(spec, ControllerSpec):
            if spec.output is None:
                data.pop("output")
            if spec.interface is None:
                data.pop("interface")
            else:
                data.pop("ports")
        # Keep the exact v1 serialization and scientific hash projection.
        if isinstance(spec, RunSpec) and spec.schema_version == 1:
            for name in ("controller", "task", "episode", "objective", "limits"):
                data.pop(name)
            for name in ("replicates", "replicate_start"):
                data["evolution"].pop(name)
            for name in ("max_active_lanes", "cpu_threads"):
                data["execution"].pop(name)
        return data
    if isinstance(spec, (tuple, list)):
        return [plain(x) for x in spec]
    if isinstance(spec, Mapping):
        return {str(k): plain(v) for k, v in spec.items()}
    return spec


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scientific_identity(spec: RunSpec) -> str:
    data = plain(spec)
    data.pop("execution")
    data.pop("recording")
    data.pop("limits", None)
    if "controller" in data:
        data["controller"].pop("capacity")
        for name in ("task", "objective"):
            inputs = data[name]["config"].get("inputs")
            if inputs is not None:
                data[name]["config"]["inputs"] = {key: {"input": f"{name}_{key}"} for key in inputs}
    data["map"].pop("tile_pairs")
    if data["spatial"]["target"] is not None:
        data["spatial"]["target"] = {"input": "target", "format": "npy-tissue-states-v1"}
    return digest(data)


def _construct(cls: type, data: dict) -> Any:
    allowed = {f.name for f in dataclasses.fields(cls)}
    extras = set(data) - allowed
    if extras:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(extras)}")
    values = dict(data)
    for name in ("environment", "genotypes", "metabolites", "initial_state"):
        if name in values and values[name] is not None:
            values[name] = tuple(values[name])
    if "injections" in values:
        values["injections"] = tuple(tuple(x) for x in values["injections"])
    if cls is ControllerSpec and "ports" in values:
        values["ports"] = _construct(PortSpec, values["ports"])
    return cls(**values)


def from_mapping(data: dict) -> RunSpec:
    values = dict(data)
    for name, cls in (
        ("model", ModelSpec),
        ("assay", AssaySpec),
        ("spatial", SpatialSpec),
        ("evolution", EvolutionSpec),
        ("execution", ExecutionSpec),
        ("recording", RecordingSpec),
        ("cell", CellSpec),
        ("map", MapSpec),
    ):
        if name in values:
            if not isinstance(values[name], dict):
                raise ValueError(f"{name} must be a table")
            values[name] = _construct(cls, values[name])
    for name, cls in (
        ("controller", ControllerSpec),
        ("task", TaskSpec),
        ("episode", EpisodeSpec),
        ("objective", ObjectiveSpec),
        ("limits", WorkLimits),
    ):
        if name in values:
            if not isinstance(values[name], dict):
                raise ValueError(f"{name} must be a table")
            values[name] = _construct(cls, values[name])
    return _construct(RunSpec, values)
