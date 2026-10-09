"""Versioned temporal, task and resource protocols, independent of placement."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    """Own configuration values and make nested collections immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("configuration requires finite JSON-compatible values")


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class WorkLimits:
    max_evolution_updates: int | None = None
    max_world_ticks: int | None = None
    max_chemical_updates: int | None = None
    max_assay_steps: int = 100_000

    def __post_init__(self) -> None:
        for name in ("max_evolution_updates", "max_world_ticks", "max_chemical_updates"):
            value = getattr(self, name)
            if value is not None:
                integer(value, name)
        integer(self.max_assay_steps, "max_assay_steps", 1)


@dataclass(frozen=True)
class ReadoutSpec:
    name: str = "motor"
    kind: str = "functional_expression"
    gene: int = 0
    protein: int | None = None
    dimer: tuple[int, int] | None = None
    metabolite: str | None = None
    stage: str = "boundary"

    def __post_init__(self) -> None:
        if not self.name or self.kind not in {
            "expression",
            "functional_expression",
            "protein",
            "dimer",
            "bound",
            "fission",
        }:
            raise ValueError("invalid readout name or kind")
        integer(self.gene, "readout.gene")
        if self.stage not in {"after_input", "after_output", "boundary"}:
            raise ValueError("readout stage must be after_input, after_output or boundary")
        if self.kind == "bound" and self.stage == "boundary":
            raise ValueError("bound readouts precede canonical boundary release")
        if self.dimer is not None:
            object.__setattr__(self, "dimer", tuple(self.dimer))


@dataclass(frozen=True)
class PortSpec:
    metabolites: tuple[str, ...] = ("00000001",)
    proteins: tuple[int, ...] = ()
    dimers: tuple[tuple[int, int], ...] = ()
    max_dose: int = 8
    input_mode: str = "pulse"
    readouts: tuple[ReadoutSpec, ...] = field(default_factory=lambda: (ReadoutSpec(),))

    def __post_init__(self) -> None:
        integer(self.max_dose, "ports.max_dose")
        if self.input_mode not in {"pulse", "replenish"}:
            raise ValueError("input_mode must be pulse or replenish")
        for met in self.metabolites:
            if not isinstance(met, str) or not 1 <= len(met) <= 8 or set(met) - {"0", "1"}:
                raise ValueError("port metabolites must be binary strings of length 1..8")
        object.__setattr__(self, "metabolites", tuple(self.metabolites))
        object.__setattr__(self, "proteins", tuple(self.proteins))
        object.__setattr__(self, "dimers", tuple(tuple(x) for x in self.dimers))
        object.__setattr__(
            self,
            "readouts",
            tuple(ReadoutSpec(**x) if isinstance(x, Mapping) else x for x in self.readouts),
        )
        names = [x.name for x in self.readouts]
        if len(set(names)) != len(names):
            raise ValueError("readout names must be unique")


@dataclass(frozen=True)
class ControllerSpec:
    mode: str = "canonical_cell"
    ports: PortSpec = field(default_factory=PortSpec)
    initial_expression: int = 0
    initial_proteins: tuple[tuple[int, int], ...] = ()
    initial_dimers: tuple[tuple[int, int, int], ...] = ()
    initial_metabolites: tuple[tuple[str, int], ...] = ()
    initial_bound: tuple[tuple[str, int, int, str, int], ...] = ()
    warmup: int = 0
    capacity: int = 4096

    def __post_init__(self) -> None:
        if self.mode not in {"canonical_cell", "regulatory"}:
            raise ValueError("controller mode must be canonical_cell or regulatory")
        integer(self.initial_expression, "initial_expression")
        integer(self.warmup, "warmup")
        integer(self.capacity, "controller.capacity", 1)
        for name in ("initial_proteins", "initial_dimers", "initial_metabolites", "initial_bound"):
            object.__setattr__(self, name, tuple(tuple(x) for x in getattr(self, name)))


@dataclass(frozen=True)
class EpisodeSpec:
    horizon: int | None = 1000
    episodes: int = 1
    dt: float = 0.05
    chemical_updates: int = 1
    panel: str = "auto"
    reduction: str = "mean"
    seed: int = 0
    reset_per_decision: bool = False

    def __post_init__(self) -> None:
        if self.horizon is not None:
            integer(self.horizon, "episode.horizon")
        integer(self.episodes, "episode.episodes", 1)
        integer(self.chemical_updates, "episode.chemical_updates", 1)
        integer(self.seed, "episode.seed")
        if self.seed >= 1 << 64:
            raise ValueError("episode.seed must fit uint64")
        if not isinstance(self.dt, (int, float)) or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("episode.dt must be finite and positive")
        if self.panel not in {"auto", "fixed", "generation"} or self.reduction != "mean":
            raise ValueError("panel must be auto, fixed or generation; reduction must be mean")


@dataclass(frozen=True)
class TaskSpec:
    kind: str = "assay"
    entrypoint: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {
            "assay",
            "temporal_probe",
            "signalling_game",
            "run_tumble",
            "tissue",
            "python",
        }:
            raise ValueError("unsupported task kind")
        if self.kind == "python" and not self.entrypoint:
            raise ValueError("python tasks require an importable entrypoint")
        object.__setattr__(self, "config", freeze(self.config))


@dataclass(frozen=True)
class ObjectiveSpec:
    kind: str = "auto"
    direction: str = "maximize"
    floor_numerator: int = 0
    floor_denominator: int = 1
    quantization: int = 1 << 24
    goal: float | None = None
    goal_operator: str = "ge"
    entrypoint: str | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    scope: str = "candidate"
    pure: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {
            "auto",
            "pattern_match",
            "metabolic_coverage",
            "bounded_reward",
            "python",
            "descriptor",
        }:
            raise ValueError("unsupported objective kind")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be maximize or minimize")
        integer(self.floor_numerator, "floor_numerator")
        integer(self.floor_denominator, "floor_denominator", 1)
        integer(self.quantization, "quantization", 1)
        if self.floor_numerator >= self.floor_denominator:
            raise ValueError("fitness floor must be in [0,1)")
        if self.goal is not None and not math.isfinite(self.goal):
            raise ValueError("goal must be finite")
        if self.goal_operator not in {"ge", "le", "eq"}:
            raise ValueError("goal_operator must be ge, le or eq")
        if self.scope not in {"candidate", "population"}:
            raise ValueError("scope must be candidate or population")
        if self.kind == "python" and not self.entrypoint:
            raise ValueError("python objectives require an importable entrypoint")
        object.__setattr__(self, "config", freeze(self.config))
