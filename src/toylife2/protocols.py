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
            "metabolite",
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
class StructuralInterface:
    """Named molecular supplies and intact target occupancy in one shared cell."""

    inputs: Mapping[str, str] = field(default_factory=dict)
    targets: Mapping[str, str] = field(default_factory=dict)
    target_copies: Mapping[str, int] = field(default_factory=dict)
    max_dose: int = 8

    def __post_init__(self) -> None:
        integer(self.max_dose, "interface.max_dose")
        identities = []
        names = []
        for attribute in ("inputs", "targets"):
            channels = dict(getattr(self, attribute))
            for name, molecule in channels.items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError("channel names must be Python identifiers")
                if (
                    not isinstance(molecule, str)
                    or not 1 <= len(molecule) <= 8
                    or set(molecule) - {"0", "1"}
                ):
                    raise ValueError("channel molecules must be binary strings of length 1..8")
                molecule = min(molecule, molecule[::-1])
                channels[name] = molecule
                identities.append(molecule)
                names.append(name)
            object.__setattr__(self, attribute, freeze(dict(sorted(channels.items()))))
        if len(set(identities)) != len(identities) or len(set(names)) != len(names):
            raise ValueError("channels require distinct names and canonical molecule identities")
        if not self.targets:
            raise ValueError("structural interface requires at least one target")
        if set(self.target_copies) - set(self.targets):
            raise ValueError("target_copies contains an unknown target")
        copies = {name: self.target_copies.get(name, 1) for name in self.targets}
        for count in copies.values():
            integer(count, "target copies")
            if count > self.max_dose:
                raise ValueError("target copies exceed max_dose")
        object.__setattr__(self, "target_copies", freeze(copies))

    def ports(self) -> PortSpec:
        readouts = []
        for name, molecule in self.targets.items():
            readouts.extend(
                (
                    ReadoutSpec(name, "bound", metabolite=molecule, stage="after_output"),
                    ReadoutSpec(
                        f"target_free:{name}",
                        "metabolite",
                        metabolite=molecule,
                        stage="after_output",
                    ),
                    ReadoutSpec(
                        f"target_fission:{name}",
                        "fission",
                        metabolite=molecule,
                        stage="after_output",
                    ),
                )
            )
        return PortSpec(
            metabolites=tuple((*self.inputs.values(), *self.targets.values())),
            max_dose=self.max_dose,
            input_mode="replenish",
            readouts=tuple(readouts),
        )

    def controller(self, **kwargs) -> ControllerSpec:
        return ControllerSpec(interface=self, **kwargs)

    def encode(self, inputs: Mapping) -> dict:
        import numpy as np

        if set(inputs) != set(self.inputs):
            raise ValueError("supply every named input, including explicit zeros")
        result = {}
        for name, molecule in self.inputs.items():
            counts = np.asarray(inputs[name])
            if (
                counts.dtype.kind not in "iu"
                or np.any(counts > self.max_dose)
                or np.any(counts < 0)
            ):
                raise ValueError("named inputs require integer counts within max_dose")
            result[molecule] = counts.copy()
        result.update(
            {molecule: self.target_copies[name] for name, molecule in self.targets.items()}
        )
        return result

    def outputs(self, readouts: Mapping) -> dict:
        return {name: readouts[name] for name in self.targets}

    def diagnostics(self, readouts: Mapping) -> dict:
        return {
            name: {
                "bound": readouts[name],
                "free": readouts[f"target_free:{name}"],
                "fission": readouts[f"target_fission:{name}"],
            }
            for name in self.targets
        }

    def endpoint_information(self, tables) -> dict:
        """Describe source annotations and exact contact-energy similarities."""
        import itertools

        import numpy as np

        channels = {**self.inputs, **self.targets}
        endpoints = {}
        similarities = []
        for name, molecule in channels.items():
            index = tables.met_index.get(molecule)
            endpoints[name] = {
                "molecule": molecule,
                "contact_table_present": index is not None,
                "source_unbreakable_annotation": (
                    bool(tables.met_unbreakable[index]) if index is not None else None
                ),
            }
        for first, second in itertools.combinations(channels, 2):
            a, b = (tables.met_index.get(channels[name]) for name in (first, second))
            similarities.append(
                {
                    "channels": [first, second],
                    "identical_protein_contacts": (
                        bool(
                            np.array_equal(
                                tables.protein_met_energy[a], tables.protein_met_energy[b]
                            )
                        )
                        if a is not None and b is not None
                        else None
                    ),
                    "identical_dimer_contacts": (
                        bool(np.array_equal(tables.dimer_met_energy[a], tables.dimer_met_energy[b]))
                        if a is not None and b is not None
                        else None
                    ),
                }
            )
        return {"endpoints": endpoints, "similarities": similarities}


SINGLE_CELL_CONTRACT = "single-cell-response-v1"


@dataclass(frozen=True)
class BinaryOutputSpec:
    """Assumed binary apparatus reading intact markers at after_output.

    bound_free compares (free, bound); bound_pair compares the occupied counts
    of the two ordered markers. permutation assigns those alternatives to acts.
    """

    representation: str = "bound_free"
    markers: tuple[str, ...] = ("act",)
    rule: str = "proportional"
    permutation: tuple[int, int] = (0, 1)

    def __post_init__(self):
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "permutation", tuple(self.permutation))
        size = {"bound_free": 1, "bound_pair": 2}.get(self.representation)
        if size is None or len(self.markers) != size or len(set(self.markers)) != size:
            raise ValueError("binary output requires one bound_free or two bound_pair markers")
        if any(not isinstance(name, str) or not name.isidentifier() for name in self.markers):
            raise ValueError("output markers require declared channel names")
        if self.rule not in {"proportional", "majority"}:
            raise ValueError("output rule must be proportional or majority")
        if any(type(x) is not int for x in self.permutation) or self.permutation not in (
            (0, 1),
            (1, 0),
        ):
            raise ValueError("output permutation must be (0,1) or (1,0)")

    @property
    def readouts(self):
        if self.representation == "bound_free":
            return (f"target_free:{self.markers[0]}", self.markers[0])
        return self.markers


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
    interface: StructuralInterface | None = None
    output: BinaryOutputSpec | None = None

    def __post_init__(self) -> None:
        if isinstance(self.output, Mapping):
            object.__setattr__(self, "output", BinaryOutputSpec(**self.output))
        if isinstance(self.interface, Mapping):
            object.__setattr__(self, "interface", StructuralInterface(**self.interface))
        if self.interface is not None:
            ports = self.interface.ports()
            if self.ports != PortSpec() and self.ports != ports:
                raise ValueError("structural interface supplies its own ports")
            if self.mode != "canonical_cell":
                raise ValueError("structural interface requires canonical_cell chemistry")
            object.__setattr__(self, "ports", ports)
        if self.output is not None:
            if self.interface is None or set(self.output.markers) - set(self.interface.targets):
                raise ValueError("binary output requires markers from a structural interface")
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
        if self.kind == "run_tumble":
            raise ValueError(
                "run_tumble has been removed; its records require the earlier release. "
                "Use chemotaxis for new experiments."
            )
        if self.kind not in {
            "assay",
            "temporal_probe",
            "signalling_game",
            "cue_reading",
            "chemotaxis",
            "tissue",
            "python",
        }:
            raise ValueError("unsupported task kind")
        if self.kind == "python" and not self.entrypoint:
            raise ValueError("python tasks require an importable entrypoint")
        if self.kind in {"cue_reading", "chemotaxis"}:
            from .molecular_worlds import resolve_task

            object.__setattr__(self, "config", resolve_task(self.kind, self.config))
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
