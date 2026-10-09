"""Serializable numerical worlds and genotype-independent scenario randomness.

World callbacks return observations, simultaneous numerical state, and bounded rewards.
Built-in stochastic draws address panel, scenario, tick and role, so candidates share
the same environmental history even when they are evaluated in separate batches.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from .protocols import TaskSpec
from .random import Stream


def numerical(value: Any) -> Any:
    """Copy a checkpoint tree, rejecting opaque callback state."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufUS":
            raise ValueError("world arrays must have numerical or string dtypes")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError("world state must be finite")
        return value.copy()
    if isinstance(value, Mapping):
        return {str(k): numerical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numerical(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        result = value.item() if isinstance(value, np.generic) else value
        if isinstance(result, float) and not np.isfinite(result):
            raise ValueError("world state must be finite")
        return result
    raise ValueError("world callbacks must return serialized numerical state")


class ScenarioRandom:
    def __init__(self, seed: int, panel: int, scenarios: np.ndarray):
        if type(panel) is not int or panel < 0:
            raise ValueError("panel must be a nonnegative integer")
        self.stream = Stream(seed)
        self.panel = panel
        self.scenarios = np.asarray(scenarios, dtype=np.uint64)

    def uniform(self, tick: int, role: str, width: int | None = None) -> np.ndarray:
        ids = self.scenarios
        if width is not None:
            ids = ids[..., None] * np.uint64(width) + np.arange(width, dtype=np.uint64)
        return self.stream.uniform(ids, role=f"world:{role}", trial=self.panel, event=tick)

    def normal(self, tick: int, role: str, width: int | None = None) -> np.ndarray:
        u = self.uniform(tick, role + ":normal-radius", width)
        v = self.uniform(tick, role + ":normal-phase", width)
        return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)


class TemporalProbe:
    def __init__(
        self,
        config: Mapping,
        shape: tuple[int, ...],
        agents: int,
        rng: ScenarioRandom,
        dt: float,
        kernels: bool,
    ):
        self.config, self.shape, self.agents, self.rng = config, shape, agents, rng
        self.kernels = kernels
        self.sequence = tuple(config.get("sequence", (0, 1, 1, 0)))
        if not self.sequence or any(type(x) is not int or x < 0 for x in self.sequence):
            raise ValueError("temporal_probe sequence must contain nonnegative integer doses")
        self.delay = int(config.get("delay", 1))
        if self.delay < 0:
            raise ValueError("temporal_probe delay must be nonnegative")

    def reset(self) -> dict:
        return {"tick": 0}

    def observe(self, state: dict) -> dict:
        tick = state["tick"]
        shape = self.shape + ((self.agents,) if self.agents > 1 else ())
        return {"dose": np.full(shape, self.sequence[tick % len(self.sequence)], np.int32)}

    def advance(self, state: dict, actions: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        tick = state["tick"]
        target = int(self.sequence[(tick - self.delay) % len(self.sequence)] > 0)
        if self.kernels:
            from ._world_kernels import temporal_commit

            reward = np.empty(np.prod(self.shape, dtype=int), np.float32)
            cast(Any, temporal_commit)(
                np.ascontiguousarray(actions.reshape(-1, self.agents), dtype=np.int32),
                reward,
                target,
            )
            reward = reward.reshape(self.shape)
        elif self.agents > 1:
            reward = np.mean(actions == target, axis=-1)
        else:
            reward = np.asarray(actions == target, dtype=np.float64)
        return {"tick": tick + 1}, reward, {"target": np.full(self.shape, target, np.int32)}


class SignallingGame:
    """Alternating send/receive ticks with a private bit retained for correctness."""

    def __init__(
        self,
        config: Mapping,
        shape: tuple[int, ...],
        agents: int,
        rng: ScenarioRandom,
        dt: float,
        kernels: bool,
    ):
        if agents != 2:
            raise ValueError("signalling_game requires an explicit two-agent axis")
        self.config, self.shape, self.rng, self.kernels = config, shape, rng, kernels

    def _private(self, round_index: int) -> np.ndarray:
        # Each adjacent pair of rounds has one bit of each value per scenario.
        first = self.rng.uniform(round_index // 2, "signalling-private") >= 0.5
        return np.asarray(first ^ bool(round_index % 2), dtype=np.int32)

    def reset(self) -> dict:
        return {"tick": 0, "private": self._private(0), "delivered": np.zeros(self.shape, np.int32)}

    def observe(self, state: dict) -> dict:
        phase = state["tick"] % 2
        dose = np.zeros((*self.shape, 2), np.int32)
        mask = np.zeros_like(dose)
        if phase == 0:
            dose[..., 0] = state["private"]
            mask[..., 0] = 1
        else:
            dose[..., 1] = state["delivered"]
            mask[..., 1] = 1
        return {"dose": dose, "mask": mask, "phase": np.full(self.shape, phase, np.int32)}

    def advance(self, state: dict, actions: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        actions = np.ascontiguousarray(actions.reshape(-1, 2), dtype=np.int32)
        private = np.ascontiguousarray(state["private"].reshape(-1), dtype=np.int32)
        delivered = np.array(state["delivered"].reshape(-1), dtype=np.int32, copy=True)
        reward = np.zeros(private.shape, np.float32)
        phase = state["tick"] % 2
        if self.kernels:
            from ._world_kernels import signalling_commit

            cast(Any, signalling_commit)(actions, private, delivered, reward, phase)
        elif phase == 0:
            delivered[:] = actions[:, 0]
        else:
            reward[:] = actions[:, 1] == private
            delivered[:] = 0
        tick = state["tick"] + 1
        if phase == 1:
            private = self._private(tick // 2).reshape(-1)
        next_state = {
            "tick": tick,
            "private": private.reshape(self.shape),
            "delivered": delivered.reshape(self.shape),
        }
        return (
            next_state,
            reward.reshape(self.shape),
            {
                "judged": np.full(self.shape, phase == 1, bool),
                "message": actions[:, 0].reshape(self.shape)
                if phase == 0
                else np.zeros(self.shape, np.int32),
            },
        )


class RunTumble:
    """A periodic Gaussian-patch habitat with stationary two-tick tumbles."""

    def __init__(
        self,
        config: Mapping,
        shape: tuple[int, ...],
        agents: int,
        rng: ScenarioRandom,
        dt: float,
        kernels: bool,
    ):
        if agents != 1:
            raise ValueError("run_tumble uses one agent per independent world")
        self.shape, self.rng, self.dt, self.kernels = shape, rng, dt, kernels
        self.domain = float(config.get("domain", 16.0))
        self.speed = float(config.get("speed", 1.0))
        self.diffusion = float(config.get("rotational_diffusion", 0.05))
        self.tau = float(config.get("field_tau", 5.0))
        self.high_hazard = float(config.get("high_motor_hazard", 0.2))
        self.low_hazard = float(config.get("low_motor_hazard", 2.0))
        if (
            self.domain <= 0
            or self.tau <= 0
            or min(self.speed, self.diffusion, self.high_hazard, self.low_hazard) < 0
        ):
            raise ValueError("habitat parameters require positive domain/tau and nonnegative rates")
        self.centers = np.asarray(
            config.get("centers", ((4, 4), (4, 12), (12, 4), (12, 12))), dtype=np.float32
        )
        if self.centers.shape != (4, 2) or not np.all(np.isfinite(self.centers)):
            raise ValueError("habitat requires four finite two-dimensional patch centers")

    def reset(self) -> dict:
        return {
            "tick": 0,
            "position": (self.rng.uniform(0, "position", 2) * self.domain).astype(np.float32),
            "theta": (self.rng.uniform(0, "heading") * (2 * np.pi)).astype(np.float32),
            "tumble": np.zeros(self.shape, np.int32),
            "latent": self.rng.normal(0, "field-initial", 4).astype(np.float32),
        }

    def field(self, state: dict) -> np.ndarray:
        delta = state["position"][..., None, :] - self.centers
        delta -= self.domain * np.floor(delta / self.domain + 0.5)
        amplitude = 1.0 / (1.0 + np.exp(-state["latent"]))
        return (
            0.05 + np.sum(amplitude * np.exp(-0.5 * np.sum(delta * delta, axis=-1)), axis=-1)
        ) / 4.05

    def observe(self, state: dict) -> dict:
        exposure = self.field(state)
        return {"dose": np.floor(8 * exposure).astype(np.int32), "exposure": exposure}

    def advance(self, state: dict, actions: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        next_state = numerical(state)
        tick = state["tick"]
        u = self.rng.uniform(tick, "tumble-hazard").astype(np.float32)
        heading = self.rng.uniform(tick, "tumble-heading").astype(np.float32)
        rotation = self.rng.normal(tick, "rotation").astype(np.float32)
        field_noise = self.rng.normal(tick, "field-update", 4).astype(np.float32)
        if self.kernels:
            from ._world_kernels import habitat_commit

            reward = np.empty(np.prod(self.shape, dtype=int), np.float32)
            cast(Any, habitat_commit)(
                next_state["position"].reshape(-1, 2),
                next_state["theta"].reshape(-1),
                next_state["tumble"].reshape(-1),
                next_state["latent"].reshape(-1, 4),
                self.centers,
                np.ascontiguousarray(actions.reshape(-1), dtype=np.int32),
                u.reshape(-1),
                heading.reshape(-1),
                rotation.reshape(-1),
                field_noise.reshape(-1, 4),
                reward,
                self.dt,
                self.domain,
                self.speed,
                self.diffusion,
                self.tau,
                self.high_hazard,
                self.low_hazard,
            )
            reward = reward.reshape(self.shape)
        else:
            decay = np.exp(-self.dt / self.tau)
            next_state["latent"][:] = (
                decay * state["latent"] + np.sqrt(1 - decay * decay) * field_noise
            )
            active = state["tumble"] > 0
            next_state["tumble"][active] -= 1
            complete = active & (next_state["tumble"] == 0)
            next_state["theta"][complete] = 2 * np.pi * heading[complete]
            hazard = np.where(actions != 0, self.high_hazard, self.low_hazard)
            start = ~active & (u < -np.expm1(-hazard * self.dt))
            next_state["tumble"][start] = 1
            moving = ~active & ~start
            next_state["theta"][moving] += np.sqrt(2 * self.diffusion * self.dt) * rotation[moving]
            theta = next_state["theta"]
            velocity = self.speed * self.dt * np.stack((np.cos(theta), np.sin(theta)), axis=-1)
            next_state["position"][moving] = (
                state["position"][moving] + velocity[moving]
            ) % self.domain
            reward = self.field(next_state)
        next_state["tick"] = tick + 1
        return (
            next_state,
            reward,
            {
                "exposure": reward.copy(),
                "tumbling": ((state["tumble"] > 0) | (next_state["tumble"] > 0)).astype(np.int32),
            },
        )


class PythonWorld:
    """Importable factory; callback state is copied and validated at every boundary."""

    def __init__(
        self, task: TaskSpec, shape: tuple[int, ...], agents: int, rng: ScenarioRandom, dt: float
    ):
        assert task.entrypoint is not None
        module, separator, name = task.entrypoint.partition(":")
        if not separator:
            module, _, name = task.entrypoint.rpartition(".")
        factory = getattr(importlib.import_module(module), name)
        self.callback = factory(
            config=dict(task.config), shape=shape, agents=agents, random=rng, dt=dt
        )
        for name in ("reset", "observe", "advance"):
            if not callable(getattr(self.callback, name, None)):
                raise ValueError(f"Python world requires {name} callback")

    def reset(self) -> dict:
        return numerical(self.callback.reset())

    def observe(self, state: dict) -> dict:
        return numerical(self.callback.observe(numerical(state)))

    def advance(self, state: dict, actions: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        next_state, reward, channels = self.callback.advance(numerical(state), actions.copy())
        return numerical(next_state), np.asarray(reward, dtype=float), numerical(channels)


def make_world(
    task: TaskSpec,
    shape: tuple[int, ...],
    agents: int,
    rng: ScenarioRandom,
    dt: float,
    *,
    kernels: bool = True,
):
    kinds = {
        "temporal_probe": TemporalProbe,
        "signalling_game": SignallingGame,
        "run_tumble": RunTumble,
    }
    if task.kind == "python":
        return PythonWorld(task, shape, agents, rng, dt)
    if task.kind not in kinds:
        raise ValueError(f"{task.kind} is not a closed-loop world")
    return kinds[task.kind](task.config, shape, agents, rng, dt, kernels)
