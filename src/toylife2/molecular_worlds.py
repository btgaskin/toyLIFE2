"""Discrete cue tasks and two field profiles for evolving molecular cells.

Fields use dimensionless units. The fluctuating field is a concrete realization
motivated by Celani et al. (2011), DOI 10.1007/s10955-011-0251-6. The periodic
exponential source adapts Alonso and Kirkegaard (2024), DOI 10.1093/pnasnexus/pgae235.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def resolve_task(kind: str, config: Mapping) -> dict:
    config = dict(config)
    from .protocols import SINGLE_CELL_CONTRACT

    contract = config.get("experiment_contract")
    if contract is not None and contract != SINGLE_CELL_CONTRACT:
        raise ValueError("unsupported single-cell experiment contract")
    if kind == "cue_reading":
        defaults: dict[str, Any] = dict(
            fixed_map=[0, 1],
            input_channels=["cue"],
            action_outputs=["act"],
            profile="basic",
            source_model="binary_matching",
        )
    else:
        defaults = dict(
            field="fluctuating",
            domain=16.0,
            speed=1.0,
            rotational_diffusion=0.05,
            sensor_scale=0.5,
            hazard_zero=2.0,
            hazard_one=0.2,
            mean=0.5,
            amplitude=0.4,
            modes=[[1, 0], [0, 1], [1, 1], [1, -1]],
            correlation_time=3.0,
            static=True,
            source_peak=1.0,
            source_length=4.0,
            arrival_radius=0.4,
            input_channel="cue",
            action_outputs=["motor"],
        )
    if contract is not None:
        defaults.pop("action_outputs", None)
        if kind == "cue_reading":
            defaults.pop("fixed_map", None)
            defaults.update(cue_doses=[[0], [1]], settling_steps=0)
    defaults.update(config)
    if kind == "chemotaxis":
        defaults.setdefault("motion_model", "ecoli_spherical_v1")
        if defaults["motion_model"] != "ecoli_spherical_v1":
            raise ValueError("chemotaxis uses motion_model=ecoli_spherical_v1")
        for name, value in dict(
            tumble_mean_duration=0.14,
            tumble_rotational_diffusion=3.5,
            tumble_speed_fraction=0.0,
            micrometers_per_length_unit=20.0,
            time_unit="seconds",
            motion_max_substep=0.01,
            motion_event_capacity=8,
        ).items():
            defaults.setdefault(name, value)
        numerical_mode(defaults)
    return defaults


def numerical_mode(config):
    """Resolve legacy settings without adding fields to their scientific identity."""
    engine = config.get("world_engine", "numpy")
    precision = config.get("world_precision", "float64")
    if engine not in {"numpy", "quadrants"} or precision not in {"float32", "float64"}:
        raise ValueError("chemotaxis requires numpy/quadrants and float32/float64")
    from ._world_random import random_math

    random_math(config)
    if config.get("reference_world", False):
        if config.get("world_engine") == "quadrants":
            raise ValueError("reference_world conflicts with world_engine=quadrants")
        engine = "numpy"
    return engine, precision


class CueReading:
    """Exhaustive prescribed nature states, one independent case per scenario."""

    def __init__(self, config, shape, agents, rng, dt, kernels):
        if agents != 1:
            raise ValueError("cue_reading uses one cell per independent episode")
        self.shape, self.rng = shape, rng
        self.binary = "experiment_contract" in config
        self.inputs = tuple(config["input_channels"])
        self.settling_steps = config.get("settling_steps", 0)
        if type(self.settling_steps) is not int or self.settling_steps < 0:
            raise ValueError("settling_steps must be a nonnegative integer")
        if self.binary:
            self.doses = np.asarray(config["cue_doses"])
            self.states = 2
            self.outputs = ("action",)
            if (
                not self.inputs
                or len(self.inputs) > 2
                or len(set(self.inputs)) != len(self.inputs)
                or self.doses.shape != (2, len(self.inputs))
                or self.doses.dtype.kind not in "iu"
                or np.any(self.doses < 0)
                or np.any(self.doses > np.iinfo(np.int32).max)
            ):
                raise ValueError("binary cue task requires two explicit nonnegative dose rows")
            if len(self.inputs) == 2 and (
                np.any(np.count_nonzero(self.doses, axis=1) != 1)
                or np.count_nonzero(np.any(self.doses > 0, axis=0)) != 2
            ):
                raise ValueError("two cue identities must be supplied mutually exclusively")
        else:
            self.mapping = np.asarray(config["fixed_map"])
            self.outputs = tuple(config["action_outputs"])
            self.states = len(self.mapping)
            self.settling_steps = config.get("settling_steps", 0)
            if type(self.settling_steps) is not int or self.settling_steps < 0:
                raise ValueError("settling_steps must be a nonnegative integer")
            if (
                not self.inputs
                or not self.outputs
                or len(self.inputs) > 2
                or len(self.outputs) > 2
                or self.states != 1 << len(self.outputs)
                or self.mapping.dtype.kind not in "iu"
                or self.mapping.shape != (self.states,)
                or np.any((self.mapping < 0) | (self.mapping >= 1 << len(self.inputs)))
                or len(set(self.inputs)) != len(self.inputs)
                or len(set(self.outputs)) != len(self.outputs)
            ):
                raise ValueError("cue task requires binary channels and a valid fixed code table")
            if shape[-1] != self.states:
                raise ValueError("exhaustive cue evaluation requires one episode per nature state")

        if shape[-1] != self.states:
            raise ValueError("exhaustive cue evaluation requires one episode per nature state")

    def reset(self):
        # Episode axis is the final world axis, independent of candidate identities.
        nature = np.broadcast_to(np.arange(self.states, dtype=np.int32), self.shape).copy()
        return dict(tick=0, nature=nature)

    def observe(self, state):
        if self.binary:
            return {
                "inputs": {
                    name: self.doses[state["nature"], i].astype(np.int32)
                    for i, name in enumerate(self.inputs)
                }
            }
        code = self.mapping[state["nature"]]
        return {
            "inputs": {
                name: ((code >> (len(self.inputs) - i - 1)) & 1).astype(np.int32)
                for i, name in enumerate(self.inputs)
            }
        }

    def advance(self, state, actions):
        correct = np.asarray(actions == state["nature"], dtype=np.float64)
        component = {
            name: (
                ((actions >> (len(self.outputs) - i - 1)) & 1)
                == ((state["nature"] >> (len(self.outputs) - i - 1)) & 1)
            )
            for i, name in enumerate(self.outputs)
        }
        return (
            dict(tick=state["tick"] + 1, nature=state["nature"].copy()),
            correct,
            {
                "judged": state["tick"] >= self.settling_steps,
                "nature": state["nature"].copy(),
                "correct": correct,
                "component_correct": component,
            },
        )


class Chemotaxis:
    """Shared noisy molecular sensing and run/tumble coupling for two fields."""

    def __init__(self, config, shape, agents, rng, dt, kernels):
        if agents != 1:
            raise ValueError("chemotaxis uses one cell per independent world")
        self.config, self.shape, self.rng, self.dt = dict(config), shape, rng, dt
        self.engine, self.precision = numerical_mode(config)
        from ._world_random import random_math

        self.random_math = random_math(config)
        self.dtype = np.dtype(self.precision)
        if config["motion_model"] != "ecoli_spherical_v1":
            raise ValueError("chemotaxis uses motion_model=ecoli_spherical_v1")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("chemotaxis dt must be finite and positive")
        for name in ("tumble_mean_duration", "micrometers_per_length_unit", "motion_max_substep"):
            if not math.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if config["time_unit"] != "seconds":
            raise ValueError("ecoli_spherical_v1 uses seconds")
        if (
            not math.isfinite(config["tumble_rotational_diffusion"])
            or config["tumble_rotational_diffusion"] < 0
        ):
            raise ValueError("tumble_rotational_diffusion must be finite and nonnegative")
        if not math.isfinite(config["tumble_speed_fraction"]) or not (
            0 <= config["tumble_speed_fraction"] <= 1
        ):
            raise ValueError("tumble_speed_fraction must be in [0,1]")
        capacity = config["motion_event_capacity"]
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("motion_event_capacity must be an integer in 1..64")
        self.motion_steps = math.ceil(dt / config["motion_max_substep"])
        if self.motion_steps > 1000:
            raise ValueError("motion requires at most 1000 angular substeps per world decision")
        self.event_capacity = capacity
        self.motion_stride = 2 + 3 * self.motion_steps
        self.draw_columns = 4 + len(config["modes"]) + capacity * self.motion_stride
        self.kind = config["field"]
        if self.kind not in {"fluctuating", "source"}:
            raise ValueError("chemotaxis field must be fluctuating or source")
        for name in ("domain", "sensor_scale", "correlation_time", "source_length"):
            if not math.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "speed",
            "rotational_diffusion",
            "hazard_zero",
            "hazard_one",
            "amplitude",
            "arrival_radius",
        ):
            if not math.isfinite(config[name]) or config[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not math.isfinite(config["mean"])
            or not math.isfinite(config["source_peak"])
            or not 0 < config["source_peak"] <= 1
            or not 0 < config["mean"] - config["amplitude"]
            or config["mean"] + config["amplitude"] > 1
        ):
            raise ValueError("field concentrations require positive bounds within [0,1]")
        modes = np.asarray(config["modes"])
        if (
            modes.ndim != 2
            or modes.shape[1] != 2
            or not len(modes)
            or modes.dtype.kind not in "iu"
            or np.any(np.all(modes == 0, axis=1))
        ):
            raise ValueError("modes require nonzero two-dimensional integer wavevectors")
        if type(config["static"]) is not bool:
            raise ValueError("static must be boolean")
        self.wavevectors = (modes * (2 * np.pi / config["domain"])).astype(self.dtype)
        self.constants = np.asarray(
            [
                config["domain"],
                config["speed"] * dt,
                np.sqrt(2 * config["rotational_diffusion"] * dt),
                config["hazard_zero"],
                config["hazard_one"],
                config["sensor_scale"],
                config["mean"],
                config["amplitude"],
                config["source_peak"],
                config["source_length"],
                np.sqrt(2 * dt / config["correlation_time"]),
                2 * np.pi,
                config["rotational_diffusion"],
                dt,
                config.get("tumble_mean_duration", 0.14),
                config.get("tumble_speed_fraction", 0.0),
                config["tumble_rotational_diffusion"],
                config["motion_max_substep"],
                self.motion_steps,
                self.event_capacity,
            ],
            dtype=self.dtype,
        )
        if not np.all(np.isfinite(self.constants)) or not np.all(np.isfinite(self.wavevectors)):
            raise ValueError("chemotaxis parameters exceed the selected world precision")
        if np.any(self.constants[[0, 5, 9]] <= 0):
            raise ValueError("chemotaxis scales underflow in the selected world precision")
        self._kernel_buffers = None
        self._draw_layout = None
        self._prepared_draw = None

    def _host_draws(self, tick):
        if self._prepared_draw is None or self._prepared_draw[0] != tick:
            from .resident_world import RandomBlockCache

            # A one-tick cache also reuses the shared runner cache when available.
            engine = getattr(self, "_engine", None)
            cache = engine._world_random_cache if engine is not None else RandomBlockCache(0)
            values = cache.block(self, tick, 1)
            self._prepared_draw = tick, values[0]
        return self._prepared_draw[1][self.rng._inverse]

    @property
    def buffer_bytes(self):
        return (
            getattr(self._draw_layout, "bytes", 0)
            + (0 if self._prepared_draw is None else self._prepared_draw[1].nbytes)
            + getattr(self._kernel_buffers, "bytes", 0)
        )

    def reset(self):
        state: dict[str, Any] = dict(
            tick=0,
            position=(self.rng.uniform(0, "chem-position", 2) * self.config["domain"]).astype(
                self.dtype
            ),
            theta=(self.rng.uniform(0, "chem-heading") * 2 * np.pi).astype(self.dtype),
            phases=(self.rng.uniform(0, "chem-phases", len(self.wavevectors)) * 2 * np.pi).astype(
                self.dtype
            ),
            source=(self.rng.uniform(0, "chem-source", 2) * self.config["domain"]).astype(
                self.dtype
            ),
        )
        polar = (2 * self.rng.uniform(0, "chem-polar") - 1).astype(self.dtype)
        radial = np.sqrt(np.maximum(1 - polar * polar, 0))
        state["orientation"] = np.stack(
            (radial * np.cos(state["theta"]), radial * np.sin(state["theta"]), polar), -1
        )
        state["tumble_remaining"] = np.zeros(self.shape, self.dtype)
        return state

    def distance(self, positions, source):
        delta = positions - source
        delta -= self.config["domain"] * np.floor(delta / self.config["domain"] + 0.5)
        return np.sqrt(np.sum(delta * delta, axis=-1))

    def field(self, state):
        if self.kind == "source":
            return self.config["source_peak"] * np.exp(
                -self.distance(state["position"], state["source"]) / self.config["source_length"]
            )
        if self.precision == "float32":
            value = np.zeros(state["position"].shape[:-1], np.float32)
            for i, wave in enumerate(self.wavevectors):
                angle = state["position"][..., 0] * wave[0]
                angle = angle + state["position"][..., 1] * wave[1]
                angle = angle + state["phases"][..., i]
                value += np.cos(angle)
            return self.constants[6] + self.constants[7] * (value / len(self.wavevectors))
        angles = np.einsum("...d,md->...m", state["position"], self.wavevectors) + state["phases"]
        return self.config["mean"] + self.config["amplitude"] * np.mean(np.cos(angles), axis=-1)

    def observe(self, state):
        from ._world_random import FLOAT32

        if self.random_math == FLOAT32 and self.engine == "quadrants":
            from .resident_world import WorldBuffers

            if self._kernel_buffers is None:
                self._kernel_buffers = WorldBuffers(self, getattr(self, "_engine", None))
            return self._kernel_buffers.observe(state)
        exposure = self.field(state)
        draw = self._host_draws(state["tick"])[..., 0]
        cue = (draw < exposure / (self.config["sensor_scale"] + exposure)).astype(np.int32)
        return {"inputs": {self.config["input_channel"]: cue}, "exposure": exposure}

    def advance(self, state, actions):
        if np.any((actions != 0) & (actions != 1)):
            raise ValueError("chemotaxis actions must be binary")
        if self.engine == "quadrants":
            return self._advance_quadrants(state, actions)
        return self._advance_motion(state, actions)

    def _advance_quadrants(self, state, actions):
        from .resident_world import WorldBuffers

        if self._kernel_buffers is None:
            self._kernel_buffers = WorldBuffers(self, getattr(self, "_engine", None))
        buffers = self._kernel_buffers
        buffers.upload(state)
        buffers.advance(actions, state["tick"])
        return buffers.result(state["tick"] + 1)

    def _advance_motion(self, state, actions):
        """Continuous phase events, geodesic angular substeps, and planar projection.

        Event capacity bounds the resident draw allocation. Exhaustion raises
        before publishing state; it never discards a transition or time segment.
        """
        p, tick = self.constants, state["tick"]
        shape = self.shape
        orientation = state["orientation"].copy()
        position = state["position"].copy()
        remaining = state["tumble_remaining"].copy()
        time_left = np.full(shape, p[13], self.dtype)
        tumble = np.zeros(shape, bool)
        hazard = np.where(actions != 0, self.config["hazard_one"], self.config["hazard_zero"])
        hazard = hazard.astype(self.dtype)
        draws = self._host_draws(tick)
        base = 4 + len(self.wavevectors)

        for event in range(self.event_capacity):
            start = base + event * self.motion_stride
            active = time_left > 0
            if not np.any(active):
                break
            is_tumble = active & (remaining > 0)
            waiting = -np.log(draws[..., start]) / np.where(hazard > 0, hazard, 1)
            waiting = np.where(hazard > 0, waiting, np.inf)
            duration = np.minimum(time_left, np.where(is_tumble, remaining, waiting))
            duration = np.where(active, duration, 0).astype(self.dtype)
            new_tumble = active & ~is_tumble & (waiting < time_left)
            diffusion = np.where(is_tumble, p[16], p[12])
            velocity = (p[1] / p[13]) * np.where(is_tumble, p[15], 1)
            segment_left = duration.copy()
            for step in range(self.motion_steps):
                h = (
                    segment_left
                    if step == self.motion_steps - 1
                    else np.minimum(segment_left, p[17])
                )
                noise = draws[..., start + 2 + 3 * step : start + 5 + 3 * step]
                tangent = noise - np.sum(noise * orientation, axis=-1)[..., None] * orientation
                w = np.sqrt(2 * diffusion * h)[..., None] * tangent
                angle = np.sqrt(np.sum(w * w, axis=-1))
                sinc = np.ones(shape, self.dtype)
                np.divide(np.sin(angle), angle, out=sinc, where=angle > 0)
                candidate = np.cos(angle)[..., None] * orientation + sinc[..., None] * w
                candidate /= np.sqrt(np.sum(candidate * candidate, axis=-1))[..., None]
                orientation = np.where((h > 0)[..., None], candidate, orientation)
                position += (velocity * h)[..., None] * orientation[..., :2]
                segment_left = np.maximum(segment_left - h, 0)
            tumble |= is_tumble & (duration > 0)
            remaining = np.where(is_tumble, np.maximum(remaining - duration, 0), remaining)
            sampled_duration = -p[14] * np.log(draws[..., start + 1])
            remaining = np.where(new_tumble, sampled_duration, remaining)
            time_left = np.maximum(time_left - duration, 0)
        if np.any(time_left > 0):
            raise RuntimeError(
                "chemotaxis motion event capacity exhausted; increase motion_event_capacity"
            )
        position -= p[0] * np.floor(position / p[0])
        phases = state["phases"].copy()
        if self.kind == "fluctuating" and not self.config["static"]:
            phases += p[10] * draws[..., 4:base]
            phases -= p[11] * np.floor(phases / p[11])
        theta = np.arctan2(orientation[..., 1], orientation[..., 0])
        next_state = dict(
            tick=tick + 1,
            position=position,
            theta=theta - p[11] * np.floor(theta / p[11]),
            orientation=orientation,
            phases=phases,
            source=state["source"].copy(),
            tumble_remaining=remaining,
        )
        exposure = self.field(next_state)
        channels = {
            name: next_state[name].copy()
            for name in ("position", "theta", "phases", "source", "orientation", "tumble_remaining")
        }
        channels.update(exposure=exposure.copy(), tumbling=tumble.astype(np.int32))
        if self.kind == "source":
            distance = self.distance(position, next_state["source"])
            channels.update(
                source_distance=distance, arrived=distance <= self.config["arrival_radius"]
            )
        return next_state, exposure, channels
