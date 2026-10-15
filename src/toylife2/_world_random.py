"""Addressed chemotaxis draws: shared layout and bounded, batched host generation."""

from __future__ import annotations

import numpy as np

from .random import philox

LEGACY = "numpy_f64_v1"
FLOAT32 = "box_muller_f32_v1"


def random_math(config):
    value = config.get("world_random_math", LEGACY)
    if value not in {LEGACY, FLOAT32}:
        raise ValueError("world_random_math requires numpy_f64_v1 or box_muller_f32_v1")
    if value == FLOAT32 and config.get("world_precision", "float64") != "float32":
        raise ValueError("box_muller_f32_v1 requires world_precision=float32")
    return value


def uniform32(words):
    """Open float32 uniforms from the existing low/high Philox word ordering."""
    hi = words[..., 1].astype(np.float32) * np.float32(2**-32)
    lo = np.right_shift(words[..., 0], 12).astype(np.float32) + np.float32(0.5)
    lo *= np.float32(2**-52)
    return np.minimum(hi + lo, np.float32(1 - 2**-24))


def uniform64(words):
    value = words[..., 0].astype(np.uint64) | (words[..., 1].astype(np.uint64) << 32)
    return ((value >> 12).astype(np.float64) + 0.5) / (1 << 52)


def normal(u, v, math):
    if math == FLOAT32:
        return np.sqrt(np.float32(-2) * np.log(u)) * np.cos(np.float32(2 * np.pi) * v)
    return np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)


class DrawLayout:
    """One descriptor per packed column; role keys retain their scientific names."""

    def __init__(self, world):
        self.math = random_math(world.config)
        self.roles = []
        self.requests = {}
        self.descriptors = np.full((world.draw_columns, 4), -1, np.int32)

        def add(column, width, role, gaussian=False):
            radius = len(self.roles)
            self.roles.append("world:" + role + (":normal-radius" if gaussian else ""))
            phase = -1
            if gaussian:
                phase = len(self.roles)
                self.roles.append("world:" + role + ":normal-phase")
            self.requests.setdefault(width, []).append((column, radius, phase))
            for axis in range(width):
                self.descriptors[column + axis] = radius, phase, width, axis

        add(0, 1, "chem-sensor")
        modes = len(world.wavevectors)
        if world.kind == "fluctuating" and not world.config["static"]:
            add(4, modes, "chem-phase-update", True)
        for event in range(world.event_capacity):
            start = 4 + modes + event * world.motion_stride
            add(start, 1, f"chem-motion-wait:{event}")
            add(start + 1, 1, f"chem-motion-duration:{event}")
            for step in range(world.motion_steps):
                add(start + 2 + 3 * step, 3, f"chem-motion-rotation:{event}:{step}", True)
        self.ids = {}
        for width in self.requests:
            if self.math == FLOAT32 and len(world.rng._unique):
                if int(world.rng._unique.max()) > ((1 << 64) - width) // width:
                    raise ValueError("expanded chemotaxis component ID exceeds uint64")
            self.ids[width] = world.rng._unique[:, None] * np.uint64(width) + np.arange(
                width, dtype=np.uint64
            )
        self.bytes = self.descriptors.nbytes + sum(a.nbytes for a in self.ids.values())

    def keys(self, world, tick, count):
        keys = np.empty((count, len(self.roles), 2), np.uint32)
        for offset in range(count):
            for index, role in enumerate(self.roles):
                keys[offset, index] = world.rng.stream.key(
                    role=role, trial=world.rng.panel, event=tick + offset
                )
        return keys

    def generate(self, world, tick, count, *, scratch_bytes=8 * 1024 * 1024):
        """Tile independent counters; the budget includes Philox's round temporaries."""
        if scratch_bytes < 512 * max(self.requests):
            raise MemoryError("world random scratch cannot hold one component family")
        result = np.zeros((count, len(world.rng._unique), world.draw_columns), world.dtype)
        keys = self.keys(world, tick, count)
        conversion = uniform32 if self.math == FLOAT32 else uniform64
        # 256 bytes per counter bounds the live uint64 Philox and transform arrays.
        capacity = max(1, scratch_bytes // 256)
        for width, requests in self.requests.items():
            scenarios = len(world.rng._unique)
            offsets = max(1, min(count, capacity // max(1, 2 * width * scenarios)))
            for first in range(0, count, offsets):
                last = min(first + offsets, count)
                size = max(1, capacity // (2 * width * (last - first)))
                for begin in range(0, scenarios, size):
                    end = min(begin + size, scenarios)
                    batch = max(1, capacity // (2 * width * (last - first) * (end - begin)))
                    for start in range(0, len(requests), batch):
                        selected = requests[start : start + batch]
                        roles = [r for _, u, v in selected for r in (u, v) if r >= 0]
                        ids = self.ids[width][begin:end]
                        counters = np.zeros(
                            (last - first, len(roles), end - begin, width, 4), np.uint32
                        )
                        counters[..., 0] = ids & np.uint64(0xFFFFFFFF)
                        counters[..., 1] = ids >> np.uint64(32)
                        values = conversion(
                            philox(counters, keys[first:last, roles, None, None, :])
                        )
                        index = 0
                        for column, _, phase in selected:
                            # Keep legacy transforms on the established scenario/component slices.
                            for offset in range(last - first):
                                value = values[offset, index]
                                if phase >= 0:
                                    value = normal(value, values[offset, index + 1], self.math)
                                elif column:
                                    value = np.clip(
                                        value.astype(world.dtype),
                                        np.finfo(world.dtype).tiny,
                                        np.nextafter(world.dtype.type(1), world.dtype.type(0)),
                                    )
                                result[first + offset, begin:end, column : column + width] = value
                            index += 2 if phase >= 0 else 1
        return result


def layout(world):
    if getattr(world, "_draw_layout", None) is None:
        world._draw_layout = DrawLayout(world)
    return world._draw_layout
