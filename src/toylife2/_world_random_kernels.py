"""Stateless float32 scenario draws written directly into resident world buffers."""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false, reportOperatorIssue=false
from __future__ import annotations

import quadrants as qd

from ._output_kernels import multiply_high, philox_pair


@qd.func
def open_uniform(first, second):
    hi = qd.cast(second, qd.f32) * qd.f32(2.3283064365386963e-10)
    lo = (qd.cast(first >> 12, qd.f32) + qd.f32(0.5)) * qd.f32(2.220446049250313e-16)
    return qd.min(hi + lo, qd.f32(0.9999999403953552))


@qd.kernel(fastcache=True)
def fill_draws(
    descriptors: qd.types.NDArray[qd.i32, 2],
    ids: qd.types.NDArray[qd.u32, 2],
    keys: qd.types.NDArray[qd.u32, 3],
    draws: qd.types.NDArray[qd.f32, 3],
    count: qd.i32,
):
    for tick, scenario, column in qd.ndrange(draws.shape[0], draws.shape[1], draws.shape[2]):
        radius, phase = descriptors[column, 0], descriptors[column, 1]
        value = qd.f32(0.0)
        if tick < count and radius >= 0:
            width, axis = (
                qd.cast(descriptors[column, 2], qd.u32),
                qd.cast(descriptors[column, 3], qd.u32),
            )
            low = ids[scenario, 0] * width
            high = ids[scenario, 1] * width + multiply_high(ids[scenario, 0], width)
            final_low = low + axis
            high += qd.cast(final_low < low, qd.u32)
            a, b = philox_pair(
                final_low, high, keys[tick, radius, 0], keys[tick, radius, 1], qd.u32(0)
            )
            value = open_uniform(a, b)
            if phase >= 0:
                a, b = philox_pair(
                    final_low, high, keys[tick, phase, 0], keys[tick, phase, 1], qd.u32(0)
                )
                value = qd.sqrt(qd.f32(-2.0) * qd.log(value)) * qd.cos(
                    qd.f32(6.283185307179586) * open_uniform(a, b)
                )
        draws[tick, scenario, column] = value
