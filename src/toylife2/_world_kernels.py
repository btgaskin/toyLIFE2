"""Numerical built-in world updates on the initialized Quadrants CPU/Metal runtime."""

# Quadrants' DSL annotations describe compiled arrays, not Python call arguments.
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false

from __future__ import annotations

import quadrants as qd


@qd.kernel(fastcache=True)
def temporal_commit(
    actions: qd.types.NDArray[qd.i32, 2], reward: qd.types.NDArray[qd.f32, 1], target: qd.i32
) -> None:
    for lane in range(actions.shape[0]):
        correct = 0
        for agent in range(actions.shape[1]):
            if actions[lane, agent] == target:
                correct += 1
        reward[lane] = qd.cast(correct, qd.f32) / actions.shape[1]


@qd.kernel(fastcache=True)
def signalling_commit(
    actions: qd.types.NDArray[qd.i32, 2],
    private: qd.types.NDArray[qd.i32, 1],
    delivered: qd.types.NDArray[qd.i32, 1],
    reward: qd.types.NDArray[qd.f32, 1],
    phase: qd.i32,
) -> None:
    for lane in range(private.shape[0]):
        reward[lane] = 0.0
        if phase == 0:
            delivered[lane] = actions[lane, 0]
        else:
            if actions[lane, 1] == private[lane]:
                reward[lane] = 1.0
            delivered[lane] = 0


@qd.kernel(fastcache=True)
def habitat_commit(
    position: qd.types.NDArray[qd.f32, 2],
    theta: qd.types.NDArray[qd.f32, 1],
    tumble: qd.types.NDArray[qd.i32, 1],
    latent: qd.types.NDArray[qd.f32, 2],
    centers: qd.types.NDArray[qd.f32, 2],
    motor: qd.types.NDArray[qd.i32, 1],
    hazard_u: qd.types.NDArray[qd.f32, 1],
    heading_u: qd.types.NDArray[qd.f32, 1],
    rotation_z: qd.types.NDArray[qd.f32, 1],
    field_z: qd.types.NDArray[qd.f32, 2],
    reward: qd.types.NDArray[qd.f32, 1],
    dt: qd.f32,
    domain: qd.f32,
    speed: qd.f32,
    rotational_diffusion: qd.f32,
    tau: qd.f32,
    high_hazard: qd.f32,
    low_hazard: qd.f32,
) -> None:
    for lane in range(theta.shape[0]):
        decay = qd.exp(-dt / tau)
        for patch in range(4):
            latent[lane, patch] = (
                decay * latent[lane, patch] + qd.sqrt(1.0 - decay * decay) * field_z[lane, patch]
            )
        if tumble[lane] > 0:
            tumble[lane] -= 1
            if tumble[lane] == 0:
                theta[lane] = heading_u[lane] * 6.283185307179586
        else:
            hazard = low_hazard
            if motor[lane] != 0:
                hazard = high_hazard
            if hazard_u[lane] < 1.0 - qd.exp(-hazard * dt):
                tumble[lane] = 1
            else:
                theta[lane] += qd.sqrt(2.0 * rotational_diffusion * dt) * rotation_z[lane]
                position[lane, 0] += speed * dt * qd.cos(theta[lane])
                position[lane, 1] += speed * dt * qd.sin(theta[lane])
                for axis in range(2):
                    position[lane, axis] -= domain * qd.floor(position[lane, axis] / domain)
        field = 0.05
        for patch in range(4):
            squared = 0.0
            for axis in range(2):
                delta = position[lane, axis] - centers[patch, axis]
                delta -= domain * qd.floor(delta / domain + 0.5)
                squared += delta * delta
            amplitude = 1.0 / (1.0 + qd.exp(-latent[lane, patch]))
            field += amplitude * qd.exp(-0.5 * squared)
        reward[lane] = field / 4.05
