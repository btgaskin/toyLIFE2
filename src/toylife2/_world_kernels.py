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
