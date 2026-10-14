"""Binary action sampling from molecular counts; chemistry remains canonical."""

from __future__ import annotations

import numpy as np

ROLE = "output:binary-counts-v1"


def bounded_draw(stream, indices, upper, *, panel, decision):
    """Unbiased uint32 rejection, addressed by episode, decision and attempt."""
    bounds, ids = np.broadcast_arrays(np.asarray(upper, np.uint64), np.asarray(indices, np.uint64))
    if np.any((bounds < 1) | (bounds > np.iinfo(np.int32).max)):
        raise ValueError("output count total must fit positive int32")
    threshold = (np.uint64(1 << 32) % bounds).astype(np.uint32)
    pending = np.ones(bounds.shape, bool)
    result = np.zeros(bounds.shape, np.uint32)
    attempt = 0
    while np.any(pending):
        word = stream.words(ids[pending], role=ROLE, trial=panel, event=decision, attempt=attempt)[
            ..., 0
        ]
        accepted = word >= threshold[pending]
        positions = np.flatnonzero(pending)[accepted]
        result.flat[positions] = word[accepted] % bounds.flat[positions]
        pending.flat[positions] = False
        attempt += 1
    return result


def observations(spec, readouts):
    """Resolve the two observed counts and their action-one probability."""
    counts = [np.asarray(readouts[name]) for name in spec.readouts]
    if counts[0].shape != counts[1].shape or any(
        a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > np.iinfo(np.int32).max)
        for a in counts
    ):
        raise ValueError("output observations require matching nonnegative integer counts")
    first, second = (a.astype(np.int64) for a in counts)
    total = first + second
    if np.any(total > np.iinfo(np.int32).max):
        raise OverflowError("output count total exceeds int32 admission")
    if spec.rule == "majority":
        a = np.where(first == second, 1, first > second).astype(np.int64)
        b = np.where(first == second, 1, second > first).astype(np.int64)
    else:
        a, b = np.where(total == 0, 1, first), np.where(total == 0, 1, second)
    probability = b / (a + b)
    if spec.permutation[0]:
        probability = 1 - probability
    return (
        a,
        b,
        {
            "output_counts": np.stack((first, second), axis=-1),
            "action_probability_one": probability,
        },
    )


def decode(spec, readouts, rng, decision):
    """Return actions and the observed counts/probability of action one."""
    a, b, diagnostics = observations(spec, readouts)
    draw = bounded_draw(rng.stream, rng.scenarios, a + b, panel=rng.panel, decision=decision)
    actions = (draw < b).astype(np.int32) ^ spec.permutation[0]
    return actions, diagnostics
