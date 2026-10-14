"""Versioned Philox streams addressed by scientific identity, independent of batching."""

from __future__ import annotations

import hashlib
import json

import numpy as np

SCHEME = "philox4x32-10-logical-v1"
_MASK = np.uint64(0xFFFFFFFF)


def philox(counter: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Random123 Philox4x32-10; final axes contain four and two uint32 words."""
    c = np.asarray(counter, dtype=np.uint32).astype(np.uint64)
    k = np.asarray(key, dtype=np.uint32).astype(np.uint64)
    if c.shape[-1] != 4 or k.shape[-1] != 2:
        raise ValueError("Philox requires four counter words and two key words")
    c, k = np.broadcast_arrays(c, np.concatenate((k, k), axis=-1))
    k = k[..., :2].copy()
    for _ in range(10):
        p0, p1 = c[..., 0] * np.uint64(0xD2511F53), c[..., 2] * np.uint64(0xCD9E8D57)
        c = np.stack(
            (
                (p1 >> np.uint64(32)) ^ c[..., 1] ^ k[..., 0],
                p1 & _MASK,
                (p0 >> np.uint64(32)) ^ c[..., 3] ^ k[..., 1],
                p0 & _MASK,
            ),
            axis=-1,
        )
        k = (k + np.asarray((0x9E3779B9, 0xBB67AE85), dtype=np.uint64)) & _MASK
    return c.astype(np.uint32)


class Stream:
    """Each draw has a role, trial, event, and stable individual index.

    SHA-256 maps the seed and role namespace to a 64-bit Philox key. Counters
    hold individual index (low/high) and rejection attempt (low/high).
    """

    def __init__(self, seed: int):
        if type(seed) is not int or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be a uint64 integer")
        self.seed = seed

    def key(self, *, role: str, trial: int = 0, event: int = 0) -> np.ndarray:
        """The two Philox key words; shared by host and resident apparatus."""
        if trial < 0 or event < 0:
            raise ValueError("logical indices must be nonnegative")
        namespace = json.dumps(
            [SCHEME, self.seed, role, trial, event], separators=(",", ":")
        ).encode()
        return np.frombuffer(hashlib.sha256(namespace).digest()[:8], dtype="<u4")

    def words(
        self,
        indices: np.ndarray | int,
        *,
        role: str,
        trial: int = 0,
        event: int = 0,
        attempt: int = 0,
    ) -> np.ndarray:
        if trial < 0 or event < 0 or attempt < 0 or attempt >= 1 << 64:
            raise ValueError("logical indices must be nonnegative and attempt must fit uint64")
        key = self.key(role=role, trial=trial, event=event)
        ids = np.asarray(indices)
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("individual indices must be nonnegative integers")
        ids = ids.astype(np.uint64)
        counter = np.empty((*ids.shape, 4), dtype=np.uint32)
        counter[..., 0] = ids & _MASK
        counter[..., 1] = ids >> np.uint64(32)
        counter[..., 2] = attempt & 0xFFFFFFFF
        counter[..., 3] = attempt >> 32
        return philox(counter, key)

    def integers(self, upper: int, indices: np.ndarray | int, **namespace) -> np.ndarray:
        """Exact unbiased bounded integers using independent rejection counters."""
        if type(upper) is not int or not 1 <= upper <= (1 << 64) - 1:
            raise ValueError("upper must be in [1, 2**64-1]")
        ids = np.asarray(indices)
        answer = np.empty(ids.shape, dtype=np.uint64)
        pending = np.ones(ids.shape, dtype=bool)
        limit = ((1 << 64) // upper) * upper
        attempt = 0
        while np.any(pending):
            words = self.words(ids[pending], attempt=attempt, **namespace)
            values = words[..., 0].astype(np.uint64) | (words[..., 1].astype(np.uint64) << 32)
            accepted = np.ones(values.shape, bool) if limit == 1 << 64 else values < limit
            positions = np.flatnonzero(pending)
            answer.flat[positions[accepted]] = values[accepted] % np.uint64(upper)
            pending.flat[positions[accepted]] = False
            attempt += 1
        return answer

    def uniform(self, indices: np.ndarray | int, **namespace) -> np.ndarray:
        """Binary64 open-interval uniforms, suitable for event waiting times."""
        words = self.words(indices, **namespace)
        value = words[..., 0].astype(np.uint64) | (words[..., 1].astype(np.uint64) << 32)
        # 52 bits + half a unit are exactly representable even at the top end.
        return ((value >> np.uint64(12)).astype(np.float64) + 0.5) / (1 << 52)

    def genotypes(
        self, count: int, genes: int, *, trial: int = 0, role: str = "initial_genotypes"
    ) -> np.ndarray:
        ids = np.arange(count * genes, dtype=np.uint64).reshape(count, genes)
        return (self.words(ids, role=role, trial=trial)[..., 0] & 0xFFFFF).astype(np.uint32)
