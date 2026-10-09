"""Ordered genes: four promoter bits followed by sixteen coding bits; 0=P, 1=H."""

from __future__ import annotations

import numpy as np


def encode(sequences: str | list[str] | tuple[str, ...]) -> np.ndarray:
    scalar = isinstance(sequences, str)
    seqs = [sequences] if scalar else list(sequences)
    if not seqs or not isinstance(seqs[0], str) or not seqs[0] or len(seqs[0]) % 20:
        raise ValueError("genotypes must contain a positive multiple of twenty bits")
    length = len(seqs[0])
    if any(len(s) != length or set(s) - {"0", "1"} for s in seqs):
        raise ValueError("genotypes must be equal-length binary strings")
    result = np.asarray([[int(s[g:g+20], 2) for g in range(0, length, 20)]
                         for s in seqs], dtype=np.uint32)
    return result[0] if scalar else result


def validate(genotypes: np.ndarray) -> np.ndarray:
    raw = np.asarray(genotypes)
    if raw.ndim < 1 or raw.shape[-1] < 1 or raw.size == 0:
        raise ValueError("genotypes require nonempty leading axes and a final gene axis")
    if raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= 1 << 20):
        raise ValueError("each gene must be an integer in [0, 2**20)")
    return np.array(raw, dtype=np.uint32, order="C", copy=True)


def decode(genotypes: np.ndarray) -> str | np.ndarray:
    genes = validate(genotypes)
    flat = genes.reshape(-1, genes.shape[-1])
    strings = ["".join(f"{int(g):020b}" for g in row) for row in flat]
    if genes.ndim == 1:
        return strings[0]
    return np.asarray(strings).reshape(genes.shape[:-1])


def mutation(genotype: np.ndarray, position: int) -> np.ndarray:
    genes = validate(genotype)
    if type(position) is not int or not 0 <= position < 20 * genes.shape[-1]:
        raise ValueError("mutation position is outside genotype")
    genes[..., position // 20] ^= np.uint32(1 << (19 - position % 20))
    return genes


def neighbours(genotypes: np.ndarray) -> np.ndarray:
    genes = validate(genotypes)
    return np.stack([mutation(genes, p) for p in range(20 * genes.shape[-1])], axis=-2)
