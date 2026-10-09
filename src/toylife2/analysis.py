"""Versioned finite-string complexity and exact periodicity summaries."""

from __future__ import annotations

import math

import numpy as np

from .results import ArrayResult

COMPLEXITY_PROTOCOL = "author-dictionary-terminal-v1"


def dictionary_count(sequence: str) -> int:
    """Author's incremental dictionary convention, including terminal insertion."""
    if not sequence or set(sequence) - {"0", "1"}:
        raise ValueError("complexity requires a nonempty binary string")
    dictionary = set(sequence)
    word = sequence[0]
    for symbol in sequence[1:]:
        combined = word + symbol
        if combined in dictionary:
            word = combined
        else:
            dictionary.add(combined)
            word = symbol
    dictionary.add(word + "$")
    return len(dictionary) - 1


def complexity(sequence: str) -> float:
    if not sequence or set(sequence) - {"0", "1"}:
        raise ValueError("complexity requires a nonempty binary string")
    scale = math.log2(len(sequence))
    if len(set(sequence)) == 1:
        return scale
    return 0.5 * scale * (dictionary_count(sequence) + dictionary_count(sequence[::-1]))


def bits(states: np.ndarray, width: int = 2) -> str:
    values = np.asarray(states)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 1 << width):
        raise ValueError("states must be integers fitting the bit width")
    return "".join(format(int(value), f"0{width}b") for value in values.ravel(order="C"))


def temporal_period(trajectory: np.ndarray) -> dict[str, int | None]:
    """First repeated complete tissue state in an autonomous trajectory.

    Callers must restrict to ticks after an imposed pulse; continuous forcing
    needs an augmented state that includes the forcing phase.
    """
    values = np.asarray(trajectory)
    if values.ndim != 2:
        raise ValueError("trajectory must have shape [time, cells]")
    seen: dict[bytes, int] = {}
    for tick, row in enumerate(values):
        key = row.tobytes()
        if key in seen:
            return {"transient": seen[key], "period": tick - seen[key]}
        seen[key] = tick
    return {"transient": None, "period": None}


def pattern_summary(patterns: np.ndarray, spec=None) -> dict:
    values = np.asarray(patterns)
    flat = values.reshape(-1, *values.shape[-2:])
    autonomous_start = 0
    if spec is not None and spec.morphogen == "explicit" and spec.injections:
        autonomous_start = max(tick for tick, _, _ in spec.injections)
    periods = []
    for row in flat:
        period = temporal_period(row[autonomous_start:])
        if period["transient"] is not None:
            period["transient"] += autonomous_start
        periods.append(period)
    return {"complexity_protocol": COMPLEXITY_PROTOCOL,
            "period_autonomous_start": autonomous_start,
            "complexities": [complexity(bits(row)) for row in flat],
            "periods": periods}


def neighbourhood_summary(base: ArrayResult, adjacent: ArrayResult) -> dict:
    """Summarize already evaluated point mutations without evaluating them again."""
    if not base.complete or not adjacent.complete:
        raise RuntimeError("robustness requires complete phenotypes")
    if adjacent.values.shape[:-2] != base.values.shape[:-1] \
            or adjacent.values.shape[-1] != base.values.shape[-1]:
        raise ValueError("base and neighbour phenotype dimensions differ")
    n_bits = adjacent.values.shape[-2]
    source = base.values.reshape(-1, base.values.shape[-1])
    nearby = adjacent.values.reshape(-1, n_bits, base.values.shape[-1])
    neutral = np.all(nearby == source[:, None, :], axis=-1)
    counts = []
    for row, mask in zip(nearby, neutral, strict=True):
        distinct, abundance = np.unique(row[~mask], axis=0, return_counts=True)
        counts.append({"phenotypes": distinct, "counts": abundance})
    return {"robustness": neutral.mean(axis=-1).reshape(base.values.shape[:-1]),
            "neutral_mutations": neutral.sum(axis=-1).reshape(base.values.shape[:-1]),
            "nonneutral_histograms": counts, "histogram_shape": base.values.shape[:-1]}
