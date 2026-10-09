"""Reproducible warm-call timings for the public phenotype paths.

Run from the repository root with ``PYTHONPATH=src python benchmarks/phenotypes.py``.
The first call at each shape captures compilation and cache loading. The
second call times the same shape, including host/device transfers and outputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from toylife2.maps import enumerate_map
from toylife2.runtime import Engine
from toylife2.spatial import patterns
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tables import Tables


def _timed(label: str, operation) -> dict[str, object]:
    start = time.perf_counter()
    first = operation()
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    second = operation()
    warm_seconds = time.perf_counter() - start
    a = first["total_genotypes"] if isinstance(first, dict) else first.values
    b = second["total_genotypes"] if isinstance(second, dict) else second.values
    if not np.array_equal(a, b):
        raise AssertionError(f"{label} changed across identical calls")
    return {"first_seconds": first_seconds, "warm_seconds": warm_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--map-pairs", type=int, default=16)
    parser.add_argument("--cache-dir", default=".cache/toylife2-bench")
    args = parser.parse_args()
    if args.samples < 1 or args.map_pairs < 1:
        parser.error("samples and map-pairs must be positive")
    started = time.perf_counter()
    tables = Tables.load()
    table_seconds = time.perf_counter() - started
    execution = ExecutionSpec(
        backend=args.backend,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        cache_dir=args.cache_dir,
    )
    rng = np.random.default_rng(60213)
    genes2 = rng.integers(0, 1 << 20, size=(args.samples, 2), dtype=np.int32)
    genes3 = rng.integers(0, 1 << 20, size=(args.samples, 3), dtype=np.int32)
    engine = Engine(tables, execution)
    result = {
        "backend": args.backend,
        "host": platform.platform(),
        "quadrants": engine.qd.__version__,
        "table_digest": tables.digest,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "map_pairs": args.map_pairs,
        "table_load_seconds": table_seconds,
        "regulatory": _timed("regulatory", lambda: engine.regulatory(genes3)),
        "metabolic": _timed(
            "metabolic",
            lambda: engine.metabolic(genes3, ("00010001", "00000011", "01101010", "10100101")),
        ),
        "patterns": _timed(
            "patterns",
            lambda: patterns(
                tables,
                genes2,
                SpatialSpec(cells=31, steps=64),
                execution,
            ),
        ),
        "weighted_map": _timed(
            "weighted_map",
            lambda: enumerate_map(
                tables,
                execution,
                stop_pair=args.map_pairs,
                tile_pairs=args.map_pairs,
            ),
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
