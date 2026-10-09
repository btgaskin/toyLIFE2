"""Run or resume exact 2**40 weighted-map qualification on Quadrants CPU.

From the repository root: ``PYTHONPATH=src python benchmarks/full_map.py``.
The ignored qualification directory holds an atomic checkpoint and the final
JSON report, including physical execution identity and all map histograms.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import quadrants as qd

from toylife2.maps import enumerate_map
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tables import Tables


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-pairs", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--memory-budget-mib", type=int, default=1024)
    parser.add_argument("--checkpoint-interval-tiles", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("qualification/full-map-cpu.json"))
    parser.add_argument("--cache-dir", default=".cache/toylife2-full-map")
    args = parser.parse_args()
    checkpoint_file = args.output.with_suffix(".checkpoint.json")
    checkpoint = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else None
    start = time.perf_counter()
    tables = Tables.load()
    execution = ExecutionSpec(
        backend="cpu",
        batch_size=args.batch_size,
        memory_budget_mib=args.memory_budget_mib,
        cache_dir=args.cache_dir,
    )
    spatial = SpatialSpec()

    def save(state: dict) -> None:
        _atomic_json(checkpoint_file, state)
        print(f"pair {state['next_pair']:,} / 7,349,521", flush=True)

    result = enumerate_map(
        tables,
        execution,
        spatial=spatial,
        tile_pairs=args.tile_pairs,
        checkpoint_interval_tiles=args.checkpoint_interval_tiles,
        checkpoint_callback=save,
        resume_checkpoint=checkpoint,
    )
    report = {
        "qualification": {
            "host": platform.platform(),
            "quadrants_version": list(qd.__version__),
            "backend": "cpu",
            "table_digest": tables.digest,
            "tile_pairs": args.tile_pairs,
            "batch_size": args.batch_size,
            "memory_budget_mib": args.memory_budget_mib,
            "checkpoint_interval_tiles": args.checkpoint_interval_tiles,
            "elapsed_seconds_this_invocation": time.perf_counter() - start,
        },
        "map": result,
    }
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "total_genotypes": result["total_genotypes"],
                "logic_classes": len(result["logic_histogram"]),
                "grn_classes": len(result["grn_histogram"]),
                "ca_classes": len(result["ca_histogram"]),
                "pattern_classes": len(result["pattern_histogram"]),
                "elapsed_seconds": report["qualification"]["elapsed_seconds_this_invocation"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
