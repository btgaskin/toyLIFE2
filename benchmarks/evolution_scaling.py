"""Bounded canonical two-gene Wright–Fisher scaling, including host selection.

Run with PYTHONPATH=src .venv/bin/python benchmarks/evolution_scaling.py.
Each subprocess isolates RSS and compilation. Repeated trajectories use fresh
fitness caches and the same Model. Target is a fixed 31-cell, 16-tick stripe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from toylife2 import Model
from toylife2.evolution import wright_fisher
from toylife2.specs import EvolutionSpec, ExecutionSpec, RecordingSpec, RunSpec, SpatialSpec

CASES = [
    {"name": f"population_{n}", "population": n, "trials": 1, "batch": 256}
    for n in (128, 1024, 8192, 65536)
] + [
    {"name": "trials_16_large", "population": 8192, "trials": 16, "batch": 256},
    {"name": "trials_4", "population": 1024, "trials": 4, "batch": 256},
    {"name": "batch_32", "population": 1024, "trials": 1, "batch": 32},
    {"name": "batch_1024", "population": 1024, "trials": 1, "batch": 1024},
    {"name": "independent_initial", "population": 1024, "trials": 1,
     "batch": 256, "initial": "independent_random"},
]


class CountingModel:
    def __init__(self, model):
        self.model = model
        self.calls = self.evaluations = 0
        self.seconds = 0.0
        self.batch_sizes = []

    def patterns(self, genotypes, spatial):
        started = time.perf_counter()
        result = self.model.patterns(genotypes, spatial)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        self.evaluations += len(genotypes)
        self.batch_sizes.append(len(genotypes))
        return result


def output_hash(result):
    h = hashlib.sha256()
    for name in ("populations", "history"):
        array = np.ascontiguousarray(result["arrays"][name])
        h.update(str(array.shape).encode())
        h.update(array.dtype.str.encode())
        h.update(array.tobytes())
    h.update(json.dumps(result["summary"], sort_keys=True).encode())
    return h.hexdigest()


def worker(case, backend, repeats, generations):
    started = time.perf_counter()
    execution = ExecutionSpec(backend=backend, batch_size=case["batch"],
                              memory_budget_mib=512,
                              max_steps=generations * case["trials"],
                              cache_dir=".cache/toylife2-bench")
    model = Model(execution=execution)
    setup_seconds = time.perf_counter() - started
    spatial = SpatialSpec(cells=31, steps=16)
    target = np.zeros((16, 31), dtype=np.uint8)
    target[:, 10:21] = 1
    spec = RunSpec(operation="wright_fisher", seed=60213, spatial=spatial,
                   evolution=EvolutionSpec(trials=case["trials"],
                                           population_size=case["population"],
                                           generations=generations,
                                           mutation_probability=0.01,
                                           initial=case.get("initial", "monomorphic_random")),
                   execution=execution,
                   recording=RecordingSpec(mode="replayable", trace_budget_mib=1))
    runs = []
    for repeat in range(repeats):
        counted = CountingModel(model)
        started = time.perf_counter()
        result = wright_fisher(counted, spec, target)
        elapsed = time.perf_counter() - started
        # Each retained generation summary comes from one full-population weights call.
        requests = len(result["arrays"]["history"]) * case["population"]
        if result["summary"]["history_dropped"]:
            raise AssertionError("history budget must retain all benchmark generations")
        runs.append({"repeat": repeat, "seconds": elapsed,
                     "pattern_seconds": counted.seconds,
                     "host_and_cache_seconds": elapsed - counted.seconds,
                     "fitness_requests": requests,
                     "evaluated_genotypes": counted.evaluations,
                     "cache_saved_fraction": 1 - counted.evaluations / requests,
                     "pattern_calls": counted.calls,
                     "batch_sizes": counted.batch_sizes,
                     "status": result["status"], "summary": result["summary"],
                     "output_sha256": output_hash(result)})
    hashes = {row["output_sha256"] for row in runs}
    if len(hashes) != 1:
        raise AssertionError("identical trajectories changed across repeats")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lang = model.engine.qd.lang
    assert lang is not None
    return {"status": "complete" if all(r["status"] == "complete" for r in runs)
            else "partial", "case": {**case, "initial": spec.evolution.initial},
            "backend": backend,
            "genes": 2, "generations": generations, "mutation_probability": 0.01,
            "seed": 60213, "spatial": {"cells": 31, "steps": 16,
            "target": "state_1_at_cells_10_through_20_all_ticks"},
            "setup_seconds": setup_seconds, "host": platform.platform(),
            "quadrants_version": model.engine.qd.__version__,
            "cpu_max_num_threads":
            lang.impl.current_cfg().cpu_max_num_threads,
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "table_digest": model.tables.digest,
            "process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[x["name"] for x in CASES])
    parser.add_argument("--cases", nargs="+", choices=[x["name"] for x in CASES])
    parser.add_argument("--backend", choices=("cpu", "metal", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.generations < 1 or args.timeout <= 0:
        parser.error("repeats, generations and timeout must be positive")
    if args.case:
        if args.backend == "both":
            parser.error("worker --case requires cpu or metal")
        case = next(x for x in CASES if x["name"] == args.case)
        try:
            report = worker(case, args.backend, args.repeats, args.generations)
        except Exception as exc:
            report = {"status": "error", "case": case, "backend": args.backend,
                      "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(report, sort_keys=True))
        return
    rows = []
    backends = ("cpu", "metal") if args.backend == "both" else (args.backend,)
    for backend in backends:
        for case in CASES:
            if args.cases and case["name"] not in args.cases:
                continue
            command = [sys.executable, str(Path(__file__).resolve()), "--case", case["name"],
                       "--backend", backend, "--repeats", str(args.repeats),
                       "--generations", str(args.generations)]
            completed = None
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           timeout=args.timeout)
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                if completed.returncode:
                    row["returncode"] = completed.returncode
                if completed.stderr:
                    row["stderr_tail"] = completed.stderr[-2000:]
            except subprocess.TimeoutExpired:
                row = {"status": "timeout", "backend": backend, "case": case,
                       "timeout_seconds": args.timeout}
            except Exception as exc:
                row = {"status": "error", "backend": backend, "case": case,
                       "error": str(exc), "stdout_tail": completed.stdout[-2000:]
                       if completed is not None else ""}
            rows.append(row)
            print(f"{backend} {case['name']}: {row['status']}", file=sys.stderr, flush=True)
            report = {"benchmark": "canonical_two_gene_wright_fisher_scaling",
                      "timeout_per_case_seconds": args.timeout, "rows": rows}
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    complete = [r for r in rows if r["status"] == "complete"]
    groups = {}
    for row in complete:
        case = row["case"]
        key = (case["population"], case["trials"], case.get("initial", "monomorphic_random"))
        groups.setdefault(key, []).append(row["runs"][0]["output_sha256"])
    report["matched_batch_backend_outputs"] = all(len(set(h)) == 1 for h in groups.values())
    report["complete_cases"] = len(complete)
    report["requested_cases"] = len(rows)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
