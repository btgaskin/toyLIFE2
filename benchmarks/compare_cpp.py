"""Compare unchanged author C++ routines with the public Quadrants APIs.

Run outside the sandbox for Metal: .venv/bin/python benchmarks/compare_cpp.py
Inputs, outputs, raw timings, build identity, and disagreements are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METS = ("00010001", "00000011", "01101010", "10100101")


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path = ROOT, timeout: int = 300,
         diagnostic: bool = False) -> dict:
    started = time.perf_counter()
    try:
        environment = {**os.environ, "TOYLIFE_BENCH_DIAGNOSTIC": "1"} if diagnostic else None
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=environment,
                                timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "process_seconds": time.perf_counter() - started}
    except subprocess.TimeoutExpired as error:
        return {"returncode": None, "stdout": str(error.stdout), "stderr": str(error.stderr),
                "error": "timeout", "process_seconds": time.perf_counter() - started}


def _worker(args: argparse.Namespace) -> None:
    from toylife2.results import ArrayResult
    from toylife2.runtime import Engine
    from toylife2.spatial import response_descriptors
    from toylife2.specs import ExecutionSpec
    from toylife2.tables import Tables

    genes = np.loadtxt(args.input, dtype=np.uint32, ndmin=2)
    started = time.perf_counter()
    tables = Tables.load()
    table_seconds = time.perf_counter() - started
    execution = ExecutionSpec(backend=args.worker, batch_size=args.batch_size,
                              max_steps=args.max_steps,
                              cache_dir=str(args.destination.parent / "cache" / args.worker))
    started = time.perf_counter()
    if args.worker == "cpu" and args.cpu_threads is not None:
        import quadrants as qd

        from toylife2._floating import preserve_host_floats

        Path(execution.cache_dir).mkdir(parents=True, exist_ok=True)
        with preserve_host_floats():
            qd.init(arch=qd.cpu, cpu_max_num_threads=args.cpu_threads, default_ip=qd.i32,
                    fast_math=False, enable_fallback=False, raise_on_templated_floats=True,
                    offline_cache=True, src_ll_cache=True,
                    offline_cache_file_path=execution.cache_dir)
    engine = Engine(tables, execution)
    initialization_seconds = time.perf_counter() - started

    def operation():
        if args.workload == "regulatory":
            return engine.regulatory(genes)
        if args.workload == "metabolic":
            return engine.metabolic(genes, METS)
        return response_descriptors(tables, genes, execution, engine=engine)

    started = time.perf_counter()
    first = operation()
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    last = operation()
    calibration_seconds = time.perf_counter() - started
    loops = min(256, max(1, math.ceil(args.seconds / max(calibration_seconds, 1e-9))))
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for _ in range(loops):
            last = operation()
        durations.append(time.perf_counter() - started)
    values = last.values if isinstance(last, ArrayResult) else last
    first_values = first.values if isinstance(first, ArrayResult) else first
    np.testing.assert_array_equal(values, first_values)
    status = {}
    if isinstance(last, ArrayResult):
        assert isinstance(first, ArrayResult)
        unique, counts = np.unique(last.status, return_counts=True)
        status = dict(zip(unique.tolist(), counts.tolist(), strict=True))
        np.testing.assert_array_equal(last.status, first.status)
    lang = engine.qd.lang
    assert lang is not None
    _write(args.destination, {
        "backend": args.worker, "workload": args.workload, "rows": len(genes),
        "genes": genes.shape[1], "batch_size": args.batch_size, "loops": loops,
        "table_load_seconds": table_seconds, "initialization_seconds": initialization_seconds,
        "first_seconds": first_seconds, "calibration_seconds": calibration_seconds,
        "durations_seconds": durations, "output": np.asarray(values).tolist(),
        "status_counts": status, "table_digest": tables.digest,
        "quadrants": engine.qd.__version__, "metabolites": METS,
        "cpu_max_num_threads": lang.impl.current_cfg().cpu_max_num_threads,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                  * (1 if sys.platform == "darwin" else 1024),
        "resident_table_bytes": engine._table_bytes,
    })


def _stats(result: dict) -> dict:
    samples = [value / result["loops"] for value in result["durations_seconds"]]
    return {"median_seconds": statistics.median(samples), "min_seconds": min(samples),
            "max_seconds": max(samples), "samples_seconds": samples,
            "iqr_seconds": float(np.quantile(samples, 0.75) - np.quantile(samples, 0.25)),
            "loops": result["loops"], "first_seconds": result["first_seconds"],
            "process_peak_rss_bytes": result.get("process_peak_rss_bytes"),
            "resident_table_bytes": result.get("resident_table_bytes")}


def _controller(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    upstream = ROOT / "upstream" / "toylife"
    sources = [ROOT / "benchmarks" / "cpp_author.cpp", upstream / "toy_plugin.cpp",
               upstream / "helper_functions.cpp", upstream / "patterns" / "patternsplugin.cpp"]
    original_hashes = {str(path.relative_to(ROOT)): _hash(path) for path in sources}
    snapshot = destination / "sources"
    snapshot.mkdir()
    for path in [*sources, ROOT / "benchmarks" / "compare_cpp.py",
                 ROOT / "benchmarks" / "cpp_repairs.py", upstream / "toy_plugin.h",
                 upstream / "helper_functions.h", upstream / "patterns" / "patternsplugin.h"]:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    from toylife2.records import _source_archive

    _source_archive(destination / "toylife2-source.zip")
    if args.cpp_variant == "iterator-safe":
        import difflib

        from cpp_repairs import repair_author

        original = sources[1].read_text()
        text = repair_author(original)
        (destination / "iterator-repair.diff").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile="upstream/toylife/toy_plugin.cpp", tofile="toy_plugin_iterator_safe.cpp")))
        patched = destination / "toy_plugin_iterator_safe.cpp"
        patched.write_text(text)
        sources[1] = patched
    binary = destination / "cpp_author"
    flags = ["-std=c++11", "-O3", "-DNDEBUG", "-I", str(upstream)]
    command = ["clang++", *flags, *(str(path) for path in sources), "-o", str(binary)]
    build = _run(command)
    _write(destination / "build.json", {"command": command, **build})
    if build["returncode"] != 0:
        raise RuntimeError("C++ build failed; see build.json")
    sanitizer = destination / "cpp_author_sanitized"
    sanitized_command = ["clang++", "-std=c++11", "-O1", "-g",
                         "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                         "-I", str(upstream), *(str(path) for path in sources),
                         "-o", str(sanitizer)]
    sanitized_build = _run(sanitized_command)
    _write(destination / "sanitizer-build.json", {"command": sanitized_command,
                                                 **sanitized_build})
    host = {name: _run(command) for name, command in {
        "hardware": ["sysctl", "hw.model", "hw.ncpu", "hw.memsize", "machdep.cpu.brand_string"],
        "power_thermal": ["pmset", "-g", "therm"],
        "compiler": ["clang++", "--version"],
        "upstream_commit": ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        "upstream_changes": ["git", "-C", str(upstream), "status", "--porcelain"],
    }.items()}
    metadata = {"schema_version": 1, "seed": 60213, "repeats": args.repeats,
                "target_repeat_seconds": args.seconds, "platform": platform.platform(),
                "python": sys.version, "host": host, "load_start": os.getloadavg(),
                "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
                "original_source_sha256": original_hashes, "cpp_variant": args.cpp_variant,
                "harness_sha256": _hash(Path(__file__)), "cases": [],
                "timing_boundary": "loaded tables and prepared inputs; public evaluation, "
                                   "allocation, transfers, and completed host outputs; "
                                   "excludes file IO",
                "baseline_threads": 1,
                "requested_cpu_threads": args.cpu_threads,
                "max_steps": args.max_steps,
                "selection": "fixed-seed unfiltered random genotypes; metabolic first two rows "
                             "are predeclared cleavage and cycle fixtures"}
    if args.agreement_subset_from:
        metadata["selection"] = "secondary postselected agreement cohort; all rows whose " \
                                "saved outputs agree across C++, CPU, and Metal"
        metadata["agreement_source"] = str(args.agreement_subset_from.resolve())
        metadata["agreement_source_sha256"] = _hash(
            args.agreement_subset_from / "comparison.json")
    cases = [(workload, n_genes, size) for workload in args.workloads
             for size in args.sizes for n_genes in args.gene_counts
             if workload != "spatial_descriptor" or n_genes == 2]
    for case_index, (workload, n_genes, size) in enumerate(cases):
        label = f"{workload}-g{n_genes}-n{size}"
        print(label, flush=True)
        folder = destination / label
        folder.mkdir()
        role = ("regulatory", "metabolic", "spatial_descriptor").index(workload)
        rng = np.random.default_rng(np.random.SeedSequence([60213, role, n_genes]))
        genes = rng.integers(0, 1 << 20, (size, n_genes), dtype=np.uint32)
        if workload == "metabolic":
            fixtures = np.zeros((2, n_genes), dtype=np.uint32)
            fixtures[:, :min(3, n_genes)] = np.asarray(
                [[907050, 1048465, 769001], [1005689, 644498, 941775]])[:, :n_genes]
            genes[:2] = fixtures
        selection = None
        if args.agreement_subset_from:
            origin = args.agreement_subset_from.resolve() / label
            original_genes = np.loadtxt(origin / "inputs.txt", dtype=np.uint32, ndmin=2)
            np.testing.assert_array_equal(genes, original_genes)
            cpp = np.asarray(json.loads((origin / "cpp.json").read_text())["output"])
            keep = np.ones(size, dtype=bool)
            for backend in ("cpu", "metal"):
                candidate = json.loads((origin / f"{backend}.json").read_text())
                if candidate["status_counts"].get("incomplete", 0):
                    raise ValueError("agreement selection requires complete source outputs")
                values = np.asarray(candidate["output"])
                if values.shape != cpp.shape:
                    raise ValueError("agreement source output shapes differ")
                keep &= np.all(cpp == values, axis=1)
            genes = genes[keep]
            selection = {"original_rows": size, "retained_rows": len(genes),
                         "excluded_rows": np.flatnonzero(~keep).tolist(),
                         "original_input_sha256": _hash(origin / "inputs.txt"),
                         "method": "postselected exact full-output agreement; no timing filter"}
            _write(folder / "selection.json", selection)
        inputs = folder / "inputs.txt"
        np.savetxt(inputs, genes, fmt="%d")
        case = {"name": label, "workload": workload, "genes": n_genes, "rows": len(genes),
                "input_sha256": _hash(inputs), "measurements": {}, "comparisons": {}}
        if selection is not None:
            case["selection"] = selection
        if sanitized_build["returncode"] == 0 and n_genes <= 16:
            log = _run([str(sanitizer), workload, str(inputs), "1",
                        str(folder / "sanitized.json"), "1"], cwd=upstream, diagnostic=True)
            _write(folder / "sanitizer-process.json", log)
            case["sanitizer_returncode"] = log["returncode"]
        elif n_genes > 16:
            case["cpp_coverage"] = "benchmark adapter coverage is capped at 16 genes"
        # Rotate measurement order to avoid giving one implementation every first slot.
        order = ["cpp", *args.backends]
        order = order[case_index % len(order):] + order[:case_index % len(order)]
        case["order"] = order
        for backend in order:
            output = folder / f"{backend}.json"
            if backend == "cpp":
                if case.get("sanitizer_returncode") != 0:
                    case["measurements"][backend] = {"error": "C++ diagnostic probe failed; "
                                                            "timed execution skipped"}
                    continue
                probe = folder / "cpp-probe.json"
                log = _run([str(binary), workload, str(inputs), "1", str(probe), "1"],
                           cwd=upstream)
                _write(folder / "cpp-probe-process.json", log)
                if log["returncode"] != 0:
                    case["measurements"][backend] = {"error": "C++ probe failed", **log}
                    continue
                first = json.loads(probe.read_text())
                sanitized = json.loads((folder / "sanitized.json").read_text())
                if first["output"] != sanitized["output"]:
                    case["measurements"][backend] = {"error": "optimized and diagnostic "
                                                            "C++ outputs disagree"}
                    continue
                case["cpp_diagnostic_output_agreement"] = True
                seconds = first["durations_seconds"][0]
                loops = min(256, max(1, math.ceil(args.seconds / max(seconds, 1e-9))))
                command = [str(binary), workload, str(inputs), str(args.repeats),
                           str(output), str(loops)]
                log = _run(command, cwd=upstream)
            else:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                           "--workload", workload, "--input", str(inputs), "--destination",
                           str(output), "--repeats", str(args.repeats), "--seconds",
                           str(args.seconds), "--batch-size", str(args.batch_size)]
                command += ["--max-steps", str(args.max_steps)]
                if args.cpu_threads is not None:
                    command += ["--cpu-threads", str(args.cpu_threads)]
                log = _run(command, timeout=args.timeout)
            _write(folder / f"{backend}-process.json", {"command": command, **log})
            if log["returncode"] != 0:
                case["measurements"][backend] = {"error": "evaluation failed", **log}
                continue
            result = json.loads(output.read_text())
            case["measurements"][backend] = {**_stats(result),
                                              "process_seconds": log["process_seconds"]}
        baseline = folder / "cpp.json"
        if baseline.exists():
            cpp = np.asarray(json.loads(baseline.read_text())["output"])
            for backend in args.backends:
                output = folder / f"{backend}.json"
                if not output.exists():
                    continue
                result = json.loads(output.read_text())
                candidate = np.asarray(result["output"])
                equal_shape = cpp.shape == candidate.shape
                mismatch_rows = np.flatnonzero(np.any(cpp != candidate, axis=1)) \
                    if equal_shape else np.arange(size)
                complete = not result["status_counts"].get("incomplete", 0)
                equivalent = equal_shape and not len(mismatch_rows) and complete
                ratio = case["measurements"]["cpp"]["median_seconds"] \
                    / case["measurements"][backend]["median_seconds"]
                case["comparisons"][backend] = {
                    "exact_output_agreement": equivalent, "mismatching_rows": len(mismatch_rows),
                    "mismatching_values": int(np.count_nonzero(cpp != candidate))
                                          if equal_shape else None,
                    "status_counts": result["status_counts"],
                    "same_input_time_ratio": ratio,
                    "equivalent_workload_speedup": ratio if equivalent else None,
                    "examples": [{"row": int(i), "genotype": genes[i].tolist(),
                                  "cpp": cpp[i].tolist(), "quadrants": candidate[i].tolist()}
                                 for i in mismatch_rows[:10]],
                }
        metadata["cases"].append(case)
        metadata["load_end"] = os.getloadavg()
        metadata["upstream_source_unchanged"] = all(
            _hash(ROOT / path) == digest for path, digest in original_hashes.items()
            if path.startswith("upstream/"))
        _write(destination / "comparison.json", metadata)
    print(destination / "comparison.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "qualification" / "cpp-benchmark")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "metal"), default=["cpu", "metal"])
    parser.add_argument("--cpp-variant", choices=("author", "iterator-safe"), default="author")
    parser.add_argument("--agreement-subset-from", type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 4096, 65536])
    parser.add_argument("--gene-counts", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--workloads", nargs="+",
                        choices=("regulatory", "metabolic", "spatial_descriptor"),
                        default=["regulatory"])
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.1)
    parser.add_argument("--worker", choices=("cpu", "metal"))
    parser.add_argument("--workload", choices=("regulatory", "metabolic", "spatial_descriptor"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.seconds <= 0 or min(args.sizes) < 2 or args.batch_size < 1:
        parser.error("use at least three repeats, positive timing/batch size, and sizes >= 2")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.worker:
        _worker(args)
    else:
        _controller(args)


if __name__ == "__main__":
    main()
