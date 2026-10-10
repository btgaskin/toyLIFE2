"""Reproduce the historical spatial map and isolate canonical rule differences.

Run from the checkout with PYTHONPATH=src. Generated author variants stay in
qualification/; the pinned upstream checkout and production rules are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from toylife2.maps import _pattern_id, _spatial_protocol
from toylife2.spatial import ca_from_descriptors
from toylife2.specs import SpatialSpec
from toylife2.tables import Tables

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/toylife"
PIN = "9e17c92ea05f2e3dd63bf437400ddd8e61694ff9"
SOURCE_SHA = "33c7f29a37b1f5023c1942e010fa20de490333c5d964031eb0ec30ae6880b90a"
PAIRS = 2711**2
VARIANTS = (
    ("source-safe.json", "safe historical source"),
    ("source-break-fixed.json", "restore state-4 break"),
    ("source-identity-fixed.json", "break + identity"),
    ("variant-3.json", "break + binding eligibility"),
    ("variant-4.json", "break + eligibility + identity"),
    ("variant-5.json", "break + eligibility + identity + polymerase tie"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(directory: Path) -> dict:
    """Apply exact patches to copies; fail if the expected source has changed."""
    original = UPSTREAM / "patterns/patternsplugin.cpp"
    if sha(original) != SOURCE_SHA:
        raise ValueError("unexpected author source; review patches against the new source")
    safety = []
    source = original.read_text()
    for protein in ("p1", "p2"):
        before = (
            f"double energy_dimers2=dim_bond_energy[dimer2]+prot_energies[{protein}]"
            f"+prot_energies[{protein}];"
        )
        after = before.replace("=", "=dimer2 >= 0 ? ", 1).replace(";", " : 0.0;")
        if source.count(before) != 1:
            raise ValueError("memory-safety patch does not match uniquely")
        source = source.replace(before, after)
        safety.append({"kind": "memory_safety", "before": before, "after": after})
    start = source.index("std::vector<int> PatternsPlugin::logic_proms(")
    separator = "////////////////////////////////////////////////////////////////////////\n"
    end = source.index("//COARSENING", start) - len(separator)
    before = "    }//case 4\n    case 5:"
    after = "      break;\n    }//case 4\n    case 5:"
    fixed = source[start:end].replace("logic_proms(", "logic_proms_fixed(")
    if fixed.count(before) != 1:
        raise ValueError("fallthrough patch does not match uniquely")
    fixed = fixed.replace(before, after)
    eligible = fixed.replace("logic_proms_fixed(", "logic_proms_eligible(")
    admission = []
    pattern = r"double b[12]=std::min\(((?:prot|dim)_prom_energies\[i\]\[\w+\]).*?;"
    for match in list(re.finditer(pattern, eligible)):
        old = match.group(0)
        new = old.replace("=", f"={match.group(1)} < 0.0 ? ", 1)
        new = new[:-1] + " : 0.0;"
        eligible = eligible.replace(old, new, 1)
        admission.append({"before": old, "after": new})
    if len(admission) != 10:
        raise ValueError("expected ten candidate-admission expressions")
    generated = source[:start] + eligible + "\n\n" + fixed + "\n" + source[start:]
    target = directory / "patterns"
    target.mkdir(parents=True, exist_ok=True)
    (target / "patternsplugin.cpp").write_text(generated)
    header = (UPSTREAM / "patterns/patternsplugin.h").read_text()
    declaration = "  std::vector<int> logic_proms(int i, int p1, int p2);"
    header = header.replace(
        declaration,
        declaration + "\n  std::vector<int> logic_proms_fixed(int i, int p1, int p2);"
        + "\n  std::vector<int> logic_proms_eligible(int i, int p1, int p2);",
    )
    (target / "patternsplugin.h").write_text(header)
    for name in ("helper_functions.cpp", "helper_functions.h"):
        shutil.copyfile(UPSTREAM / name, directory / name)
    patches = {
        "upstream_commit": PIN,
        "source_sha256": SOURCE_SHA,
        "generated_sha256": sha(target / "patternsplugin.cpp"),
        "memory_safety_patches": safety,
        "fixed_variant_patch": {"before": before, "after": after},
        "eligibility_patches": admission,
    }
    write_json(directory / "patches.json", patches)
    return patches


def trajectories(rules: np.ndarray) -> np.ndarray:
    """Independent synchronous NumPy simulation of the published tissue assay."""
    state = np.zeros((len(rules), 31), dtype=np.int32)
    result = np.empty((len(rules), 100, 31), dtype=np.int32)
    for tick in range(100):
        incoming = state.copy()
        if tick == 0:
            incoming[:, 15] |= 2
        left = np.zeros_like(state)
        right = np.zeros_like(state)
        left[:, 1:] = incoming[:, :-1]
        right[:, :-1] = incoming[:, 1:]
        state = np.take_along_axis(rules, left * 16 + incoming * 4 + right, axis=1)
        result[:, tick] = state
    return result


def decode(raw: dict) -> dict:
    hist = {
        "".join(str((int(code) >> (2 * i)) & 3) for i in range(6))
        + str(int(code) >> 12): weight for code, weight in raw["histogram"].items()
    }
    desc = np.asarray([[int(c) for c in key] for key in hist], dtype=np.int32)
    rules = ca_from_descriptors(desc, "A")
    unique, inverse = np.unique(rules, axis=0, return_inverse=True)
    protocol = _spatial_protocol(SpatialSpec())
    patterns = [_pattern_id(frame, protocol) for frame in trajectories(unique)]
    logic, cas, pats = defaultdict(int), defaultdict(int), defaultdict(int)
    for (key, weight), rule, index in zip(hist.items(), rules, inverse, strict=True):
        logic[key[:6]] += weight
        cas["".join(map(str, rule))] += weight
        pats[patterns[index]] += weight
    return {
        "logic_histogram": dict(logic), "grn_histogram": hist,
        "ca_histogram": dict(cas), "pattern_histogram": dict(pats),
    }


def difference(left: dict, right: dict) -> dict:
    keys = left.keys() | right.keys()
    return {
        "common_classes": len(left.keys() & right.keys()),
        "left_only_classes": len(left.keys() - right.keys()),
        "right_only_classes": len(right.keys() - left.keys()),
        "changed_weight_bins": sum(left.get(k, 0) != right.get(k, 0) for k in keys),
        "weight_l1_distance": sum(abs(left.get(k, 0) - right.get(k, 0)) for k in keys),
    }


def historical_checks() -> dict:
    results = UPSTREAM / "patterns/results"
    mapping = dict(line.split() for line in (results / "GRN_ca_1.txt").read_text().splitlines())
    desc = np.asarray([[int(c) for c in key] for key in mapping], dtype=np.int32)
    compiled = ca_from_descriptors(desc, "A")
    mismatches = sum(
        "".join(map(str, rule)) != expected
        for rule, expected in zip(compiled, mapping.values(), strict=True)
    )
    ca_ids = dict(
        line.split() for line in (results / "cellular_automata.txt").read_text().splitlines()
    )
    ca_patterns = dict(
        line.split() for line in
        (results / "ca_phen_31_100_0_point_only_P1.txt").read_text().splitlines()
    )
    ca_ids = {rule: ca_id for rule, ca_id in ca_ids.items() if ca_id in ca_patterns}
    lines = (results / "phen_id_31_100_0_point_only_P1.txt").read_text().splitlines()
    saved = {}
    for offset in range(0, len(lines), 101):
        pattern_id = lines[offset].split()[0]
        saved[pattern_id] = np.asarray(
            [[int(c) for c in line] for line in lines[offset + 1:offset + 101]],
            dtype=np.int32,
        )
    rules = np.asarray([[int(c) for c in key] for key in ca_ids], dtype=np.int32)
    tissue_mismatches = sum(
        not np.array_equal(frame, saved[ca_patterns[ca_id]])
        for frame, ca_id in zip(trajectories(rules), ca_ids.values(), strict=True)
    )
    return {
        "descriptors_checked": len(mapping), "ca_mismatches": mismatches,
        "automata_checked": len(ca_ids), "saved_patterns": len(saved),
        "trajectory_mismatches": tissue_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "qualification/parity-source")
    parser.add_argument("--canonical", type=Path, default=ROOT / "evidence/full-map-cpu.json")
    parser.add_argument("--reuse-counts", action="store_true",
                        help="analyze existing counters; requires identical generated source")
    parser.add_argument("--sanitizer-smoke", action="store_true",
                        help="also compile and exercise selected intervals with ASan/UBSan")
    args = parser.parse_args()
    directory = args.output.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True,
    ).strip()
    if commit != PIN:
        raise ValueError("unexpected author checkout commit")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=UPSTREAM, check=True)
    previous = None
    if args.reuse_counts:
        previous = json.loads((directory / "patches.json").read_text())
    patches = prepare(directory)
    if previous is not None and previous != patches:
        raise ValueError("saved counters were generated with different source patches")
    command = [
        "clang++", "-std=c++11", "-O3", "-DNDEBUG", "-I", str(directory / "patterns"),
        str(ROOT / "benchmarks/parity_spatial.cpp"),
        str(directory / "patterns/patternsplugin.cpp"),
        str(directory / "helper_functions.cpp"), "-o", str(directory / "counter"),
    ]
    if not args.reuse_counts:
        subprocess.run(command, check=True)
    canonical = json.loads(args.canonical.read_text())["map"]
    tables = Tables.load(cache_dir=ROOT / "qualification/tables")
    if tables.digest != canonical["table_digest"]:
        raise ValueError("canonical evidence uses different lookup tables")
    if _spatial_protocol(SpatialSpec()) != canonical["spatial_digest"]:
        raise ValueError("canonical evidence uses a different tissue protocol")
    sanitizer = []
    if args.sanitizer_smoke:
        sanitized = command.copy()
        sanitized[2:4] = ["-O1", "-g", "-fsanitize=address,undefined",
                          "-fno-omit-frame-pointer"]
        sanitized[-1] = str(directory / "counter_sanitized")
        subprocess.run(sanitized, check=True)
        intervals = [(v, 0, 4096) for v in (0, 1, 3, 5)]
        intervals += [(5, a * 2711, (a + 1) * 2711) for a in (90, 174)]
        for variant, start, stop in intervals:
            output = directory / f"asan-{variant}-{start}.json"
            run = subprocess.run(
                [sanitized[-1], str(variant), str(start), str(stop), str(output)],
                cwd=UPSTREAM / "patterns", text=True, capture_output=True, check=True,
            )
            (directory / f"asan-{variant}-{start}.stderr.txt").write_text(run.stderr)
            sanitizer.append({"variant": variant, "start_pair": start, "stop_pair": stop,
                              "exit_code": run.returncode, "output_sha256": sha(output)})
    maps, rows = [], []
    for index, (filename, label) in enumerate(VARIANTS):
        path = directory / filename
        if not args.reuse_counts:
            print(f"Enumerating variant {index}: {label}", flush=True)
            subprocess.run(
                [str(directory / "counter"), str(index), "0", str(PAIRS), str(path)],
                cwd=UPSTREAM / "patterns", check=True,
            )
        raw = json.loads(path.read_text())
        if (raw["variant"], raw["start_pair"], raw["stop_pair"], raw["total_genotypes"]) != (
            index, 0, PAIRS, 1 << 40,
        ):
            raise ValueError(f"incomplete or mismatched counter: {path}")
        decoded = decode(raw)
        if any(sum(h.values()) != 1 << 40 for h in decoded.values()):
            raise ValueError("counter histogram does not conserve genotype multiplicity")
        maps.append(decoded)
        rows.append({
            "variant": index, "label": label, "counter_sha256": sha(path),
            "counts": {field.removesuffix("_histogram"): len(h) for field, h in decoded.items()},
            "canonical_comparison": {
                field: difference(h, canonical[field]) for field, h in decoded.items()
            },
        })
        print(label, rows[-1]["counts"], flush=True)
    checks = historical_checks()
    historical_keys = {
        line.split()[0] for line in
        (UPSTREAM / "patterns/results/GRN_abundance.txt").read_text().splitlines()
    }
    checks["grn_key_symmetric_difference"] = len(
        historical_keys ^ maps[0]["grn_histogram"].keys()
    )
    summary = {
        "schema": 1, "upstream_commit": PIN, "patches": patches,
        "driver_sha256": sha(ROOT / "benchmarks/parity_spatial.cpp"),
        "script_sha256": sha(Path(__file__)), "canonical_sha256": sha(args.canonical),
        "table_digest": canonical["table_digest"], "spatial_digest": canonical["spatial_digest"],
        "ordered_pairs": PAIRS, "total_genotypes_per_variant": 1 << 40,
        "counts_reused": args.reuse_counts, "historical_checks": checks, "variants": rows,
        "sanitizer_smoke": sanitizer,
        "historical_to_break_fixed": {
            field: difference(maps[0][field], maps[1][field]) for field in maps[0]
        },
        "adjacent_variant_comparisons": {
            f"{a}_to_{b}": difference(maps[a]["grn_histogram"], maps[b]["grn_histogram"])
            for a, b in ((0, 1), (1, 2), (1, 3), (3, 4), (4, 5))
        },
    }
    write_json(directory / "reconciliation.json", summary)
    if checks["ca_mismatches"] or checks["trajectory_mismatches"]:
        raise AssertionError("historical downstream comparison failed")
    if checks["grn_key_symmetric_difference"]:
        raise AssertionError("safe source does not reproduce historical GRN identities")
    if any(h != canonical[field] for field, h in maps[5].items()):
        raise AssertionError("combined diagnostic changes do not reproduce canonical histograms")


if __name__ == "__main__":
    main()
