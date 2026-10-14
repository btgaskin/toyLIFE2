"""Sample process memory beside running experiments without changing their records.

Example: python examples/monitor_molecular_runs.py --watch PID=results/my-run
RSS is sampled resident memory; brief peaks between samples can be missed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def process_stats(pids):
    result = subprocess.run(
        ["ps", "-p", ",".join(map(str, pids)), "-o", "pid=,rss=,pcpu=,etime=,lstart="],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError(f"process sampling failed: {result.stderr.strip()}")
    rows = {}
    for line in result.stdout.splitlines():
        pid, rss, cpu, elapsed, started = line.split(maxsplit=4)
        rows[int(pid)] = dict(
            rss_bytes=int(rss) * 1024,
            cpu_percent=float(cpu),
            process_elapsed=elapsed,
            process_started=started.strip(),
        )
    return rows


def progress(root, cached):
    """Read only committed checkpoint metadata, and only when it changes."""
    path = root / "evolution" / "manifest.json"
    if not path.exists():
        return cached
    try:
        manifest = json.loads(path.read_text())
        relative = manifest.get("artifacts", {}).get("checkpoint")
        if not relative or cached.get("checkpoint") == relative:
            return cached
        checkpoint = json.loads((root / "evolution" / relative).read_text())
        base = checkpoint["state"]["session_state"]["base"]
        return dict(checkpoint=relative, checkpoint_progress=base["total"])
    except (OSError, ValueError, KeyError):
        # Writers may be replacing metadata; try again at the next sample.
        return cached


def save_summary(root, summary):
    temporary = root / "memory-summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(root / "memory-summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="append", required=True, metavar="PID=RUN_DIRECTORY")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    watches = {}
    for value in args.watch:
        pid_text, directory = value.split("=", 1)
        pid, root = int(pid_text), Path(directory).resolve()
        if pid <= 0 or not root.is_dir():
            parser.error("each watch requires a positive PID and an existing run directory")
        if pid in watches:
            parser.error("each PID must be supplied once")
        watches[pid] = (root, {})
    initial = process_stats(watches)
    for pid, (root, _) in list(watches.items()):
        if pid not in initial:
            parser.error(f"PID {pid} is not running")
        summary_path = root / "memory-summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if previous and previous.get("process_started") != initial[pid]["process_started"]:
            parser.error(f"{root} already contains telemetry for another process")
        spec_path = root / "evolution" / "spec.json"
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        summary = dict(
            pid=pid,
            process_started=initial[pid]["process_started"],
            monitoring_started=previous.get("monitoring_started", datetime.now(UTC).isoformat()),
            sample_interval_seconds=args.interval,
            memory_budget_bytes=spec.get("execution", {}).get("memory_budget_mib", 0) * 1024**2,
            samples=previous.get("samples", 0),
            peak_sampled_rss_bytes=previous.get("peak_sampled_rss_bytes", 0),
            status="observing",
            measurement="ps RSS; sampled resident memory, not total allocation or an exact peak",
        )
        watches[pid] = (root, summary)
        source = root / "memory-monitor.py"
        if not source.exists():
            source.write_bytes(Path(__file__).read_bytes())
    cached = {}
    while watches:
        rows = process_stats(watches)
        now = datetime.now(UTC).isoformat()
        for pid, (root, summary) in list(watches.items()):
            row = rows.get(pid)
            if row is None or row["process_started"] != summary["process_started"]:
                summary.update(
                    status="process_exited" if row is None else "pid_reused", ended_at=now
                )
                save_summary(root, summary)
                del watches[pid]
                continue
            cached[pid] = progress(root, cached.get(pid, {}))
            sample = dict(timestamp=now, pid=pid, **row, **cached[pid])
            with (root / "memory.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            summary.update(
                samples=summary["samples"] + 1,
                last_sample=sample,
                peak_sampled_rss_bytes=max(summary["peak_sampled_rss_bytes"], row["rss_bytes"]),
            )
            save_summary(root, summary)
        if watches:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
