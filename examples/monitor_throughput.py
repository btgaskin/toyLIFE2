"""Write live throughput from committed molecular-run checkpoints.

Usage: python examples/monitor_throughput.py RUN_DIRECTORY --pid PID --launch RECEIPT
Reads small checkpoint arrays and process statistics; never touches GPU state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    root = args.run.resolve()
    spec = tomllib.loads((root / "profile.toml").read_text())
    launch = json.loads(args.launch.read_text())
    launched = timestamp(launch["launched_at"])
    initial = launch.get("initial_counters", {"generations": 0, "world_ticks": 0})
    target = spec["evolution"]["trials"] * spec["evolution"]["generations"]
    evaluation_work = spec["episode"]["episodes"] * spec["episode"]["horizon"]
    samples = [{"time": launched, **initial}]
    seen = set()
    latest = None
    while True:
        try:
            evolution = root / "evolution"
            manifest = json.loads((evolution / "manifest.json").read_text())
            for revision in manifest["checkpoint_revisions"]:
                if revision in seen:
                    continue
                path = evolution / "checkpoints" / revision / "checkpoint.json"
                checkpoint = json.loads(path.read_text())
                session = checkpoint["state"]["session_state"]
                payload = session["payload"]

                def array(reference, *, path=path, checkpoint=checkpoint):
                    return np.load(path.parent / checkpoint["arrays"][reference["__array__"]],
                                   allow_pickle=False, mmap_mode="r")

                cursors = array(payload["cursors"]).tolist()
                work = session["base"]["total"]
                latest = {"time": timestamp(checkpoint["created_at"]),
                          "generations": sum(cursors), "world_ticks": work["world_ticks"],
                          "chemical_updates": work["chemical_updates"],
                          "trial_generations": cursors, "checkpoint": str(path.relative_to(root))}
                samples.append(latest)
                seen.add(revision)
            process = subprocess.run(
                ["ps", "-p", str(args.pid), "-o", "rss=,pcpu=,etime="],
                capture_output=True, text=True, check=False)
            if process.stderr.strip():
                raise RuntimeError(process.stderr.strip())
            alive = bool(process.stdout.strip())
            if latest is not None:
                now = time.time()
                window = [row for row in samples if row["time"] >= latest["time"] - 300]
                previous = window[0] if len(window) > 1 else samples[max(0, len(samples) - 2)]
                duration = latest["time"] - previous["time"]
                generation_rate = ((latest["generations"] - previous["generations"]) / duration
                                   if duration > 0 else 0)
                world_rate = ((latest["world_ticks"] - previous["world_ticks"]) / duration
                              if duration > 0 else 0)
                elapsed = latest["time"] - launched
                remaining = max(0, target - latest["generations"])
                rss, cpu, process_elapsed = process.stdout.split() if alive else ("0", "0", "ended")
                complete = (root / "experiment.json").exists()
                phase = ("complete" if complete else "evolving" if remaining else "assessment")
                if not alive and not complete:
                    phase = "process_ended; inspect run log"
                report = {**latest, "updated_at": datetime.now(UTC).isoformat(), "pid": args.pid,
                          "phase": phase, "backend": spec["execution"]["backend"],
                          "progress_percent": 100 * latest["generations"] / target,
                          "world_decisions_per_second": world_rate,
                          "equivalent_genome_evaluations_per_second": world_rate / evaluation_work,
                          "trial_generations_per_minute": generation_rate * 60,
                          "mean_generation_seconds": spec["evolution"]["trials"] / generation_rate
                          if generation_rate else None,
                          "overall_world_decisions_per_second":
                          (latest["world_ticks"] - initial["world_ticks"]) / elapsed
                          if elapsed > 0 else 0,
                          "estimated_training_seconds_remaining": remaining / generation_rate
                          if generation_rate else None,
                          "checkpoint_age_seconds": now - latest["time"],
                          "rss_bytes": int(rss) * 1024, "cpu_percent": float(cpu),
                          "process_elapsed": process_elapsed,
                          "rate_window_seconds": duration}
                eta = report["estimated_training_seconds_remaining"]
                eta_text = f"{eta / 60:.1f} minutes" if eta is not None else "waiting for progress"
                text = (f"# Chemotaxis progress\n\nUpdated {report['updated_at']}\n\n"
                        f"- Phase: {phase}; backend: {report['backend']}\n"
                        f"- Generations per trial: {latest['trial_generations']} / "
                        f"{spec['evolution']['generations']}\n"
                        f"- Training progress: {report['progress_percent']:.1f}%\n"
                        f"- Recent throughput: {world_rate:,.0f} world decisions/s; "
                        f"{world_rate / evaluation_work:.1f} full-genome evaluations/s equivalent\n"
                        f"- Recent time per generation across all trials: "
                        f"{report['mean_generation_seconds'] or 0:.1f} s\n"
                        f"- Estimated training time remaining: {eta_text}\n"
                        f"- Sampled RSS: {int(rss) / 1024:.0f} MiB; CPU: {cpu}%\n"
                        f"- Checkpoint age: {report['checkpoint_age_seconds']:.0f} s\n\n"
                        "Rates use committed work over a window of up to five minutes. "
                        "ETA changes with genome diversity and reaction graph size; "
                        "final held-out assessment adds time.\n")
                atomic(root / "progress.json", json.dumps(report, indent=2) + "\n")
                atomic(root / "progress.md", text)
                with (root / "throughput.jsonl").open("a") as output:
                    output.write(json.dumps(report) + "\n")
                print(text, flush=True)
            if not alive:
                return
        except (OSError, ValueError, KeyError) as error:
            print(f"Progress temporarily unavailable: {error}", flush=True)
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
