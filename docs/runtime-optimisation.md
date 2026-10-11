# Runtime optimisation measurements

Measured on 2026-10-11 on this arm64 Mac, using one CPU thread and physical
Metal with backend fallback disabled. Baseline source is repository revision
`88e9f0f56ce4eb88add62ae941c2c4e5b6961337`, extracted into an isolated source tree.
The same benchmark script runs both source versions in fresh processes.

| Backend | Workload | Baseline median | Optimised median | Speedup |
|---|---|---:|---:|---:|
| CPU | 32 chemistry updates | 0.1305 s | 0.0176 s | 7.42× |
| CPU | 32 complete decisions, K=1 | 0.1471 s | 0.0445 s | 3.30× |
| CPU | Three-generation campaign | 1.0258 s | 0.1863 s | 5.51× |
| METAL | 32 chemistry updates | 0.1664 s | 0.0190 s | 8.77× |
| METAL | 32 complete decisions, K=1 | 0.2309 s | 0.0498 s | 4.64× |
| METAL | Three-generation campaign | 1.4373 s | 0.2584 s | 5.56× |

The chemistry and complete-decision workloads use 128 independent eight-gene
genomes, eight episodes per genome, and 32 updates/decisions. Chemistry receives
alternating zero/one doses of `0001`, one copy of `0111`, and reads bound `0111`
after output. Complete decisions use the existing run/tumble task and controller.
Five warm samples follow initial kernel materialisation. These measurements
include input upload, readout transfer, and synchronization; complete decisions
also include world updates and retained-state capture.

Campaigns use two trials, 128 individuals, two genes, eight episodes, 16 decisions
per evaluation, K=4 chemistry updates per decision, three generations, independent
random initialization, and mutation probability 0.2. Three campaigns use seeds
484–486. Timings include evaluation construction, world/chemistry execution,
fitness handling, selection, and mutation after benchmark kernels are warmed.
Final population digests and rollout score/position digests match the baseline
for every paired run. Scientific signalling interfaces remain a separate change.

## Internal choices

- Shared metadata: the eight-gene chemistry cohort reserves 7,948,304 bytes,
  versus 20,033,552 bytes with replicated device metadata, a 60.3% reduction.
  Baseline reservation was 9,089,024 bytes. These are admitted cohort bytes,
  including conservative construction/scratch allowance; Engine tables, shared
  host plans, and rollout retained-state buffers have separate accounting.
  The two-gene probe reserves 2,754,576 bytes versus the baseline 2,482,176 bytes:
  transactional diagnostics cost more here, where sharing offers less leverage.
- Eight-update blocks: at K=16, CPU complete decisions took 0.2014 s with blocks
  versus 0.3125 s with single updates. The initial sweep’s Metal block run varied noticeably;
  four alternating fresh-process follow-ups with ten samples each gave medians
  0.33035/0.32933 s for single updates and 0.12602/0.12605 s for eight-update blocks.
  Raw samples preserve outliers. Full recording and uncertain admission retain
  checked single updates.
- Size cohorts: at 1,024 eight-gene genomes and eight episodes, nine power-of-two
  cohorts reduced summed reservations from 66,830,352 to 35,542,600 bytes. CPU
  chemistry timing changed from 0.1332 to 0.1296 s, while Metal increased from
  0.0389 to 0.1445 s. Production retains one cohort. `--buckets` benchmarks
  chemistry cohorts; complete rollout timings in that mode still use one cohort.

These are measurements of the recorded workloads. First-process latency remains
separate in each JSON: existing on-disk kernel caches were retained, and a first
step can include new compilation. The initial sweep, retained in `first-sweep/`,
recorded first optimised chemistry steps of 0.817 s on CPU and 0.774 s on Metal.
The final sweep, using those existing caches, recorded 0.0569 s and 0.0101 s.
Subsequent warm samples are shown above.
Genomes with larger closures, inspection frequency, full recording, thread count,
and memory limits can change the balance.

## Reproduce and inspect

```sh
.venv/bin/python benchmarks/runtime_optimisation.py --backend cpu --threads 1
.venv/bin/python benchmarks/runtime_optimisation.py --backend metal --threads 1
.venv/bin/python benchmarks/runtime_optimisation.py --backend metal --threads 1 \
  --chemical-updates 16 --block-size 8 --repeats 10
.venv/bin/python benchmarks/runtime_optimisation.py --backend cpu --threads 1 \
  --genes 2 --updates 16 --chemical-updates 4 --generations 3 --repeats 3
```

For baseline comparisons, extract the recorded revision and use its source:

```sh
mkdir -p /tmp/toylife-runtime-baseline
git archive 88e9f0f src | tar -x -C /tmp/toylife-runtime-baseline
ln -s "$PWD/upstream" /tmp/toylife-runtime-baseline/upstream
PYTHONPATH=/tmp/toylife-runtime-baseline/src .venv/bin/python \
  benchmarks/runtime_optimisation.py --backend cpu --threads 1
```

Raw parameters, timing samples, hashes, cache metrics, kernel counts,
and reservations are in [benchmark results](../benchmarks/results/runtime-optimisation/).
The benchmark fingerprints sorted package Python sources:

- Baseline: `a7a921d7e39413d5bcc9bb2ebe37592dc371edd0b3252d6aef536d0245e84d10`
- Optimised: `7d6572d1dcb2ca49c04647bafed228b5361b1ea4b1183e0b2927c303cb372f93`

## Validation

- CPU suite: 246 passed, two subtests passed.
- Physical Metal contract suite: 80 passed, fallback disabled.
- CPU and Metal session qualification: temporal, run/tumble, signalling,
  population, pending-decision replay, and population replay passed.
- Wheel: installed tables and source archive match; installed replay passed.
- Ruff, Pyright, and whitespace checks passed.
- Astra reviewed the implementation and final allocation/retry fixes.

Added tests cover immutable plan reuse and independent episodes; complete random
addresses; scalar/block agreement for canonical and regulatory modes, pulse and
replenishment, and readout stages; partial quotas and admission failure prefixes;
full traces and wide expression; reusable warm reset state; failed readout and
retention transactions; field-selective snapshots; construction admission; and
idle-cache collection after Engine disposal.
