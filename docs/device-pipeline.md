# Device placement and transfers

`backend="auto"` is the default for `ExecutionSpec`, new molecular profiles, and
the molecular CLI. Availability detection selects Metal first on Apple hardware
and CUDA first elsewhere, with native CPU as the fallback. Explicit `cpu`, `metal`,
and `cuda` selections stay strict. The concrete choice replaces `auto` when the
execution specification is constructed, so saved specifications capture placement
and continuation retains it. A process owns one backend; initialization and
kernel failures remain visible rather than changing placement mid-run.

## Chemotaxis decision pipeline

New molecular chemotaxis profiles declare `world_engine="quadrants"` and
`world_precision="float32"`. These settings enter the task identity. Older
undeclared configurations keep their NumPy/float64 world behaviour. Explicit
Quadrants/float64 worlds run on CPU; GPU backends reject that combination.

For replayable structural controllers with one cue and one motor, `run()` executes
up to eight complete decisions in one kernel. Each lane senses its current
exposure, runs the configured number of canonical chemical updates, reads bound
motor occupancy, and advances its world. Reward exposure becomes the next cue's
exposure. Block lengths stop at the horizon, work limits, or checkpoint boundary.

| Data | Placement and traffic |
| --- | --- |
| Molecular tables and compiled reaction graph | Uploaded at setup; immutable graph plans are shared across episode lanes. |
| World constants, wavevectors, source positions and scenario mapping | Uploaded at cohort setup and reused. |
| Inventories, expression, staged occupancy and compact motor readouts | Device resident through each decision block; state copies used for commit happen on the device. |
| Position, heading, phases, exposure and last action | Device resident with separate candidate and committed frames. |
| Random draws | Host Philox retains the original seed/panel/scenario/tick/role addresses and normal transform. One upload per block contains unique scenarios, shared by genotype lanes. |
| Rewards | One download per block, accumulated on the host in float64 and in tick order. |
| Diagnostics and full molecular state | Materialised when requested by `step()`, snapshots, checkpoints, or detailed recording. |

The engine owns a bounded random-block cache, capped at the smaller of 16 MiB and
one thirty-second of its memory budget. Shared admission can reclaim it. Buffers
are reused within a cohort; new cohorts can reuse graph and random plans. The
routine path avoids separate per-decision cue uploads, readout downloads, action
uploads, and repeated world-array transfers.

Warmup, reset-per-decision, custom codecs, descriptors, and full traces use the
checked decision path where their contracts require intermediate host data.
Held-out inspection preallocates trajectory arrays. Inspecting every decision
still deliberately downloads each requested diagnostic.

Prepared blocks survive checkpoint publication failure. The visible world stays
at the last committed boundary while the prepared chemistry and candidate world
remain available for retry. Checkpoints capture the numerical mode and backend.

## Other runners

| Runner | Execution and transfer boundaries |
| --- | --- |
| Regulatory truth tables | Quadrants parallelises genotype/state lanes in memory-admitted batches; one truth-table download per batch. |
| Metabolic assays | Small-genome assays run on Quadrants. Larger genomes batch reachable regulatory states and metabolite lanes; cycle detection downloads next expression bits. |
| Spatial pattern assays | A complete tissue trajectory runs within a device kernel, with one trajectory download per batch; morphogen data is uploaded once per call. |
| Interactive tissue sessions | Device state is double buffered, with injection upload and tissue download at each public step. |
| Episodic evaluation | Pure candidate requests are deduplicated, scenario panels are shared, and independent genome cohorts are batched. Admission failures retry smaller cohorts. |
| Population evolution | NumPy vectorises integer parent selection, mutation and fitness reductions. Phenotype evaluation uses the selected device; ready replicates sharing a panel are scored together. |
| Weighted map enumeration | Device promoter responses and spatial trajectories surround vectorised host descriptor construction, weighted counts, deduplication and histogram merging. |
| Named batch runs | A bounded scheduler advances child sessions fairly on one engine. Each child batches its own numerical lanes. |

All these runners use the shared execution specification for automatic GPU
placement. Small workloads can spend more time on setup and transfers than on
numerical work; select CPU explicitly when measuring that crossover.

## Qualification

Run correctness in fresh backend processes. The Modal CLI harness allocates one
ephemeral T4, verifies CUDA selection, runs the numerical contracts, and returns
logs, XML, package versions and source hashes under `qualification/modal-cuda/`:

```sh
.venv/bin/python tests/check_cpu.py -q
.venv/bin/python tests/check_metal.py
modal run tests/modal_cuda.py
```

The resident-world suite covers source/static/dynamic fields, multiple chemical
updates, block sizes and cohort widths, quotas, resume, checkpoint write failures,
warmup, reset, detailed traces, and fixed-action agreement with the NumPy reference.

For timing, compare matched float32 modes and identical genomes/panels. The
benchmark supports `--world-engine numpy|quadrants`, `--field source|static|dynamic`,
`--block-size`, cohort width, episodes, and complete generations:

```sh
.venv/bin/python benchmarks/runtime_optimisation.py --backend metal \
  --genes 10 --episodes 16 --updates 1000 --population 64 --repeats 7
```

Measure complete workloads in fresh processes after competing GPU workloads
finish. Use identical genomes, panels, precision, thread counts and memory budgets
for paired comparisons. Report setup, steady execution, checkpoint writes and
detailed inspection separately, and retain timing samples alongside outputs and
source fingerprints. The [recorded runtime measurements](runtime-optimisation.md)
cover an earlier protocol and identify their source revisions.
