# Phenotype execution and performance

`Engine` keeps immutable geometry and energy tables in Quadrants arrays and runs
one lane per genotype and regulatory state or metabolite assay. The Python API
returns arrays with the caller's leading batch dimensions. `ExecutionSpec`
controls backend, batch size, memory budget, kernel cache, and iteration caps;
these settings do not change genotype or scientific identity.

Regulatory chemistry and metabolic competition use integer tenths of an energy
unit. Thus candidate ordering and equal-energy annihilation have exact integer
comparisons on CPU and Metal. The CPU computes the small regulatory attractor
from the Quadrants transition table. In metabolic assays, a surviving metabolite
is evaluated through all attractor phases. Repeated phase-boundary inventories
stop intact cycles exactly. After each expression reset, at most one molecule
per gene can be produced, so the exact state history fits within
`comb(n_genes + n_genes + n_genes*(n_genes+1)//2, n_genes)` inventory states.
For three genes this is 220 entries per assay lane, even with the default
100,000-step cap. The memory budget admits and chunks workloads before array
allocation; a single assay that cannot fit raises `MemoryError`.

Spatial evaluation compiles each two-gene response descriptor to a 64-entry
local rule, then advances all cells synchronously in a Quadrants kernel.
Diffusion tracks free A, free B, and preformed AB separately. The `patterns`
function accepts an existing `Engine` to reuse resident tables across runs.
`SpatialSpec` selects an initial tissue and either the standard central
morphogen or explicit `(tick, cell, state)` injections. An injection adds its
state bits (B=1, A=2, both=3) before that tick's synchronous update. Continuous
and explicit schedules can inject free A/B into a cell holding preformed AB;
these trajectories use a 216-entry local rule over six possible input states.
The public 64-entry rule remains the summary over ordinary cell outputs 0–3.

`enumerate_map` tiles ordered pairs of the 2,711 folded protein identities.
It evaluates 16 A-promoter and 16 B-promoter responses per pair, combines
their independent outputs into 256 descriptors, and weights by the number of
coding sequences for each protein. The full interval covers exactly `2**40`
two-gene genotypes. A dedicated Quadrants kernel forms each pair's six fixed
molecular inventories once, then evaluates all 16 promoter sequences over
them. It accumulates six-output logic, seven-symbol GRN,
cellular-automaton, and tissue-pattern histograms. Pattern identifiers hash
the spatial protocol and the integer trajectory. Mapping each distinct
descriptor to its cellular automaton avoids constructing and deduplicating
256 full rule rows per pair. A repeated automaton reuses its trajectory.
The weighted NumPy accumulation uses float64 internally; each sum is at most
`2**40`, so its integer value is exactly representable. `start_pair` and
`stop_pair` provide deterministic, independently mergeable shards. A caller
can persist each tile through `checkpoint_callback` and restore it through
`resume_checkpoint`; `max_tiles` returns an explicit partial result with
the next pair cursor.

Run the repeatable local benchmark with:

```sh
PYTHONPATH=src python benchmarks/phenotypes.py \
  --samples 1024 --batch-size 256 --max-steps 100000 --map-pairs 256
```

The script uses a fixed seed and reports table loading, first-call latency,
and warm-call latency for regulatory, four-metabolite assays, 31-cell by
64-step patterns, and a weighted-map shard. Timings include output transfer.
The first call also includes specialization and cache work. A measured CPU
run on this macOS 26.4.1 arm64 host with Quadrants 1.3.3 and table digest
`a15ef34c9284b8b903da17156f534c887a2a3dab94b15e6985702baf26a341c7`
gave these warm wall-clock times:

| Workload | Work | Warm time |
|---|---:|---:|
| Regulatory | 1,024 three-gene truth tables | 0.0022 s |
| Metabolic | 1,024 three-gene genotypes × 4 metabolites | 0.0097 s |
| Patterns | 1,024 two-gene genotypes × 31 cells × 64 steps | 0.0141 s |
| Weighted map | 256 ordered protein pairs × 256 promoter pairs and patterns | 0.0050 s |

These timings characterize this warm sample and are not a throughput guarantee
for highly reactive metabolic genotypes. A complete CPU map run with 2,048
pairs per tile, 4,096 batch size, and a 1 GiB budget took 90.28 seconds on
this host. All four histograms summed to `2**40`; the run found 1,955 logic
signatures, 2,183 seven-symbol GRNs, 557 local CAs, and 191 tissue patterns.
The ignored `qualification/full-map-cpu.json` contains the full histograms,
rule-to-pattern associations, protocol and table digests, and execution
metadata. These are counts for the canonical ruleset in this package. The
[completed source comparison](parity.md) explains their difference from the
published totals through isolated promoter-rule changes.

The complete recorded Metal map took 79.09 seconds with the same 2,048-pair
tiles, 4,096 batch size, and 1 GiB budget. Its histograms and rule-to-pattern
associations match the complete recorded CPU run, which took 82.58 seconds.
These are single end-to-end observations, including host map aggregation
and recording; repeated matched measurements are required to estimate a
backend speedup. The records are `qualification/full-map-metal-record` and
`qualification/full-map-record`.

CPU and physical Metal checks cover
regulatory expression, metabolic fission and cycles, diffusion neighborhood
inputs, and exact weighted-map shards. See [qualification](qualification.md)
for the commands and saved evidence.

See [C++ comparison and scaling](cpp-benchmark.md) for matched regulatory
cohorts, unfiltered outputs, gene-size observations, and evolutionary reuse
counts. The [evolution guide](evolution.md) gives concrete trial/population
configuration; the [session runner](runner.md) describes joint trial scheduling
and bounded compatible experiment groups.

## Stateful experiment execution

Schema-2 chemistry batches independent controllers and scenarios into packed
persistent device buffers. Evolution scores ready replicate lanes through one
shared evaluator; pure repeated genomes reuse a bounded FIFO result cache.
Contextual games batch fixed matchups and retain complete-population barriers.
Named batches coalesce compatible fixed assays across child runs. Generations and
world time remain ordered state transitions. These mechanisms reduce redundant
work and enable batched kernel execution without prescribing a hardware speedup.

For control tasks, benchmark the complete perception–chemistry–action loop at the
intended gene count, scenario count, K chemical updates, and recording mode.
Resident chemotaxis keeps chemistry and world buffers on device across decision
blocks; detailed traces materialize observations. Account for transfers and
checkpoint costs when choosing CPU or Metal. Record cold compilation separately
from warm complete evaluation, include
validity and work counters, and retain identical logical seeds across comparisons.
Existing fixed-assay C++ timings describe their original recorded workloads.


## Shared numerical memory admission

One Engine accounts for persistent session buffers, resident lazy table packs,
fixed-assay outputs, and numerical scratch. Session release frees its reservation;
resident tables remain available for later assays. Table uploads preflight the
whole requested pack before allocating, so a failed admission preserves the
existing Engine buffers and counters. The budget covers managed numerical arrays;
Python interpreter overhead, driver/compiler allocations, and serialization are
outside this numerical accounting.

Use one `BatchRunner` for compatible work to share tables and runtime state. Named
fixed assays may coalesce into a larger cohort. Their child records preserve the
same result arrays and summaries as standalone runs, and record cohort size as
execution metadata. A named batch keeps the same molecular source pack on
recovery; ordinary child-record resume restores the archived pack automatically.

## Persistent molecular controllers

Chemistry sessions compile immutable host plans once per distinct genome,
controller specification, table digest, and explicit initial inventory. One Engine
shares these plans across sessions. Active batches pin their plans; an LRU retains
idle plans within the smaller of 32 MiB and one sixteenth of the execution budget.
Plan storage participates in Engine admission, including construction estimates.

Within a cohort, identical plans share device reaction and promoter metadata.
Each lane keeps its own inventory, expression, reaction diagnostics, and stage
counts. Reaction energy groups, promoter order, and readout selections are compiled
once. Readout kernels return the selected values without downloading whole
inventories. Host state is materialised when requested by inspection, full traces,
or checkpoint serialization.

Replayable execution groups up to eight chemical updates in one dispatch when a
conserved-mass bound proves count and demand safety for every update. The block
ends at the decision boundary, remaining work quota, or safe admission prefix.
Full traces retain one-update boundaries. When the conservative bound cannot
admit a block, the existing exact inventory admission handles the next update.
Pulse inputs enter the first update of a decision; replenished inputs are set at
each update. Readout stage and chemical work counters retain their original meaning.

Completed state and candidate state have separate buffers. Counts and diagnostics
commit together after successful readout. Rollouts retain committed and reset
chemistry in reusable device buffers, and capture a candidate before recording
publication. Failed transfers or retention leave the preceding committed state
available for inspection and retry. Checkpoints materialise these buffers through
the existing payload schema.

Scenario-addressed Philox draws compute each unique scenario address once and
expand to the original episode layout. The random stream, tick, role, width, and
component determine each draw independently of candidate ordering. Batched host
generation groups integer work across roles and decisions, with scratch tiled to
fit the shared memory budget. Immutable host blocks use a bounded cache.

New float32 chemotaxis profiles use `world_random_math="box_muller_f32_v1"`.
Direct generation uploads compact role keys and writes uniforms and normals
into the existing draw buffer. CUDA uses this path by default. CPU and Metal use
batched NumPy under the same declared mathematics. GPU buffers are reused without
retaining an additional device-block cache. Compact role-key tables share the bounded host cache across
cohorts, avoiding repeated namespace hashing for the same panel and decisions.
The older `numpy_f64_v1` transforms remain available and apply
when the setting is absent. Backend arithmetic can change trajectories; controls
check distributions, motion and consistent scenario coupling.

`benchmarks/chemotaxis_random.py` compares completed fixed-genotype rollouts using
`--provider host` and `--provider direct`. Use fresh backend processes and the same
scenario counts, fields and panel policy; `--reuse-panel` exercises repeated-cohort
reuse. Receipts separate first execution from warm samples and include source
hashes, upload/readback counts and backend identity. These measurements include
session construction and completed numerical publication, with disk recording
outside the timing interval. They launch no evolution.

The Metal default favours repeated-cohort reuse: direct generation improved
changing-panel timings but regressed the warm-cache guard beyond its 5% threshold.
The benchmark can select it explicitly. The CUDA direct path passed both the
changing-panel improvement gate and repeated-panel guard. Backend selection still
defaults to Metal/CUDA followed by native CPU; this policy selects random preparation
within that backend.

Use `benchmarks/runtime_optimisation.py` for cold construction, warm chemistry,
complete run/tumble decisions, and optional complete Wright–Fisher campaigns.
Run each backend in a fresh process. `--block-size 1`, `--metadata-layout replicated`,
and `--buckets` expose internal benchmark alternatives. Production uses shared
metadata, an eight-update maximum, and one unsplit cohort: measured size buckets
reduced buffer storage but increased execution time on the tested workloads.
See [runtime optimisation measurements](runtime-optimisation.md) for the recorded
workloads, raw samples, and validation.
