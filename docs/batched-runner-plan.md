# Nested batched runner design and implementation

Status: Schema version 2 session, evaluator and batch APIs are implemented. The
legacy version 1 runner remains available. CPU acceptance tests exercise shared
candidate evaluation, population wavefronts, contextual matchups, bounded logical
work quotas and portable checkpoint continuation. See [evolution.md](evolution.md)
and [records.md](records.md) for the implemented contracts.

## Implemented API

`BatchRunner` owns one Model/Engine and its immutable lookup pack. It constructs
chemical, tissue, rollout and population sessions and reusable evaluators. A
`PopulationSession` stores populations as `[replicate, individual, gene]`, with
independent lane cursors and observation validity. Its scheduler gathers current
unobserved lanes, evaluates duplicate genomes in bounded candidate chunks, and
advances each ready replicate at most once per step. `max_active_lanes` and
`batch_size` are execution settings; they do not enter random addresses.

`Evaluator.evaluate(genotypes, candidate_ids=None, panel=0, identities=None,
agent_axis=None)` preserves candidate axes and ID order. Pure candidate objectives
use complete-result deduplication and a bounded FIFO cache. Episodic cohorts retain
their own pending rollout state under request permutation and checkpoint resume.
Population-scoped signalling scores freeze the roster and batch bounded matchup
cohorts across balanced roles before a selection barrier.

The scientific RNG addresses remain the original seed/role/replicate/event/
individual addresses for legacy transitions. Generation-specific episode panels
have their own exogenous scenario addresses. Exact rational selection and a
canonical pinned last-K history make completed CPU outputs independent of cohort
width and observation arrival order.

The CLI includes version 2 temporal runs and rectangular/named batch definitions.
Records pin input bytes, lookup tables and custom-component source/dependencies.
Custom-world Python boundaries and user codecs remain explicit host execution;
built-in world arrays and chemical phase kernels use the selected Quadrants
backend. CPU acceptance establishes the tested contracts; Metal qualification and
performance measurements remain separate tasks.

## Earlier design rationale

The following sections retain the reviewed design discussion. Proposed options
such as `execution.evolution_schedule` describe that earlier interface proposal;
the implemented version 2 population API selects its wavefront through sessions
and uses `max_active_lanes` for its physical cohort bound.

## Purpose and execution model

Provide the useful semantics of `vmap`: independent simulation lanes have
explicit batch axes, share compatible computation, and retain their own state
and random streams. Implement this through mapped simulation primitives and
an explicit scheduler. Quadrants kernels already execute many lanes; wrapping
the current Python trial loop in another Python loop would retain its dispatch
overhead. An arbitrary Python-function tracing transform is outside this design.

For Wright–Fisher, population state has shape
`[*experiment_axes, trials, individuals, genes]`. Selection reduces only the
individual axis within one trial. Genes form the final structural axis.
Generation is an ordered scan over state, rather than an independent mapped
axis. Tissue time is a second ordered scan inside the phenotype assay.

For ten trials of one hundred individuals with two genes, one population
wave is `[10, 100, 2]`. Each of one hundred generation transitions scores the
active population lanes, batches the unique missing genotypes, then advances
each trial independently. The run observes generation zero and the final
generation: 100,000 offspring updates and 101,000 individual score requests.
Caching and deduplication determine actual phenotype evaluations. A wave may
require several kernel chunks; no assumption of one kernel launch is needed.

## Recommended public interface

1. Add `execution.evolution_schedule = "sequential" | "wavefront"`.
   Initially retain `sequential` as the default and make wavefront explicit.
   Existing configurations and source-pinned records continue to describe the
   current algorithm. After acceptance, evaluate a default change separately.
2. Introduce `EvolutionBatch` for array state and immutable lane identities,
   and a reusable `BatchRunner` context that owns one Model/Engine, table pack,
   fitness caches, memory planner, and recording adapters. Homogeneous input
   supports arbitrary leading axes. Validate shapes and axis ownership once.
3. Add a recorded `run_batch` entry point for an explicit rectangular array
   or keyed collection of RunSpecs. Preserve one child run per scientific
   experiment and a batch manifest mapping axes, member IDs, records, and
   status. A workspace remains a collection of runs; its directory order has
   no scheduling or random-stream meaning.
4. Add TOML batch definitions and `toylife2 batch validate/run/inspect/resume`.
   Start with explicit named members and referenced existing run TOMLs, then
   add deterministic grid expansion. Reject implicit array broadcasting of
   scientific settings. CLI batch options set execution defaults; member
   scientific values remain explicit and recorded.

Separate batch physical controls from member work quotas. A `BatchExecutionSpec`
owns backend placement, the shared memory/cache budget, unique-genotype chunk
size, and active-lane limit. Each child retains its own `max_steps`/`max_events`
quota per invocation. Resolve physical overrides once during validation and
record every child's effective execution settings. Reject conflicting backend
requests in a single Engine context; an outer coordinator can partition them
into separately initialized workers. The initial API has no implicit aggregate
batch transition cap.

Proposed Python usage, rather than an existing API:

```python
with BatchRunner(execution=execution) as runner:
    result = runner.run_batch(
        {"condition_a": spec_a, "condition_b": spec_b},
        workspace="study",
    )
```

Every member retains its seed and `trial_start`. A batch axis is an arrangement
of members, not a source of randomness. Named dimension metadata accompanies
results, but numerical hot paths use NumPy arrays and simple offsets.

The layout records an explicit coordinate-to-member map, for example
`(condition="a", repeat="r2") -> "a/r2"`, then each member's trial IDs.
Coordinates remain labels with a declared order. Moving an array position
does not change a member identity, seed, or trial range. Grid expansion
requires declared seeds/trial ranges for each member; using a repeated seed
and trial range is an explicit matched-stream choice.

## Homogeneous wavefront scheduler

Extract one trial transition into a tested pure step: current population plus
integer fitness weights and logical RNG addresses produce offspring, parent
indices, and mutation positions. The sequential runner calls this same step,
so there is one selection and mutation implementation.

Replace the scalar trial/generation cursor with arrays of logical trial IDs,
generation cursors, population revisions, observed horizons, arrivals, and
statuses. A scheduler selects ready lanes, gathers their populations, obtains
fitness, and scatters weights using original individual order. Each ready
trial advances by at most one transition in a wave. Completed or blocked lanes
are masked out. Resumed lanes can have different generations; fitness is a
pure assay and can batch them together.

Represent population generation, observed-through generation, and terminal
status independently. A lane at its final population generation is ready for
one final observation; that observation consumes no transition quota. Mark it
complete only after the observation and required recording commit. Arrivals,
history, horizons, and scientific status update at observation commit, while
speculative evaluation updates only execution/cache state.

Use `Fitness.weights` on `[active_trials, individuals, genes]` for the first
slice. Preserve its complete-result requirement, integer denominator, and
bounded insertion-ordered cache. A true LRU policy is a separate later tuning
decision. Deduplicate across the gathered trial set before chunking.
Where feasible, gather through offsets rather than repeatedly stacking all
populations; use bounded reusable work buffers for necessary copies.

Keep exact per-trial int64 cumulative weights, overflow checks, unbiased
bounded parent draws, `searchsorted(side="right")`, zero-sum uniform selection,
and one-bit mutation unchanged. Retain original individual order for prefix
sums. Host selection is the initial implementation. Genome/state arrays stay
on the host at this stage; the phenotype assay uses Quadrants CPU or Metal.

`max_steps` remains the total number of committed trial-generation transitions
per child invocation, so the example needs 1,000 transitions. At a quota smaller
than one wave, advance a deterministic subset. Persist a stable round-robin
cursor so repeated small-budget resumes distribute progress across lanes.
Identify this cursor by stable member and logical-trial IDs, with a defined
canonical ID order, rather than a mutable flattened slot. Adding or reordering
members does not redistribute an existing child's quota. A child resume gets
a new invocation quota; crash replay restores committed counters and cannot
spend transitions twice within the same invocation.
Speculative fitness evaluation spends no transition quota. Its cache and
metrics are execution state, and an evaluated population becomes an observed
generation only when the observation is committed.

## Random identity and equivalence

Keep `philox4x32-10-logical-v1` byte-for-byte: each original member's seed,
role, `trial_start + trial`, generation/event, original individual index, and
rejection attempt identify a draw. Do not add batch position or experiment ID
to the existing namespace. Two members using identical seeds and trial IDs
deliberately share random streams, which can provide matched comparisons.
Independent repeats use explicit different seeds or disjoint trial ranges.

Batch member IDs are immutable bookkeeping identities, separate from physical
array positions. Reordering members, chunks, or lanes never renumbers logical
trials or individuals. New vectorized RNG helpers construct the same existing
keys for each lane and broadcast counters; they must match the scalar Stream
on rejection cases and uint64 bounds before replacing per-lane host calls.

Completed sequential and wavefront runs must match populations, exact weights,
parents, mutations, arrivals, horizons, trial statuses, and history values.
Sort output history by original trial and generation, preserving current
trial-major results. Pin the history retention K at record creation from its
declared recording/resource budget, and store it in every checkpoint. Retain
the bounded canonical last-K rows and `history_dropped` count despite different
observation order. Store each lane's observed-through generation separately
and maintain retained rows incrementally in a bounded keyed structure, such
as a min-heap plus row map keyed by `(trial, generation)`. Do not truncate
arrival-order history and sort it afterward: canonical rows may already have
been discarded. Count unique observations independently of retained rows,
then derive `history_dropped` from that count. Store the observed cursor
so rescoring a checkpoint population cannot duplicate an observation whose
history row has been dropped. Resuming with another memory or trace budget
does not change K or restore dropped rows; admission must still accommodate
the pinned retained history. An explicit future retention-policy change
must record that earlier dropped rows remain unavailable. Partial runs
report each trial's actual cursor and status; several trials can be partial
simultaneously. Scheduling determines the frontier reached at a work cap.
It does not change a completed trajectory.

## Compatible groups and ragged experiments

Initially batch trials of one RunSpec. Then let the outer coordinator group
members with an identical fitness context: operation, rules and table digest,
gene count, tissue geometry, initial/injection/diffusion protocol, target
content hash, and exact fitness floor. Input path and member name are excluded
from numerical cache identity. A cache of weights must include this context
and the ordered genotype; weights from different targets cannot be reused.

Use per-member denominators and exact weights. A group may share phenotype
work when its members have different mutation rates, horizons, or populations;
those settings belong to the transition step. Shared seeds never imply shared
mutable populations. Incompatible groups dispatch independently through the
same admitted backend context. A process uses one initialized Quadrants
backend; CPU and Metal groups need separate processes.

Unequal populations use flat `[total_individuals, genes]` storage plus lane
offsets and lengths. Parent selection operates within each segment. Different
gene counts or tissue shapes use separate groups, rather than padding large
genomes or tissues to the maximum size. Rectangular results retain their named
axes; ragged results retain offsets, member/trial IDs, and typed child arrays.

Only later add target-independent phenotype deduplication: members with the
same tissue protocol can share one pattern evaluation and score it against
several targets. Score and release each bounded pattern chunk immediately.
Avoid retaining a full-trajectory cache for every genotype by default.

## Resource planning and throughput

Separate `max_active_lanes` from `batch_size`: the former bounds trial state
in one scheduling wave, while the latter caps unique genotype evaluation
chunks. Admit populations, next-population copies, offsets, weight and prefix
arrays, RNG temporaries, caches, histories, device tables, kernel scratch,
uploads/downloads, and pending recording data against a shared budget. Model
current peak buffer lifetimes, rather than only final result sizes.

Stream large studies through bounded active cohorts. Persist dormant member
state and its cursor before eviction; a resume selects the same scientific
streams regardless of cohort width. Chunking must never alter population
selection or omit individuals. Use compact ordered two-gene keys where useful;
retain general ordered gene tuples for higher-gene assays.

Start with one worker per GPU, resident immutable tables, cross-trial
deduplication, and bounded reusable buffers. CPU worker count and thread count
require explicit admission to avoid oversubscription. Do not share a mutable
Engine concurrently or silently launch a process per trial. CPU/Metal choice
remains explicit; record the actual backend and thread configuration.

Add counters for score requests, within-wave duplicate requests, cached reuse,
new genotype evaluations, chunks, launches, active lanes, memory admission,
and selection/assay/recording time. Define each count separately. Larger active
sets provide more useful work per dispatch. Store these counters as execution
metadata and exclude them from scientific-output equality checks. Runtime
also depends on genotype
diversity, cache reuse, tissue length, recording, and hardware. No timing
prediction is attached to the ten-by-hundred-by-hundred example.

The current five-or-more-gene metabolic path dispatches per genotype. Nested
runner scheduling alone cannot remove that inner bottleneck. Optimize reached
inventory packing as a separately validated kernel task, preserving padded
local phase indices, distinct Brent/exact-history modes, and memory admission.
The existing prototype is unintegrated and unvalidated. Two-gene spatial
evolution remains the first supported fitness protocol.

## Checkpoints, records, and full traces

Define a versioned evolution checkpoint payload with scheduler version,
immutable lane identities and layout, per-lane cursors/statuses, populations,
arrivals, committed observations/history, round-robin cursor, invocation IDs,
and committed quota counters. Fitness caches
are optional recomputable accelerators. The checkpoint must be sufficient
without them. Validate array dimensions, identity uniqueness, cursor bounds,
and compatibility before resuming. Recompute target and table content hashes.

Retain scalar-cursor readers and the sequential path for legacy state.
Source-pinned reproduce/resume rules continue to apply. Importing legacy state
into new source needs a dedicated validated conversion command, rather than
ordinary resume: current resume requires matching package-source hashes.
The converter verifies the old source/checkpoint and algorithm/RNG identity,
records old and new source hashes plus conversion version in a child record,
and preserves completed prefixes, retained history, observations, and each
trial's actual population. It derives the pinned K from the original record
and cannot recover dropped history. Unrecognized payloads fail conversion.
Execution may change chunk width or backend only through compatibility checks,
without modifying the source record.

For replayable mode, update in-memory state at each wave boundary. Make a
durable atomic checkpoint when an active lane reaches its configured
generation checkpoint interval, at a cohort eviction, or at completion,
quota exhaustion, or controlled interruption. Checkpoint the member's complete
current lane state at that boundary. This preserves checkpoint controls without
serializing every population after every generation. A crash restarts from
the last committed populations and logical cursors. Recomputing unfinished
waves produces the same draws. An assay failure leaves the last committed
state available; it never becomes zero fitness.

For full mode, extend the recorder to accept an ordered list of transition
events in one boundary transaction. Pair each event with its lane ID and
generation, parents, mutations, exact weights, and before/after populations.
Serialize in deterministic wave/lane order; expose canonical trial/generation
keys for comparison with legacy trial-major traces. Trace byte ordering is
versioned execution metadata and will differ between schedules.

Choose the transition prefix whose required full rows fit the trace budget
before advancing those lanes. A pending terminal/observation row must also fit
or remain explicitly pending. Commit that prefix and its resulting state
atomically; do not advance transitions whose required rows were excluded.
Persist pending prepared work only if needed, with a defined replay contract.
The first wavefront release supports replayable mode; explicit wavefront/full
combinations raise a clear validation error until this recorder is implemented.

For several child records, atomicity applies per child. A batch manifest is a
reconcilable index of child commits, never a promise of a cross-directory
transaction. Persist member failure/status independently and recover the
index from child manifests after a crash. Blocked or failed members are
explicitly reported; other independent members can continue by configured
failure policy. No result aggregate treats an unfinished member as complete.

## Implementation sequence and acceptance

1. Extract shared pure Wright–Fisher transition logic and exact observation
   handling. Existing sequential tests and full-trace behavior must pass.
2. Add homogeneous wavefront replayable mode, per-lane state/results, canonical
   bounded history, counters, and checkpoint payload. Keep explicit opt-in.
3. Verify exact sequential/wavefront parity at every generation with nonconstant
   fitness, both initialization modes, zero-weight fallback, mutation extremes,
   duplicate genotypes, cache eviction, and integer admission bounds.
4. Verify quota boundaries smaller than a wave, mixed resumed cursors,
   interruption/crash injection, backend and chunk changes, stable sharding,
   lane permutation with IDs preserved, history truncation, and malformed
   checkpoints. Use the independent scalar phenotype reference in small cases;
   CPU/Metal agreement alone cannot establish chemistry correctness.
5. Add keyed multi-experiment orchestration, compatible groups, ragged offsets,
   batch manifests, TOML/CLI, and per-member source/input/result capture. Test
   targets and floors that intentionally differ to expose cache contamination.
6. Add transactional multi-event full recording and trace-budget prefix tests,
   including terminal events and exact recovery with increased trace budgets.
7. Extend neutral walks by batching independent proposals and SSWM by batching
   resident/neighbour fitness. Each SSWM lane retains its own event time and
   pending event; no common time step is imposed. Retain current stable host
   floating-point acceptance calculations and random streams.
8. Consider batched RNG and GPU-resident selection only after profiling reveals
   a material host bottleneck. Device selection requires exact integer scans,
   existing Philox/rejection behavior, overflow handling, and accepted Metal
   integer support. It is an optional later backend, not a prerequisite for
   efficient cross-trial phenotype batching.

Sol reviews the implementation slices in parallel with independent parity and
record-recovery checks. Astra reviews the design and final scheduling,
identity, admission, and transaction changes. Performance tuning follows
correctness; structural counters can establish reduced dispatch and duplicate
work without publishing hardware-independent speed claims.

## Review resolutions

Astra's design review accepted the shared phenotype work plus exact host
transition mechanism. The plan incorporates its requested resolutions:
per-child quotas and stable scheduling IDs; explicit observation/terminal
state; pinned canonical history with deduplication cursors; interval-controlled
durable checkpoints; dedicated source-aware legacy conversion; and separate
transactional full-trace support. The current cache eviction policy is
preserved. Nested execution and larger-gene kernel optimization remain
separate implementation tasks. No additional workload timing was run for
this plan or its ten-trial example.
