# Evolution and sampling protocols

Schema version 1 evolutionary workflows use the two-gene tissue phenotype as their fitness
assay. A target is a `uint8` NumPy array with shape `[steps, cells]` and states
0–3. The run copies this file into its record and hashes its bytes. The score is
the fraction of matching cell-time states, with a configured `fitness_floor`:

`fitness(g) = max(matching_states / (steps * cells), fitness_floor)`.

The default tissue is the [2020 protocol](model.md): 31 open-boundary cells,
100 updates, A diffusion, and a central A pulse. A different tissue geometry,
injection schedule, target, or floor defines a different selection problem.
The accepted model choices and historical source comparisons are in
[decisions.md](decisions.md).

## Reproducible random streams

Quadrants initialization enables flush-to-zero on its calling thread. The
runtime saves and restores the host C floating-point environment around that
initialization, preserving IEEE subnormal handling for the Python SSWM math.
Chemistry kernels compare integer energies. Hosts without the standard C
floating-point environment API fail initialization explicitly.

`philox4x32-10-logical-v1` addresses each draw by the seed, role, logical trial,
event or generation, individual, and rejection attempt. A SHA-256 namespace
derives the Philox key. Uniform draws use 52 random bits plus half a unit to
lie strictly between zero and one. Bounded integer draws use rejection, so each
integer in the requested range has equal probability. Batch size, execution
backend, and checkpoint interval do not enter these addresses.

`trial_start` offsets logical trial IDs when independent runs shard an
experiment. Keep seed and scientific settings fixed and assign disjoint trial
ranges to obtain the same random streams as an unsharded run. The recorded RNG
scheme identifies the stream algorithm; a checkpoint with another scheme or
algorithm is rejected.

## Wright–Fisher

At each generation, the algorithm scores all individuals, records first target
arrival (including generation zero), and samples `population_size` parents with
replacement in proportion to fitness. Each offspring mutates with probability
`mutation_probability`; a mutating offspring flips exactly one uniformly
chosen bit among the `20 * genes` genome bits. Mutation is applied to every
selected offspring, including offspring from a zero-fitness parent when all
parent weights are zero.

Fitness is represented as an exact rational integer weight. The denominator is
the least common multiple of the number of target states and the decimal
fitness-floor denominator. Parent sampling uses integer cumulative weights;
the implementation checks signed 64-bit admission before accumulation. If
every weight is zero, parents are sampled uniformly and the mutation step still
runs. The recorded history rows contain trial, generation, mean fitness,
maximum fitness, and number of distinct genotypes. The checkpoint stores every
individual at a generation boundary.

`initial = "monomorphic_random"` starts each trial with one seeded founder;
`"independent_random"` samples each individual independently. An explicit
genotype can supply a founder. The starting population is part of the
scientific definition.

### Ten independent 100-individual trials

This Wright–Fisher configuration follows ten independent populations of 100
individuals for 100 generation transitions each. Supply `target.npy` as a
`uint8` array of shape `[100, 31]` with tissue states 0–3; the target and
spatial settings together define the fitness assay.

```toml
schema_version = 1
operation = "wright_fisher"
seed = 20200603

[model]
genes = 2

[spatial]
cells = 31
steps = 100
boundary = "open"
diffusion = "A"
morphogen = "central_A_pulse"
target = "target.npy"

[evolution]
trials = 10
population_size = 100
generations = 100
mutation_probability = 0.01
initial = "monomorphic_random"

[execution]
backend = "cpu"
batch_size = 256
memory_budget_mib = 1024
max_steps = 1000

[recording]
mode = "replayable"
```

Save this as `wf-10x100x100.toml`, then run
`toylife2 validate wf-10x100x100.toml` and
`toylife2 run wf-10x100x100.toml --destination wf-10x100x100`.
The target path must resolve when the run starts; the record copies and hashes
the target. Choose `initial = "independent_random"` if each trial should start
with 100 independently sampled genomes instead of one seeded founder copied
to its population.

The run makes 10 × 100 = 1,000 population transitions, creating 100,000
offspring in total. It requests 100 fitness scores at each of 101 generations
per trial, or 101,000 individual score requests. `max_steps = 1000` covers the
transitions across all trials. The runner processes one trial and generation at
a time; within each fitness call it deduplicates genotypes, evaluates missing
ones in batches of up to `batch_size`, and reuses scores from a bounded cache.
Memory admission uses `memory_budget_mib`, which also bounds that cache.
Actual work depends on the genotypes encountered, cache reuse, execution
backend, and available hardware resources, so these counts do not determine a
portable runtime.

For separate records, keep the seed, target, and scientific settings fixed,
set `trials` to the shard size, and give each shard a disjoint `trial_start`
(for example, 0 and 5 for two five-trial shards). Logical trial IDs then keep
the random streams stable across shard layouts.

The [nested runner plan](batched-runner-plan.md) proposes joint fitness
evaluation across independent trials and experiments, with per-trial
generation cursors and unchanged logical random streams.

## Strong-selection, weak-mutation (SSWM)

SSWM stores one resident genotype per trial. Each one-bit neighbor has a
mutation rate equal to its fixation acceptance divided by the number of genome
bits. The acceptance is evaluated in log space using the declared
`population_size`, including stable equal-fitness and zero-fitness limits. The
sum of neighbor rates sets an exponential waiting time; a second draw selects
the accepted neighbor by rate. Time is measured in mutation-time units, with
one attempted one-bit mutation per resident per unit before fixation weighting.

The process records a target present at time zero as an arrival at zero. It
applies a sampled transition only when its event time is within
`time_horizon`. A sampled event beyond the horizon remains pending in the
checkpoint, together with its candidate position and waiting-time draw. This
preserves the trajectory if a run is resumed with a longer horizon. A finite
event exactly at the horizon is applied. Recorded history rows contain trial,
event time, fitness, and event index. Reported `terminal_times` are the last
applied event times; `pending_events` describe sampled future transitions.

## Neutral walk

Each event proposes a uniformly chosen one-bit neighbor. The candidate is
accepted exactly when its declared metabolic or regulatory phenotype matches
the resident phenotype. The history records trial, event index, mutated bit,
and acceptance. The same logical stream and checkpoint scheme provide
deterministic continuation.

## Resource and recording controls

`max_steps` and `max_events` bound work per invocation. A reached cap returns a
partial run with a checkpoint, so the result does not masquerade as a
biological absence. Host admission estimates all trial and population state
and requires it to fit within half of `memory_budget_mib`; larger studies can
use disjoint `trial_start` shards. Pattern fitness is batched and cached by
genotype with a bounded insertion-ordered cache. History retention is bounded
by the smaller of `trace_budget_mib` and one eighth of the memory budget; the
summary reports dropped early history rows.

`recording.mode = "replayable"` retains the effective specification, source,
target, results, and committed checkpoints. `"full"` additionally emits
matched before/after transition rows: parent indices and mutated bits for a
Wright–Fisher generation, resident and candidate for SSWM, and proposal and
acceptance for a neutral walk. The trace has a strict byte budget. When the
next required row will exceed it, the run becomes partial and holds that row in
`pending_trace` in the committed checkpoint. A resume with a larger recording
budget writes the pending row before advancing. Records retain the latest two
committed checkpoint revisions. See [records.md](records.md) for integrity and
portable replay details.

The [2020 study](https://doi.org/10.1098/rsif.2019.0843) motivates the
two-gene pattern landscape. These configurable simulation protocols are
explicitly identified in each record; matching a published numerical result
also requires its target, initial distribution, sample size, and stopping rule.


## Schema version 2 population sessions

`PopulationSession(model, spec, evaluator=None, limits=None, recording=None,
checkpoint=None, target=None)` runs the Wright–Fisher, SSWM and neutral protocols
as observable state machines. `observe()` evaluates the current population;
`step()` advances each ready replicate by at most one transition and observes
its resulting population; `run()` stops at completion or a resource boundary.
`state` exposes cursors, statuses and score validity. `snapshot()` copies requested
host arrays, and `checkpoint()` returns a portable numerical checkpoint.

Schema version 2 separates `task`, `episode`, `objective` and `controller`.
A task defines the world and its chemical ports, an episode defines its horizon
and scenario panel, and an objective defines reported scores and selection.
`evolution.replicates` and `replicate_start` alias the legacy trial fields.
Generation zero and the final generation are observed. Zero requested generations
therefore evaluate the founders and consume zero evolutionary updates.

The evaluator preserves ordered gene identities and candidate IDs. Complete pure
candidate scores share a bounded FIFO cache and duplicate candidates share assay
work. The context includes lookup-table content, target content, task/controller
protocols, objective configuration and captured custom-component identity.
Declared component input files contribute content hashes, so copied record paths
retain the same context. Population objectives freeze the full roster before
scoring and use an observation barrier before selection.

Built-in objectives include exact tissue pattern matching, metabolic coverage of
all supplied environment entries (including repetitions), and bounded episodic
reward. Bounded rewards use `floor(D * reward + 0.5)`, with default `D = 2**24`.
Exact Wright–Fisher weights apply the rational floor using a common denominator.
The signalling game retains integer success counts and judged-round denominators.
Reported scores and selection fitness are distinct fields. A custom objective
returns `ObjectiveOutput`; Wright–Fisher requires explicit nonnegative integer
weights with a fixed denominator, and SSWM requires explicit nonnegative finite
selection fitness. Minimizing reported scores therefore use a declared selection
mapping. Neutral walks compare the complete assay phenotype or an explicitly
recorded discrete action/message descriptor.

`EpisodeSpec.panel="auto"` uses generation panels for stochastic episodic Wright–Fisher
and a fixed panel for standalone evaluation, SSWM and neutral walks. The same
panel/scenario random draws are shared across candidates. A contextual signalling
population uses a seeded stable ordering, up to four cyclic opponents per focal
individual, and both sender/receiver roles; a single individual plays itself.
The rational floor is applied after averaging these outcomes.

`WorkLimits` bounds evolutionary transitions, logical world ticks, chemical
updates and assay convergence work. The first three counters accumulate through
all calls in one session invocation; checkpoint construction starts a fresh
invocation and retains total counters. World and chemical counts include all
candidate/scenario/agent lanes. New candidate cohorts narrow when the remaining
quota admits fewer lanes. An assay or rollout that stops early remains invalid
for selection. An already advanced population retains its unobserved revision
for retry, so recovery cannot sample another transition before that observation.

`execution.max_active_lanes` bounds ready population lanes in a wave;
`execution.batch_size` bounds distinct candidate or matchup cohorts. Completed
CPU tests cover legacy trajectory/history equality, exact integer selection,
cohort/chunk changes, pinned history retention and checkpoint continuation.
The three legacy functions in `evolution.py` retain their version 1 execution
path and source-pinned recording behavior.
