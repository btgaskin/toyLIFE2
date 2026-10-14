# Evolving molecular cells

These experiments evolve complete toyLIFE cells. A genotype specifies every
promoter and protein sequence. Named molecular ports connect a cell's shared
chemistry to a task: the world supplies ligand copies, molecular interactions
determine intact target occupancy, and the task turns those counts into acts.
Receptor, regulator and effector describe functions that an evolved protein can
perform through its interactions.

## Run and inspect

```sh
uv sync --extra reports
uv run toylife2 molecular basic --destination workspaces/basic
uv run toylife2 molecular cue --destination workspaces/cue
uv run toylife2 molecular cue4 --destination workspaces/cue4
uv run toylife2 molecular chemotaxis --destination workspaces/chemotaxis
uv run toylife2 molecular chemotaxis --field source --destination workspaces/source
```

All these profiles contain one cell per independent episode. The signalling
network is internal to that cell. Multiple interacting cells are a later
experiment.

### Results and continuation

Without `--destination`, runs receive a unique directory under `results/`.
`--workspace binary-cells` groups them under `results/binary-cells/runs/`;
an explicit workspace path is also accepted. `--name first` supplies a readable
run name. Workspaces support the existing `workspace status` and `workspace check`
commands, including the experiment arrays and nested runtime records.

```sh
uv run toylife2 molecular basic --workspace binary-cells --name first --generations 5
uv run toylife2 molecular --resume results/binary-cells/runs/first \
    --generations 25 --workspace binary-cells --name extended
uv run toylife2 workspace status results/binary-cells
uv run toylife2 workspace check results/binary-cells
```

Generation targets are totals: extending from five to 25 advances 20 further
generations. The population, random coordinates, evaluations, counters and
history continue from the saved checkpoint. A new parent-linked record preserves
the original run. Its molecular task and mutation protocol remain fixed;
execution placement and work budgets can change. Resume with the same target
to continue a quota-interrupted run. Checkpoint intervals are configured by
`--set recording.checkpoint_interval=1`; the runtime also saves a final checkpoint
on normal completion and budget stops. `PopulationSession.extension(25)` provides
the corresponding typed spec and numerical checkpoint for Python integrations.

### Optional cell playback

```sh
uv run toylife2 molecular chemotaxis --animate --frame-stride 5 \
    --workspace chemotaxis --name first
```

`animation.html` is an offline, dependency-free Canvas view. Play, pause and scrub
through one matched held-out scenario, showing a small rod bacterium, heading,
trail, field and molecular readouts. Initial and final selected cells occupy
separate views of the same scenario. Every fifth saved world step is displayed
in this example; sampling changes the rendering rather than the simulation.
The final frame is always retained. Playback is an optional output and adds an
initial-cell inspection only when requested. Binary cue reports always include
initial and final response tables.

### Swimming and tumbling

Chemotaxis uses one motion model, `ecoli_spherical_v1`. A cell carries a unit
swimming direction in three dimensions. Motion on the existing periodic XY
world is its planar projection; the concentration field is invariant along Z.
The sensor receives a noisy local-concentration cue through its molecular port.
Motor occupancy selects the run-to-tumble rate (default 2 or 0.2 per second).

Runs have rotational diffusion 0.05 rad²/s. Tumbles pause translation and rotate
the direction by active rotational diffusion, default 3.5 rad²/s. Each tumble
has an exponential duration with mean 0.14 s. Time is in seconds; one position
unit is 20 µm, so default speed 1 corresponds to 20 µm/s. Physical parameters
are explicit in `task.config`.

These defaults follow [Saragosti et al. (2012)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0035412).
Their historical angle-distribution fit used discretized observed durations.
Our continuous-duration reduction predicts mean directional correlation
`1/(1 + 2 D_tumble mean_duration) ≈ 0.505`; it does not reproduce that measured
angular histogram by construction.

Phase transitions use exponential waiting times within each world decision.
Geodesic angular increments subdivide each phase segment into at most 0.01 s
steps (`motion_max_substep`). Translation uses each increment's final direction.
`motion_event_capacity`, default 8, bounds the phase events per decision and the
random-draw storage. Exhaustion fails the step before candidate state is committed;
increase this capacity for unusually large rates or decision intervals. The
remaining tumble duration and full orientation are included in checkpoints.

Playback reconstructs the same analytic concentration field from saved phases
or source coordinates. Its fixed 0–1 colour scale makes higher concentrations
darker. Fourier landscapes have several peaks; the `source` profile has one
periodic exponential source, marked with a cross. Rod drawings are schematic.
Results recorded before this motion change retain their original source and
describe their original dynamics; new evaluations produce separate records.

The default `smoke` preset runs two independent populations of four cells for
two generations; chemotaxis uses two training episodes of 20 decisions. It checks
the complete workflow. `--preset pilot` uses population 64, 100 generations,
two replicates, and sixteen 1,000-decision chemotaxis training episodes. Both use
independent random initial genomes and mutation probability 0.01 per offspring,
with one bit flipped in a mutating offspring. All promoter and coding bits can
mutate. Gene count defaults to two and can be set with `--genes`.

Use `--set evolution.population_size=128` and other ordinary RunSpec overrides
to configure a run. Automatic placement selects Metal on Apple hardware, CUDA
where available, and native CPU otherwise. Use `--backend cpu|metal|cuda` to select
explicit placement. New chemotaxis profiles run Quadrants float32 world updates
and chemistry in resident decision blocks. Cue-reading worlds retain their host
calculations. Their profiles report world placement separately from chemistry;
see the [device pipeline guide](device-pipeline.md).

Each destination contains a resolved `profile.toml`, evolution record, selected
final-population genomes, fitness history, held-out rollout record,
`behaviour.npz`, `history_probes.npz`, an `experiment.json` summary, and
`report.html`. With the reports extra, standalone SVG plots show evolutionary
scores, chemistry under sustained histories, and chemotaxis trajectories and
sensing. Arrays preserve replicate and scenario axes. Artifact hashes cover
the additional experiment arrays; ordinary runtime records retain their source,
table, checkpoint and integrity machinery. A budget-interrupted evolution keeps
its record and stops selection, so stale evaluations cannot select new genomes.

### Whole-gene duplication and deletion

Molecular Wright–Fisher runs can evolve gene count through an optional
`evolution.gene_copy` block. Each gene remains one four-bit promoter followed by
one sixteen-bit protein-coding sequence. `model.genes` sets the founder length.

```toml
[evolution.gene_copy]
duplication_probability = 0.01
deletion_probability = 0.01
min_genes = 1
max_genes = 20
```

These example rates are experiment choices. Both structural rates default to
zero. Configure the same block on the CLI with
`--set evolution.gene_copy.duplication_probability=0.01` and
`--set evolution.gene_copy.deletion_probability=0.01`.

After fitness-weighted parent sampling, an offspring receives at most one event:
a point mutation with probability `mutation_probability`, a duplication, a
deletion, or unchanged inheritance. The three event probabilities must sum to at
most one. A point mutation samples uniformly among the active genome's `20*g`
bits. Duplication samples one of its `g` gene copies and inserts an identical
copy immediately after it. Deletion samples and removes one gene copy. Identical
copies remain separate sampling targets. A structural event at a size bound
leaves the offspring unchanged and increments the blocked-event count.

Copy number enters the existing canonical chemistry through the active genes.
The same protein sequences retain their molecular identities; their translated
copy counts can change. These operators add no length penalty or resource cost.
They apply to candidate-scored molecular cue and chemotaxis tasks with a
structural interface and empty initial expression masks.

Mixed-length populations are grouped by active length for the existing fixed-width
evaluators. The groups share work quotas, cache admission and a complete-population
selection barrier. Checkpoints include active counts and pending evaluations.
Leaving out the block preserves the original point-mutation protocol and its
serialization. An explicit block, including zero structural rates, identifies
the gene-copy protocol in saved records.

Selections use paired `selected_genomes.npy` and `selected_gene_counts.npy`
arrays, with corresponding `initial_` arrays and a `selection.json` format marker.
Read them with `toylife2.genotypes.load_selected()`, which returns a `GenomeBatch`
for this format. Its `active(index)` method returns the exact selected genome;
zero-filled storage beyond its active count has no molecular meaning. Reports
show the selected lengths and proposed, applied and blocked event counts.
`gene_count_history.npy` records population length distributions, with replicate
and generation coordinates in `gene_copy_statistics.json`.

Held-out assessments are recorded under `inspection/genes-<count>/`, each with
the exact gene count in its child specification. The parent summary maps child
rows back to selection IDs. Parent behaviour arrays and playback retain that
selection order and include active counts alongside expression masks.

## Named ports

```python
from toylife2 import BatchRunner, StructuralInterface

interface = StructuralInterface(
    inputs={"cue": "0001"}, targets={"motor": "1111"},
)
with BatchRunner() as runner:
    cell = runner.chemistry([907050, 1048465], interface.controller())
    cell.step(interface.encode({"cue": 1}))
    occupancy = interface.outputs(cell.last_readouts)["motor"]
    diagnostics = interface.diagnostics(cell.last_readouts)
```

`step()` advances one canonical chemical update. `encode()` requires every
input name, including explicit zeros, and accepts integer arrays with batch
axes. Input and target molecules have distinct canonical identities; reversal
aliases normalize to a single identity. Target copies default to one, with
`target_copies={"motor": 2}` available. `max_dose` defaults to eight.

The generated ports replenish free input and target inventories at each
chemical update. This models a supplied molecular bath: previous free supply is
replaced, while canonical reaction and boundary rules determine bound inventories
and fragments. These supply counts describe an environmental intervention,
rather than a closed-cell conservation law. Intact target occupancy is measured
at `after_output`, before boundary complex release. Occupancy can include
pre-translation complexes that persist into the output phase. Translation
replaces the free protein inventory; the boundary releases bound ligands and
removes their bound protein partners. Free counts, target fission,
expression and inventories are available for inspection. A source-table
"unbreakable" annotation does not disable executable cleavage reactions.

The interface is stateless. Cellular history resides in the chemistry. Existing
custom codecs support other coupling rules, and lower-level PortSpec APIs support
pulse supply and alternative readout stages. Python and TOML construct the same
typed declaration; generated ports are resolved from that declaration once.

For two channels, declare two ligands and two targets. `cue4` evolves a single
cell containing both channels; it does not concatenate separate circuits. The
task specifies `action_outputs=["act_a", "act_b"]`, interpreting occupancy > 0
as bits in most-significant-first order. Count thresholds and choice rules belong
to the task. Both channel pairs participate in the same intracellular network.

## Cue tasks and fitness

| Profile | World-to-cue map | Required act |
|---|---|---|
| `basic` | 0 → 0, 1 → 1 | Nature state |
| `cue` | 0 → 1, 1 → 0 | Nature state |
| `cue4` | 0 → 00, 1 → 01, 2 → 11, 3 → 10 | Nature state's two-bit representation |

Every nature state receives an independent empty-start chemistry episode. The
cue is held for eight observations, each advancing one chemical update. Intact
target occupancy is decoded as a binary act after every update. Fitness measures
exact agreement across judged observations and equally represented states;
`cue4` also reports component correctness. Joint agreement supplies its fitness.
Fitness is `max(mean_reward, 1e-8)`, preserving a small random-selection floor.
Evolutionary reward influences reproduction between episodes. It is not fed
into the cell as an intracellular training signal.

The cue definitions follow the CUE fixed-map structure in
[self-assembling-games](https://github.com/btgaskin/self-assembling-games) (B4);
the four-state extension uses B3's coding
motivation. The molecular controller and Wright–Fisher adaptation are this
experiment's mechanisms. The source game's reinforcement learner remains a
separate comparison. These profiles study conditional acts within one cell.

Four additional persistent-chemistry probes supply histories 0011, 1111, 1100
and 0000, holding each observation for eight chemical updates. Paired histories
end with identical current cues. Their saved expression and occupancy traces
help distinguish a history-dependent response from a current-cue mapping.
These probes are inspections of selected genomes and do not affect selection.

## Chemotaxis and feedback

Each decision samples ligand availability once from
`Bernoulli(c / (sensor_scale + c))` at the current position. That zero/one dose
is supplied for the configured number of chemical updates. Evolved proteins
interact with the ligand and motor target. Occupied motor target selects tumble
hazard 0.2; an unoccupied target selects hazard 2.0 per second. Run waiting times
are exponential. Finite tumbles pause swimming for an exponentially distributed
duration with mean 0.14 seconds while active rotational diffusion turns the cell.
The three-dimensional orientation projects onto the square torus, with speed 1,
run rotational diffusion 0.05 rad²/s and `dt=0.05` seconds. The motion model above
describes the substeps and state carried across decisions.

Feedback is the physical loop: cell action changes its path, and the path changes
later ligand observations. Temporal comparison can emerge through persistent
chemistry. The world supplies no gradient sign, temporal comparator or reward
input. The sampling law supplies concentration-dependent encounters; actual
ligand–protein contacts use toyLIFE's deterministic chemistry. Its `sensor_scale`
is an external sensitivity parameter, not an evolved receptor's physical Kd.

Two field profiles use this same body, sensing, motor and fitness:

* `fluctuating`: `c = 0.5 + 0.4 * mean(cos(k·x + phase))` on a domain of
  side 16, with integer modes (1,0), (0,1), (1,1), (1,-1). Initial phases are
  uniform and remain fixed by default. Set `task.config.static=false` to enable
  phase increments with variance `2*dt/3`, giving each mode an exponential
  temporal correlation of time scale 3. The profile name identifies the Fourier
  field family; its `static` setting controls temporal variation.
  This finite, positive Fourier field is inspired by
  [Celani et al. (2011)](https://doi.org/10.1007/s10955-011-0251-6).
* `source`: `c = exp(-r/4)` around a random source, using minimum-image distance
  on the same torus. Arrival within radius 0.4 is recorded without ending an
  episode. This periodic source adapts the source-search setting of
  [Alonso and Kirkegaard (2024)](https://doi.org/10.1093/pnasnexus/pgae235).

Both are dimensionless experimental adaptations. Their explicitly recorded
field, boundaries, sensor and discrete motor timing define the implemented
experiment; matching an original paper's quantitative results would require a
separate reproduction protocol.

Fitness is mean post-movement concentration, bounded within [0,1], with the same
minimum selection floor. Selection uses a generation-addressed common scenario panel.
For equal episode durations, this ranks cells identically to cumulative exposure
`dt * sum(concentration)`. Each candidate is averaged over the same sixteen
training scenarios in a pilot; initial positions, headings and field layouts
vary across scenarios. Static training separates spatial sensing from field
motion; temporal variation can then be introduced as a separate experiment.
The final population's best cell in each replicate is inspected on 32 fixed
held-out scenarios with a new seed and panel. Reports include matched constant
actions, cue, inverse cue, random actions, and a declared two-timescale temporal
policy, plus a chemistry-reset-each-decision intervention. Baselines share
scenario-addressed field and motion randomness. The temporal policy uses
exponential cue averages with time constants 0.2 and 1.0, initialized at 0.5;
it chooses the low-tumble motor state when the fast average exceeds the slow one.
Different policies produce different paths despite shared initial draws.
Compare each evolved score with controls within its field
profile, since the two fields have different concentration distributions.

## Scaling

Population, replicate, episode and channel axes use the existing batch runtime.
Immutable chemistry tables are shared; each cell retains its own molecular
state. Configurable work and memory budgets continue to govern chemistry,
world buffers and trace capture. Detailed inspections are deliberately confined
to selected cells. Large campaigns can use the lower-level evaluator and record
only chosen inspections. The NumPy world runner provides a reference for the
device-resident chemotaxis implementation described in the
[device pipeline guide](device-pipeline.md).


## Fixed random ports and response timing

New molecular profiles sample ports once with `--interface-seed 0`, independently
of the evolutionary seed. The default `--port-pool four_site_active` contains nine
reversal-canonical four-site molecules with negative protein or dimer contacts.
`--port-pool all_lengths_active` uses the same contact rule across supported
lengths four through eight. Sampling is uniform over canonical identities without
replacement, so the broader pool gives longer lengths greater representation.
Contact eligibility does not guarantee an empty-start response. Equivalent contact
profiles remain in the pool; their similarities and cleavage are reported.

```sh
uv run toylife2 molecular cue --genes 10 --port-pool all_lengths_active \
    --interface-seed 17 --response-updates 16 --settling-updates 2 \
    --workspace molecular-cells --name longer-cue
```

The cue is held throughout settling and response. Settling observations are
recorded but excluded from fitness. Each response is binary; scoring measures the
fraction of correct responses rather than thresholding an averaged occupancy.
Targets are replenished each update and participate in binding competition.
Cleavage gives no intact occupancy and can accompany a correct action zero.
Fission is reported separately. Chemotaxis retains an instantaneous binary motor
command for each world decision and its existing separate chemistry clock.

Explicit `StructuralInterface` values remain available through `profile_spec`'s
`interface` argument. A verified inverse-task fixture uses genotype
`[360861, 503167]`, cue `00000011`, and target `1111`: occupancy alternates zero/one
without the cue and remains zero with it, giving 0.75 over eight observations.
Legacy horizon-one/eight-chemistry-update specifications retain endpoint scoring.
Continuations retain the saved molecules, task, and timing.

The matched calibration runner first screens 128 common random genomes at windows
4, 8, and 16, then runs four fixed port conditions per pool with mutation rates
0.001, 0.01, and 0.1. Each pilot uses 256 cells, 100 generations, ten genes, and
four evolutionary replicates. Its JSON receipt records timing-screen score
frequencies, final scores, best historical scores, counters, and elapsed times.

```sh
uv run python examples/calibrate_molecular.py --destination results/interface-calibration
```

Selected-cell inspections inherit the configured recording mode and trace budget.
The default replayable mode records trajectories, readouts, seeds, and checkpoints
without full molecular event traces. Use `--set 'recording.mode="full"'` to opt
into those traces, and `--set recording.trace_budget_mib=1024` for a larger
allowance. Chemotaxis pilots allocate 512 MiB per full record by default.

### Memory-aware evaluation

`execution.batch_size` sets the maximum number of distinct genomes in an
evaluation cohort. Each genome is replicated across `episode.episodes` independent
worlds. The engine checks projected chemistry and world allocations against
`execution.memory_budget_mib`; this is an allocation budget rather than a process
RSS limit.

For pure, independently scored candidates, a rollout-construction memory failure
now halves the cohort and retries before any simulation work has run. Further
cohorts in that request use the smaller width. Scenario identities, molecular
updates, trial counts and selection rules remain the same. Retry counts appear
in evaluator metrics as `memory_batch_retries`. A single genome with all its
episodes must still fit; failures during simulation are reported normally.
Population-scoped objectives and interacting-agent evaluations retain their
complete cohorts.

Generation-panel evolution discards completed score-cache entries from panels
that active or pending replicates cannot revisit. Current and pending panels are
retained; chemistry graph plans remain reusable across generations. Fixed-panel
caches retain their existing bounded policy. When several evaluators share a score
cache, pruning also retains that bounded policy so sibling populations keep
their reusable scores.

Choose the budget with headroom for Python, recordings, other applications and
concurrent runners. Measure resident-memory peaks and end-to-end throughput when
raising batch widths; a larger budget helps only when it permits useful batches.

### Live memory sampling

Attach the optional standard-library sampler to an existing process:

```sh
python examples/monitor_molecular_runs.py --watch PID=results/my-run --interval 10
```

Repeat `--watch` to monitor several runs in one lightweight process. Each run gets
`memory.jsonl` with timestamped RSS, CPU percentage and committed checkpoint
progress, plus `memory-summary.json` with current and highest sampled RSS. The
sampler stores its source alongside these files and stops watching a process
when it exits. It does not change simulation state or recorded scientific inputs.
RSS measures resident process memory; sampling can miss brief peaks. The recorded
allocation budget and RSS measure different aspects of memory use.
