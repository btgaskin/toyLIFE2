# Evolving molecular response machinery

These experiments evolve molecular sensing and control machinery within an assumed
viable cellular background. The evolving genome specifies the simulated promoters
and protein sequences. Fixed molecular supplies expose the machinery to cues and
provide output markers. Canonical toyLIFE interactions determine molecular states;
a declared output rule converts those states into actions; and the experiment
determines how performance affects reproduction.

The experiment contract is `single-cell-response-v1`. The molecular rules remain
`toylife-canonical-v1`, described in the [model](model.md). Earlier molecular
experiment data and checkpoints are deprecated and remain archived. The molecular
entrypoint requires the current contract for new runs and continuation.

## Three scientific layers

| Layer | What it owns |
|---|---|
| Molecular model | Genes, inventories, binding, regulation, translation, cleavage and turnover |
| Experimental realization | Molecular supplies, environment, output apparatus, actions and task performance |
| Evolutionary process | Founders, reproductive contributions, parent sampling, variation and replacement |

The causal chain is **genome → chemistry → molecular observations → action → task
performance → reproductive contribution**. Execution placement and recording
support this chain. They do not determine its biological meaning.

A *molecular cell* is the simulated intracellular state. The *viable background*
provides physiological support outside the evolving module. A *cue* is an
experimentally supplied molecular input. An *output marker* is a supplied molecule
whose observed state informs an action. A *signal* should be described through the
particular information-bearing mechanism being studied. The present assays study
single-cell responses; they do not introduce molecular communication between cells.

## What the experiment adds

| Addition | How it works | Why it is added |
|---|---|---|
| Fixed molecular interface | Supplies named cue and marker counts to the existing chemistry | Connects environmental conditions to structurally realized intracellular responses |
| Binary output apparatus | Reads intact marker counts after the output reaction phase and chooses an action | Makes the assumed coupling to downstream apparatus explicit |
| Production measurement | Counts expressed gene copies translated at every canonical update | Measures production activity separately from task performance and reproduction |
| Resolved methods record | Saves configuration, rule versions, source, random seeds and a generated methods summary | Makes the numerical settings and biological premises inspectable together |

The supplies are maintained by assumed background apparatus. They sit outside the
evolving genome and can be present inside the simulated molecular inventory. Their
identities, doses and output assignment stay fixed throughout a lineage. Proteins
acquire sensing, regulatory or output roles through their molecular interactions.

## Molecular updates and supplies

One canonical update has two reaction phases with promoter evaluation and
translation between them. Translation replaces free proteins and dimers. Each
active gene implies one translated chain; nonfolding products are omitted from the
functional inventory. The output reaction phase then resolves binding and cleavage.

Marker observations are taken at `after_output`, before boundary release.
Canonical release returns bound metabolites to the free inventory and removes
their protein partners. Surviving fission fragments can remain. Free environmental
metabolites that never bound return to the environment. This molecular turnover is
separate from replacement of organisms between evolutionary generations.

`StructuralInterface` specifies cue and marker identities, marker copy counts and
available observations. It uses `replenish`: the declared free supplies are set at
each chemical update. Cleavage and turnover therefore matter. The observed intact
pool is not assumed to be a conserved copy of the supplied pool.

## Assumed binary output apparatus

`ControllerSpec.output` owns the observation-to-action rule. `TaskSpec` owns the
meaning and consequences of the resulting action. Both output representations
produce **one binary action**, independently of the number of input molecules.

| Representation | First count | Second count |
|---|---|---|
| `bound_free` | Free intact marker | Bound intact marker |
| `bound_pair` | Bound marker A | Bound marker B |

For counts `(n0, n1)`, the apparatus supports:

- **`majority`:** choose the larger count; choose either alternative equally on a tie.
- **`proportional`:** choose alternative 1 with probability `n1 / (n0 + n1)`;
  choose either alternative equally when both counts are zero.

The founding `permutation`, either `[0, 1]` or `[1, 0]`, assigns alternatives to
actions. For counts `(3, 1)` and assignment `[0, 1]`, majority always chooses action
0 and proportional chooses action 1 one quarter of the time. The two rules agree
on zero/one observed-count cases. One supplied copy alone does not guarantee those
observations under every chemical trajectory.

Proportional is the default. New profiles sample the action assignment once from
the interface seed and store it explicitly. Both markers initially receive one
copy, with equal supplies in the paired representation.

The single-marker fraction means **bound among surviving intact markers**. Counts
can stand in for relative concentrations under an assumed common volume. The
normalization and action rule describe downstream apparatus; they add no binding
reaction, kinetic rate or protein actuator. Cleavage can influence action by
changing the observations available to that apparatus.

```python
from toylife2 import BinaryOutputSpec, StructuralInterface

interface = StructuralInterface(inputs={"cue": "00101110"}, targets={"act": "0111"})
controller = interface.controller(
    output=BinaryOutputSpec(
        representation="bound_free",
        markers=("act",),
        rule="proportional",
        permutation=(0, 1),
    )
)
```

Draws occur once per world decision after its chemical updates. Philox streams use
fixed panel, episode, decision and rejection-attempt coordinates. Matched candidates
share random coordinates; identical genotypes may share an evaluated estimate.
Batching, retries and caching preserve that estimate. This convention evaluates
genotypes on matched episodes rather than independently simulating every copy's life.

## Tasks and clocks

| Profile | Prescribed cue conditions | Required action |
|---|---|---|
| `basic` | Absent / present | 0 / 1 |
| `cue` | Present / absent | 0 / 1 |
| `cue_alternatives` | Cue A / cue B, supplied mutually exclusively | 0 / 1 |
| `chemotaxis` | Sampled local-concentration cue | Action selects the configured tumble-onset rate |

Binary cue profiles exhaust both Nature states, one episode per condition. Cue
identities and output representation are independent choices. The two-cue profile
adds no implicit both-present or neither-present Nature states. Doses are stored as
two explicit rows in `task.config.cue_doses`.

Chemical updates are discrete molecular transitions. World decisions request an
action after `episode.chemical_updates` transitions. Episodes provide repeated task
conditions or trajectories. Evolutionary trials are independent population replicates;
evolutionary generations perform parent sampling and offspring replacement. A world's
physical `dt` does not calibrate molecular kinetics.

Cue profiles use fixed matched panels. `settling_steps` are unscored world decisions
before the response interval. All response decisions are scored by default. Task
performance is mean sampled reward; the bounded objective quantizes it and applies
the recorded reproductive-weight floor. Rewards do not enter intracellular chemistry.

For cue assays, action probabilities also yield expected accuracy: average the
probability of the correct action over scored decisions and conditions. This
integrates out output sampling because actions do not change the cue or chemistry.
Sampled reward continues to select parents. Reports distinguish sampled accuracy
from expected accuracy. Probabilities along a sampled chemotaxis trajectory do not
establish expected total performance in an action-dependent world.

## Initialization, variation and competition

`independent_random` initializes a population of independently sampled genomes.
`monomorphic_random` copies one random genome across the population. The cloned
genome is the first member of the corresponding independently sampled population
under the same seed. These are experimental starting conditions. Repeated copies
of one genotype are not independent observations of initial solution accessibility.

Evolution defaults to fixed-size, globally competing, nonoverlapping Wright–Fisher
generations. Parent sampling uses the declared reproductive weights. Zero total
weight invokes uniform sampling. Demographic sampling already supplies drift.
Genotype loss describes loss of inherited variants within this model.

### Whole-gene duplication and deletion

Each gene contains a four-bit promoter and sixteen-bit coding sequence. Enable
single-event structural variation through:

```sh
uv run toylife2 molecular cue --preset pilot \
  --set evolution.mutation_probability=0.10 \
  --set evolution.gene_copy.duplication_probability=0.01 \
  --set evolution.gene_copy.deletion_probability=0.01
```

One categorical proposal is drawn per offspring: point mutation, duplication,
deletion or unchanged inheritance. Point mutation flips one uniformly selected bit
among the active genes. Duplication inserts the entire selected promoter/coding unit
adjacent to its source; deletion removes the entire selected unit. Bounds default
to 1–20 active genes. A blocked proposal leaves the inherited genome unchanged and
is counted separately. Proposed, applied and blocked counts are recorded.

Enabling more operators changes total variation input at these specified rates.
Holding point probability per offspring fixed while changing gene count changes
its per-site probability. The experiment must declare which comparison it intends.
A gene count is not assumed to equal the number of nodes in a Boolean realization:
a response can also arise through direct molecular competition.

## Production activity

`translation_events_warmup` and `translation_events_task` count active gene copies
translated during their respective intervals. `chemical_updates_warmup` and
`chemical_updates_task` supply their denominators. Task production includes settling.
Sixteen toy sequence units per translated chain give a derived sequence-unit total.

Counts cover every chemical update, including those inside device blocks. They
include nonfolding products and repeated gene copies. They describe the simulated
cell over its evaluation interval; cache reuse does not multiply them. Population
means weight per-genotype measurements by the abundance of evaluated copies.

These measurements impose no production penalty and no expression-resource limit.
Binding competition already exists within canonical chemistry. Shared transcription
or translation resources would add another interaction mechanism, capable of
changing behaviour before reproductive selection. [Qian et al. (2017)](https://doi.org/10.1021/acssynbio.6b00361)
provides motivation for a separately specified resource model. Production counts and
folding/contact energies do not provide a calibrated energetic cost.

## Run, record and inspect

```sh
uv run toylife2 molecular cue --destination results/cue-response
uv run toylife2 molecular cue_alternatives --output-representation bound_pair \
  --output-rule majority --destination results/alternative-response
uv run toylife2 molecular chemotaxis --field source --destination results/source-response
```

`--preset smoke` uses four organisms, two generations and two evolutionary trials.
The cue response lasts eight updates by default. `--preset pilot` uses 64 organisms
and 100 generations. Override population/trial counts with `--set`, and select
`--initial`, `--cue-length`, `--marker-length`, `--marker-copies`, `--output-rule`
and `--output-representation` explicitly when needed.

The automatic backend selects Metal or CUDA when available, followed by native CPU.
Resident chemotaxis keeps molecular/world state on device across decisions. Routine
production accounting transfers compact block totals rather than molecular inventories.
Full traces and detailed inspection explicitly materialize observations.

New float32 chemotaxis profiles declare
`task.config.world_random_math="box_muller_f32_v1"`. A shared addressed Philox layout
supplies sensing, event waiting times, tumble durations and rotational noise.
CPU and Metal defaults batch those draws through NumPy; CUDA fills the resident
draw buffer through Quadrants. Direct Metal generation remains available in the
qualification benchmark. Both providers use float32 Box–Muller normals and
strictly open uniforms. Initial positions, orientations, phases and sources retain
the existing host initialization. Backend arithmetic can produce different
trajectories while implementing the same stochastic process.

The setting `numpy_f64_v1` uses binary64 NumPy transforms before conversion to the
world precision. An absent setting retains that behavior for existing
configurations. The declared mathematics is part of the task identity; the selected
implementation is execution metadata. Neither choice changes tumble hazards,
diffusion constants, the angular integrator or evolutionary variation rates.

Every scientific run stores its resolved configuration, source/table identities,
checkpoints and a generated `methods.md`. Molecular inspection writes actions,
marker readouts, expression, task outcomes and production summaries. Cue inspection
also reports expected accuracy. Full reaction trajectories remain optional.

## Prepare a factorial separately

```sh
uv run python examples/single_cell_factorial.py --destination results/response-matrix
```

Preparation writes specifications and a workload estimate. It launches no campaign.
The default matrix has two initialization modes, eight operator treatments and
founder sizes 1–5: 80 conditions for each declared interface seed. Trial counts,
interface seeds, output rule and output representation are explicit. Execution
requires `--run`.

The runner archives initial populations, checks matching across operator arms,
records sampled and expected performance by generation, and reports first expected-
perfect solutions, first solution genotypes absent from the initial population,
retention, active gene count, production and operator statistics. An all-off cloned
population supplies a distinct control from all-off selection on standing variation.

Use a small qualified subset before larger campaigns. Vary molecular lengths and
sequence replicates in later stages rather than silently expanding this matrix.
Traces propose mechanisms; interventions test them. Record discovery, retention and
perturbation sensitivity separately.

## Chemotaxis and next mechanisms

Chemotaxis uses a spherical swimmer with three-dimensional orientation and a planar
concentration field, held constant along Z. Action selects the configured onset
rate of run-to-tumble transitions. Runs diffuse rotationally; tumbles pause
translation and reorient by active rotational diffusion with exponentially
sampled durations. Saved playback reconstructs the same field and motion parameters.
See the resolved `ecoli_spherical_v1` task settings for rates, durations, substeps
and event capacity. Time is in seconds; default speed is 20 µm/s under the declared
length conversion. The [Saragosti et al. (2012)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0035412)
study motivates this motion model. Chemical updates retain their own abstract clock.

A persistent protein actuator is the next chemistry design milestone. Its design
must state scaffold identity/copies, binding partners, persistence through
translation and boundary release, activation, turnover and reset, with a minimal
mechanistic example and invariants. Such a model receives its own molecular-rule
identifier after those choices are specified.

Resource constraints, two-cell transport, local competition, alternative replacement
timing, novelty/QD and game integration each require a concrete experiment and an
explicit mechanism. The current contract supplies the interpretable single-cell
baseline against which those additions can be compared.
