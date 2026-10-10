# Methods and experiment design

toyLIFE2 provides a versioned artificial chemistry and experiment protocols built
around it. This guide describes what an assay measures, how a simulation advances,
and which settings a researcher must record. The [README](../README.md) contains
installation, examples, full article citations, and links to saved measurements.

## Research basis and model identity

[Arias et al. (2014)](https://doi.org/10.1038/srep07549) introduced toyLIFE as a
multilevel genotype–phenotype model connecting proteins, gene regulation, and
metabolism. [Catalán et al. (2018)](https://doi.org/10.1098/rsif.2017.0516) studied
the revised molecular model's robustness and evolvability;
[its supplementary methods](https://doi.org/10.6084/m9.figshare.c.3950983.v1)
and the bundled [author definition](../upstream/toylife/toyLIFE.pdf) describe
that molecular universe. [Catalán, Manrubia, and Cuesta (2020)](https://doi.org/10.1098/rsif.2019.0843)
embedded two-gene regulatory circuits in a one-dimensional tissue and studied
pattern abundance and evolutionary accessibility.

The product ruleset is `toylife-canonical-v1`, with lookup data pinned to author
commit `9e17c92ea05f2e3dd63bf437400ddd8e61694ff9`. The revised folding rule
allows equally stable configurations with the same perimeter to represent a
functional protein. This differs from the original 2014 folding definition.
[Model decisions](decisions.md) specify all accepted interpretations and source
exceptions, including curated binding geometry. A change to the chemistry is a
scientific model change; backend and chunk size are execution choices.

## Genotypes and molecular representation

A genotype is an ordered collection of genes. Each gene has four promoter bits
and sixteen coding bits. Coding bits denote polar (`0`) and hydrophobic (`1`)
units. The chain folds compactly on a 4 × 4 lattice; perimeter and folding
energy define functional protein identity. Protein ID zero denotes an unfolded
product. An active gene retains its expression bit even if its product is unfolded.

Public arrays contain one integer in `[0, 2**20)` per gene, with genes on the
last axis. `encode()` converts binary strings to this representation. Leading
axes identify independent candidates or explicitly declared coupled agents.
Ordered genes preserve promoter/product associations throughout evaluation.

Lookup tables represent folded proteins, perimeters, dimers, promoter contacts,
metabolite contacts, and fission geometry. The package reads the source tables,
checks their shapes and consistency, and converts energies to integer tenths.
The HP contact energies are −2, −0.3, and 0; exact integer comparisons avoid
floating-point ambiguity at ties. The interaction threshold is strictly below
−2.6. The [model reference](model.md) lists array shapes and source exceptions.

## One chemical update

A cell carries expression bits and molecular inventories. One complete update:

1. Admit reaction candidates from the phase-start inventory and form complexes.
2. Evaluate every promoter from the same post-reaction snapshot.
3. Replace free proteins with functional products from active genes and clear free dimers.
4. Perform the second reaction phase, retaining intermediate bound complexes.
5. Release bound metabolites and retain surviving fission fragments at the boundary.

Free environmental metabolites that never bound return to the environment.
Promoter binding does not consume its competitors. A readout declares whether
it samples `after_input`, `after_output`, or the completed `boundary`; bound
complexes are available at the first two stages.

Only valid interactions enter competition. At a promoter, copies of one species
constitute one competing identity. A tied strongest group is excluded from that
promoter evaluation and the next group is considered; molecules remain present.
For reactions, copy counts determine capacity, including two copies per
homodimer. Each energy group computes candidate capacities and combined resource
demand before applying reactions. Candidates involving an over-demanded reactant
are rejected. Later groups use updated inventories for already admitted
candidates; newly possible reaction types wait until the next phase.

These phase, identity, and resource rules define the
[scientific contract](model.md#scientific-contract). Scalar reference, fixed
assays, and persistent kernels implement the same chemistry.

## Phenotype assays

### Regulatory function

For `G` genes, a complete regulatory assay tests all `2**G` Boolean expression
states and returns the next-expression table in ascending input-state order.
Every leading genotype axis is preserved. A two-gene input shaped `[R, N, 2]`
therefore returns values shaped `[R, N, 4]`.

Full truth tables grow exponentially. Admission checks truth-state and memory
budgets before evaluation. Larger-genome metabolic assays instead follow the
trajectory reached from all-off and find its repeating regulatory phases.
These are different computations and should be named explicitly in comparisons.

### Metabolic phenotype

A metabolic assay tests each declared metabolite separately from every phase of
the regulatory attractor reached from all-off. Cleavage in any phase establishes
a positive phenotype. The simulation tracks the complete chemical state, including
metabolite-bound states; repetition establishes an intact result. Caps produce
`incomplete` status. Larger state spaces use a constant-memory cycle proof,
which can require more updates than the first observed repeat.

The supplied universe contains 274 reversal-equivalent metabolites of lengths
four through eight. Specify the environment and its ordering when reporting a
subset assay. A molecule and its reverse share chemical identity; repeated
environment entries can still contribute repeatedly to a coverage objective.

### Tissue patterns

The two-gene spatial assay evaluates six molecular inputs: empty, free B, free A,
AB (or free A+B when AB cannot form), AB+free A, and AB+free B. A seventh descriptor
symbol records AB availability. Inputs with preformed AB keep that assembly
fixed during promoter evaluation. The descriptor and diffusion rule determine a
64-entry radius-one cellular automaton. Tissue cells then update synchronously.

The default 2020-style protocol uses 31 cells, 100 updates, open edges, A
diffusion, and a central A pulse at tick zero. Output rows are **post-update**
states; the initial state is not an extra output row. Tissue states 0–3 record
functional A/B expression. Geometry, diffusion, starting state, and later
injections are configurable scientific settings.

### Weighted genotype–phenotype map

The two-gene map covers `2**40` ordered binary genomes by factorizing coding
multiplicities. It enumerates all 7,349,521 ordered protein pairs, evaluates the
16 promoters for each gene, and weights promoter combinations by the exact
number of coding sequences producing each protein. Histograms retain logic
signatures, seven-symbol regulatory descriptors, local rules, and pattern hashes.

Every completed histogram must sum to 1,099,511,627,776 genome weights. Work
intervals can be tiled and resumed. Merging requires adjacent intervals and
matching table and spatial identities. [Qualification](qualification.md) and
[parity reconciliation](parity.md) describe complete enumeration and comparisons.

## Evolutionary protocols

The built-in protocols expose mutations, selection, founder distributions,
fitness mappings, and stopping conditions in configuration:

- **Wright–Fisher:** score a population, sample parents with replacement using
  fitness weights, then independently mutate each selected offspring with the
  declared probability. A mutating offspring flips one uniformly chosen genome
  bit. Generation zero and the final generation are observed. Zero total parent
  weight gives uniform parent sampling.
- **SSWM:** retain one resident per replicate, evaluate one-bit neighbors, use
  population-size-dependent fixation acceptance to define substitution rates,
  and sample an exponential waiting time and accepted neighbor. A transition
  beyond the time horizon stays pending for continuation. Time is measured in
  the protocol's mutation-time units.
- **Neutral walk:** propose a uniformly chosen one-bit neighbor and accept it
  when the declared phenotype matches the resident's phenotype. Episodic neutral
  walks require a declared discrete descriptor for comparison.

For a tissue target, fitness is the fraction of matching cell-time states with a
configured floor. The target is a copied, hashed `uint8` array of shape
`[steps, cells]`. Other objectives include metabolic coverage and bounded
controller rewards. Reported scores and selection fitness are separate fields;
custom objectives must declare the selection mapping. Wright–Fisher sampling
uses exact integer weights. [Evolution](evolution.md) specifies arithmetic,
fixation limits, mutations, founders, and checkpoint semantics.

Ten replicates of 100 individuals over 100 generation transitions create
100,000 offspring and request 101,000 scores including founders. Those requests
can share completed assays through deduplication and bounded caches. Assay work
depends on genotype diversity, context, mutation, and cache retention.

## Perception–action experiments

A rollout places each candidate controller into a world, with an optional episode
axis for repeated scenarios. One decision observes the committed world, encodes
sensor inputs into chemical ports, performs `chemical_updates` updates, decodes
an action, and advances the world. Inventories persist between decisions unless
a declared reset policy restores initialized controller state. World `dt` and
chemical updates per decision are separate settings.

Built-in worlds provide delayed temporal decisions, two-agent signalling, and
run/tumble motion in a fluctuating field. Ports, motor rules, reward, and world
dynamics are experimental choices documented in [the runner guide](runner.md).
The locomotion task is a dimensionless control environment; it does not implement
the E. coli biochemical transduction pathway. Importable worlds, codecs, and
objectives support other numerical tasks through the same state/checkpoint interface.

An explicit agent axis joins controllers into a coupled world. Actions are decoded
before the simultaneous world update. Population-dependent objectives freeze
the roster and complete its scoring barrier before selection. This prevents
scheduling from changing who interacts or which fitness is used.

## Batching, randomness, and continuation

`BatchRunner` shares immutable tables and a Quadrants runtime. Fixed assays
preserve leading array axes. Population sessions gather ready work across
replicates into bounded candidate or matchup cohorts. Persistent chemistry uses
packed reaction buffers; CPU and Metal world kernels currently transfer NumPy
state at decision boundaries. Custom Python callbacks execute on the host.
Covered integer chemistry and map outputs agree exactly across CPU and Metal.
Floating-point world trajectories can diverge across backends; common logical
random draws do not guarantee identical continuous states.

Fixed chemistry assays are deterministic. Random sampling and evolution use
`philox4x32-10-logical-v1`, with draws addressed by seed and logical roles,
replicates, transitions, individuals, and attempts. Execution placement and
chunk size do not enter those addresses. Episodic environments use panel,
scenario, tick, and role namespaces; candidates within a panel receive common
environmental randomness. Generation panels change between generations for
stochastic Wright–Fisher evaluation, while fixed panels support standalone
assays, SSWM, and neutral walks.

Work limits count evolutionary transitions, logical world ticks, chemical
updates, and convergence work. An unfinished assay remains invalid for selection.
Checkpoints preserve pending chemical decisions, future SSWM events, scoring
rosters, numerical state, and random addresses. Resume starts new invocation
quotas while retaining cumulative work. [Records](records.md) defines durable
transactions, trace budgets, portable capture, and source identities.

## Validation and performance measurement

Qualification combines independently regenerated fold/contact geometry,
scalar–kernel comparisons, conservation of weighted genome counts, backend
agreement, and checkpoint/reproduction checks. Curated assembly and ligand
ambiguity remains source-pinned; the geometry checks identify the coverage they
regenerate. [Qualification](qualification.md) lists completed checks and saved records.

Historical source variants diagnose differences between canonical rules and
reported maps. The isolated state-4 break repair matches the final paper's
aggregate counts; the canonical promoter decisions yield a different map.
Matching final publication class identities and abundances requires its final
generating source or data. [Parity](parity.md) gives the isolated comparisons.

The [C++ benchmark protocol](cpp-benchmark.md) identifies the frozen inputs,
iterator repairs, compiler flags, CPU threads, warm repeats, and timing boundary.
Its primary comparisons retain every genotype and report disagreements. Secondary
cohorts are selected for full output agreement before independent timing.
Measurements include completed public evaluation and host output; initialization,
compilation, and file loading are recorded separately. Report the cohort and
hardware when quoting a speedup. Persistent-session throughput is a separate
workload from the saved fixed-assay timings.

## Reporting an experiment

Retain the run record and report:

- toyLIFE2 version and commit, ruleset, table content identity, and relevant article citations;
- operation, genotype dimensions, founders or supplied genomes, input files, and seed;
- phenotype definition, tissue geometry or world dynamics, inputs, episode panels, and horizon;
- objective, selection mapping, mutation protocol, population size, and transition count;
- backend, CPU threads, chunk/cohort sizes, work and memory budgets, and recording mode;
- completion/status arrays, actual work, checkpoints, and retained or dropped trace/history detail;
- for benchmarks, hardware, cold/warm timing boundary, repeat statistics, cohort selection, and output agreement.

CLI runs capture these settings through the effective spec and manifests. Direct
Python sessions need an explicit recording workflow, described in [runner.md](runner.md).
For stochastic studies, keep evaluation panels and selection settings explicit
so that repeated trajectories measure the intended environmental variation.
