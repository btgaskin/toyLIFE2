# Canonical model decisions and flagged comparisons

The canonical rule set is `toylife-canonical-v1`. It joins the revised folding
definition in the bundled `upstream/toylife/toyLIFE.pdf` with the six-state
spatial definition in Catalán, Manrubia, and Cuesta (2020). The PDF explicitly
states that its folding tie rule differs from the original 2014 toyLIFE
publication: equally stable folds with the same perimeter remain functional.
The bundled protein table expresses this revised universe. The 2020 work adds
the two chemical inputs AB+A and AB+B needed to derive spatial cellular
automata from two-gene regulatory networks.

**Canonical interpretation and published parity.** The accepted decisions
below define an executable, versioned interpretation. Complete CPU enumeration
of this interpretation gives 2,183 GRNs, 557 automata, and 191 default patterns,
while the final 2020 article reports 2,152, 453, and 176. The
[spatial reconciliation](parity.md) recovers the publication totals by restoring
the author's missing state-4 break, then reproduces every canonical histogram
weight with three isolated competition changes. The product follows one
canonical chemistry, defined by the [scientific contract](model.md#scientific-contract).
Historical source behaviour is retained in qualification tools to explain
results and isolate implementation differences. The molecular lookup source is
pinned to author
commit `9e17c92ea05f2e3dd63bf437400ddd8e61694ff9`; regeneration and phenotype
tests establish the comparisons they cover. [qualification.md](qualification.md)
records the completed map and the remaining scientific gates.

| Area | Canonical decision | Source and reason |
| --- | --- | --- |
| Folding | Use the supplied revised protein IDs, and independently regenerate the 38 compact fold classes and revised perimeter tie rule | `toyLIFE.pdf`, pp. 2–4; `data/protein_gene.txt`; `data/perimeters.txt` |
| Energies | Store integer tenths and compare exactly | The HP contacts are −2, −0.3, and 0; the source's `d_equal` and `d_less` tolerate floating error at `1e-8` (`helper_functions.cpp:41–47`) |
| Binding threshold | An interaction must be strictly below −2.6, hence at most −2.7 in tenth units | `toyLIFE.pdf`, p. 4; curated lookup entries |
| Promoters | Include polymerase as a real candidate; exclude tied strongest identities from that promoter evaluation, then reconsider the next energy group; tie exclusion does not consume molecules | PDF annihilation rule, pp. 4–5; C++ `toy_plugin.cpp:361–435` excludes weaker-than-polymerase binders and does not explicitly resolve a polymerase tie |
| Promoter admission | Admit only molecules with a valid negative promoter contact before ranking complete bound energies | `patterns/patternsplugin.cpp:372–373`, `411–412`, and related branches rank nonbinders before checking contact; the isolated admission fix explains the 453→557 CA and 176→191 pattern count changes |
| Promoter identity | Group copies of the same protein as one competing identity | PDF identity rule; the spatial source can annihilate same-ID nondimerizing copies; the isolated fix changes 24 GRN abundance bins |
| Promoter resources | Evaluate every promoter against one snapshot; binders are released, then each active gene yields one product | PDF dynamics, p. 7; C++ `toy_plugin.cpp:362–449` |
| Unfolded coding | Preserve the active gene bit and omit functional protein ID zero from chemical inventories | `data/protein_gene.txt`; C++ `toy_plugin.cpp:438–449` keeps expression bits while inserting protein zero |
| Dimers | A same-ID dimer consumes two copies and has capacity `floor(count/2)` | PDF identity and stoichiometric rules, pp. 4–5; C++ candidate multiplicity incorrectly uses `count` (`toy_plugin.cpp:487–525`, `591–602`) |
| Energy-group conflicts | Recompute each candidate's capacity from current availability, sum simultaneous demand, and reject candidates whose reactants are over-demanded | PDF stoichiometric rule; C++ erases map entries during iteration in `toy_plugin.cpp:510–525`, `693–716` |
| Dimer–metabolite fission | Compare every unambiguous subunit attachment after adding the actual subunit's fold energy; emit the two split fragments as a recorded event | PDF metabolism, p. 7; source `prot_breaking.txt` retains only one interaction-minimum attachment before complete-energy comparison (`toy_plugin.cpp:637–659`) |
| Full cell cycle | Translate the current promoter response before the second reaction phase | PDF step dynamics, p. 7; C++ `promoter_expression` mutates the protein inventory (`toy_plugin.cpp:437–449`) |
| Metabolic termination | Stop at first fission, loss of the assay molecule, or repetition of the complete step-boundary chemical state; report a cap as incomplete | PDF step dynamics; C++ `2^genes` cutoff (`toy_plugin.cpp:779–801`) does not cover metabolite-bound states |
| Metabolite identity | Canonicalize a string and its reverse, including fission fragments | PDF p. 9; source tables contain 274 canonical species |
| Tissue input | Keep preformed AB in inputs 3–5; an added A/B in inputs 4–5 does not displace it during promoter evaluation | 2020 paper's two extra inputs; `patterns/patternsplugin.cpp:319–623` |
| Tissue boundaries | Open edges supply zero beyond the tissue; ring edges wrap modulo cell count | `patterns/gpmap_morphogen_choose_diffusion.cpp:127–185`; source reads a negative left index before applying its open-edge guard |
| Time-limited evolution | Apply a sampled transition only when its event time lies within the horizon; count target present at time zero as an arrival at zero | `patterns/simulations_pattern113.cpp:79–92` and `simulations_prob_of_arrival_patterns.cpp:79–130` otherwise apply overshooting events and omit time-zero arrivals |

The exact C++ execution trace is not a portable model target where it erases
`std::map` entries while iterating (`toy_plugin.cpp:421–426`, `510–525`,
`693–716`, `750–765`). The model keeps the stated biological decision rule and
tests it with a deterministic reference implementation.

**Published aggregate comparison reconciled.** The committed
`patterns/results/GRN_abundance.txt` contains 1,472 seven-symbol GRNs, while
the final [2020 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7328398/)
reports 2,152. Both give 453 cellular automata for the P1-diffusion setting.
The source's `patterns/patternsplugin.cpp:403–471` falls through from state 4
into state 5, so the state-4 result can be spuriously activated by a B+dimer
response. Its state-4 and state-5 branches also read dimer energy at index
`-1` before checking whether a homodimer exists (`:405`, `:473`). A
lookup-backed counterexample uses proteins 47 and 264, heterodimer 44, and
promoter 3: the intended state-4 response is zero, while fallthrough makes it
one. Exhaustive enumeration with safe index checks and the missing break
restored gives exactly 2,152 GRNs, 453 CAs, and 176 patterns. The 453-CA total
contains 442 historical rules, removes 11, and adds 11. Adding valid-contact
admission, same-identity grouping, and dimer/polymerase tie annihilation to the
copied author code gives exact agreement with all four canonical weighted
histograms. [parity.md](parity.md) records the six isolated variants, concrete
counterexamples, and the remaining final-source identity comparison.

**Published example files.** The source abundance program writes
`GRN_abundance.txt` and `GRN_ca_1.txt` (`patterns/grn_abundance.cpp:480–493`),
while downstream readers request `logic_abundance.txt` and `logic_ca_1.txt`
(`patterns/simulationsplugin.cpp:80–88` and
`patterns/gpmap_morphogen_choose_diffusion.cpp:49–96`). The committed outputs
are historical comparison artifacts. Reproduction workflows record source
digest, ruleset, dimensions, and all scientific parameters so that new results
can be compared without replacing those artifacts.

**Evolution and analysis protocol identifiers.** The TOML specification
records the chosen sampling rules as scientific parameters. The current
Wright–Fisher protocol samples `population_size` parents with fitness weights
per generation, applies one random bit flip to each offspring selected by
`mutation_probability`, and records target arrival at generation zero when
present. The SSWM protocol uses the declared population size in fixation
probabilities, samples one-bit neighbors by their rates, and records an event
only when its waiting time is within `time_horizon`. It retains a pending
event at the horizon for exact continuation. Random draws use
`philox4x32-10-logical-v1`, addressed by seed, role, trial, event, individual,
and rejection attempt; batch size and backend do not enter those addresses.
The finite-string measure is `author-dictionary-terminal-v1`, with terminal
dictionary insertion and average forward/reverse counts. These identifiers,
the fitness floor, target, tissue geometry, mutation rule, and seed belong to
the recorded scientific identity. Execution caps and batch size govern
completion and resource use; cap-exhausted results remain partial.

**Scope of geometry regeneration.** The independent fold calculation matches
all 65,536 coding assignments, including the all-polar chain's explicit
unfolded status, and all functional folding energies. Independent contact
geometry matches every one of 1,522,172 supplied breaker rows whose perceived
sequence appears in a dimer–metabolite binding. The curated dimer and
ligand-binding tables additionally encode symmetry and ambiguity choices;
their IDs, orientation fields, and source digest remain part of the canonical
input. Qualification of the full batched implementation checks agreement with
the scalar rules, complete weighted-map coverage, exact multiplicity conservation,
backend agreement, and reproducibility. Published aggregates provide a separate
source comparison whose rule differences are recorded explicitly.

The additional binding check reconstructs all listed dimer bond energies,
all protein–promoter energies, and all protein–metabolite energies exactly.
It finds three protein–promoter polymerase annotations on perimeter 375 that
disagree with the adjacent-side contact calculation; the source's false
annotations are retained. Dimer pair membership and dimer ligand binding
depend on assembly orientation and occluded-surface ambiguity that are not
recorded in the provided files. The supplied dimer tables therefore remain
the authority for those discrete decisions.

## Temporal sessions and experimental tasks (schema 2)

The temporal core uses the same `toylife-canonical-v1` molecular decisions.
Each update selects reaction candidates from the phase-start inventory, resolves
integer-energy groups by simultaneous demand and available capacity, evaluates
all promoters from a shared snapshot, replaces free proteins with the expressed
functional products, clears free dimers, performs the second reaction phase,
then releases bound metabolites and retains fission fragments. Intermediate bound
complexes survive between phases. The boundary contains no bound complexes.
Readouts therefore declare `after_input`, `after_output`, or `boundary`; bound
readouts use one of the first two stages. An unfolded gene can have an expression
bit of one while producing no functional protein.

Persistent memory consists of expression and admitted molecular inventories.
Protein accumulation, phosphorylation, higher-order complexes and permanent
receptor/motor assemblies require additional versioned rules. The current
chemotaxis task projects a three-dimensional swimming orientation onto a periodic
square with molecular input ports and a motor readout. Finite tumbles turn the
cell through active rotational diffusion; remaining duration and orientation
persist across decisions. Its motion, concentration field, and score are declared
task parameters. The intracellular controller follows toyLIFE chemistry; the
[molecular cells guide](molecular-cells.md) defines the E. coli-inspired body model.
The signalling game uses two simultaneous agents, alternating sender/receiver
roles, with messages delivered at the next decision boundary. Task defaults and
source are captured in each run. Explicit `reset_per_decision` provides a
memory-reset experimental control.

Three clocks are recorded separately: a chemical update advances each logical
controller, a world tick advances each coupled world, and an evolutionary
transition advances one replicate. Quotas count these logical units, including
episode and agent multiplicities. A zero quota performs no corresponding update;
generation-zero observations remain available when their evaluation quotas allow
completion. A zero-length episode has score zero and no successful decision.

For episodic Wright–Fisher selection, `panel=auto` selects a generation panel:
population g is evaluated on panel g, offspring g+1 on panel g+1. Fixed assays
use a fixed panel. SSWM and neutral walks use fixed panels and frozen opponents.
Shared exogenous environmental randomness excludes genotype and candidate ID.
Task-specific endogenous state and decisions still differ between controllers.
Floating world trajectories reproduce under the same pinned backend and runtime;
CPU/Metal floating trajectories can diverge. Integer molecular transitions are
qualified against the scalar reference.

Selection uses declared objectives. Bounded rewards in [0,1] are quantized with
`floor(D*x + 0.5)`, default D=2**24; exact integer weights and a fixed common
denominator carry a rational fitness floor. SSWM uses the declared raw fitness
mapping. A Python objective must supply selection weights when Wright–Fisher
selection needs them; direction metadata alone does not manufacture weights.
Neutrality uses a discrete descriptor. Contextual self-play scores both roles
against a frozen roster with a complete-population barrier, and disables
per-genotype fitness caching. Pure candidate objectives use a bounded FIFO cache
keyed by evaluation context and ordered genome.

Version-1 specification serialization and scientific hashes retain their original
projection. Version-2 records include numerical session checkpoints and importable
research component archives. Full recording commits the trace and corresponding
state in one manifest revision; an indivisible trace boundary that exceeds its
budget remains pending. External optimizer state includes every RNG cursor and
one outstanding request; a tell or commit failure restores the post-ask state.
Batch manifests index independently committed child records. Recovery may
recompute physical work after the last accepted checkpoint while preserving
logical random addresses and scientific state.

Packed chemistry vocabulary capacity, live numerical buffer budgets, scenario
cohorts, and checkpoint/trace retention are explicit resource controls. Canonical
chemistry buffers persist on the device; built-in world kernels use NumPy arrays
at decision boundaries and currently incur transfers on Metal. This implementation
provides batching and continuation mechanisms; new control workloads need matched
end-to-end timing before any acceleration figure is attached to them.
