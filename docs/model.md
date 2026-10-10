# toyLIFE2 model

toyLIFE2 implements the revised toyLIFE molecular universe and its two-gene
spatial model. A gene has a four-bit promoter and sixteen-bit coding sequence;
`0` is polar and `1` is hydrophobic. The source tables in `upstream/toylife`
define folding, binding geometry, and metabolite interactions. The loader
converts every energy to an integer number of tenths and checks table shape and
perimeter consistency. The original source remains unchanged.

## Scientific contract

The product model is `toylife-canonical-v1`. Only a molecule with a valid
interaction enters energy competition. A protein ID, a sorted pair of protein
IDs for a dimer, or polymerase identifies one promoter competitor. Multiple
copies of that species establish its presence; they do not create additional
competing identities. Copy counts determine reaction capacity, including two
copies for each homodimer.

Reaction candidates are admitted from the inventory at the start of a phase.
Energy groups are processed from strongest to weakest. Each group calculates
all candidate capacities and their combined resource demand against the current
inventory before accepting or applying any reaction. A candidate requiring an
over-demanded reactant is rejected. Capacities for later groups use the updated
inventory, so products can contribute to already admitted reactions. Additional
reaction types wait until the next phase. Promoter competition evaluates exact
integer energies: a tied strongest group is excluded from that evaluation and
the next group is considered. The
molecules remain in the inventory. Reaction conflicts and promoter ties thus
have explicit rules for their respective contexts.

Every promoter reads the same post-reaction snapshot. Translation then replaces
the free protein inventory before the second reaction phase. The spatial assay
can begin with preformed AB and an extra free A or B; that assembled input is
held fixed during promoter evaluation. These are defined starting conditions
for the same binding rules.

Scalar, batched, and persistent execution implement this contract. Leading
batch axes retain independent states; chunking and backend placement preserve
the logical chemistry. Runs record scientific parameters and logical random
addresses separately from execution settings. Historical author-source variants
live in qualification tools and explain earlier results. The
[model decisions](decisions.md) record the detailed rules and source-table
exceptions, and [qualification](qualification.md) records their validation.

## Tables and scalar interface

`Tables.load(source=None, cache_dir=None)` reads the bundled source by default.
The installed wheel carries the lookup files in
`toylife2/_vendor/toylife`; a source checkout reads `upstream/toylife`.
Both locations produce the same table digest. The wheel also carries the
author PDF, C++ sources, source license notice, and the package's build lock
and decisions document for portable run provenance.
Supplying `cache_dir` writes an indexed binary cache keyed by a SHA-256 digest
of all molecular source tables. The returned NumPy arrays are read-only. The
main arrays are:

| Array | Shape | Meaning |
| --- | ---: | --- |
| `gene_to_protein` | 65,536 | Coding sequence to protein ID; zero is unfolded |
| `protein_perimeter`, `protein_energy` | 2,711 | Perimeter ID and folding energy |
| `perimeter_sides` | 379 × 8 | Four sides in each of two geometry orientations |
| `dimer_id` | 379 × 379 | Dimer ID for a sorted perimeter pair, `-1` absent |
| `dimer_energy` | 20,207 | Dimer interaction energy |
| `protein_promoter_energy` | 16 × 379 | Promoter interaction energy |
| `dimer_promoter_energy` | 16 × 20,207 | Dimer–promoter interaction energy |
| `protein_met_energy` | 274 × 379 | Protein–metabolite interaction energy |
| `dimer_met_energy` | 274 × 20,207 | Dimer–metabolite interaction energy |
| `dimer_met_seq`, `dimer_met_pos`, `dimer_met_orientation` | 274 × 20,207 | Bound contact geometry and fission position |
| `breaking_energy` | 3,169 × 379 × 2 | Unique best contact for each possible dimer constituent |

The breaking array is indexed through `breaking_seq_to_index[0:65536]`.
`dimer_met_breakable` distinguishes a valid all-zero contact sequence from
an absent sequence. Each table entry contains an interaction energy; the
simulator adds the actual folded proteins' energies when it compares complete
complexes. Protein–promoter and protein–metabolite energies are indexed by
perimeter because all protein IDs with that perimeter share those interactions.

`ReferenceModel(tables)` is the scalar scientific oracle. It accepts one
encoded genotype, such as `genotypes.encode("0" * 40)`, and provides:

| Method | Result |
| --- | --- |
| `regulatory(genotype)` | Unsigned integer next-state table in ascending input-state order |
| `attractor(genotype)` | Repeating expression states reached from state zero, including `(0,)` |
| `metabolic(genotype, environment, max_steps)` | Cleavage bits and `complete`/`incomplete` status for each metabolite |
| `promoter_response(genotype, proteins, dimers)` | Gene-expression bits for an assembled inventory |
| `spatial_descriptor(genotype)` | Six chemical-input outputs plus dimer-availability bit |
| `step(genotype, CellState(...))` | Next cell state, expression bits, and fission fragments |

Expression bits record promoter activation even when an active gene codes for
unfolded protein zero. The cellular protein inventory omits that product. A
two-gene tissue output applies a separate functional-product mask.

An explicit cell step first forms complexes among free input molecules, then
evaluates every promoter against the same free-molecule snapshot. It replaces
free proteins and dimers with translated products, resolves a second reaction
phase, and carries free proteins, free dimers, and metabolites released from
complexes into the next step. Metabolites introduced by fission remain
available. Free environmental metabolites that never bound are returned to
the environment. `CellState` uses reversal-canonical metabolite strings.

The metabolic phenotype tests one metabolite at a time from each phase of the
regulatory attractor. A successful fission scores one. A repeated complete
chemical state establishes an intact result. A step cap returns `incomplete`
so that resource limits remain visible in results.

For larger genomes, the Quadrants assay follows only regulatory states
reachable from all-off. It stores expression as one bit per gene, gathers the
repeating phase inventories, and evaluates those phases with the same
metabolic chemistry kernel. This path admits genomes beyond the signed
31-bit truth-table limit when their quadratic molecule scratch fits the
declared memory budget. Small finite state spaces retain every chemical
boundary state and stop at the first repeat. Larger spaces use a
constant-memory Brent cycle proof. A step cap can return `incomplete` before
that proof finishes, even when a repeated state occurred earlier. Cleavage
remains a definitive phenotype when observed.

The independent `regeneration.validate_regeneration(tables)` qualification
enumerates all 552 Hamiltonian 4×4 walks and their 38 symmetry classes, folds
all 65,536 coding sequences, and checks energies and functional assignments.
It also checks more than 1.5 million source breaking rows that occur in the
supplied dimer–metabolite universe. The expected report has 52,549 functional
coding sequences, 379 distinct perimeters, and zero mismatches. This check
derives folds and contacts from lattice geometry; it does not replace the
source's curated dimer and ligand disambiguation tables.

`regeneration.validate_binding_regeneration(tables)` independently computes
all 20,207 listed dimer bond energies, all 6,064 protein–promoter energies,
and all 103,846 protein–metabolite energies from four-site contacts. These
energies match every corresponding source cell. The promoter calculation
counts distinct sites on each distinct orientation; metabolite windows tied
at one protein site are one attachment. A neighboring-side polymerase rule
agrees with every source annotation except promoter/perimeter pairs
`(10, 375)`, `(12, 375)`, and `(14, 375)`. The source annotations govern
those three cases. The dimer table's admitted perimeter pairs and its
ligand-binding annotations remain curated geometry inputs because the
source does not include the generator's complete assembly-orientation and
surface-occlusion rules.

The two-gene spatial descriptor has inputs `0` empty, `1` free B, `2` free A,
`3` AB (or free A+B if AB cannot form), `4` AB+free A, and `5` AB+free B.
When AB cannot form, inputs 4 and 5 can instead form AA or BB from the two
copies of the added species. A preformed AB remains assembled while promoters
are evaluated. The seventh descriptor entry records whether AB can form.
The tissue model derives a 64-entry radius-one cellular automaton from this
descriptor and the selected diffusion rule.

The default source includes 274 reversal-equivalent metabolites of lengths
four through eight. Sixty are marked unbreakable. The default metabolic assay
evaluates the other 214; callers can supply an explicit subset, including an
unbreakable signaling molecule.
