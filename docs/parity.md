# Spatial parity reconciliation

The count difference has a reproducible explanation. An isolated copy of the
pinned author code recovers the final paper's **2,152 GRNs / 453 automata /
176 patterns** after restoring a missing `break`. Three further promoter-rule
changes recover the current canonical map, including exact abundances in all
four histograms across `2**40` genomes. Production chemistry remains
`toylife-canonical-v1`.

The experiment was completed on 10 October 2026. Its [saved summary](../evidence/parity-reconciliation.json)
records source and output hashes, patches, comparisons, and sanitizer checks.
The [reproduction script](../benchmarks/reconcile_parity.py) generates separate
source copies and uses the [native counter](../benchmarks/parity_spatial.cpp).

## The three reference points

The pinned [author checkout](https://github.com/PabloCatalan/toylife/tree/9e17c92ea05f2e3dd63bf437400ddd8e61694ff9)
contains the 2019 results associated with the [preprint](https://doi.org/10.1101/817908).
The [final 2020 paper](https://doi.org/10.1098/rsif.2019.0843) reports a larger
GRN count and four additional tissue patterns. The present implementation
uses documented canonical decisions for promoter competition.

| Isolated author-source variant | Six-output signatures | Seven-symbol GRNs | P1 automata | Tissue patterns |
| --- | ---: | ---: | ---: | ---: |
| Safe historical source | 1,271 | 1,472 | 453 | 172 |
| Restore state-4 `break` | 1,948 | 2,152 | 453 | 176 |
| Break + competing-identity rule | 1,948 | 2,152 | 453 | 176 |
| Break + binding eligibility | 1,948 | 2,176 | 557 | 191 |
| Break + eligibility + identity | 1,948 | 2,176 | 557 | 191 |
| Break + eligibility + identity + polymerase ties | 1,955 | 2,183 | 557 | 191 |

Every row covers all **7,349,521 ordered protein pairs**, all **256 ordered
promoter pairs**, and the coding-sequence multiplicity of each protein. The
2,711 protein IDs comprise 2,710 functional proteins and the unfolded sentinel.
Each of the four weighted histograms sums to **1,099,511,627,776**.

A six-output signature records the two expression bits under each of six
chemical inputs. A seven-symbol GRN also records whether the encoded proteins
can form AB. That flag affects transport and therefore belongs to the spatial
descriptor. The author abundance files use this seven-symbol key. Reporting
both counts keeps the molecular truth table and its assembly context explicit;
the final paper's aggregate is recovered by the seven-symbol count above.

## Where the differences arise

**Missing state-4 break.** In `patterns/patternsplugin.cpp:403–471`, the AB+A
branch falls through into AB+B. The second branch can activate an output that
the first branch leaves off. Proteins **47 / 264**, promoter **3**, provide a
lookup-backed example: AB+A should be off while AB+B is on. Restoring the
break changes the historical 1,472 / 453 / 172 totals to 2,152 / 453 / 176.
This isolated change explains the aggregate preprint-to-final shift. Establishing
the authors' actual revision history would require their final generating source.

The automaton total stays at 453, but its membership changes: **442** rules are
shared, **11** are removed, and **11** are added. Of the shared rules, **18**
change weighted abundance. All 172 historical trajectories remain, and four
new trajectories appear. Count agreement therefore needs class comparisons.

**Binding eligibility before energy ranking.** The source's multi-candidate
branches rank `contact + folding energy` before checking for a valid promoter
contact. A stable molecule with contact zero can outrank a real binder and
force a polymerase fallback. The canonical implementation admits only valid
binders into competition.

For proteins **47 / 1303**, promoter **14**, protein 47 has contact −4.3 and
fold energy −10.6; its bound total is −14.9 and it represses. Protein 1303 has
fold energy −16.0 and contact zero. The source ranks the nonbinder first and
falls back to polymerase at −13.9, activating expression. Canonical competition
selects the eligible repressor. Fixing admission in the isolated source gives
**557 automata and 191 patterns**, accounting for the entire difference in
these two distinct counts from the break-fixed author variant.

**Competing identity.** When A and B encode the same nondimerizing protein,
the source can treat their equal-energy copies as competing identities and
annihilate both. Canonical competition groups copies by protein identity.
Protein **174**, promoter **3**, is an activating example. This change affects
**24 GRN abundance bins** in the eligibility-corrected map while preserving
the distinct counts. Its histogram L1 difference is **12,569,432**; this counts
weight removed and added, rather than the number of individual genomes changed.

**Dimer–polymerase ties.** The source's AB+A and AB+B branches clip candidate
energies relative to polymerase at zero. At equality they can fall back to
polymerase. Canonical competition removes tied strongest identities and then
considers remaining candidates. Proteins **90 / 2350**, promoter **5**, form
dimer 442 with total bound energy −15.0, exactly equal to polymerase. Removing
that tied pair leaves A repressing in AB+A and B activating in AB+B.

There are **10 ordered pair/promoter dimer ties** in the supplied tables
(five unordered pairs, all at promoter 5). Applying the tie rule after admission
and identity changes affects **26 GRN abundance bins**, with L1 difference
**31,744**, and changes the distinct GRN count from 2,176 to 2,183. The resulting
logic, GRN, automaton, and pattern histograms equal the recorded canonical map
exactly, including every weight.

## Checks that locate the change upstream of tissue dynamics

The current CA compiler reproduces all **1,472** committed historical
descriptor-to-rule mappings exactly. An independent synchronous NumPy tissue
simulation reproduces all **453** historical P1 automata's saved trajectories,
covering the **172** historical patterns with zero differences.

The assay has **31 cells**, **100 updates**, open boundaries, P1 diffusion,
and an A pulse at cell 15 on update zero. Each trajectory records the 100
post-update rows. These checks cover the actual cell states and schedules,
so trajectory hashing and transient handling do not explain the count gap.
The source-table ordering is also compatible with the current protein-to-
perimeter lookup.

Before enumeration, two reads of nonexistent homodimers at index `-1` are
guarded in the copied source. The safe historical variant reproduces the
committed historical GRN key set exactly. Historical abundance text is rounded;
the new counter retains exact integer weights. ASan/UBSan checks pass on six
selected counter intervals. The diagnostic source preserves the original
tables and generates every semantic variant explicitly.

Lookup-backed regression fixtures check the five examples above against both
the scalar reference and the batched Quadrants paths. Spatial and map tests
pass with these additions.

## Reproduce

From a checkout with the pinned submodule, Python dependencies, and Clang:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/reconcile_parity.py --sanitizer-smoke
```

The script enumerates six source variants and writes counters, patch identities,
and `qualification/parity-source/reconciliation.json`. It verifies the upstream
commit and tracked source, lookup digest, tissue protocol, complete intervals,
genome-weight conservation, historical comparisons, and final canonical
histogram equality. Its counter uses promoter-response frequencies to collapse
the 256 promoter combinations without changing their weights. This is a
reconciliation experiment; its runtime is not an author-C++ performance benchmark.

With existing counters from the same generated source:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/reconcile_parity.py --reuse-counts
.venv/bin/python -m pytest -q tests/test_spatial.py tests/test_maps.py
```

## Product model and diagnostic comparisons

The product follows one canonical chemistry with a
[documented scientific contract](model.md#scientific-contract): valid interactions
enter competition, copies share their molecular identity, resource demand
resolves simultaneously, and each phase has defined inputs and outputs.
The scalar, batched, and persistent paths implement these principles. Their
consistent execution and reproducible results qualify the product model.

The isolated author-source variants remain diagnostic tools. They explain how
historical behaviours affect the map and provide a comparison when implementation
changes are reviewed. They do not add configuration choices to the research
runner. The existing APIs, defaults, rule identifier, and scientific hash
projection continue to describe the canonical chemistry.

The break-fixed diagnostic recovers all final aggregate counts. Exact
final-publication class identities and abundances would require the final
generating data or source for a separate publication-reproduction claim.
The canonical model's relationship to those aggregates is already established
by the complete weighted-map comparison and the concrete fixtures above.
