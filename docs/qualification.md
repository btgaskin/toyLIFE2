# Scientific qualification

Qualification connects three kinds of evidence: geometry regenerated from
first principles, genotype-level agreement between scalar and batched rules,
and aggregate agreement with the published two-gene map. Each result is tied
to a source digest, protocol, backend, and completed work interval. A run is
scientifically qualified for a claim only when the gate for that claim passes.
Selected saved results are included in [recorded evidence](../evidence/README.md),
along with their export manifest and backend identities.

| Gate | Required observation | Current status |
| --- | --- | --- |
| Fold and contact geometry | All 65,536 coding sequences and relevant breaking contacts reproduce the bundled source, including the revised folding tie rule | Implemented and targeted checks pass; run `toylife2 qualify` for a saved report |
| Binding energies | All listed dimer, protein–promoter, and protein–metabolite energies match independently computed contacts; three known polymerase annotations retain source values | Implemented and checked; curated assembly and ligand ambiguity stays source-pinned |
| Scalar vs batched phenotype | Regulatory, metabolic, cell, diffusion, and map-shard fixtures agree with the independent reference at exact integer energies and statuses | Targeted CPU tests pass; fixture coverage is reported separately from exhaustive map coverage |
| Full two-gene map | Complete `2**40` weighted genomes; conservation in every histogram; reconcile published GRN, CA, and pattern counts and identities | Enumeration complete; **publication parity open**: canonical totals are 2,183 GRNs, 557 CAs, and 191 patterns |
| Metal execution | Physical device initializes Quadrants and matches CPU outputs over representative and full-map shards | Passed outside the sandbox: numerical contracts, recorded oracle checks, and the complete map match CPU |
| Reproduction | Portable record verifies exact source, tables, dependency identity, inputs, checkpoints, and output arrays | CPU run/resume/reproduce checks pass; a fresh installed wheel reproduces a CPU record and matches checkout tables and source archive |

The geometry validator enumerates all 552 Hamiltonian walks on the 4×4
lattice, their 38 symmetry classes, all 65,536 coding assignments, and the
source's relevant dimer–metabolite breaking rows. Expected totals include
52,549 functional coding sequences and 379 perimeters. The binding validator
checks 20,207 dimer bond energies, 6,064 protein–promoter energies, and
103,846 protein–metabolite energies. The polymerase exceptions are
`(10, 375)`, `(12, 375)`, and `(14, 375)`; the source annotations are the
canonical inputs for these entries. [model.md](model.md) explains the table
shapes and [decisions.md](decisions.md) explains the exceptions.

## Full-map comparison

The final [2020 article](https://doi.org/10.1098/rsif.2019.0843) reports
2,152 GRNs, 453 cellular automata, and 176 tissue patterns for its published
two-gene setting. The pinned [2019 author
code](https://github.com/PabloCatalan/toylife/tree/9e17c92ea05f2e3dd63bf437400ddd8e61694ff9)
contains a preprint-stage GRN abundance file with 1,472 classes; its
associated preprint reported 172 patterns. The author-code aggregate is a
historical comparison artifact, while the final publication is the gate for
the final protocol. The local implementation corrects specified source-code
fallthrough and undefined-iterator cases according to the written model, so
matching any aggregate requires a term-by-term protocol and rule comparison.

Run or continue the CPU enumeration from the repository root:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/full_map.py
```

The script writes an atomic `qualification/full-map-cpu.checkpoint.json` and,
on completion, `qualification/full-map-cpu.json`. It records the table digest,
Quadrants version, backend, interval cursor, weighted genotype total, four
histograms, and elapsed time. A completed result must cover all 7,349,521
ordered protein pairs and `2**40` weighted genomes. Every histogram must sum
to that genome total. Distinct-count agreement alone is insufficient: compare
the GRN, local-rule, and pattern definitions, per-class multiplicities, and
representative trajectories. Shards can be merged only when their intervals
are adjacent and their table and spatial digests agree.

On 9 October 2026, the CPU enumeration completed all 7,349,521 ordered pairs
and exactly 1,099,511,627,776 weighted genomes in 90.28 seconds. Its totals
were 1,955 six-response logic signatures, 2,183 seven-symbol GRNs, 557 local
rules, and 191 pattern hashes. Each of the four histograms conserves the full
genome total. The result, associations, configuration, and host details are
in `qualification/full-map-cpu.json`. Two independent audits covering 150
uniform and 500 bonded protein pairs, each across all sixteen promoters,
found no descriptor differences from the scalar reference.

A second complete enumeration through the configured run recorder took
82.58 seconds and produced the same histograms. Its captured source, tables,
configuration, outputs, and manifest are in `qualification/full-map-record`.
The independently regenerated geometry and binding checks are saved in
`qualification/cpu-oracle-record`; all covered comparisons pass, with the
three documented polymerase annotations retained. These measured runtimes
describe this CPU host and workload; they establish no speedup relative to
the original implementation.

Physical Metal qualification also passed on this host on 9 October 2026
when run outside the sandbox. `qualification/metal-oracle-record` matches
the CPU record's genotype, regulatory, metabolic, and status arrays exactly.
`qualification/full-map-metal-record` covers the same complete interval in
79.09 seconds; all four histograms, rule-to-pattern associations, pattern
rules, and scientific protocol fields match CPU exactly. Only elapsed time
differs between the map summaries. Both Metal records pass integrity checks.
The earlier sandboxed attempt failed during device initialization with
`maxBufferLength=0`; elevated execution successfully accesses the GPU.

These canonical results differ from the final article's 2,152/453/176
GRN/CA/pattern totals. The accepted polymerase tie interpretation, corrected
homodimer stoichiometry, and corrected six-state dynamics remain explicit
rule choices in [decisions.md](decisions.md). They require a comparison with
the final authors' generating code or complete supplementary protocol before
publication parity can be claimed. The implementation retains this discrepancy
as an open scientific gate; it does not modify rules to force aggregate counts.

## Other checks and claim levels

Run the repository tests with `.venv/bin/python -m pytest -q` and static
checks with `.venv/bin/ruff check src tests benchmarks` and
`.venv/bin/python -m pyright`. `toylife2 qualify`
creates a run record of oracle and geometry comparisons for selected inputs.
After building a wheel, `.venv/bin/python tests/check_wheel.py` installs it
outside the checkout and verifies table identity, identical source packing,
and a recorded CPU run followed by reproduction.
Run `.venv/bin/python tests/check_metal.py` in a fresh process with GPU
access to execute the existing runtime, reachable-trajectory, spatial, and
map tests on Metal. The harness selects Metal for their execution specs;
the Engine disables backend fallback. All 25 tests passed on this host,
including positive cleavage, 32-gene expression, constant-memory cycle
proof, diffusion modes, driven injections, and map resume. Results are saved
in `qualification/metal-tests.json` and `qualification/metal-tests.xml`.
The harness also includes a fresh-process CPU preinitialization regression
test, which checks native architecture aliases and rejects backend mismatch.
`benchmarks/phenotypes.py` records repeatable cold and warm CPU timings for
specified workloads; timings from one host describe that host and workload.

A phenotype result with `complete` status is a result under the recorded
canonical rules and resource limits. A partial result states which units or
traces remain unfinished. Aggregate publication parity requires the full-map
gate above. Accelerated-backend parity requires a device that successfully
initializes and runs the kernels. These distinctions keep each scientific
claim tied to the work actually completed.

## Persistent session qualification

The schema-2 implementation adds CPU tests for heterogeneous persistent chemistry,
staged readouts and inputs, tissue stepping, world/controller decisions, exact
selection, contextual scoring barriers, numerical ask/tell recovery, full nested
traces, source/data capture, and explicit version-1 conversion. The session replay
checks run on each backend in a fresh process:

```sh
.venv/bin/python tests/qualify_sessions.py --backend cpu --output qualification/sessions-cpu.json
.venv/bin/python tests/qualify_sessions.py --backend metal --output qualification/sessions-metal.json
```

The saved checks cover batched temporal probes, run/tumble worlds, signalling games,
and population continuation, including an unfinished chemical decision. Metal is
selected explicitly with fallback disabled. Exact continuation is checked within
each backend; floating world paths are not required to coincide across backends.
Separate Metal chemistry checks matched 1,024 heterogeneous scalar inventories,
recurrent reactions, every tissue boundary/diffusion mode, repeated multi-species
pulse/replenish inputs, and an 80-gene expression-clamp checkpoint.

These checks qualify correctness of the covered session operations. Existing
benchmark figures continue to describe their recorded fixed-assay workloads.

Final implementation checks on 9 October 2026 passed 178 CPU tests, Ruff, and
Pyright (39 source files, zero errors or warnings). The built wheel was installed
under a separate temporary directory and tested outside the checkout: recorded
partial rollout, linked continuation, integrity verification, and reproduction all
passed. The Metal session qualification also passed on the final exported runner.
