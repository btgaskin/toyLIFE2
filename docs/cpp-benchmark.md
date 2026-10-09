# C++ comparison and scaling benchmarks

The comparison uses the author checkout at
`9e17c92ea05f2e3dd63bf437400ddd8e61694ff9`, an Apple M5 Mac with ten CPU
cores and 32 GiB of memory, and Quadrants 1.3.3. Inputs, raw outputs, repeat
durations, compiler commands, source identities, and disagreement examples
are retained under `qualification/` locally. The scripts recreate the inputs
from seed 60213; no source modification is made inside the author checkout.
Selected machine-readable summaries, including the unfiltered comparisons,
are available in [recorded evidence](../evidence/README.md). Its manifest
identifies the local source records and the exported hashes.

## Baselines and numerical agreement

The unchanged author implementation fails ASan on the frozen regulatory and
metabolic inputs: `ToyPlugin::dimerization` advances an erased iterator.
For example, the two-gene regulatory draw fails at row 1, genotype
`[186347, 171462]`, and the three-gene draw fails at row 3,
`[917393, 237919, 918169]`. The positive cleavage fixture
`[907050, 1048465, 769001]` also triggers this failure. Timing this execution
would give no valid original-author speedup estimate.

A separate generated copy applies nine targeted iterator-lifetime repairs:
two promoter competition loops, two dimerization loops, one metabolic
competition loop, and four inventory cleanup loops. The exact replacements
are in `benchmarks/cpp_repairs.py`; each verifies the pinned source text.
The generated source has SHA-256
`a6c190926c1f3eae675e70ab5196b5f5ac0b682eefad2fd6a886a8f61d43089e`.
Astra reviewed the diff: conditions, energy calculations, lookup expressions,
and chemical actions are preserved. Sanitizer and optimized outputs must
agree before a case is timed. First and repeated outputs must remain stable.

The author's spatial descriptor API additionally indexes an absent dimer at
`-1`, producing a sanitizer-confirmed buffer underflow in `logic_proms`.
The first frozen spatial row is `[555454, 355242]`. That case remains an
unavailable C++ timing baseline. Handling absent dimers would require an
additional declared rule choice. The original tissue executable also uses
negative neighbor indexing and consumes precomputed automata; it supplies
no directly matched full tissue or weighted-map timing here.

## Timing boundary

Each sample times the public evaluation call after tables and inputs are
loaded. C++ preparation converts integer genes to binary strings outside
timing; the author method parses those strings inside timing. Quadrants
timing includes input admission, array allocation, upload, kernel completion,
download, and public result construction. Both return all requested values.
File I/O and validation remain outside the timed interval.

The primary warm comparisons use seven repeat blocks, with an adaptive
number of whole evaluations targeting 100 ms per block, capped at 256.
Measurements record every duration, median, interquartile range, and range.
Table loading, initialization, first-call compilation, and process elapsed
time are recorded separately. Backend order rotates across workloads.
Task-owned caches isolate cases. The C++ baseline uses
`clang++ -std=c++11 -O3 -DNDEBUG`; sanitizer builds use ASan and UBSan at
`-O1`. The C++ routine is single-threaded; the main CPU configuration uses
ten Quadrants threads, with a separate one-thread experiment.

These measurements characterize the specified inputs on this host. GPU
activity was not measured with hardware counters. Recorded host load and
thermal diagnostics accompany the results; repeated samples describe runtime
variation for a fixed draw rather than variation across genotype populations.

## Unfiltered workloads

The primary comparison retains every genotype in each draw. Each row below
is a full evaluation of the specified population; times are milliseconds.
Both Quadrants backends have the same disagreement counts against repaired
C++. Every recorded assay completed.

| Workload | Genotypes | Repaired C++ | CPU, 10 threads | Metal | Disagreeing genotypes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regulatory, 2 genes | 256 | 0.156 | 0.382 | 2.350 | 0 |
| Regulatory, 3 genes | 256 | 0.414 | 0.382 | 2.350 | 0 |
| Regulatory, 2 genes | 4,096 | 2.436 | 0.552 | 2.387 | 1 |
| Regulatory, 3 genes | 4,096 | 6.645 | 0.904 | 2.283 | 2 |
| Regulatory, 2 genes | 65,536 | 41.371 | 3.202 | 3.400 | 3 |
| Regulatory, 3 genes | 65,536 | 109.704 | 10.240 | 5.860 | 18 |
| Metabolic, 3 genes × 4 metabolites | 256 | 3.414 | 1.812 | 9.778 | 2 |
| Metabolic, 3 genes × 4 metabolites | 4,096 | 51.000 | 12.569 | 13.693 | 15 |

Ratios on differing outputs are same-input implementation timing ratios.
The machine-readable comparison leaves `equivalent_workload_speedup` empty
for those cases. The differences remain available for rule reconciliation.
The metabolic environment, in order, is `00010001`, `00000011`, `01101010`,
and `10100101`; its first two rows are declared cleavage/cycle fixtures.

Results: `qualification/cpp-benchmark/comparison.json` records original-source
failures; `qualification/cpp-benchmark-repaired/comparison.json` records the
repaired comparison. The repaired run uses a maximum batch size of 65,536.

## Secondary agreement cohorts

A second experiment retains every row whose complete saved outputs agree
across C++, CPU, and Metal. Selection uses outputs, never timings, and all
excluded indices are saved before the new timings. Three two-gene and
eighteen three-gene rows are excluded. Each retained cohort is then validated
again and benchmarked independently.

| Workload | Retained genotypes | Repaired C++ | CPU, 10 threads | Metal |
| --- | ---: | ---: | ---: | ---: |
| Regulatory, 2 genes | 65,533 | 40.859 ms | 3.302 ms / 12.4× | 3.440 ms / 11.9× |
| Regulatory, 3 genes | 65,518 | 111.050 ms | 10.474 ms / 10.6× | 9.553 ms / 11.6× |

These speedups apply to the postselected shared-output cohorts. Agreement
selection can change the distribution of molecular behaviors, so this result
does not establish a general speedup across all genotypes. The separately
timed one-thread CPU cases give 7.8× and 3.1× speedups against their paired
C++ measurements. Cross-run wall times differ; use each record's own paired
measurements when calculating its ratio.

Results: `qualification/cpp-benchmark-agreement/comparison.json` and
`qualification/cpp-benchmark-agreement-one-thread-fixed/comparison.json`.

## Recreating a comparison

Run Metal workloads in an environment with physical GPU access. Each
destination must be new; records are retained rather than overwritten.

```sh
uv run python benchmarks/compare_cpp.py --cpp-variant author \
  --workloads regulatory --gene-counts 2 3 --sizes 256 4096 65536 \
  --batch-size 65536 --destination qualification/author-new
uv run python benchmarks/compare_cpp.py --cpp-variant iterator-safe \
  --workloads regulatory --gene-counts 2 3 --sizes 256 4096 65536 \
  --batch-size 65536 --destination qualification/repaired-new
uv run python benchmarks/compare_cpp.py --cpp-variant iterator-safe \
  --sizes 65536 --batch-size 65536 \
  --agreement-subset-from qualification/repaired-new \
  --destination qualification/agreement-new
```

Use `--backends cpu --cpu-threads 1` for the one-thread comparison.
Metabolic assays use `--workloads metabolic`; `--gene-counts`, `--sizes`,
`--batch-size`, and `--max-steps` specify the scaling experiment. Complete
regulatory tables contain `2**genes` entries per genotype. Larger-gene
metabolic assays instead follow reached regulatory trajectories; these are
distinct workloads. C++ adapter coverage is capped at sixteen genes.

`benchmarks/evolution_scaling.py` measures full canonical two-gene spatial
Wright–Fisher runs, including phenotype caches and host selection. Its cases
vary population, trial count, batching, and founder diversity. This supplies
evolutionary scaling measurements independently of the phenotype microbenchmarks.
An end-to-end C++ evolution speedup requires an additional matched baseline.

## Gene-count scaling

The completed exploratory metabolic sweep uses 128 genotypes, four metabolites,
maximum batch size 128, a 64-step cap, and three repeat blocks. Every CPU and
Metal assay completed, with exact agreement between these backends at all five
gene counts. The first two genotypes are fixed cleavage/cycle fixtures; the
remaining draws use the frozen seed. Warm medians are milliseconds:

| Genes | Repaired C++, 1 thread | CPU, 10 threads | Metal | Genotypes differing from C++ |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.683 | 19.460 | 12.742 | 0 |
| 4 | 3.330 | 4.002 | 9.939 | 1 |
| 8 | 134.786 | 367.132 | 1,627.810 | 8 |
| 16 | Diagnostic timed out | 559.511 | 2,632.593 | Unavailable |
| 32 | Outside adapter coverage | 581.080 | 2,663.679 | Unavailable |

This sweep ran under substantial host load: recorded load averages rose from
7.91/15.57/11.65 to 18.66/22.27/16.65. Treat its times as exploratory observations
under those conditions. The sixteen-gene C++ sanitizer probe reached its
300-second timeout at row 84; optimized timing was skipped. This supplies no
estimate of optimized C++ throughput at sixteen genes.

For five or more genes, the current implementation follows regulatory cycles
without materializing the full truth table. It constructs cycle inventories
and dispatches metabolic assays separately for each genotype. This reduces
state-space storage but exposes host dispatch and transfer overhead, especially
on Metal. Batching reachable inventories across genotypes is a separate
optimization task. `benchmarks/reachable_batched.py` contains an unvalidated,
unintegrated prototype; it contributes no production result or timing here.

Results: `qualification/metabolic-gene-scaling/comparison.json`, with full
per-backend values and statuses in each case directory. A reached trajectory
can remain short as gene count rises; these observations characterize these
draws and do not determine a general gene-count complexity curve.

## Evolutionary workload reuse

The completed two-gene spatial Wright–Fisher sweep varied population size
(128, 1,024, 8,192), trial count (one or four), chunk size (32, 256, 1,024),
and founder diversity over sixteen generation transitions. All fourteen
CPU/Metal cases completed. Scientific outputs match across backends and
the chunk-size controls. Results are in `qualification/evolution-scaling.json`.

| Example | Individual fitness requests | Phenotype evaluations | Evaluations avoided |
| --- | ---: | ---: | ---: |
| One trial, 128 individuals, one founder | 2,176 | 18 | 99.17% |
| One trial, 1,024 individuals, one founder | 17,408 | 118 | 99.32% |
| One trial, 8,192 individuals, one founder | 139,264 | 562 | 99.60% |
| Four trials, 1,024 individuals each, one founder per trial | 69,632 | 275 | 99.61% |
| One trial, 1,024 independently sampled individuals | 17,408 | 1,183 | 93.20% |

Avoided evaluations combine within-call deduplication and bounded cached reuse
across generations and trials. Initial diversity and mutation rate determine
how much work is reusable. These counts explain a useful scaling mechanism
for evolution; they supply no matched end-to-end C++ evolution speedup.

For the illustrative ten-trial, hundred-individual, hundred-generation study,
use the [configuration and guidance](evolution.md#ten-independent-100-individual-trials).
Its wall time depends on hardware, active genotype diversity, tissue duration,
cache reuse, and recording. No timing prediction is attached to that example.
