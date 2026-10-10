# toyLIFE2

toyLIFE2 is an independent, modernized research implementation of toyLIFE: an
artificial chemistry in which binary genes encode lattice proteins, molecular
interactions regulate expression, and dimers can cleave metabolites. A Python
interface and batched [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)
kernels run on CPU and Apple Metal. NumPy arrays carry typed inputs and results.
Recorded experiments capture configurations, source, lookup tables, inputs,
checkpoints, and outputs.

**Early research preview (0.1.0).** The core supports fixed assays, evolutionary
populations, and persistent controller–world experiments. APIs and record formats
are developing. This project builds on the
[author implementation](https://github.com/PabloCatalan/toylife); its name identifies
an independent implementation and implies no author endorsement.

The molecular model follows the revised definition used by
[Catalán, Wagner, Manrubia, and Cuesta (2018)](https://doi.org/10.1098/rsif.2017.0516).
The two-gene tissue assay follows the protocol in
[Catalán, Manrubia, and Cuesta (2020)](https://doi.org/10.1098/rsif.2019.0843).
The [methods guide](docs/methods.md) explains both and the added experimental
interfaces. [Model decisions](docs/decisions.md) document the canonical rules and
[parity reconciliation](docs/parity.md) explains their relationship to published results.

[Quick start](#install-and-run) · [Methods](#model-and-methods) ·
[Batching and evolution](#populations-trials-and-scaling) ·
[Results](#qualification-and-published-results) · [Documentation](#documentation)

## Modernized for research

- **Batched execution:** regulatory, metabolic, and tissue assays accept arrays
  with nested population and replicate dimensions. Weighted maps enumerate
  independently mergeable tiles. Execution budgets bound memory and work.
- **Evolutionary experiments:** Wright–Fisher populations, strong-selection,
  weak-mutation (SSWM) trajectories, and neutral walks use explicit mutation and
  selection protocols. Deduplication and bounded caches reuse completed assays.
  Logical Philox random streams preserve draws across chunk sizes and backends.
- **Stateful experiments:** persistent cell chemistry, tissue stepping, and
  perception–action loops support changing environments and coupled agents.
  `BatchRunner` shares tables and a runtime across compatible work.
- **Reproducible workflows:** typed TOML, CLI overrides, workspaces, atomic
  checkpoints, resume, integrity checks, reproduction, and HTML reports.
- **Maintainable structure:** genotype encoding, immutable lookup tables,
  molecular rules, kernels, protocols, and recording are separate modules.
  A scalar reference and independently regenerated geometry check the implementation.

## Install and run

Use Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/). CPU is the default;
Metal requires a compatible Apple device. The repository is currently private,
so cloning requires access. Model data and author source are pinned in a Git submodule:

```sh
git clone --recurse-submodules https://github.com/btgaskin/toyLIFE2.git
cd toyLIFE2
uv sync --locked
# For an existing clone: git submodule update --init --recursive
```

Run a small regulatory assay and inspect its saved results:

```sh
uv run toylife2 validate examples/regulatory.toml
uv run toylife2 run examples/regulatory.toml --destination runs/regulatory-example
uv run toylife2 inspect runs/regulatory-example
uv run toylife2 report runs/regulatory-example
```

The destination must be new or empty. `inspect` checks file integrity and reports
status, array inventory, and provenance. `report` writes an HTML file beside the
run directory. The example configuration is:

```toml
schema_version = 1
operation = "regulatory"
seed = 20201003
samples = 8

[model]
genes = 2

[execution]
backend = "cpu"
batch_size = 256

[recording]
mode = "replayable"
```

Sampling needs an explicit seed. To supply exact genomes, use a `genotypes` list
of binary strings with `20 * genes` bits each. Typed overrides change fields and
save the effective configuration. For example, on Apple hardware:

```sh
uv run toylife2 run examples/regulatory.toml --destination runs/regulatory-metal \
  --set 'execution.backend="metal"'
```

Schema 1 defines fixed assays and the original evolutionary workflows. Schema 2
adds persistent chemistry, tissue, task, and population sessions. Existing
[example configurations](examples/) provide starting points for both;
[the CLI guide](docs/cli.md) describes all commands and fields.

## Model and methods

Each gene contains a four-bit promoter and a sixteen-bit coding sequence.
Coding bits represent polar (`0`) or hydrophobic (`1`) units that fold on a
4 × 4 lattice. Protein identity is its folded perimeter and energy; the pinned
lookup tables encode folding, binding geometry, and metabolite interactions.
Energies use integer tenths for exact comparisons.

One cell update forms complexes, evaluates all promoters from a common snapshot,
translates active genes, performs a second reaction phase, and releases bound
metabolites and surviving fission fragments. Only valid interactions enter energy
competition. Molecular identities compete at promoters; copy counts set reaction
capacity. Reactions at equal energy resolve shared resource demand simultaneously.
The [model contract](docs/model.md#scientific-contract) specifies ties, phase
boundaries, and inventory handling.

| Experiment | Method and output | Starting configuration |
| --- | --- | --- |
| Regulatory assay | Enumerate Boolean input states and return the next-expression table | [regulatory.toml](examples/regulatory.toml) |
| Metabolic assay | Test declared metabolites from regulatory attractor phases; return cleavage bits and completion status | [metabolic.toml](examples/metabolic.toml) |
| Tissue pattern | Derive a two-gene local rule; update a line of cells synchronously and retain cell-time states | [pattern_2020.toml](examples/pattern_2020.toml) |
| Weighted map | Enumerate protein/promoter classes and weight them by exact coding multiplicities | [weighted-map.toml](examples/weighted-map.toml) |
| Stateful task | Alternate chemical processing and world updates with persistent internal state | [v2-temporal.toml](examples/v2-temporal.toml) |
| Evolution | Evaluate populations, select parents or substitutions, mutate, and record transitions | [v2-evolution.toml](examples/v2-evolution.toml) |

The default tissue assay uses 31 cells, open boundaries, A diffusion, and a
central A pulse at tick zero. Its output contains 100 **post-update** rows.
Other geometry and input schedules define different experiments.

The [methods guide](docs/methods.md) describes assays, evolution, control,
randomness, measurement, and reporting choices. The [model reference](docs/model.md)
provides lookup shapes and scalar APIs.

## Python batches

The final genotype axis holds ordered genes encoded as integers in `[0, 2**20)`.
Every leading axis is preserved. A regulatory assay with two genes returns four
input-state transitions per genotype:

```python
import numpy as np
from toylife2 import ExecutionSpec, Model

genomes = np.random.default_rng(42).integers(
    0, 2**20, size=(2, 8, 2), dtype=np.uint32
)  # [replicate, individual, gene]
model = Model(execution=ExecutionSpec(backend="cpu", batch_size=16))
result = model.regulatory(genomes)
assert result.complete
print(result.values.shape)  # (2, 8, 4)
```

`Model` also exposes metabolic assays, tissue patterns, one-bit neighborhoods,
and robustness. `ArrayResult` owns read-only values and status arrays. Check
status before using capped or interrupted assays as fitness. A shared runner
admits live session buffers, resident table uploads, assay outputs, and numerical
scratch against one memory budget. Fixed assays share the runner's Engine;
compatible named assays can coalesce while retaining standalone result artifacts.

## Stateful control and nested experiments

`BatchRunner` creates sessions that can advance one decision at a time:

```python
import numpy as np
from toylife2 import BatchRunner, EpisodeSpec, TaskSpec

genomes = np.random.default_rng(42).integers(
    0, 2**20, size=(10, 100, 2), dtype=np.uint32
)  # [replicate, individual, gene]
with BatchRunner() as runner:
    session = runner.rollout(
        genomes,
        task=TaskSpec(kind="temporal_probe"),
        protocol=EpisodeSpec(horizon=100, episodes=1, chemical_updates=4),
        seed=42,
    )
    session.step()  # one perception–action decision
    session.run()  # continue to the horizon or a resource boundary
    scores = session.aggregate_scores()  # [10, 100]
```

Each decision observes the current world, encodes molecular inputs, advances
chemistry for the declared number of updates, decodes actions, and advances the
world simultaneously. Internal state persists between decisions. Chemical
updates, world ticks, and evolutionary transitions have separate work quotas;
a partial decision retains staged inputs and random addresses for continuation.

Built-in tasks include a delayed temporal probe, two-agent signalling, and
run/tumble locomotion in a fluctuating field. These are experimental toy tasks;
the run/tumble task uses declared sensor and motor rules rather than reproducing
the E. coli molecular pathway. Their definitions are in the
[runner guide](docs/runner.md) and [model decisions](docs/decisions.md).

```sh
uv run toylife2 run examples/v2-temporal.toml --destination runs/temporal
uv run toylife2 run examples/v2-evolution.toml --destination runs/evolution
uv run toylife2 run-batch examples/batch.toml --destination runs/batch
```

Importable Python worlds, codecs, and objectives extend the numerical state
interface. External optimizers use a transactional `ask`/`tell` adapter.
Records capture declared component source, helper modules, dependencies, and
input files. See [the runner guide](docs/runner.md) for inspection, checkpointing,
custom components, and recording Python sessions.

## Populations, trials, and scaling

Evolutionary replicates, population members, and stochastic episodes are separate
axes. Schema-2 population sessions score ready replicates together in bounded
cohorts and advance generations through `step()`. A fixed deterministic assay
needs one completed evaluation per distinct genome and context. Optional episodes
add repeated trajectories in stochastic worlds. Coupled games declare an agent
axis and complete population scoring before selection.

Ten independent trials of 100 individuals over 100 generation transitions
produce 100,000 offspring. Observing founders and every resulting generation
requests 101,000 individual scores; deduplication and caches determine the
number of actual assays. The [evolution guide](docs/evolution.md) includes that
configuration and explains founders, mutation, selection, and stopping rules.
Schema-1 evolutionary runs process trials sequentially and batch their assays;
schema 2 also combines ready work across replicates.

Recorded two-gene tissue experiments show how diversity affects reuse. With
1,024 individuals over 16 transitions, 17,408 requests needed 118 phenotype
evaluations from a single founder, or 1,183 with independently sampled founders.
An 8,192-member single-founder run needed 562 evaluations for 139,264 requests.
CPU and Metal matched populations and histories across tested chunk sizes.
See [evolution scaling data](evidence/evolution-scaling.json).

Gene count changes the workload: a full regulatory table has `2**genes` states,
while large-genome metabolic assays follow reached trajectories. The recorded
metabolic sweep covers 2, 4, 8, 16, and 32 genes. Persistent chemistry sessions
instead compile admitted reaction vocabularies into packed device buffers.
World kernels currently transfer NumPy state at decision boundaries, so control
throughput depends on chemical work per decision and batch size. The
[scaling guide](docs/cpp-benchmark.md#gene-count-scaling) describes these distinct
workloads; evolutionary wall time also depends on diversity and recording.

## Runs and workspaces

A standalone run owns its captured inputs and model data. A workspace groups
named runs and shares immutable table packs:

```sh
uv run toylife2 workspace init research
uv run toylife2 run examples/v2-temporal.toml --workspace research --name temporal
uv run toylife2 workspace check research
uv run toylife2 resume research/runs/temporal --destination runs/continued
uv run toylife2 reproduce runs/regulatory-example runs/regulatory-reproduced
```

Records contain the effective TOML/JSON specification, source and dependency
identity, lookup tables, copied inputs, result arrays, summaries, and checkpoints.
`recording.mode="full"` additionally retains emitted event detail within a byte
budget. Resource limits produce explicit partial status. Resume uses the last
committed checkpoint and new invocation quotas.

Scientific identity includes the model, protocol, seed, actual molecular table
content, copied inputs, and explicit rollout panels and numerical identities.
Recorded Python sessions use a matching `RunSpec` plus a destination or workspace;
`runner.from_spec(spec, destination="runs/control")` is the shortest route.
Alternate source table packs supplied through `Tables` are captured and restored
from the record for resume and reproduction.
Execution placement and chunk size are recorded separately, supporting recovery
with different physical budgets. Covered integer chemistry and map outputs agree
exactly across CPU and Metal; floating-point world trajectories can diverge.
See [record format](docs/records.md) for retained files, portable records, and
integrity checks and a [recorded Python example](docs/runner.md#recording-python-sessions).
`reproduce` uses the captured tables and inputs and requires the recorded source,
Python runtime, platform, and numerical dependency fingerprints. Creating a
`Model` or `BatchRunner` alone leaves recording to the caller.

## Qualification and published results

The product uses one canonical chemistry, `toylife-canonical-v1`. Qualification
covers independent fold/contact regeneration, scalar–kernel fixtures, persistent
sessions, checkpoint reproduction, and complete two-gene weighted enumeration.
The recorded CPU and physical Metal maps agree exactly in all four histograms
and conserve all `2**40` genome weights.

| Map | Seven-symbol gene regulatory networks | Cellular automata | Tissue patterns |
| --- | ---: | ---: | ---: |
| Final 2020 article, reported counts | 2,152 | 453 | 176 |
| Isolated author source with missing state-4 `break` restored | 2,152 | 453 | 176 |
| Current canonical chemistry | 2,183 | 557 | 191 |

The source variant reproduces the article's reported aggregate counts. Comparing
its class identities and abundances with the final publication requires the
final generating source or data. Three further isolated promoter-rule changes
reproduce every canonical histogram weight: valid-contact admission, grouping
copies by molecular identity, and dimer/polymerase tie handling. Historical
variants remain qualification diagnostics. The
[reconciliation](docs/parity.md), [saved comparison](evidence/parity-reconciliation.json),
and [qualification guide](docs/qualification.md) give the rules, counterexamples,
protocols, and completed checks.

## Performance measurements

The following warm regulatory evaluations were recorded on an Apple M5 Mac with
10 CPU cores, 32 GiB memory, and Quadrants 1.3.3. Values are medians over seven
repeat blocks in milliseconds. The primary draws retain all genotypes; the
secondary cohorts retain rows whose complete outputs agree across all three paths.

| Cohort | Genes | Genotypes | Repaired C++, 1 thread | CPU, 10 threads | Metal | Genotypes differing from C++ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unfiltered | 2 | 65,536 | 41.371 | 3.202 | 3.400 | 3 |
| Unfiltered | 3 | 65,536 | 109.704 | 10.240 | 5.860 | 18 |
| Agreement-selected | 2 | 65,533 | 40.859 | 3.302 | 3.440 | 0 |
| Agreement-selected | 3 | 65,518 | 111.050 | 10.474 | 9.553 | 0 |

For the secondary shared-output cohorts, CPU acceleration was 12.4× and 10.6×,
and Metal acceleration was 11.9× and 11.6×. A separate one-thread CPU comparison
measured 7.8× and 3.1×. Agreement selection changes the genotype distribution;
these ratios apply to the recorded cohorts. The unfiltered differences and
outputs remain available in [the benchmark evidence](evidence/README.md).

The baseline is a separate copy of the pinned author C++ with nine mechanical
iterator-lifetime repairs. The unchanged source fails sanitizers on the frozen
samples. Timings include public evaluation, allocation, and completed host output;
C++ parses prepared binary genotype strings inside evaluation. File I/O, table
loading, and first-call compilation are recorded separately. Small batches can
favor C++, and CPU often has lower latency than Metal.

The complete canonical weighted map took 82.58 seconds on CPU and 79.09 seconds
on Metal in recorded runs; these have no matched C++ full-map timing baseline.
The [benchmark methods](docs/cpp-benchmark.md) define repairs, timing boundaries,
input selection, and reproduction commands. The [performance guide](docs/performance.md)
describes execution tuning and current bottlenecks.

## Documentation

| Guide | Contents |
| --- | --- |
| [Methods](docs/methods.md) | Model, assays, evolutionary and control protocols, reporting checklist |
| [CLI](docs/cli.md) | TOML fields, overrides, workspaces, resume, reproduction, reports |
| [Runner](docs/runner.md) | Persistent APIs, axes, clocks, quotas, tasks, external optimizers |
| [Evolution](docs/evolution.md) | Wright–Fisher, SSWM, neutral walks, scoring, logical randomness |
| [Model](docs/model.md) | Chemical contract, tables, scalar APIs, tissue rules |
| [Decisions](docs/decisions.md) | Accepted interpretations, source exceptions, experimental task choices |
| [Architecture](docs/architecture.md) | Modules, data flow, runtime ownership |
| [Records](docs/records.md) | File layout, identities, traces, checkpoint transactions |
| [Qualification](docs/qualification.md) and [parity](docs/parity.md) | Validation and historical comparisons |
| [Benchmarks](docs/cpp-benchmark.md) and [evidence](evidence/README.md) | Measurement protocols and machine-readable results |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, static checks, packaging |

## Articles and attribution

- **Original framework:** Arias, C. F., Catalán, P., Manrubia, S., and Cuesta, J. A.
  (2014). [toyLIFE: a computational framework to study the multi-level organisation
  of the genotype-phenotype map](https://doi.org/10.1038/srep07549).
  *Scientific Reports* **4**, 7549.
- **Revised molecular model and metabolic landscapes:** Catalán, P., Wagner, A.,
  Manrubia, S., and Cuesta, J. A. (2018). [Adding levels of complexity enhances
  robustness and evolvability in a multilevel genotype–phenotype map](https://doi.org/10.1098/rsif.2017.0516).
  *Journal of the Royal Society Interface* **15**(138), 20170516.
  [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5805960/) ·
  [Supplementary methods](https://doi.org/10.6084/m9.figshare.c.3950983.v1).
- **Spatial patterns and evolutionary accessibility:** Catalán, P., Manrubia, S.,
  and Cuesta, J. A. (2020). [Populations of genetic circuits are unable to find the
  fittest solution in a multilevel genotype–phenotype map](https://doi.org/10.1098/rsif.2019.0843).
  *Journal of the Royal Society Interface* **17**(167), 20190843.
  [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7328398/).

When reporting research, cite the articles relevant to the model and assay,
identify toyLIFE2 by version and commit, and retain the run's scientific identity,
configuration, and input hashes. This connects a result to its actual chemistry
and protocol. The [methods checklist](docs/methods.md#reporting-an-experiment)
lists the remaining details needed to reproduce an experiment.

## Licence

The [author code and lookup tables](https://github.com/PabloCatalan/toylife) are
pinned at commit `9e17c92ea05f2e3dd63bf437400ddd8e61694ff9`. The author code
declares GNU GPL version 3 or later. This implementation and its included author
lookup data are distributed under the same terms; see [LICENSE](LICENSE).
Source attribution and documented model decisions accompany the implementation.
