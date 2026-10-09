# toyLIFE2

toyLIFE2 is an independent, modernized research implementation of toyLIFE with a Python
interface and batched Quadrants kernels for CPU and Apple Metal. NumPy carries
typed inputs and saved results. Runs capture
configuration, source, lookup tables, inputs, and results for inspection,
checkpoint recovery, and reproduction.

**Early research preview (0.1.0).** The core supports batched assays,
evolutionary populations, and stateful controller–world experiments. APIs and
record formats are still developing. Scientific qualification and the
remaining publication-parity differences are tracked in
[qualification](docs/qualification.md). This project builds on the original
[toyLIFE research and author implementation](https://github.com/PabloCatalan/toylife);
the name identifies this independent project and implies no author endorsement.

The molecular rules follow the revised toyLIFE model described by Catalán,
Wagner, Manrubia, and Cuesta (2018). The two-gene tissue protocol follows
Catalán, Manrubia, and Cuesta (2020): a 31-cell line, 100 updates, open ends,
protein A diffusion, and a central protein A pulse at the start. Both papers and
the precise rule decisions are linked in [the decisions document](docs/decisions.md).

## Modernized for research

- **Batched execution:** regulatory and metabolic assays and spatial patterns
  accept arrays of genotypes, including nested trial and population dimensions.
  Weighted maps use tiled, independently mergeable pair enumeration.
  Execution budgets bound allocations and unfinished work carries an explicit status.
- **Evolutionary experiments:** Wright–Fisher populations, SSWM, and neutral
  walks use recorded protocols and logical Philox random streams. Cached
  genotype evaluations reuse fitness across repeated individuals and generations.
- **Reproducible workflows:** typed TOML configuration and CLI overrides,
  named workspaces, run records, atomic checkpoints, resume, source and table
  capture, integrity checks, reproduction, and HTML reports.
- **Maintainable model boundaries:** separate genotype encoding, immutable
  lookup tables, molecular rules, device kernels, protocols, and recording.
  An independent scalar reference and regenerated geometry validate the kernels.

Version-2 sessions add persistent, multi-species cell chemistry on CPU and
Metal, incremental tissue updates, and controller–world rollouts. A shared
`BatchRunner` owns the runtime and tables. It batches compatible evaluations
across independent evolutionary replicates and retains ordered genomes,
episode states, and pending requests through checkpoint recovery.

## Measured acceleration

On an Apple M5 Mac with 10 CPU cores and Quadrants 1.3.3, a secondary
postselected agreement cohort gave the following median warm evaluation times over seven
repeats. Each evaluation returns every regulatory transition for every
genotype, including allocation and completed host output transfers.

| Regulatory workload | Author C++ (repaired, 1 thread) | Quadrants CPU, 10 threads | Quadrants Metal |
| --- | ---: | ---: | ---: |
| 65,533 two-gene genotypes | 40.86 ms | 3.30 ms (12.4×) | 3.44 ms (11.9×) |
| 65,518 three-gene genotypes | 111.05 ms | 10.47 ms (10.6×) | 9.55 ms (11.6×) |

The single-threaded C++ baseline uses the pinned author code with nine
mechanical iterator-lifetime repairs in a separate generated copy. The
unchanged source triggers sanitizer errors on the frozen samples. Cohorts
retain genotypes whose full outputs agree across C++, CPU, and Metal: three
two-gene and eighteen three-gene rows were excluded from the original 65,536
rows. This postselection describes a shared-output workload; the unfiltered
comparisons and their differences remain recorded. A separate one-thread
Quadrants CPU comparison measured 7.8× and 3.1× acceleration for these cohorts.

These are host-specific measurements of warm batch evaluation. The C++ path
parses prepared binary genotype strings inside evaluation; file I/O, table
loading, and first-call compilation are reported separately. Small batches
can favor C++, and CPU often has lower latency than Metal. The complete
canonical weighted map covers `2**40` genomes in 82.58 seconds on CPU and
79.09 seconds on Metal, with exactly matching histograms; those timings have
no matched C++ full-map baseline.

See the [benchmark protocol and results](docs/cpp-benchmark.md),
[performance guide](docs/performance.md), and
[scientific qualification](docs/qualification.md) for scopes and evidence.

## Populations, trials, and scaling

A study with 10 trials, 100 individuals per trial, and 100 generations makes
100,000 offspring updates. The runner reuses cached fitness and evaluates new
unique genotypes in chunks, which can greatly reduce phenotype computation
when populations contain repeated genotypes. In the recorded two-gene spatial
experiment with 8,192 individuals and 16 generations, 139,264 individual
fitness requests required 562 phenotype evaluations. CPU and Metal produced
identical populations and histories across the tested batch sizes.

The version-2 runner scores ready replicates together in bounded cohorts.
Evolutionary replicates, population members, and stochastic episodes are
separate axes; generations advance through repeated `step()` calls. A
100-member population needs one deterministic evaluation per distinct genome
for a fixed assay. Optional episodes add repeated trajectories for stochastic
worlds. Coupled games use explicit agent axes and a complete-population scoring
barrier. The [runner guide](docs/runner.md) and [evolution guide](docs/evolution.md)
explain these layouts, quotas, and continuation. The version-1 CLI path preserves
its original sequential trial protocol.

Gene count changes the work itself: full regulatory tables have `2**genes`
states, whereas larger-gene metabolism follows reached trajectories. The
recorded metabolic sweep covers 2, 4, 8, 16, and 32 genes. That fixed-assay generic
path constructs and evaluates inventories per genotype. The new persistent
chemistry sessions compile admitted reaction vocabularies into packed device
buffers and batch chemical updates across controllers and episodes. These are
different workloads; the existing timing results apply to the fixed assays.
World kernels currently transfer NumPy state at decision boundaries, so
end-to-end control performance depends on chemical work per decision and batch size.
The [scaling results](docs/cpp-benchmark.md#gene-count-scaling) describe this
bottleneck and preserve the complete outputs and comparison differences.

## Install and run

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are required.
The original model data and source are pinned in a Git submodule. Clone with
submodules, or initialize them in an existing checkout:

```sh
git clone --recurse-submodules https://github.com/btgaskin/toyLIFE2.git
cd toyLIFE2
# For an existing clone: git submodule update --init --recursive
```

From this checkout:

```sh
uv sync --locked
uv run toylife2 validate examples/regulatory.toml
uv run toylife2 run examples/regulatory.toml --destination runs/regulatory-example
uv run toylife2 inspect runs/regulatory-example
uv run toylife2 report runs/regulatory-example
```

Every run needs an explicit random seed when it samples genotypes. An exact
genotype can be supplied in a configuration without sampling. You can change a
field at the command line with a typed TOML override, such as
`--set execution.backend='"metal"'`. The effective configuration is saved in
the run record.

Use `uv run toylife2 workspace init research` to create a named workspace;
pass `--workspace research` to `run`, `profile`, or `qualify` to place a run there.
See [CLI guide](docs/cli.md) and [record format](docs/records.md).
The [architecture](docs/architecture.md) and [evolution guide](docs/evolution.md)
describe execution and sampling; [qualification](docs/qualification.md) tracks
the scientific comparison gates.

## Stateful control and nested experiments

```python
import numpy as np
from toylife2 import BatchRunner, EpisodeSpec, TaskSpec

with BatchRunner() as runner:
    # [replicate, individual, gene]; scenarios are an additional episode axis.
    genomes = np.zeros((10, 100, 2), dtype=np.int32)
    session = runner.rollout(
        genomes,
        task=TaskSpec(kind="temporal_probe"),
        protocol=EpisodeSpec(horizon=100, episodes=1, chemical_updates=4),
        seed=42,
    )
    session.run()
    scores = session.aggregate_scores()  # [10, 100]
```

Each decision perceives the current world, encodes molecular inputs, advances
chemistry for the declared number of updates, decodes an action, and advances
the world simultaneously. Internal state persists between decisions. Chemical
updates, world ticks, and evolutionary transitions have separate work quotas.
A partial tick retains its staged inputs and random addresses for exact
continuation. Run/tumble locomotion and two-agent signalling are included as
small research tasks. Their ports and action rules are explicit experimental
choices, described in [the decisions document](docs/decisions.md).

```sh
uv run toylife2 run examples/v2-temporal.toml --destination runs/temporal
uv run toylife2 run examples/v2-evolution.toml --destination runs/evolution
uv run toylife2 run-batch examples/batch.toml --destination runs/batch
```

Importable Python worlds, codecs, and objectives extend the same numerical
state interface. External optimizers use a transactional `ask`/`tell` adapter;
records capture their declared source, helper modules, dependencies, and input
files. See [the runnable guide](docs/runner.md) for full APIs and recording.

## Scope

The core model maps binary genes to lattice proteins, protein and dimer
interactions, Boolean regulation, and metabolic outcomes. The spatial layer
maps two-gene regulatory functions to cellular automata and expression
patterns. Evolutionary runs use recorded mutation and selection settings.
Configuration controls tissue geometry, input schedules, execution placement,
and recording. Scientific identity excludes hardware and chunk size, so CPU
and Metal executions of the same experiment can be compared directly.

The 2019 author-code results are retained as historical reference data. Its
published preprint reported 1,472 GRNs and 172 patterns; the final 2020 paper
reports 2,152 GRNs and 176 patterns, with 453 automata in both. This version
gap is recorded explicitly in the [decisions document](docs/decisions.md).

The completed CPU enumeration under the accepted canonical decisions gives
2,183 GRNs, 557 automata, and 191 patterns. Strict publication parity remains
open while these differences are reconciled. CPU and physical Metal checks
pass, including exact agreement across the complete weighted map. Metal
access on this host requires execution outside the sandbox.
See [qualification](docs/qualification.md).

## Sources and licence

- Arias et al., [*Scientific Reports* 4, 7549 (2014)](https://doi.org/10.1038/srep07549)
- Catalán et al., [*Journal of the Royal Society Interface* 15, 20170516 (2018)](https://doi.org/10.1098/rsif.2017.0516)
- Catalán et al., [*Journal of the Royal Society Interface* 17, 20190843 (2020)](https://doi.org/10.1098/rsif.2019.0843)
- [Author code and lookup tables](https://github.com/PabloCatalan/toylife)

The author code declares GNU GPL version 3 or later. This implementation and
its included author lookup data are distributed under the same terms.
The author checkout is pinned at `9e17c92ea05f2e3dd63bf437400ddd8e61694ff9`;
its source and model decisions retain their attribution.

See [contributing](CONTRIBUTING.md) for development checks and
[recorded evidence](evidence/README.md) for the published benchmark summaries.
