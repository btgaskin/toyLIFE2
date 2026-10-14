# Command-line workflow

Resume can increase the physical trace quota with
`toylife2 resume RUN --destination NEW_RUN --trace-budget-mib 512`. The captured
scientific identity remains the same. A pending full-trace event is committed
before further transitions. Cell expression arrays have shape
`[samples, steps, genes]` with one bit per gene and explicit completion status.

A TOML file defines the scientific experiment and its physical execution. The
parser rejects unknown fields and invalid values. Use `validate` before a long
run:

```sh
toylife2 validate examples/pattern_2020.toml
toylife2 run examples/pattern_2020.toml --workspace research --name central-pulse
toylife2 workspace status research
toylife2 workspace check research
```

`run`, `profile`, and `qualify` accept `--destination PATH` for a standalone
record or `--workspace PATH` with an optional `--name NAME`. With neither
option, the current directory becomes a workspace and the run goes in `runs/`.
An existing nonempty destination is never overwritten. `profile` measures
execution and `qualify` runs extended scientific checks; both produce records.

`--set dotted.path=value` applies a TOML literal and records the resulting
effective specification. For strings, quote both the shell argument and the
TOML value:

```sh
toylife2 run examples/pattern_2020.toml --set 'execution.backend="metal"'
toylife2 run examples/pattern_2020.toml --set 'spatial.cells=61'
```

`resume CHECKPOINT` resumes the exact saved scientific experiment. Pass
`--destination PATH` for a new record and `--backend auto|cpu|metal|cuda` only to change
execution placement. `reproduce RECORD DESTINATION` verifies and reruns a saved
record. `inspect RECORD` prints status, source and input hashes, actual backend,
array inventory, and integrity results. `report RECORD` creates a static HTML
report beside the record; `report --compare RECORD... --destination comparison.html`
creates a provenance comparison from saved records.

`tables check` validates the author lookup files and prints their content
digest. `tables generate` validates them and writes a derived cache to
`.cache/toylife2`; `--source` and `--cache-dir` select alternate paths. A table
cache is an execution optimization. The source lookup files define the model
and are included in each portable run record.

For an evolutionary target, set `spatial.target` to a `.npy` file containing
`uint8` states with shape `[spatial.steps, spatial.cells]` and values 0 through
3. A saved pattern array from a completed pattern run can serve as that file.
The run copies the target into `inputs/` and includes its content hash in the
record's scientific identity. Historical pattern IDs such as 113 depend on a
particular published catalogue; the target file defines the actual selection
objective.

The entry points `python -m toylife2` and `toylife2` are equivalent. Use
`toylife2 --help` for the full command syntax.

## Configuration fields

The top-level fields are `schema_version` (1 or 2; default 1), `operation`, `seed`,
`samples`, and `genotypes`. An explicit `genotypes` list contains binary
strings of length `20 * model.genes`; seeded sampling fills other workflows.
Schema-1 operations are `regulatory`, `metabolic`, `neighbourhood`,
`patterns`, `weighted_map`, `wright_fisher`, `sswm`, `cell`, and `neutral_walk`.
Schema 2 also supports `chemistry`, `tissue`, `rollout`, and `external`, plus
generalized evolutionary sessions. The workflow validates operation-specific
constraints before execution. See [Temporal and batch runs](#temporal-and-batch-runs)
and the [runner guide](runner.md) for schema-2 fields and protocols.

| TOML table | Fields and meaning |
| --- | --- |
| `[model]` | `rules` identifies the canonical rule set; `genes` sets genome size. Published spatial and evolutionary protocols use two genes. |
| `[assay]` | `phenotype` is `metabolic` or `regulatory`; `environment` lists 4–8-symbol binary metabolites for metabolic assays. |
| `[spatial]` | `cells`, `steps`, `boundary` (`open` or `ring`), `diffusion` (`none`, `A`, `B`, `both`, `all`), `morphogen` (`none`, central A/B pulse or continuous input, or `explicit`), optional `initial_state` (one state per cell), `injections` (`[tick,cell,state]` triples for `explicit`), and optional `target` `.npy` path. |
| `[evolution]` | `trials`, `trial_start`, `population_size`, `generations`, `mutation_probability`, `mutation`, `fitness_floor` (in `[0,1)`), `time_horizon`, and `initial` define the selected evolutionary protocol. |
| `[execution]` | `backend` (`auto`, `cpu`, `metal`, or `cuda`; default `auto`), `batch_size`, `memory_budget_mib`, `max_steps`, `max_events`, `max_truth_states`, and `cache_dir` control placement and resource guards. |
| `[recording]` | `mode` (`replayable` or `full`), `checkpoint_interval`, and `trace_budget_mib` control retained detail. |
| `[cell]` | `steps`, `initial_expression`, and `metabolites` configure one-cell molecular dynamics. |
| `[map]` | `start_pair`, optional `stop_pair`, and `tile_pairs` select a bounded weighted-map interval. |

The defaults are serialized into each run's `spec.toml`; inspect that file for
the exact effective values. `replayable` records the complete spec, source,
lookup pack, inputs, seed and checkpoints. `full` additionally retains bounded
per-event detail up to `trace_budget_mib` when the workflow emits it. Trace
truncation is recorded explicitly and makes a run partial when the trace is required. Resource
guards produce explicit partial results or errors; they do not change model
rules.

## Temporal and batch runs

Schema version 2 adds `chemistry`, `tissue`, `rollout`, generalized evolutionary
sessions, and `external` ask/tell runs. See [runner.md](runner.md) and the
`examples/v2-*.toml` files for their protocols.

```sh
toylife2 run-batch examples/batch.toml --destination runs/nested
toylife2 run-batch examples/batch.toml --destination runs/nested --resume
toylife2 resume runs/temporal --destination runs/continued --max-chemical-updates 100000
toylife2 convert-checkpoint runs/legacy-neutral runs/converted-neutral
```

A batch TOML file contains `[runs]`, mapping stable names to configuration paths
relative to that file. Children share one execution configuration and runtime;
compatible fixed assays coalesce across runs. Work quota flags accept zero.
Resume starts a new invocation quota and preserves total logical counters.
External run `evolution.generations` sets the number of completed ask/tell updates;
recording requires an importable `task.config.adapter_entrypoint` factory.

## Current molecular experiment contract

`molecular` creates `single-cell-response-v1` experiments. Profiles are `basic`,
`cue`, `cue_alternatives` and `chemotaxis`. Use `--output-representation bound_free|bound_pair`,
`--output-rule proportional|majority`, `--marker-length 4..8`, `--marker-copies N`
and `--initial independent_random|monomorphic_random`. Founding output assignments
are sampled from `--interface-seed` and saved explicitly. Deprecated molecular
profiles/checkpoints require their archived implementation. See the
[molecular methods](molecular-cells.md) for the experiment boundary.
