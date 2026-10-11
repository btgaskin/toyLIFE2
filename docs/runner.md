# Persistent sessions and batches

`BatchRunner` owns one table pack and one Quadrants runtime. It creates interactive
chemical, tissue, task, and evolutionary sessions, and it can record their state
through the same checkpoint format. Schema version 2 TOML files select these
protocols; version 1 files retain their existing fixed-assay behavior.

## Run a task

From the repository root:

```sh
.venv/bin/toylife2 run examples/v2-temporal.toml --destination /tmp/toylife-temporal
.venv/bin/toylife2 run examples/v2-signalling.toml --destination /tmp/toylife-signalling
.venv/bin/toylife2 run examples/v2-evolution.toml --destination /tmp/toylife-evolution
.venv/bin/toylife2 run-batch examples/batch.toml --destination /tmp/toylife-batch
```

Each destination must be new. The batch file names child configurations relative
to its own directory. All children use the same execution configuration. The batch
index reports each child's latest record and completion status; each child record
contains its own durable checkpoints. `run-batch --resume` continues partial
children and retains completed children.

The Python interface supports inspection between decisions:

```python
import numpy as np
from toylife2.batch import BatchRunner
from toylife2.protocols import EpisodeSpec, TaskSpec, WorkLimits

genes = np.array([[907050, 1048465, 769001], [1005689, 644498, 941775]], np.int32)
with BatchRunner() as runner:
    session = runner.rollout(
        genes,
        task=TaskSpec(kind="run_tumble"),
        protocol=EpisodeSpec(horizon=1000, episodes=8, chemical_updates=2),
        limits=WorkLimits(max_world_ticks=160),
        seed=123,
    )
    session.run()
    print(session.state)                  # clocks, scores, status, work counters
    print(session.observe())              # sensors from the committed world
    snapshot = session.snapshot(fields=("world", "scores"), include_pending=True)
    checkpoint = session.checkpoint()
    continued = runner.resume(checkpoint, limits=WorkLimits(max_world_ticks=320))
    continued.run()
```

`step()` advances one complete world decision when its work and recording budgets
permit it. `run(steps=n)` makes at most `n` step attempts. `run()` follows the declared
horizon or a world-tick limit. An unbounded episode requires one of these stopping
conditions. `snapshot()` copies selected numerical state into owned, read-only
arrays. The runner's numerical memory budget covers live session reservations,
resident tables, fixed assay outputs, and temporary numerical work together.
Release completed sessions with `runner.release(session)` to free their
reservation before admitting another workload.
`include_pending=True` adds staged observations, chemical progress, and any prepared
world update; selecting fields also selects which snapshot entries are returned.

## Axes, clocks, and persistent chemistry

The final genotype axis contains ordered genes. Each leading genotype position
creates an independent world. With `episodes=E`, controller arrays have shape
`[*candidate_axes, E, G]`; scores and status have shape `[*candidate_axes, E]`.
`aggregate_scores()` takes the mean across the episode axis.

An explicit `agent_axis` joins agents into one coupled world. For genotypes of shape
`[N, A, G]` with `agent_axis=1`, controllers have shape `[N, E, A, G]` and world scores
have shape `[N, E]`. The signalling TOML example reshapes four sampled genotypes to
`[2, 2, G]` through `task.config.batch_shape`, then joins the second axis through
`task.config.agent_axis`. This produces two candidate pairs, each with four episodes.

A decision observes the old world and its delivered messages, encodes chemical
inputs, performs `chemical_updates=K`, decodes all actions, and advances the world
simultaneously. Chemical inventories survive between decisions. Input mode `pulse`
injects once per decision; `replenish` sets admitted sensor counts at every chemical
update. The canonical chemistry rules release bound metabolites and surviving
fragments at a boundary, and discard free environmental metabolites there.

The chemical update counter measures controller work; `dt` controls world time.
Changing `K` changes chemical processing per decision while preserving the declared
world clock. `reset_per_decision=True` starts each decision from the same initialized
and warmed controller state. Warm-up consumes chemical work once per episode.
Mutable Python codec state is a serialized numerical tree and is staged with the
decision, so a quota retry preserves its encoded inputs.

Work limits count logical lanes. One chemical update costs the number of worlds
times agents; one world decision costs the number of worlds. Evolution updates
use their own counter. These quotas apply to each invocation, and checkpoints also
preserve cumulative totals. A quota below the cost of one simultaneous update
returns a partial result before that update. Resuming starts fresh invocation quotas.

A horizon of zero completes with zero reward samples, score zero, and false success.
A positive horizon completes after its declared ticks. A work or recording limit
produces a partial result with its staged chemical or world progress preserved.

## Built-in worlds and Python callbacks

`temporal_probe` repeats an integer dose sequence and scores a binary action against
the sequence at a declared delay. The example uses `[0, 8, 8, 0]`, one decision of
delay, and two chemical updates per decision. Delayed indices wrap within the
sequence, including the first decision.

`signalling_game` requires two agents. On a send tick, only the sender observes the
private bit. The next tick delivers the sender's binary message to the receiver and
scores the receiver against the saved private bit. A second sensor metabolite
encodes the active observation mask. Adjacent rounds balance private zero and one
for every scenario. The example declares 256 rounds and horizon 512; an explicit
round count `R` requires horizon `2*R`, with horizon zero available for an empty
episode. Mean reward divides by judged receive ticks.

`run_tumble` uses a dimensionless periodic square of side 16. Default world time
steps are 0.05, speed is 1, and rotational diffusion is 0.05. A high motor lowers
the tumble hazard to 0.2; a low motor uses hazard 2. A tumble occupies two stationary
ticks and draws a uniform new heading on completion. Four fixed Gaussian patches
have width 1. Their amplitudes are sigmoids of unit-variance Ornstein–Uhlenbeck
processes with time constant 5, updated by the exact discrete transition. The
field has baseline 0.05, divides by 4.05, and supplies an absolute sensor dose
`floor(8 * field)`. Reward samples the updated field at the resulting position;
the mean exposure is the task score. Agents do not deplete the field.

Built-in numerical world kernels run on Quadrants CPU or Metal. World state and
stochastic inputs currently pass through NumPy arrays at each decision; Metal use
therefore includes host/device transfers. Habitat state uses float32, and reward
aggregation uses float64. `session.profile` reports placement, axes, clocks, and
logical work counts. Timing results for earlier fixed assays describe those
measured workloads; the session API introduces no additional throughput claim.

Exogenous random draws use namespaces for panel, scenario, tick, and role. They
omit candidate and genotype identities, which gives candidates common environmental
randomness within a panel. Fixed panels reuse scenarios; generation panels change
between evolutionary generations while retaining matched scenarios within one
generation.

`TaskSpec(kind="python", entrypoint="module:factory")` imports a world factory.
It receives `config`, `shape`, `agents`, `random`, and `dt`, and returns callbacks
`reset()`, `observe(state)`, and `advance(state, actions)`. The advance callback
returns `(new_state, bounded_reward, channels)`. State and channels must be
serialized numerical trees; callbacks receive copies of the committed state.
Python callbacks execute on the host. `examples/numerical_world.py` is runnable
from the repository and demonstrates this contract.

## Recording and continuation

Replayable mode stores initial conditions, protocol identity, and checkpoints.
Full mode also records chemical inputs, expression, inventories at chemical stages,
readouts, fissions, actions, messages, rewards, and world channels. Its byte budget
bounds retained rows. Required rows that exceed the budget remain pending and pause
advancement. World publication waits for the matching recording boundary; snapshots
continue to show the previous committed world while the pending update is retained.

The CLI resumes a record's latest checkpoint into a new continuation record:

```sh
.venv/bin/toylife2 run examples/v2-temporal.toml \
  --set limits.max_chemical_updates=12 --destination /tmp/toylife-partial
.venv/bin/toylife2 resume /tmp/toylife-partial \
  --max-chemical-updates 1000 --destination /tmp/toylife-continuation
```

For a full trace paused by its byte limit, increase the retained trace budget:

```sh
.venv/bin/toylife2 resume /tmp/toylife-partial \
  --trace-budget-mib 32 --destination /tmp/toylife-larger-trace
```

## Recording Python sessions

Build the effective specification once and use it to construct the session:

```python
from toylife2 import BatchRunner, EpisodeSpec, RunSpec, TaskSpec
from toylife2.runner import reproduce

spec = RunSpec(
    schema_version=2, operation="rollout", seed=42, samples=8,
    task=TaskSpec(kind="temporal_probe"),
    episode=EpisodeSpec(horizon=20, chemical_updates=4),
)
with BatchRunner(spec.execution) as runner:
    session = runner.from_spec(spec, destination="runs/python-control")
    session.step()
    session.run()
    runner.finish(session)
reproduce("runs/python-control", "runs/python-control-replayed")
```

For explicit genomes and stochastic panels, `runner.rollout(genomes, spec=spec,
panel=7, identities=numeric_tree, destination=...)` captures the panel and typed
identity tree alongside the spec. Declare those genomes in `spec.genotypes`,
and use the same task, protocol, controller, seed, work limits, and recording
settings. Omitted settings inherit the supplied spec. A session that differs
from the supplied scientific configuration is rejected at registration.

Pass `tables=Tables.load(source=..., cache_dir=...)` to `BatchRunner` or
`toylife2.runner.run` to record an alternate source pack. Replay restores its captured
source; recording requires the executing arrays to match that source.

`BatchRunner.finish(session)` flushes a final checkpoint and result artifacts for a
recorded session. A runner context manager finishes its registered records on exit.
Interactive recording requires `spec=RunSpec(...)` and a destination or workspace,
so the record can capture the experiment's complete configuration. Unrecorded full
sessions expose `drain_trace()` for an enclosing evaluator to collect committed rows.
Requested neutral descriptors contain ordered discrete actions and messages; reward
scores stay outside the descriptor.

Chemistry plan sharing and device residency apply automatically to persistent
sessions and evolutionary populations. Reuse one `BatchRunner` to retain tables
and compiled host plans between evaluations. Episode lanes keep independent state
while sharing a genome's immutable reaction metadata. Inspection and durable
checkpointing materialise host copies when required. See
[performance](performance.md#persistent-molecular-controllers) for admission,
update blocks, and the benchmark command.
