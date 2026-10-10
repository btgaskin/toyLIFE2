# Run records

A run record is a directory with a committed `manifest.json`. It is readable
while the run is active and after completion. The manifest states whether the
run is `running`, `complete`, `partial`, `interrupted`, or `failed`. An
integrity check covers every committed file with SHA-256 and byte size.

| Path | Contents |
| --- | --- |
| `spec.json`, `spec.toml` | Effective validated run definition. |
| `source/source.zip` | Executed package source, dependency lock when present, and scientific decisions. |
| `source/tables.zip` | Exact executed lookup source files, including alternate source packs. In a workspace this is a link to an immutable content-addressed pack under `packs/`; portable exports include the bytes. |
| `inputs/` | Copied external input files and typed numerical session inputs, with their hashes. |
| `checkpoints/<revision>/checkpoint.json`, `*.npy` | Immutable algorithm/RNG state and large arrays; manifest retains the latest two committed revisions. |
| `arrays/*.npy` | Typed result arrays and per-element status arrays where applicable. |
| `traces/*.jsonl` | Immutable event chunks in `full` recording mode, bounded by `trace_budget_mib`. |
| `summary.json`, `summary.csv` | Aggregate outputs and portable scalar summary. |
| `manifest.json` | Schema, source/input/lock/environment identity, scientific identity, backend, status, artifacts, and file hashes. |

The scientific identity hashes model rules, operation, seed, assay settings,
copied input contents, captured custom components, the executed molecular source
digest, and explicit session inputs. `spec_scientific_identity` captures the
validated spec alone. New manifests declare
`scientific_identity_scheme="spec-inputs-components-tables-session-v1"`; existing
records retain their original identity projection when verified.

A recorded rollout retains its `panel` and numerical `identities` in
`session_inputs`. Arrays use typed NPY files under `inputs/session/`; tuples and
array references use a JSON state tree. Replay restores those inputs before
constructing the session. Scientific identity omits physical execution and
recording settings, including controller capacity and work quotas. The manifest
separately records the requested and actual backend and coalescing metadata.
Standalone and coalesced assays retain the same scientific summaries and arrays,
including neighborhood robustness and neutral mutation counts.

The lookup pack holds authoritative model data. Recording checks the executing
arrays against a fresh parse of their source pack, captures that source, and
checks its digest against the executing tables. Synthetic in-memory tables can
be used for model experiments; a replayable record requires their corresponding
source pack. Parsed and cached tables share shape, dtype, index, and relationship
validation. Cached arrays additionally carry a source-bound content checksum;
invalid caches rebuild from source.

A standalone record owns its pack; workspace runs share a content-addressed
pack. Portable exports include the bytes. Resume and reproduction restore the
captured tables even when the original alternate source directory has moved or
been removed. Named batch recovery uses one table pack per runner and checks it
against the captured batch and children; supply that same `Tables` pack when
recovering a batch, or resume a child directly from its record.

`reproduce` requires the executed package source, Python build, platform,
architecture, and NumPy/Quadrants/tomli-w versions and installed-distribution
fingerprints to match the record. Use the captured source and lock in the recorded
runtime environment for strict replay. A mismatch identifies the changed
component before execution. Cross-platform comparisons and backend qualification
are separate experiments with their own records.

An array result may have a paired status array. `complete` means its value is
valid. Resource-limited or interrupted units carry their own explicit status;
they are never silently converted into biological negatives. A partial record
can be inspected and reported without claiming a complete scientific result.

`toylife2 inspect RECORD` checks the manifest; `toylife2 workspace check PATH`
checks all runs in a workspace. `toylife2 reproduce RECORD DESTINATION` verifies
the source record before replay. Reports read saved arrays and summaries; they
do not rerun model rules. An HTML report defaults to a sibling of the record so
it does not modify the committed data.

The schema starts at version 1. Record writers create a new immutable
checkpoint revision, then atomically replace the manifest pointer. The current
and preceding revisions remain available; older revisions are removed after
the new manifest commits. A crash can leave an uncommitted revision directory.
Verification reports these orphan revisions without treating them as committed
data, and `resume` reads the last committed checkpoint. A full-mode transition
commits its trace chunk and corresponding checkpoint in the same manifest
update; an interrupted write leaves both as uncommitted orphans. At the trace
byte budget, the transition remains in the checkpoint as `pending_trace` and
the run becomes partial. The manifest records accepted rows, bytes, pending
rows, and any rows dropped by direct trace appends.


## Session checkpoints

Schema version 2 sessions use `Checkpoint(kind, identity, state, arrays)`.
JSON state refers to owned numeric/string NumPy arrays; object arrays are rejected.
`Checkpoint.write(directory)` and `.read(directory)` support unrecorded portable
continuation. Recorded sessions commit the same payload through the record writer.

Population checkpoints retain original replicate IDs, independent generation/event
cursors, observed revisions, exact scores and selection mappings, arrivals,
SSWM pending events, the stable scheduling cursor and bounded canonical history.
The history capacity is pinned at creation. Unique observation counts survive
history eviction and resumed rescoring, and `history_dropped` counts those evicted
observations. A converted legacy frontier can recompute missing selection mappings
without creating another observation.

A pending episodic evaluation retains its world state, staged chemical decision,
codec/adapter state and the original bounded candidate/matchup cohort. Complete
pure candidate cache entries accompany the checkpoint as bounded accelerators.
A contextual score retains the frozen roster and completed role evaluations until
the full barrier completes. Nested checkpoint payloads are repacked together so
array references remain unique across population and rollout state.

Session scientific identity includes actual table and target content and captured
custom-component source/dependency identity. Component `config.inputs` values are
normalized by content hash for identity; their current paths remain available to
the running component. Execution placement, controller capacity and work limits
remain physical settings. Invocation counters reset on constructing a resumed
session, while total scientific work counts persist.

Full nested recording drains committed chemical/world rows into population
observation boundaries. An evaluated request records candidate order and validity;
cache reuse records a request/reference without inventing world transitions.
The session's trace budget bounds buffered rows, and refused record transactions
remain pending in its checkpoint for exact retry.

Version-2 manifests declare `record_schema=2`; version-1 manifests remain readable.
`convert-checkpoint CHECKPOINT DESTINATION` explicitly qualifies an evolutionary
version-1 checkpoint and creates a linked version-2 continuation. It checks the
captured numerical core, table pack, dependency fingerprints, and state structure
before conversion. A changed numerical core requires separately qualified
migration. Some completed legacy SSWM records have evicted the only retained event
cursor; those records are inspected normally and conversion reports the missing
cursor. The converter never executes archived Python source.

`reproduce` compares final numerical artifacts, scientific summaries, and total
logical work. Invocation work counters describe how much a particular continuation
executed, so they are retained in records and excluded from reproduction equality.
