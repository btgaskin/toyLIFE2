"""Explicit, source-qualified conversion of v1 evolutionary checkpoints."""

from __future__ import annotations

import dataclasses
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .population import PopulationSession
from .random import SCHEME
from .sessions import Checkpoint, pack
from .specs import RunSpec

# These captured files determine v1 inventories, random addresses, molecule
# responses, selection and floating-point transition calculations. New temporal
# session files may legitimately be absent from a v1 source archive.
_NUMERICAL_CORE = (
    "evolution.py",
    "random.py",
    "genotypes.py",
    "tables.py",
    "reference.py",
    "runtime.py",
    "spatial.py",
    "_kernels.py",
    "_trajectory.py",
    "_floating.py",
)


@dataclass(frozen=True)
class PreparedConversion:
    spec: RunSpec
    checkpoint: Checkpoint
    parent: dict
    compatibility: str


def _int(value, name, lower=0, upper=None):
    if type(value) is not int or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid legacy {name}")
    return value


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"invalid legacy {name}")
    return float(value)


def _pending(value, time, genes):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or not {"time", "position"}.issubset(value)
        or set(value) - {"time", "position", "log_wait"}
    ):
        raise ValueError("invalid legacy pending transition")
    result = dict(value)
    if result.get("time") is not None:
        when = _finite(result["time"], "pending time")
        if when < time:
            raise ValueError("legacy pending transition precedes current time")
        _int(result.get("position"), "pending position", upper=20 * genes - 1)
    elif result.get("position") is not None:
        _int(result["position"], "pending position", upper=20 * genes - 1)
    if "log_wait" in result and not math.isfinite(result["log_wait"]):
        raise ValueError("invalid legacy pending log waiting time")
    return result


def convert_legacy_state(
    model, spec, legacy, *, target=None, execution=None, recording=None, limits=None
):
    """Prepare numeric v2 state without advancing chemistry or evolution.

    This lower-level conversion accepts already authenticated input. Recorded
    callers use ``prepare_conversion`` to qualify captured source, tables and
    numerical dependencies before constructing a continuation.
    """
    if spec.schema_version != 1 or spec.operation not in {"wright_fisher", "sswm", "neutral_walk"}:
        raise ValueError("conversion requires a v1 evolutionary checkpoint")
    state = legacy["state"]
    if state.get("algorithm") != spec.operation or state.get("rng") != SCHEME:
        raise ValueError("legacy checkpoint algorithm or RNG scheme mismatch")
    new_spec = dataclasses.replace(
        spec,
        schema_version=2,
        execution=execution or spec.execution,
        recording=recording or spec.recording,
        limits=limits or spec.limits,
    )
    trials = spec.evolution.trials
    trial = _int(state.get("trial"), "trial cursor", upper=trials)
    finished = state.get("trial_status", [])
    if finished != ["complete"] * trial:
        raise ValueError("legacy completed trial inventory disagrees with trial cursor")
    histories = state.get("history", [])
    width = 5 if spec.operation == "wright_fisher" else 4
    rows = []
    for row in histories:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError("invalid legacy retained history")
        lane = _int(row[0], "history trial", upper=trials - 1)
        if lane > trial:
            raise ValueError("legacy history refers to an unstarted trial")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in row
        ):
            raise ValueError("legacy history must contain finite numeric rows")
        if spec.operation == "wright_fisher":
            _int(row[1], "history generation", upper=spec.evolution.generations)
            _int(row[4], "history diversity", lower=1, upper=spec.evolution.population_size)
            if not 0 <= row[2] <= row[3] <= 1:
                raise ValueError("invalid legacy history fitness")
        elif spec.operation == "neutral_walk":
            _int(row[1], "history proposal", upper=spec.evolution.generations - 1)
            _int(row[2], "history mutation position", upper=20 * spec.model.genes - 1)
            _int(row[3], "history mutation acceptance", upper=1)
        else:
            _finite(row[1], "history time")
            _int(row[3], "history event cursor")
            if not 0 <= row[2] <= 1:
                raise ValueError("invalid legacy history fitness")
        rows.append(list(row))
    if "history" in legacy.get("loaded_arrays", {}):
        if not np.array_equal(np.asarray(rows), legacy["loaded_arrays"]["history"]):
            raise ValueError("legacy retained history differs from captured numeric history")
    population_key = "populations" if spec.operation == "wright_fisher" else "founders"
    population = np.asarray(legacy["loaded_arrays"][population_key])
    count = spec.evolution.population_size if spec.operation == "wright_fisher" else 1
    shape = (
        (trials, count, spec.model.genes)
        if count != 1 or spec.operation == "wright_fisher"
        else (trials, spec.model.genes)
    )
    if (
        population.shape != shape
        or population.dtype.kind not in "iu"
        or np.any(population < 0)
        or np.any(population >= 1 << 20)
    ):
        raise ValueError("invalid legacy population arrays")
    if spec.operation != "wright_fisher":
        population = population[:, None, :]
    horizons = state.get("observed_horizons", [None] * trials)
    arrivals = state.get("arrivals")
    if not isinstance(arrivals, list) or len(arrivals) != trials or len(horizons) != trials:
        raise ValueError("legacy observation and arrival axes must match trials")
    terminal = state.get("terminal_times", [None] * trials)
    pending_events = state.get("pending_events", [None] * trials)
    if len(terminal) != trials or len(pending_events) != trials:
        raise ValueError("legacy pending and terminal axes must match trials")
    cursors = np.zeros(trials, np.int64)
    times = np.zeros(trials, float)
    observed = np.full(trials, -1, np.int64)
    lane_status = np.full(trials, "ready", dtype="U12")
    pending = [None] * trials
    for lane in range(trials):
        lane_rows = [row for row in rows if row[0] == lane]
        if lane < trial:
            lane_status[lane] = "complete"
            if spec.operation == "sswm":
                times[lane] = _finite(terminal[lane], "completed terminal time")
                if lane_rows:
                    cursors[lane] = _int(lane_rows[-1][3], "completed event cursor")
                elif times[lane] != 0:
                    raise ValueError(
                        "completed SSWM event cursor was erased by legacy history retention"
                    )
                observed[lane] = cursors[lane]
                pending[lane] = _pending(pending_events[lane], times[lane], spec.model.genes)
            else:
                cursors[lane] = spec.evolution.generations
                observed[lane] = cursors[lane]
        elif lane == trial and trial < trials:
            cursor_name = "event" if spec.operation == "sswm" else "generation"
            cursors[lane] = _int(state.get(cursor_name), cursor_name + " cursor")
            if spec.operation == "sswm":
                times[lane] = _finite(state.get("time"), "time cursor")
                pending[lane] = _pending(state.get("pending"), times[lane], spec.model.genes)
                if lane_rows:
                    observed[lane] = _int(lane_rows[-1][3], "observed event cursor")
                elif horizons[lane] is not None and _finite(horizons[lane], "observed time") == 0:
                    observed[lane] = 0
            elif horizons[lane] is not None:
                observed[lane] = _int(horizons[lane], "observed generation cursor")
        if observed[lane] > cursors[lane]:
            raise ValueError("legacy observation cursor exceeds state cursor")
        if arrivals[lane] is not None:
            arrival = _finite(arrivals[lane], "arrival")
            if arrival > (times[lane] if spec.operation == "sswm" else cursors[lane]):
                raise ValueError("legacy arrival exceeds current cursor")
    session = PopulationSession(model, new_spec, target=target)
    payload = session._checkpoint_payload()
    payload.update(
        populations=population.astype(np.uint32, copy=True),
        cursors=cursors,
        times=times,
        observed=observed,
        arrivals=list(arrivals),
        lane_status=lane_status,
        pending=pending,
        evaluations=[None] * trials,
        round_robin=trial % trials,
    )
    history = []
    for row in rows:
        lane = int(row[0])
        cursor = int(row[3]) if spec.operation == "sswm" else int(row[1])
        if cursor < 0:
            raise ValueError("negative legacy history cursor")
        key = [lane + spec.evolution.trial_start, cursor]
        history.append([key, row])
    keys = [tuple(key) for key, _ in history]
    if len(set(keys)) != len(keys) or keys != sorted(keys):
        raise ValueError("legacy history has duplicate or unordered logical cursors")
    # Retention is pinned to the legacy invocation rather than the new physical
    # memory/trace placement, preserving the retained history exactly.
    capacity = max(
        1,
        min(
            spec.recording.trace_budget_mib * 1024 * 1024,
            spec.execution.memory_budget_mib * 1024 * 1024 // 8,
        )
        // 512,
    )
    if len(history) > capacity:
        raise ValueError("legacy history exceeds declared retention capacity")
    dropped = _int(state.get("history_dropped", 0), "dropped history count")
    payload.update(
        history=history,
        history_capacity=capacity,
        observation_count=len(history) + dropped,
        conversion={
            "source_schema": 1,
            "observed_horizons": list(horizons),
            "trial_cursor": trial,
            "completed_trials": list(finished),
        },
    )
    trace = state.get("pending_trace", [])
    trace = [trace] if isinstance(trace, dict) else list(trace)
    totals = dict(chemical_updates=0, world_ticks=0, evolution_updates=int(cursors.sum()))
    status = "complete" if trial == trials else "ready"
    if trace:
        status = "partial"
    arrays = {}
    encoded = pack(
        {
            "base": {
                "status": status,
                "reason": "trace_budget" if trace else None,
                "used": dict.fromkeys(totals, 0),
                "total": totals,
                "boundaries": 0,
                "pending_trace": trace,
            },
            "payload": payload,
        },
        arrays,
    )
    return new_spec, Checkpoint("population", session.identity, encoded, arrays)


def _qualify_captured_core(record):
    from .records import _hash, _software_environment, _table_archive, inspect_record

    manifest = inspect_record(record)
    package = Path(__file__).resolve().parent
    with zipfile.ZipFile(record / "source" / "source.zip") as archive:
        for name in _NUMERICAL_CORE:
            captured = f"src/toylife2/{name}"
            try:
                source = archive.read(captured)
            except KeyError as error:
                raise ValueError(f"captured v1 numerical source missing: {name}") from error
            if source != (package / name).read_bytes():
                raise ValueError(
                    f"captured v1 numerical source differs: {name}; "
                    "exact conversion requires a separately qualified migration"
                )
    with tempfile.TemporaryDirectory() as temporary:
        tables = _table_archive(Path(temporary))
        if _hash(tables) != manifest["table_pack_sha256"]:
            raise ValueError("current molecular tables differ from captured v1 pack")
    current, expected = _software_environment(), manifest["software_environment"]
    for name in ("python", "implementation", "platform", "machine"):
        if current[name] != expected[name]:
            raise ValueError(f"numerical runtime differs from v1 record: {name}")
    for name in ("numpy", "quadrants", "tomli-w"):
        if current["dependencies"].get(name) != expected["dependencies"].get(name) or current[
            "distribution_records_sha256"
        ].get(name) != expected["distribution_records_sha256"].get(name):
            raise ValueError(f"numerical dependency differs from v1 record: {name}")


def prepare_conversion(checkpoint, model, *, execution=None, recording=None, limits=None):
    """Validate a recorded v1 checkpoint and prepare an explicit linked v2 child.

    Captured code is compared as data. Conversion never imports or executes an
    older source archive. A changed numerical core requires separate qualification.
    The caller writes the returned checkpoint/spec with ``parent`` provenance.
    """
    from .records import _hash, inspect_record, read_checkpoint
    from .runner import _captured_spec, _check_source, _checkpoint_record

    record = _checkpoint_record(Path(checkpoint).expanduser().resolve())
    compatibility = "identical_source"
    try:
        _check_source(record)
    except ValueError as error:
        if not str(error).startswith("executed source differs from record"):
            raise
        _qualify_captured_core(record)
        compatibility = "identical_v1_numerical_core"
    manifest = inspect_record(record)
    actual_digest = getattr(getattr(model, "tables", None), "digest", None)
    if actual_digest is not None:
        from .tables import DEFAULT_SOURCE, _source_digest, _source_files

        captured_digest = manifest.get("table_digest") or _source_digest(
            DEFAULT_SOURCE, _source_files(DEFAULT_SOURCE)
        )
        if str(actual_digest) != str(captured_digest):
            raise ValueError(
                "conversion model molecular table identity differs from captured v1 pack"
            )
    spec = _captured_spec(record)
    loaded = read_checkpoint(checkpoint)
    converted, cp = convert_legacy_state(
        model, spec, loaded, execution=execution, recording=recording, limits=limits
    )
    committed = record / "checkpoints" / loaded["revision"] / "checkpoint.json"
    parent = {
        "relationship": "convert",
        "scientific_identity": inspect_record(record)["scientific_identity"],
        "manifest_sha256": _hash(record / "manifest.json"),
        "checkpoint_revision": loaded["revision"],
        "checkpoint_sha256": _hash(committed),
        "source_schema": 1,
        "target_schema": 2,
        "source_compatibility": compatibility,
    }
    return PreparedConversion(converted, cp, parent, compatibility)
