"""Run, checkpoint, replay, profile, and qualify concrete research records."""

from __future__ import annotations

import dataclasses
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from . import Model
from .analysis import pattern_summary
from .genotypes import encode
from .random import SCHEME, Stream
from .records import (
    begin_record,
    finalize_record,
    inspect_record,
    load_record_spec,
    read_checkpoint,
    verify_record,
)
from .specs import ExecutionSpec, RecordingSpec, RunSpec, plain
from .workspaces import new_record_path


def _inputs(spec: RunSpec) -> dict:
    if spec.schema_version == 2:
        from .components import input_files

        return input_files(spec)
    return {"target": Path(spec.spatial.target)} if spec.spatial.target else {}


def _destination(spec: RunSpec, workspace, destination) -> Path:
    if destination is not None:
        return Path(destination)
    return new_record_path(workspace or "workspaces/default", spec)


def _genotypes(spec: RunSpec) -> np.ndarray:
    if spec.genotypes:
        return encode(spec.genotypes)
    if spec.seed is None:
        raise ValueError("sampling requires a seed")
    return Stream(spec.seed).genotypes(spec.samples, spec.model.genes)


def _target(record: Path, spec: RunSpec) -> np.ndarray:
    manifest = inspect_record(record)
    target = manifest["inputs"].get("target")
    if target is None:
        raise ValueError("evolution requires spatial.target: a .npy tissue array")
    array = np.load(record / target["path"], allow_pickle=False)
    if array.shape != (spec.spatial.steps, spec.spatial.cells):
        raise ValueError("target dimensions differ from the configured tissue")
    return array


def _evaluate(model: Model, spec: RunSpec, genes: np.ndarray):
    import copy

    execution = model.engine.execution
    if spec.schema_version == 2:
        model = copy.copy(model)
        model.execution = dataclasses.replace(
            model.execution, max_steps=min(model.execution.max_steps, spec.limits.max_assay_steps)
        )
        model.engine.execution = model.execution
    try:
        if spec.operation == "regulatory":
            return model.regulatory(genes)
        if spec.operation == "metabolic":
            return model.metabolic(genes, spec.assay.environment)
        if spec.operation == "neighbourhood":
            return model.neighbourhood(
                genes, phenotype=spec.assay.phenotype, environment=spec.assay.environment
            )
        if spec.operation == "patterns":
            return model.patterns(genes, spec.spatial)
        raise ValueError("profiling/evaluation requires a batched phenotype operation")
    finally:
        model.engine.execution = execution


def _fixed_artifacts(model, spec, genes, result):
    """Use the same scientific result contract for standalone and coalesced assays."""
    arrays = {"genotypes": genes, "phenotypes": result}
    summary = {
        "samples": len(genes),
        "complete": result.complete,
        "output_shape": list(result.values.shape),
    }
    if spec.operation == "neighbourhood" and result.complete:
        from .analysis import neighbourhood_summary

        retained = genes.nbytes + result.values.nbytes + result.status.nbytes
        with model.engine._temporary_bytes(
            retained, message="neighbourhood results exceed memory budget"
        ):
            base = _evaluate(
                model, dataclasses.replace(spec, operation=spec.assay.phenotype), genes
            )
            model.engine._check_memory(
                base.values.nbytes
                + base.status.nbytes
                + result.status.nbytes
                + 5 * result.values.nbytes
                + result.values.size // result.values.shape[-1] * 16,
                message="neighbourhood summary exceeds memory budget",
            )
            neighbourhood = neighbourhood_summary(base, result)
        arrays["robustness"] = neighbourhood["robustness"]
        arrays["neutral_mutations"] = neighbourhood["neutral_mutations"]
        summary["nonneutral_histograms"] = [
            {"phenotypes": row["phenotypes"].tolist(), "counts": row["counts"].tolist()}
            for row in neighbourhood["nonneutral_histograms"]
        ]
    if spec.operation == "patterns":
        summary.update(pattern_summary(result.values, spec.spatial))
    return arrays, summary, "complete" if result.complete else "partial"


def _execute(spec: RunSpec, record: Path, checkpoint=None, tables=None) -> Path:
    if spec.schema_version == 2 and spec.operation in {
        "chemistry",
        "tissue",
        "rollout",
        "wright_fisher",
        "sswm",
        "neutral_walk",
        "external",
    }:
        from .batch import BatchRunner

        with BatchRunner(spec.execution, tables=tables) as batch:
            session = batch.from_spec(_captured_spec(record), checkpoint=checkpoint, record=record)
            if spec.operation == "chemistry":
                session.run(max(0, spec.controller.warmup + spec.cell.steps - session.update))
                if session._status != "partial":
                    session._status = "complete"
            elif spec.operation == "tissue":
                session.run(max(0, spec.spatial.steps - session.tick))
                if session._status != "partial":
                    session._status = "complete"
            elif spec.operation == "external":
                session.run(max(0, spec.evolution.generations - session.update))
                if session.update >= spec.evolution.generations:
                    session._status = "complete"
            else:
                session.run()
            finished = batch.finish(session)
            assert finished is not None
            return finished
    started = time.perf_counter()
    model = Model(execution=spec.execution, tables=tables)
    arrays, summary = {}, {}
    status = "complete"
    actual = spec.execution.backend

    def save(state, values):
        from .records import write_boundary

        events = None
        if spec.recording.mode == "full" and (
            state.get("pending_trace") or state.get("trace_event")
        ):
            details = state.get("pending_trace") or state["trace_event"]
            events = [details]
        _, accepted = write_boundary(record, state, arrays=values, events=events)
        if events:
            if not accepted:
                state["recording_limit"] = "trace_budget"
                state["pending_trace"] = details
            else:
                state.pop("pending_trace", None)
                state.pop("trace_event", None)
                state.pop("recording_limit", None)
        return accepted

    def save_map(state):
        save(state, {})

    if checkpoint and checkpoint["state"].get("pending_trace"):
        if not save(checkpoint["state"], checkpoint["loaded_arrays"]):
            return finalize_record(
                record, status="partial", summaries={"recording_limit": "trace_budget"}
            )

    if spec.operation in ("wright_fisher", "sswm", "neutral_walk"):
        from . import evolution

        function = getattr(evolution, spec.operation)
        kwargs = {"checkpoint": checkpoint, "save": save}
        if spec.operation != "neutral_walk":
            kwargs["target"] = _target(record, spec)
        result = function(model, spec, **kwargs)
        arrays, summary, status = result["arrays"], result["summary"], result["status"]
        summary["selection_backend"] = (
            "host_exact_integer" if spec.operation == "wright_fisher" else "host_binary64"
        )
    elif spec.operation == "weighted_map":
        from .maps import enumerate_map

        summary = enumerate_map(
            model.tables,
            spec.execution,
            engine=model.engine,
            spatial=spec.spatial,
            diffusion=spec.spatial.diffusion,
            start_pair=spec.map.start_pair,
            stop_pair=spec.map.stop_pair,
            tile_pairs=spec.map.tile_pairs,
            max_tiles=spec.execution.max_tiles,
            checkpoint_interval_tiles=spec.recording.checkpoint_interval,
            resume_checkpoint=checkpoint["state"] if checkpoint else None,
            checkpoint_callback=save_map,
        )
        status = summary["status"]
    elif spec.operation == "cell":
        from .cells import simulate

        genes = _genotypes(spec)
        result = simulate(model.tables, genes, spec, checkpoint=checkpoint, save=save)
        arrays, summary, status = result["arrays"], result["summary"], result["status"]
        actual = "python_reference"
    else:
        genes = _genotypes(spec)
        result = _evaluate(model, spec, genes)
        arrays, summary, status = _fixed_artifacts(model, spec, genes, result)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return finalize_record(
        record,
        arrays=arrays,
        summaries=summary,
        status=status,
        metadata={
            "execution": {
                "requested_backend": spec.execution.backend,
                "actual_backend": actual,
                "platform": platform.platform(),
            },
            "table_digest": model.tables.digest,
            "rng_scheme": SCHEME,
        },
    )


def run(
    spec: RunSpec,
    workspace=None,
    destination=None,
    checkpoint=None,
    parent=None,
    *,
    tables=None,
    session_inputs=None,
) -> Path:
    """Create provenance before execution, retaining failed/interrupted records."""
    if tables is None:
        from .tables import Tables

        tables = Tables.load(cache_dir=spec.execution.cache_dir)
    record = begin_record(
        spec,
        _destination(spec, workspace, destination),
        inputs=_inputs(spec),
        tables=tables,
        session_inputs=session_inputs,
    )
    if parent:
        from .records import annotate_record

        annotate_record(record, {"parent": parent})
    try:
        return _execute(spec, record, checkpoint, tables)
    except BaseException as error:
        finalize_record(
            record,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            summaries={"error_type": type(error).__name__, "error": str(error)},
        )
        raise


def _checkpoint_record(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            return candidate
    raise ValueError("checkpoint must belong to a recorded run")


def _check_source(record: Path) -> None:
    from .records import _hash, _source_archive

    checked = verify_record(record)
    if not checked["ok"]:
        raise ValueError("record integrity failed: " + "; ".join(checked["errors"]))
    manifest = inspect_record(record)
    if manifest.get("components"):
        from .components import verify_current

        verify_current(load_record_spec(record), manifest["components"])
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        _source_archive(folder / "source.zip")
        if _hash(folder / "source.zip") != manifest["source_sha256"]:
            raise ValueError("executed source differs from record; restore captured source first")
    from .records import _software_environment

    expected = manifest["software_environment"]
    current = _software_environment()
    for name in ("python", "implementation", "platform", "machine"):
        if current[name] != expected[name]:
            raise ValueError(f"numerical runtime differs from record: {name}")
    for name in ("numpy", "quadrants", "tomli-w"):
        if current["dependencies"][name] != expected["dependencies"][name] or current[
            "distribution_records_sha256"
        ].get(name) != expected["distribution_records_sha256"].get(name):
            raise ValueError(f"recorded numerical dependency differs: {name}")


def _captured_spec(record: Path) -> RunSpec:
    spec = load_record_spec(record)
    target = inspect_record(record)["inputs"].get("target")
    if target:
        spec = dataclasses.replace(
            spec, spatial=dataclasses.replace(spec.spatial, target=str(record / target["path"]))
        )
    if spec.schema_version == 2:
        inputs = inspect_record(record)["inputs"]
        for name in ("task", "objective"):
            component = getattr(spec, name)
            declared = component.config.get("inputs", {})
            if declared:
                config = {
                    **component.config,
                    "inputs": {
                        key: str(record / inputs[f"{name}_{key}"]["path"]) for key in declared
                    },
                }
                spec = dataclasses.replace(
                    spec, **{name: dataclasses.replace(component, config=config)}
                )
    return spec


def resume(
    checkpoint,
    destination=None,
    execution: ExecutionSpec | None = None,
    recording: RecordingSpec | None = None,
    tile_pairs: int | None = None,
    limits=None,
) -> Path:
    """Resume logical algorithms into a fresh record; physical placement may change."""
    source = _checkpoint_record(Path(checkpoint).resolve())
    _check_source(source)
    spec = _captured_spec(source)
    if execution:
        spec = dataclasses.replace(spec, execution=execution)
    if recording:
        spec = dataclasses.replace(spec, recording=recording)
    if limits is not None:
        spec = dataclasses.replace(spec, limits=limits)
    if tile_pairs is not None:
        spec = dataclasses.replace(spec, map=dataclasses.replace(spec.map, tile_pairs=tile_pairs))
    loaded = read_checkpoint(checkpoint)
    committed_path = source / "checkpoints" / loaded["revision"] / "checkpoint.json"
    if spec.operation not in (
        "wright_fisher",
        "sswm",
        "neutral_walk",
        "weighted_map",
        "cell",
        "rollout",
        "chemistry",
        "tissue",
        "external",
    ):
        raise ValueError("this operation has no evolutionary checkpoint")
    from .records import _hash, captured_tables, load_session_inputs

    with captured_tables(source) as tables:
        return run(
            spec,
            destination=destination,
            checkpoint=loaded,
            tables=tables,
            session_inputs=load_session_inputs(source),
            parent={
                "relationship": "resume",
                "scientific_identity": inspect_record(source)["scientific_identity"],
                "manifest_sha256": _hash(source / "manifest.json"),
                "checkpoint_revision": loaded["revision"],
                "checkpoint_sha256": _hash(committed_path),
            },
        )


def reproduce(record, destination) -> Path:
    """Verify executed source/dependencies, rerun captured inputs, and compare arrays."""
    source = Path(record).resolve()
    _check_source(source)
    if inspect_record(source)["status"] != "complete":
        raise ValueError("reproduce requires a complete original run")
    from .records import _hash

    source_summary = json_summary(source)
    kind = source_summary.get("kind")
    if kind in ("profile", "qualification"):
        function = profile if kind == "profile" else qualify
        output = function(_captured_spec(source), destination=destination)
    else:
        from .records import captured_tables, load_session_inputs

        with captured_tables(source) as tables:
            output = run(
                _captured_spec(source),
                destination=destination,
                tables=tables,
                session_inputs=load_session_inputs(source),
                parent={
                    "relationship": "reproduce",
                    "scientific_identity": inspect_record(source)["scientific_identity"],
                    "manifest_sha256": _hash(source / "manifest.json"),
                },
            )
    original = inspect_record(source)["artifacts"]["arrays"]
    recreated = inspect_record(output)["artifacts"]["arrays"]
    if set(original) != set(recreated):
        raise RuntimeError("reproduction produced different artifacts")
    for name, description in original.items():
        for field in ("path", "status_path"):
            if field in description:
                a = np.load(source / description[field], allow_pickle=False)
                b = np.load(output / recreated[name][field], allow_pickle=False)
                if not np.array_equal(a, b):
                    raise RuntimeError(f"reproduction differs: {name}/{field}")
    if source_summary != json_summary(output):
        raise RuntimeError("reproduction differs: scientific summary")
    return output


def convert_checkpoint(checkpoint, destination, *, execution=None, recording=None, limits=None):
    """Explicitly qualify and continue a source-compatible v1 evolution checkpoint."""
    from .conversion import prepare_conversion

    source = _checkpoint_record(Path(checkpoint).resolve())
    original = load_record_spec(source)
    model = Model(execution=execution or original.execution)
    prepared = prepare_conversion(
        checkpoint, model, execution=execution, recording=recording, limits=limits
    )
    return run(
        prepared.spec,
        destination=destination,
        checkpoint={
            "state": prepared.checkpoint.record_state(),
            "loaded_arrays": prepared.checkpoint.arrays,
        },
        parent=prepared.parent,
    )


def json_summary(record: Path) -> dict:
    import json

    path = record / "summary.json"
    if not path.is_file():
        return {}
    values = json.loads(path.read_text())
    for key in ("elapsed_seconds", "cold_seconds", "warm_seconds", "warm_median_seconds"):
        values.pop(key, None)
    # A continuation executes only its remaining physical work. The total
    # logical counters describe the reproducible scientific trajectory.
    if isinstance(values.get("counters"), dict):
        values["counters"].pop("invocation", None)
    return values


def profile(spec: RunSpec, destination=None, workspace=None) -> Path:
    """Separate cold compile/transfer and warm synchronized batched timings."""
    record = begin_record(spec, _destination(spec, workspace, destination), inputs=_inputs(spec))
    try:
        model = Model(execution=spec.execution)
        genes = _genotypes(spec)
        start = time.perf_counter()
        result = _evaluate(model, spec, genes)
        cold = time.perf_counter() - start
        warm = []
        for _ in range(5):
            start = time.perf_counter()
            _evaluate(model, spec, genes)
            warm.append(time.perf_counter() - start)
        return finalize_record(
            record,
            arrays={"genotypes": genes, "phenotypes": result},
            summaries={
                "kind": "profile",
                "cold_seconds": cold,
                "warm_seconds": warm,
                "warm_median_seconds": float(np.median(warm)),
                "samples": len(genes),
                "speedup_claim": None,
            },
            metadata={
                "table_digest": model.tables.digest,
                "execution": {"actual_backend": spec.execution.backend},
            },
            status="complete" if result.complete else "partial",
        )
    except BaseException as error:
        finalize_record(record, status="failed", summaries={"error": str(error)})
        raise


def qualify(spec: RunSpec, destination=None, workspace=None) -> Path:
    """Compare real kernels to the independent scalar oracle on recorded inputs."""
    from .reference import ReferenceModel
    from .regeneration import validate_binding_regeneration, validate_regeneration
    from .spatial import response_descriptors

    record = begin_record(spec, _destination(spec, workspace, destination), inputs=_inputs(spec))
    try:
        model = Model(execution=spec.execution)
        genes = _genotypes(spec)
        oracle = ReferenceModel(model.tables)
        truth = model.regulatory(genes)
        reference = np.stack([oracle.regulatory(g) for g in genes])
        if not np.array_equal(truth.values, reference):
            raise AssertionError("regulatory parity failed")
        geometry = plain(validate_regeneration(model.tables))
        bindings = plain(validate_binding_regeneration(model.tables))
        failures = [key for key, value in geometry.items() if key.endswith("mismatches") and value]
        failures += [
            key
            for key, value in bindings.items()
            if key.endswith("mismatches") and key != "protein_promoter_pol_mismatches" and value
        ]
        expected_annotations = [[10, 375], [12, 375], [14, 375]]
        if bindings["protein_promoter_pol_mismatches"] != expected_annotations:
            failures.append("unexpected polymerase annotation mismatch")
        if failures:
            raise AssertionError(
                "independent geometry qualification failed: " + ", ".join(failures)
            )
        reports = {
            "regulatory": True,
            "geometry": geometry,
            "binding_geometry": bindings,
            "curated_assembly_ambiguity": "source-pinned",
            "polymerase_annotation_exceptions": expected_annotations,
        }
        if genes.shape[-1] == 2:
            spatial = response_descriptors(model.tables, genes, spec.execution, engine=model.engine)
            if not np.array_equal(spatial, np.stack([oracle.spatial_descriptor(g) for g in genes])):
                raise AssertionError("spatial descriptor parity failed")
            reports["spatial"] = True
        metabolism = model.metabolic(genes, spec.assay.environment)
        reference_met = [
            oracle.metabolic(g, spec.assay.environment, spec.execution.max_steps) for g in genes
        ]
        if not np.array_equal(
            metabolism.values, np.stack([x[0] for x in reference_met])
        ) or not np.array_equal(metabolism.status, np.stack([x[1] for x in reference_met])):
            raise AssertionError("metabolic parity failed")
        reports.update(
            metabolic=True,
            published_aggregate_parity="unresolved",
            samples=len(genes),
            kind="qualification",
        )
        return finalize_record(
            record,
            arrays={"genotypes": genes, "regulatory": truth, "metabolic": metabolism},
            summaries=reports,
            metadata={
                "table_digest": model.tables.digest,
                "execution": {"actual_backend": spec.execution.backend},
            },
            status="complete" if metabolism.complete else "partial",
        )
    except BaseException as error:
        finalize_record(record, status="failed", summaries={"error": str(error)})
        raise
