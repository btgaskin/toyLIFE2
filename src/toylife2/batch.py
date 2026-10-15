"""One runtime for interactive chemistry, episodes and evolutionary experiments."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import Model
from .protocols import ControllerSpec, EpisodeSpec, TaskSpec
from .sessions import Checkpoint
from .specs import ExecutionSpec, from_mapping, plain


def _controller(data):
    from .specs import _construct

    return _construct(ControllerSpec, dict(data))


def _numeric_artifacts(value, prefix=""):
    """Flatten numerical snapshots into portable NPY artifacts."""
    result = {}
    if isinstance(value, np.ndarray) and not value.dtype.hasobject:
        result[prefix or "values"] = value
    elif isinstance(value, Mapping):
        for name, child in value.items():
            result.update(_numeric_artifacts(child, (prefix + "_" if prefix else "") + name))
    return result


class BatchRunner:
    """Own one immutable table pack, Quadrants runtime and compatible evaluators.

    Leading axes describe independent candidates, worlds and scenarios. Coupled
    agents use an explicit agent_axis; generations advance through session.step().
    Recorded sessions flush at finish() or context-manager exit.
    """

    def __init__(self, execution=None, *, model=None, tables=None, workspace=None):
        self.execution = execution or (model.execution if model is not None else ExecutionSpec())
        self.model = model or Model(tables=tables, execution=self.execution)
        if self.model.execution != self.execution:
            raise ValueError("runner and model execution configurations differ")
        self.workspace = workspace
        self._evaluators = {}
        self._records = {}
        self._sessions = []
        self._reservations = {}
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.close(error=error)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("BatchRunner is closed")

    def _register(
        self, session, spec=None, destination=None, parent=None, record=None, nested=False
    ):
        self._check_open()
        if not nested and spec is not None and session.kind in {"chemistry", "tissue", "rollout"}:
            from .runner import _genotypes

            expected = _genotypes(spec)
            shape = spec.task.config.get("batch_shape")
            if shape is not None:
                expected = expected.reshape(*shape, spec.model.genes)
            actual = getattr(session, "_source_genotypes", session.genotypes)
            matches = np.array_equal(expected, actual)
            if session.kind == "chemistry":
                matches = matches and session.spec == spec.controller
            elif session.kind == "tissue":
                matches = matches and session.spec == spec.spatial
            else:
                matches = (
                    matches
                    and session.task == spec.task
                    and session.protocol == spec.episode
                    and session.controller == spec.controller
                    and session.seed == (spec.seed or 0)
                    and session.agent_axis == spec.task.config.get("agent_axis")
                )
            matches = (
                matches and session.limits == spec.limits and session.recording == spec.recording
            )
            if record is not None:
                from .records import inspect_record, load_session_inputs
                from .specs import digest

                declared = inspect_record(record)
                matches = (
                    matches
                    and declared.get("table_source_digest", self.model.tables.digest)
                    == self.model.tables.digest
                )
                if session.kind == "rollout":
                    inputs = load_session_inputs(record)
                    from .rollout import _plain

                    matches = matches and session.panel == inputs.get("panel", 0)
                    matches = matches and digest(_plain(session.identities)) == digest(
                        _plain(inputs.get("identities"))
                    )
            if not matches:
                if record is not None:
                    from .records import finalize_record

                    finalize_record(
                        record,
                        status="failed",
                        summaries={
                            "error": "interactive session differs from its recorded specification"
                        },
                    )
                raise ValueError("interactive session differs from RunSpec; use from_spec")
        reserved = getattr(self.model.engine, "_reserved_session_bytes", 0)
        required = getattr(session, "buffer_bytes", 0)
        self.model.engine._check_memory(
            required, message="live sessions exceed the runner numerical memory budget"
        )
        self.model.engine._reserved_session_bytes = reserved + required
        self._reservations[id(session)] = required
        session._memory_reserved_bytes = required
        self._sessions.append(session)
        if (
            not nested
            and record is None
            and (destination is not None or self.workspace is not None)
        ):
            if spec is None:
                raise ValueError("recording an interactive session requires a RunSpec")
            inputs = (
                {"panel": session.panel, "identities": session.identities}
                if session.kind == "rollout"
                else None
            )
            record = self._begin(spec, destination, parent, session_inputs=inputs)
        if record is not None:
            from .records import write_boundary

            record = Path(record)
            session.record = record
            self._records[id(session)] = (record, spec, time.perf_counter())

            def commit(checkpoint, events):
                if checkpoint.kind == "external" and any(
                    row.get("type") == "external_tell" for row in events
                ):
                    assert spec is not None
                    import json

                    from .records import _jsonable, inspect_record

                    manifest = inspect_record(record)
                    additional = sum(
                        len(
                            (
                                json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
                            ).encode()
                        )
                        for row in events
                    )
                    used = manifest.get("trace", {}).get("bytes", 0)
                    limit = spec.recording.trace_budget_mib * 1024 * 1024
                    if used + additional > limit:
                        # Retain the durable post-ask optimizer until tell and
                        # its trace can commit together; rollback happens in RAM.
                        return False
                state = checkpoint.record_state()
                # Accepted checkpoints describe events already committed in the
                # matching trace revision. A rejected write retains them outside
                # session_state, which Checkpoint.from_record grafts on restore.
                state["session_state"] = dict(state["session_state"])
                state["session_state"]["base"] = {
                    **state["session_state"]["base"],
                    "pending_trace": [],
                }
                _, accepted = write_boundary(
                    record, state, arrays=checkpoint.arrays, events=events or None
                )
                return accepted

            session._record_callback = commit
            warmup = getattr(session, "_unrecorded_trace", [])
            if warmup:
                session._pending_trace.extend(warmup)
                session._unrecorded_trace = []
            session._boundary(force=True)
        return session

    def _begin(self, spec, destination=None, parent=None, session_inputs=None):
        if spec.operation == "external" and not spec.task.config.get("adapter_entrypoint"):
            raise ValueError("recorded external optimizers require an adapter_entrypoint factory")
        from .components import input_files
        from .records import begin_record
        from .workspaces import new_record_path

        path = destination or new_record_path(self.workspace or "workspaces/default", spec)
        return begin_record(
            spec,
            path,
            inputs=input_files(spec),
            parent=parent,
            tables=self.model.tables,
            session_inputs=session_inputs,
        )

    def chemistry(
        self,
        genotypes,
        controller=None,
        *,
        limits=None,
        recording=None,
        checkpoint=None,
        spec=None,
        destination=None,
    ):
        self._check_open()
        from .chemistry import ChemistrySession

        controller = controller or (spec.controller if spec else None)
        limits = limits or (spec.limits if spec else None)
        recording = recording or (spec.recording if spec else None)

        record = (
            self._begin(spec, destination)
            if spec is not None and (destination is not None or self.workspace is not None)
            else None
        )
        session = ChemistrySession(self.model, genotypes, controller, limits, recording, checkpoint)
        return self._register(session, spec, destination, record=record)

    def tissue(
        self,
        genotypes,
        spatial=None,
        *,
        limits=None,
        recording=None,
        checkpoint=None,
        spec=None,
        destination=None,
    ):
        self._check_open()
        from .tissue import TissueSession

        spatial = spatial or (spec.spatial if spec else None)
        limits = limits or (spec.limits if spec else None)
        recording = recording or (spec.recording if spec else None)

        record = (
            self._begin(spec, destination)
            if spec is not None and (destination is not None or self.workspace is not None)
            else None
        )
        session = TissueSession(self.model, genotypes, spatial, limits, recording, checkpoint)
        return self._register(session, spec, destination, record=record)

    def rollout(
        self,
        genotypes,
        task=None,
        protocol=None,
        controller=None,
        *,
        limits=None,
        recording=None,
        seed=None,
        panel=0,
        agent_axis=None,
        identities=None,
        checkpoint=None,
        spec=None,
        destination=None,
    ):
        self._check_open()
        from .rollout import RolloutSession

        task = task or (spec.task if spec else None)
        protocol = protocol or (spec.episode if spec else None)
        controller = controller or (spec.controller if spec else None)
        seed = (spec.seed or 0) if seed is None and spec else (seed or 0)
        if agent_axis is None and spec:
            agent_axis = spec.task.config.get("agent_axis")
        limits = limits or (spec.limits if spec else None)
        recording = recording or (spec.recording if spec else None)

        record = (
            self._begin(
                spec, destination, session_inputs={"panel": panel, "identities": identities}
            )
            if spec is not None and (destination is not None or self.workspace is not None)
            else None
        )
        session = RolloutSession(
            self.model,
            genotypes,
            task,
            protocol,
            controller,
            limits,
            recording,
            seed,
            panel,
            agent_axis,
            identities,
            checkpoint,
        )
        return self._register(session, spec, destination, record=record)

    def evaluator(self, spec, *, target=None):
        self._check_open()
        from .evaluation import Evaluator

        def nested_rollout(model, *args, **kwargs):
            from .rollout import RolloutSession

            return self._register(RolloutSession(model, *args, **kwargs), nested=True)

        if spec.evolution.gene_copy is not None:
            from ._variable_evaluation import VariableGeneEvaluator

            trial = VariableGeneEvaluator(
                self.model, spec, target=target, runner=self, rollout_factory=nested_rollout
            )
            self._evaluators[(trial.context, id(trial))] = trial
            return trial
        trial = Evaluator(
            self.model, spec, target=target, runner=self, rollout_factory=nested_rollout
        )
        # Physical settings and quotas may differ; cache values only within an
        # identical frozen scientific evaluation context. Each session gets its
        # own continuation and counters while sharing the pure FIFO value cache.
        existing = self._evaluators.get(trial.context)
        if existing is None:
            self._evaluators[trial.context] = trial
        else:
            trial.cache = existing.cache
            trial._shared_cache = existing._shared_cache = True
        return trial

    def release(self, session):
        """Release a completed nested evaluation from runner memory admission."""
        if id(session) in self._records:
            raise ValueError("finish a recorded session before releasing its buffers")
        released = self._reservations.pop(id(session), 0)
        session._memory_reserved_bytes = 0
        self.model.engine._reserved_session_bytes = max(
            0, getattr(self.model.engine, "_reserved_session_bytes", 0) - released
        )
        if session in self._sessions:
            self._sessions.remove(session)

    def evolution(
        self,
        spec,
        *,
        adapter=None,
        evaluator=None,
        limits=None,
        recording=None,
        checkpoint=None,
        destination=None,
        record=None,
        parent=None,
    ):
        self._check_open()
        evaluator = evaluator or self.evaluator(spec)
        if spec.operation == "external":
            from .adapters import ExternalSession

            if adapter is None:
                from .components import resolve

                entrypoint = spec.task.config.get("adapter_entrypoint")
                if not entrypoint:
                    raise ValueError("external evolution requires an adapter or adapter_entrypoint")
                adapter = resolve(entrypoint)(config=dict(spec.task.config), seed=spec.seed)
            session = ExternalSession(
                self.model,
                spec,
                adapter,
                evaluator=evaluator,
                limits=limits or spec.limits,
                recording=recording or spec.recording,
                checkpoint=checkpoint,
            )
        else:
            from .population import PopulationSession

            session = PopulationSession(
                self.model,
                spec,
                evaluator,
                limits or spec.limits,
                recording or spec.recording,
                checkpoint,
            )
        return self._register(session, spec, destination, parent, record)

    def from_spec(self, spec, *, checkpoint=None, record=None, destination=None, adapter=None):
        """Create a session from v2 TOML, preserving the configured batch layout."""
        if spec.schema_version != 2:
            raise ValueError("interactive sessions require schema_version=2")
        if spec.execution != self.execution:
            raise ValueError("use one execution configuration per BatchRunner")
        if isinstance(checkpoint, dict):
            checkpoint = Checkpoint.from_record(checkpoint)
        if spec.operation in {"wright_fisher", "sswm", "neutral_walk", "external"}:
            return self.evolution(
                spec, checkpoint=checkpoint, record=record, destination=destination, adapter=adapter
            )
        from .runner import _genotypes

        genes = _genotypes(spec)
        shape = spec.task.config.get("batch_shape")
        if shape is not None:
            shape = tuple(shape)
            if any(type(n) is not int or n <= 0 for n in shape) or np.prod(shape) != len(genes):
                raise ValueError("task.config.batch_shape must contain every sampled genotype")
            genes = genes.reshape(*shape, spec.model.genes)
        from .records import load_session_inputs

        captured = load_session_inputs(record) if record is not None else {}
        common: dict[str, Any] = dict(
            limits=spec.limits, recording=spec.recording, checkpoint=checkpoint
        )
        if spec.operation == "chemistry":
            from .chemistry import ChemistrySession

            session = ChemistrySession(self.model, genes, spec.controller, **common)
        elif spec.operation == "tissue":
            from .tissue import TissueSession

            session = TissueSession(self.model, genes, spec.spatial, **common)
        elif spec.operation == "rollout":
            from .rollout import RolloutSession

            session = RolloutSession(
                self.model,
                genes,
                spec.task,
                spec.episode,
                spec.controller,
                seed=spec.seed or 0,
                panel=captured.get("panel", 0),
                identities=captured.get("identities"),
                agent_axis=spec.task.config.get("agent_axis"),
                **common,
            )
        else:
            raise ValueError(f"{spec.operation} is a fixed assay; use runner.run or evaluator")
        return self._register(session, spec, destination, record=record)

    def resume(
        self, checkpoint, *, limits=None, recording=None, destination=None, spec=None, adapter=None
    ):
        """Restore a numerical checkpoint; invocation work quotas start afresh."""
        if not isinstance(checkpoint, Checkpoint):
            checkpoint = Checkpoint.read(checkpoint)
        data = checkpoint.payload["payload"]
        if checkpoint.kind in {"population", "external"}:
            saved = spec or from_mapping(data["spec"])
            saved = dataclasses.replace(
                saved,
                execution=self.execution,
                limits=limits or saved.limits,
                recording=recording or saved.recording,
            )
            return self.evolution(
                saved, adapter=adapter, checkpoint=checkpoint, destination=destination
            )
        common = dict(
            limits=limits,
            recording=recording,
            checkpoint=checkpoint,
            spec=spec,
            destination=destination,
        )
        if checkpoint.kind == "chemistry":
            return self.chemistry(data["genotypes"], _controller(data["spec"]), **common)
        if checkpoint.kind == "tissue":
            from .specs import SpatialSpec, _construct

            return self.tissue(data["genotypes"], _construct(SpatialSpec, data["spec"]), **common)
        if checkpoint.kind == "rollout":
            return self.rollout(
                data["genotypes"],
                TaskSpec(**data["task"]),
                EpisodeSpec(**data["protocol"]),
                _controller(data["controller"]),
                seed=data["seed"],
                panel=data["panel"],
                agent_axis=data["agent_axis"],
                identities=data["identities"],
                **common,
            )
        raise ValueError("unsupported checkpoint session kind")

    def finish(self, session, *, error=None):
        """Flush a final checkpoint and commit the numerical result artifacts."""
        item = self._records.pop(id(session), None)
        if item is None:
            return getattr(session, "record", None)
        record, spec, started = item
        from .records import finalize_record

        try:
            if error is None:
                session._boundary(force=True)
            status = (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else (
                    "failed"
                    if error
                    else "complete"
                    if session._status == "complete"
                    else "partial"
                )
            )
            snapshot = session.snapshot()
            summaries = {
                "session_kind": session.kind,
                "counters": session.counters,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if "summary" in snapshot:
                summaries.update(snapshot["summary"])
            if error:
                summaries["error"] = str(error)
            return finalize_record(
                record,
                arrays=_numeric_artifacts(snapshot),
                summaries=summaries,
                status=status,
                metadata={
                    "table_digest": self.model.tables.digest,
                    "execution": {
                        "requested_backend": self.execution.backend,
                        "actual_backend": self.execution.backend,
                        **({"world": session.profile} if session.kind == "rollout" else {}),
                    },
                },
            )
        finally:
            session._record_callback = None

    def close(self, *, error=None):
        if self._closed:
            return
        for session in self._sessions:
            if id(session) in self._records:
                self.finish(session, error=error)
        self.model.engine._reserved_session_bytes = max(
            0,
            getattr(self.model.engine, "_reserved_session_bytes", 0)
            - sum(self._reservations.values()),
        )
        for session in self._sessions:
            session._memory_reserved_bytes = 0
        self._reservations.clear()
        for evaluator in self._evaluators.values():
            if hasattr(evaluator, "close"):
                evaluator.close()
        self._closed = True

    @property
    def metrics(self):
        return {
            "sessions": len(self._sessions),
            "evaluation_contexts": len(self._evaluators),
            "table_digest": self.model.tables.digest,
            "backend": self.execution.backend,
            "live_session_buffer_bytes": sum(self._reservations.values()),
        }

    def run_batch(self, specs, *, destination, resume=False, adapters=None):
        """Advance named runs fairly with a bounded number of live sessions.

        Each child's manifest is its durable commit point. The batch manifest is
        a recoverable index, refreshed from child records after interruption.
        """
        from .records import _write_json, inspect_record, read_checkpoint

        self._check_open()
        specs = dict(specs)
        if not specs:
            raise ValueError("run_batch requires at least one named run")
        for name, spec in specs.items():
            if not name or not all(c.isalnum() or c in "_-" for c in name):
                raise ValueError("batch run names use letters, digits, '_' or '-'")
            if spec.schema_version != 2 or spec.execution != self.execution:
                raise ValueError("batch children require v2 and the runner execution configuration")
        directory = Path(destination).resolve()
        index_path = directory / "batch.json"
        if directory.exists() and any(directory.iterdir()) and not resume:
            raise FileExistsError(f"batch directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        from .specs import digest, scientific_identity

        identity = digest({name: plain(spec) for name, spec in specs.items()})
        scientific = digest({name: scientific_identity(spec) for name, spec in specs.items()})
        if index_path.exists():
            import json

            prior = json.loads(index_path.read_text())
            if (
                prior.get("table_source_digest", self.model.tables.digest)
                != self.model.tables.digest
            ):
                raise ValueError("batch molecular tables differ from the captured index")
            if prior.get("scientific_identity", prior["spec_sha256"]) != scientific:
                raise ValueError("batch specifications differ from the captured index")
        queue = list(specs)
        active = {}
        outputs = {}

        def publish():
            children = {}
            for name in specs:
                record = outputs.get(name, directory / name)
                if (record / "manifest.json").is_file():
                    manifest = inspect_record(record)
                    children[name] = {
                        "record": str(record.relative_to(directory)),
                        "status": manifest["status"],
                        "scientific_identity": manifest["scientific_identity"],
                        "latest_checkpoint": manifest.get("artifacts", {}).get("checkpoint"),
                    }
                else:
                    children[name] = {"record": name, "status": "unstarted"}
            _write_json(
                index_path,
                {
                    "batch_schema": 2,
                    "table_source_digest": self.model.tables.digest,
                    "spec_sha256": identity,
                    "scientific_identity": scientific,
                    "specs": {name: plain(spec) for name, spec in specs.items()},
                    "children": children,
                    "runtime": self.metrics,
                },
            )

        publish()
        try:
            while queue or active:
                while queue and len(active) < self.execution.max_active_lanes:
                    name = queue.pop(0)
                    spec = specs[name]
                    path = directory / name
                    cp = None
                    parent = None
                    session_inputs = None
                    if (path / "manifest.json").exists():
                        from .runner import _check_source

                        _check_source(path)
                        manifest = inspect_record(path)
                        if (
                            manifest.get("table_source_digest", manifest.get("table_digest"))
                            != self.model.tables.digest
                        ):
                            raise ValueError(
                                "batch molecular tables differ from the captured child"
                            )
                        if inspect_record(path)["status"] == "complete":
                            outputs[name] = path
                            continue
                        source = path
                        cp = (
                            read_checkpoint(source)
                            if inspect_record(source).get("artifacts", {}).get("checkpoint")
                            else None
                        )
                        # A child continuation remains a separate auditable record.
                        generation = 1
                        while (directory / f"{name}-continuation-{generation}").exists():
                            latest = directory / f"{name}-continuation-{generation}"
                            _check_source(latest)
                            source = latest
                            if inspect_record(latest)["status"] == "complete":
                                outputs[name] = latest
                            cp = (
                                read_checkpoint(latest)
                                if inspect_record(latest).get("artifacts", {}).get("checkpoint")
                                else None
                            )
                            generation += 1
                        if name in outputs:
                            continue
                        path = directory / f"{name}-continuation-{generation}"
                        from .records import _hash, load_session_inputs
                        from .runner import _captured_spec

                        session_inputs = load_session_inputs(source)
                        spec = dataclasses.replace(
                            _captured_spec(source),
                            execution=self.execution,
                            limits=spec.limits,
                            recording=spec.recording,
                        )
                        parent = {
                            "relationship": "resume",
                            "scientific_identity": inspect_record(source)["scientific_identity"],
                            "manifest_sha256": _hash(source / "manifest.json"),
                            "checkpoint_revision": cp["revision"] if cp else None,
                        }
                    if spec.operation not in {
                        "chemistry",
                        "tissue",
                        "rollout",
                        "wright_fisher",
                        "sswm",
                        "neutral_walk",
                        "external",
                    }:
                        from .records import finalize_record
                        from .results import ArrayResult
                        from .runner import _evaluate, _fixed_artifacts, _genotypes

                        def assay_key(value):
                            return digest(
                                [value.operation, value.model, value.assay, value.spatial]
                                + [value.limits.max_assay_steps]
                            )

                        members = [(name, spec, path)]
                        if spec.operation in {
                            "regulatory",
                            "metabolic",
                            "patterns",
                            "neighbourhood",
                        }:
                            for other in list(queue):
                                if (
                                    assay_key(specs[other]) == assay_key(spec)
                                    and not (directory / other).exists()
                                ):
                                    queue.remove(other)
                                    members.append((other, specs[other], directory / other))
                        records = [
                            self._begin(item, target, parent if child == name else None)
                            for child, item, target in members
                        ]
                        inputs = [_genotypes(item) for _, item, _ in members]
                        genes = np.concatenate(inputs)
                        try:
                            with self.model.engine._temporary_bytes(
                                genes.nbytes + sum(item.nbytes for item in inputs),
                                message="coalesced inputs exceed memory budget",
                            ):
                                value = _evaluate(self.model, spec, genes)
                                retained = value.values.nbytes + value.status.nbytes
                                with self.model.engine._temporary_bytes(
                                    retained, message="coalesced results exceed memory budget"
                                ):
                                    offset = 0
                                    for (child, child_spec, _), record, original in zip(
                                        members, records, inputs, strict=True
                                    ):
                                        stop = offset + len(original)
                                        self.model.engine._check_memory(
                                            value.values[offset:stop].nbytes
                                            + value.status[offset:stop].nbytes,
                                            message="coalesced child results exceed memory budget",
                                        )
                                        sliced = ArrayResult(
                                            value.values[offset:stop],
                                            value.status[offset:stop],
                                            value.metadata,
                                        )
                                        arrays, summary, status = _fixed_artifacts(
                                            self.model, child_spec, original, sliced
                                        )
                                        outputs[child] = finalize_record(
                                            record,
                                            arrays=arrays,
                                            status=status,
                                            summaries=summary,
                                            metadata={
                                                "table_digest": self.model.tables.digest,
                                                "coalesced_runs": len(members),
                                            },
                                        )
                                        del arrays, summary, sliced
                                        offset = stop
                        except BaseException as error:
                            for record in records:
                                finalize_record(
                                    record, status="failed", summaries={"error": str(error)}
                                )
                            raise
                        continue
                    record = self._begin(spec, path, parent, session_inputs=session_inputs)
                    from .runner import _captured_spec

                    child_spec = _captured_spec(record)
                    try:
                        session = self.from_spec(
                            child_spec,
                            checkpoint=cp,
                            record=record,
                            adapter=(adapters or {}).get(name),
                        )
                    except BaseException as error:
                        from .records import finalize_record

                        outputs[name] = finalize_record(
                            record, status="failed", summaries={"error": str(error)}
                        )
                        raise
                    active[name] = session
                for name, session in list(active.items()):
                    spec = specs[name]
                    if spec.operation == "chemistry":
                        target = spec.controller.warmup + spec.cell.steps
                        if session.update >= target:
                            session._status = "complete"
                    elif spec.operation == "tissue" and session.tick >= spec.spatial.steps:
                        session._status = "complete"
                    elif (
                        spec.operation == "external"
                        and session.update >= spec.evolution.generations
                    ):
                        session._status = "complete"
                    if session._status not in {"complete", "partial", "failed"}:
                        session.step()
                    if session._status in {"complete", "partial", "failed"}:
                        outputs[name] = self.finish(session)
                        active.pop(name)
                        released = self._reservations.pop(id(session), 0)
                        session._memory_reserved_bytes = 0
                        self.model.engine._reserved_session_bytes -= released
                        self._sessions.remove(session)
                publish()
        except BaseException as error:
            for name, session in list(active.items()):
                outputs[name] = self.finish(session, error=error)
            publish()
            raise
        publish()
        return outputs
