"""Persistent chemical controllers in staged, simultaneously committed worlds."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .genotypes import validate
from .protocols import ControllerSpec, EpisodeSpec, TaskSpec, WorkLimits
from .sessions import Checkpoint, SessionBase, owned
from .specs import RecordingSpec, RunSpec
from .worlds import ScenarioRandom, make_world, numerical


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _trace_size(rows) -> int:
    return len(json.dumps(_plain(rows), separators=(",", ":")).encode()) if rows else 0


def _array_bytes(value, seen=None) -> int:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_array_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(v, seen) for v in value)
    return 0


class RolloutSession(SessionBase):
    """One independent world per genome, with an explicit axis for coupled agents.

    Candidate axes precede the episode axis; the optional agent axis immediately
    precedes the gene axis in controller arrays. A chemical work cap preserves the
    staged decision, including its observations and codec state, for exact resume.
    """

    def __init__(
        self,
        model,
        genotypes,
        task: TaskSpec | None = None,
        protocol: EpisodeSpec | None = None,
        controller: ControllerSpec | None = None,
        limits: WorkLimits | None = None,
        recording: RecordingSpec | None = None,
        seed: int = 0,
        panel: int = 0,
        agent_axis: int | None = None,
        identities=None,
        checkpoint: Checkpoint | None = None,
    ):
        from .chemistry import ChemistrySession

        self.model = model
        self.task = task or TaskSpec(kind="temporal_probe")
        if protocol is None and self.task.kind == "signalling_game":
            protocol = EpisodeSpec(horizon=2 * int(self.task.config.get("rounds", 256)))
        self.protocol = protocol or EpisodeSpec()
        self.controller = controller or ControllerSpec()
        self.seed, self.panel = seed, panel
        genes = validate(genotypes)
        self._source_genotypes = owned(genes)
        self.agent_axis = agent_axis
        if agent_axis is not None:
            axis = agent_axis % genes.ndim
            if axis == genes.ndim - 1:
                raise ValueError("agent_axis must identify a leading axis, not genes")
            genes = np.moveaxis(genes, axis, -2)
            self.agents = genes.shape[-2]
            self.candidate_shape = genes.shape[:-2]
        else:
            self.agents = 1
            self.candidate_shape = genes.shape[:-1]
        self.genotypes = owned(genes)
        self.world_shape = (*self.candidate_shape, self.protocol.episodes)
        self.controller_shape = self.world_shape + (
            (self.agents,) if agent_axis is not None else ()
        )
        self._world_work = int(np.prod(self.world_shape, dtype=int))
        self._chemical_work = int(np.prod(self.controller_shape, dtype=int))
        if self._world_work == 0 or self._chemical_work == 0:
            raise ValueError("rollouts require at least one world and controller")
        episode_axis = -3 if agent_axis is not None else -2
        self._controller_genes = np.broadcast_to(
            np.expand_dims(genes, episode_axis), (*self.controller_shape, genes.shape[-1])
        )
        self.identities = numerical(identities) if identities is not None else None
        data = {
            "task": self.task,
            "episode": self.protocol,
            "controller": self.controller,
            "genes": genes,
            "seed": seed,
            "panel": panel,
            "agent_axis": agent_axis,
            "identities": self.identities,
            "tables": str(model.tables.digest),
        }
        identity_data = _plain(data)
        identity_data["controller"].pop("capacity")
        declared_inputs = identity_data["task"]["config"].get("inputs", {})
        if not isinstance(declared_inputs, dict):
            raise ValueError("task config.inputs must map names to file paths")
        if declared_inputs:
            identity_data["task"]["config"]["inputs"] = {
                name: {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                for name, path in declared_inputs.items()
            }
        from .components import capture

        captured = capture(RunSpec(schema_version=2, seed=0, task=self.task))
        identity_data["components"] = captured[1] if captured is not None else None
        identity = hashlib.sha256(
            json.dumps(identity_data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        super().__init__("rollout", identity, limits, recording, checkpoint)
        scenarios = np.broadcast_to(
            np.arange(self.protocol.episodes, dtype=np.uint64), self.world_shape
        )
        if type(seed) is not int or not 0 <= seed < 1 << 64:
            raise ValueError("seed must be a uint64 integer")
        world_seed = int.from_bytes(
            hashlib.sha256(json.dumps([seed, self.protocol.seed]).encode()).digest()[:8], "little"
        )
        rng = ScenarioRandom(world_seed, panel, scenarios)
        self.world = make_world(
            self.task,
            self.world_shape,
            self.agents,
            rng,
            self.protocol.dt,
            kernels=not bool(self.task.config.get("reference_world", False)),
        )
        self._chemistry_spec = dataclasses.replace(self.controller, warmup=0)
        self.chemistry = ChemistrySession(
            model,
            self._controller_genes,
            self._chemistry_spec,
            WorkLimits(),
            recording or RecordingSpec(),
        )
        snapshot_bytes = _array_bytes(self.chemistry._payload_view())
        # Reserve peak world working arrays and persistent initialized/committed
        # chemistry copies before constructing the numerical world state.
        self._world_peak_bytes = self._world_work * 200 + self._chemical_work * 16
        required = (
            self.chemistry.buffer_bytes
            + snapshot_bytes * 3
            + self._world_peak_bytes
            + self.genotypes.nbytes * 2
        )
        self._admit_memory(required)
        self._codec = None
        codec_entrypoint = self.task.config.get("codec_entrypoint")
        if codec_entrypoint:
            module, _, name = codec_entrypoint.partition(":")
            self._codec = getattr(importlib.import_module(module), name)(
                config=dict(self.task.config), ports=self.controller.ports
            )
        self._codec_state = numerical(self.task.config.get("codec_state", {}))
        self._world_state = numerical(self.world.reset())
        self.tick = 0
        self.reward_sum = np.zeros(self.world_shape, np.float64)
        self.reward_count = np.zeros(self.world_shape, np.int64)
        self.channels = {}
        self._actions = np.zeros(self.controller_shape, np.int32)
        self._descriptor = []
        self._descriptor_bytes = 0
        self._unrecorded_trace = []
        self._unrecorded_trace_bytes = 0
        self._warmup_events = []
        self._warmup_event_bytes = 0
        self._pending = None
        self._publishing = False
        self._committed_chemistry = self.chemistry.payload()
        self._warmup_done = 0
        self._reset_chemistry = None
        if self.task.kind == "signalling_game" and "rounds" in self.task.config:
            rounds = self.task.config["rounds"]
            if (
                type(rounds) is not int
                or rounds < 0
                or self.protocol.horizon not in (0, 2 * rounds)
            ):
                raise ValueError("signalling rounds require horizon == 2 * rounds")
        if checkpoint is not None:
            self._restore_payload(checkpoint.payload["payload"])
        if self.protocol.horizon == 0:
            self._status = "complete"
        self._admit_memory(self.buffer_bytes)

    def _admit_memory(self, required: int) -> None:
        self.model.engine._check_memory(
            required, message="persistent rollout exceeds execution memory budget"
        )

    @property
    def buffer_bytes(self) -> int:
        """Conservative numeric allocation accounting without reading device fields."""
        trees = (
            self._source_genotypes,
            self.genotypes,
            self.identities,
            getattr(self, "_world_state", None),
            getattr(self, "_committed_chemistry", None),
            getattr(self, "_reset_chemistry", None),
            getattr(self, "_pending", None),
            getattr(self, "reward_sum", None),
            getattr(self, "reward_count", None),
            getattr(self, "_actions", None),
            getattr(self, "channels", None),
            getattr(self, "_codec_state", None),
            getattr(self, "_descriptor", None),
            getattr(self, "_warmup_events", None),
            getattr(self, "_unrecorded_trace", None),
            self._pending_trace,
        )
        return self.chemistry.buffer_bytes + self._world_peak_bytes + _array_bytes(trees)

    @property
    def status(self) -> np.ndarray:
        return owned(np.full(self.world_shape, self._status, dtype="U16"))

    @property
    def scores(self) -> np.ndarray:
        scores = np.divide(
            self.reward_sum,
            self.reward_count,
            out=np.zeros_like(self.reward_sum),
            where=self.reward_count > 0,
        )
        return owned(scores)

    @property
    def success(self) -> np.ndarray:
        return owned(
            (self.status == "complete")
            & (self.reward_count > 0)
            & (self.scores >= float(self.task.config.get("goal", 1.0)))
        )

    @property
    def descriptor(self):
        return owned(tuple(self._descriptor))

    @property
    def profile(self) -> dict:
        reference = self.task.kind == "python" or bool(
            self.task.config.get("reference_world", False)
        )
        return {
            "world_execution": "python_host" if reference else "quadrants",
            "backend": "host" if reference else self.model.execution.backend,
            "world_float_dtype": "float32" if self.task.kind == "run_tumble" else "none",
            "world_buffers": "numpy_at_boundaries",
            "world_transfers": "per_decision" if not reference else "host_only",
            "reward_accumulator_dtype": "float64",
            "buffer_bytes": self.buffer_bytes,
            "world_axes": self.world_shape,
            "controller_axes": self.controller_shape,
            "chemical_updates_per_decision": self.protocol.chemical_updates,
            "dt": self.protocol.dt,
            "randomness": "philox:panel/scenario/tick/role",
            "counters": self.counters,
        }

    def aggregate_scores(self) -> np.ndarray:
        return owned(np.mean(self.scores, axis=-1))

    def drain_trace(self) -> list[dict]:
        """Transfer full committed rows to an enclosing evaluator or recorder."""
        rows = owned(self._unrecorded_trace)
        self._unrecorded_trace = []
        self._unrecorded_trace_bytes = 0
        return rows

    def _drain_chemical_events(self):
        drain = getattr(self.chemistry, "drain_trace", None)
        return numerical(drain()) if drain is not None and self.recording.mode == "full" else []

    def _trace_admitted(self, staged_bytes: int) -> bool:
        if (
            staged_bytes + self._unrecorded_trace_bytes + self._descriptor_bytes
            > self.recording.trace_budget_mib * 1024 * 1024
        ):
            self._status, self._reason = "partial", "trace_budget"
            return False
        return True

    @property
    def state(self) -> dict:
        return {
            "tick": self.tick,
            "status": self.status,
            "reason": self._reason,
            "counters": self.counters,
            "scores": self.scores,
            "success": self.success,
            "pending": None
            if self._pending is None
            else {
                "tick": self._pending["tick"],
                "chemical_cursor": self._pending["chemical_cursor"],
            },
        }

    def observe(self) -> dict:
        return owned(numerical(self.world.observe(numerical(self._world_state))))

    def snapshot(self, fields=None, include_pending: bool = False) -> dict:
        # Borrow existing arrays first; derived arrays and all owning copies are
        # allocated only after the selected fields pass shared admission.
        result = {
            "tick": self.tick,
            "world": self._world_state,
            "chemistry": self._committed_chemistry,
            "codec": self._codec_state,
            "actions": self._actions,
            "channels": self.channels,
            "reward_sum": self.reward_sum,
            "reward_count": self.reward_count,
            "scores": None,
            "status": None,
            "success": None,
        }
        if include_pending:
            result["pending"] = (
                None
                if self._pending is None and self._warmup_done >= self.controller.warmup
                else {
                    **(self._pending or {}),
                    "warmup_done": self._warmup_done,
                    "chemistry": self.chemistry._payload_view(),
                }
            )
        if fields is not None:
            result = {name: result[name] for name in fields}
        count = self.reward_sum.size
        derived_bytes = sum(
            size for name, size in (
                ("scores", self.reward_sum.nbytes), ("status", count * 64), ("success", count)
            ) if name in result
        )
        scratch = count * 3 if "scores" in result or "success" in result else 0
        if "success" in result and "scores" not in result:
            scratch += self.reward_sum.nbytes
        required = self.model.engine._snapshot_bytes(result) + 2 * derived_bytes + scratch
        self.model.engine._check_snapshot_memory(self, required)
        scores = None
        if "scores" in result or "success" in result:
            scores = np.divide(
                self.reward_sum, self.reward_count, out=np.zeros_like(self.reward_sum),
                where=self.reward_count > 0,
            )
        if "scores" in result:
            result["scores"] = scores
        if "status" in result:
            result["status"] = np.full(self.world_shape, self._status, dtype="U16")
        if "success" in result:
            assert scores is not None
            result["success"] = (
                (self._status == "complete") & (self.reward_count > 0)
                & (scores >= float(self.task.config.get("goal", 1.0)))
            )
        return owned(result)

    def _encode(self, observation: dict, codec_state):
        if self._codec is not None:
            inputs, state = self._codec.encode(numerical(observation), numerical(codec_state))
            return numerical(inputs), numerical(state)
        ports = self.controller.ports
        raw_dose = np.asarray(observation.get("dose", 0), dtype=np.int32)
        if self.agent_axis is not None and self.agents == 1 and raw_dose.shape == self.world_shape:
            raw_dose = raw_dose[..., None]
        dose = np.broadcast_to(raw_dose, self.controller_shape)
        if self.task.kind == "signalling_game":
            dose = dose * ports.max_dose
        inputs = {}
        if ports.metabolites:
            if np.any((dose < 0) | (dose > ports.max_dose)):
                raise ValueError("observation dose outside configured ports.max_dose")
            inputs[ports.metabolites[0]] = dose.copy()
        if len(ports.metabolites) > 1 and "mask" in observation:
            inputs[ports.metabolites[1]] = np.asarray(observation["mask"]) * ports.max_dose
        return inputs, codec_state

    def _decode(self, readouts: Mapping, codec_state):
        if self._codec is not None:
            actions, state = self._codec.decode(numerical(readouts), numerical(codec_state))
            actions = np.asarray(actions)
            if actions.dtype.kind not in "biu":
                raise ValueError("codec actions must be discrete integers")
            if np.any((actions < 0) | (actions > np.iinfo(np.int32).max)):
                raise ValueError("codec actions must fit nonnegative int32")
            actions = actions.astype(np.int32)
            if actions.shape != self.controller_shape:
                raise ValueError("codec actions must preserve all controller axes")
            return actions, numerical(state)
        name = self.task.config.get("action_readout", next(iter(readouts), None))
        if name not in readouts:
            raise ValueError("action decoder requires a configured controller readout")
        threshold = float(self.task.config.get("action_threshold", 0))
        actions = (np.asarray(readouts[name]) > threshold).astype(np.int32)
        if actions.shape != self.controller_shape:
            raise ValueError("readout shape does not match controller axes")
        return actions, codec_state

    def step(self):
        if self._pending_trace and not self._flush_world_trace():
            return self._report({"tick": self.tick})
        if self.protocol.horizon is not None and self.tick >= self.protocol.horizon:
            self._status = "complete"
        if self._status == "complete":
            return self._report({"tick": self.tick}, {"scores": self.scores})
        self._status, self._reason = "ready", None
        if not self._allow("world_ticks", self._world_work):
            return self._report({"tick": self.tick}, {"scores": self.scores})
        # Warm-up is persistent and uses the same chemical quota as decision updates.
        while self._warmup_done < self.controller.warmup:
            if not self._trace_admitted(self._warmup_event_bytes):
                return self._report({"tick": self.tick})
            if not self._allow("chemical_updates", self._chemical_work):
                return self._report({"tick": self.tick}, {"scores": self.scores})
            warmup_report = self.chemistry.step({})
            warmup_rows = self._drain_chemical_events()
            self._warmup_events.extend(warmup_rows)
            self._warmup_event_bytes += _trace_size(warmup_rows)
            if warmup_report.status in {"partial", "failed"}:
                if warmup_report.transition and warmup_report.transition.get("advanced", False):
                    self._consume("chemical_updates", self._chemical_work)
                    self._warmup_done += 1
                self._status, self._reason = warmup_report.status, warmup_report.reason
                return self._report({"tick": self.tick})
            self._consume("chemical_updates", self._chemical_work)
            self._warmup_done += 1
        if not self._trace_admitted(self._warmup_event_bytes):
            return self._report({"tick": self.tick})
        if self._reset_chemistry is None:
            self._reset_chemistry = self.chemistry.payload()
        if self._pending is None:
            if self.protocol.reset_per_decision:
                from .chemistry import ChemistrySession

                self.chemistry = ChemistrySession(
                    self.model,
                    self._controller_genes,
                    self._chemistry_spec,
                    WorkLimits(),
                    self.recording,
                )
                self.chemistry.restore_payload(self._reset_chemistry)
            observation = numerical(self.world.observe(numerical(self._world_state)))
            inputs, codec = self._encode(observation, self._codec_state)
            self._pending = {
                "tick": self.tick,
                "chemical_cursor": 0,
                "observation": observation,
                "inputs": inputs,
                "codec": codec,
                "chemical_events": self._warmup_events,
                "chemical_bytes": self._warmup_event_bytes,
            }
            self._warmup_events = []
            self._warmup_event_bytes = 0
        pending = self._pending
        if "prepared" in pending:
            return self._finish_prepared()
        while pending["chemical_cursor"] < self.protocol.chemical_updates:
            if not self._trace_admitted(pending["chemical_bytes"]):
                return self._report({"tick": self.tick})
            if not self._allow("chemical_updates", self._chemical_work):
                return self._report(
                    {"tick": self.tick, "chemical_cursor": pending["chemical_cursor"]},
                    {"scores": self.scores},
                )
            inject = (
                self.controller.ports.input_mode == "replenish" or pending["chemical_cursor"] == 0
            )
            report = self.chemistry.step(pending["inputs"] if inject else {})
            chemical_rows = self._drain_chemical_events()
            pending["chemical_events"].extend(chemical_rows)
            pending["chemical_bytes"] += _trace_size(chemical_rows)
            if report.status in {"partial", "failed"}:
                if report.transition and report.transition.get("advanced", False):
                    pending["chemical_cursor"] += 1
                    self._consume("chemical_updates", self._chemical_work)
                self._status, self._reason = report.status, report.reason
                return self._report(
                    {"tick": self.tick, "chemical_cursor": pending["chemical_cursor"]}
                )
            pending["chemical_cursor"] += 1
            self._consume("chemical_updates", self._chemical_work)
        if not self._trace_admitted(pending["chemical_bytes"]):
            return self._report({"tick": self.tick})
        actions, codec = self._decode(self.chemistry.last_readouts, pending["codec"])
        world_actions = (
            actions[..., 0] if self.agent_axis is not None and self.agents == 1 else actions
        )
        world, reward, channels = self.world.advance(numerical(self._world_state), world_actions)
        world, channels = numerical(world), numerical(channels)
        reward = np.asarray(reward, dtype=np.float64)
        if reward.shape != self.world_shape or not np.all(np.isfinite(reward)):
            raise ValueError("world reward must be finite and preserve world axes")
        if np.any((reward < 0) | (reward > 1)):
            raise ValueError("world rewards must be bounded in [0,1]")
        judged = np.broadcast_to(
            np.asarray(channels.get("judged", True), dtype=bool), self.world_shape
        )
        transition = {
            "type": "world_tick",
            "tick": self.tick + 1,
            "observation": pending["observation"],
            "actions": actions,
            "reward": reward,
            "channels": channels,
        }
        pending["prepared"] = {
            "world": world,
            "reward": reward,
            "judged": judged.copy(),
            "channels": channels,
            "codec": numerical(codec),
            "actions": actions.copy(),
            "transition": transition,
            "boundary_waiting": False,
            "boundary_done": False,
            "events": pending["chemical_events"] + [transition],
        }
        return self._finish_prepared()

    def _finish_prepared(self):
        """Publish a staged transition only after its required recording boundary."""
        assert self._pending is not None
        prepared = self._pending["prepared"]
        unrecorded_rows = []
        unrecorded_bytes = 0
        if self.recording.mode == "full" and self._record_callback is None:
            unrecorded_rows = prepared["events"]
            unrecorded_bytes = _trace_size(unrecorded_rows)
            if not self._trace_admitted(unrecorded_bytes):
                return self._report({"tick": self.tick}, {"scores": self.scores})
        descriptor = None
        descriptor_bytes = 0
        if bool(self.task.config.get("record_descriptor", False)):
            message = np.asarray(prepared["channels"].get("message", []), dtype=np.int32)
            descriptor = {"actions": prepared["actions"].copy(), "message": message.copy()}
            # Include conservative per-row Python container/array-header overhead,
            # so scalar traces also have a useful capacity bound.
            descriptor_bytes = prepared["actions"].nbytes + message.nbytes + 1024
            if (
                self._descriptor_bytes
                + descriptor_bytes
                + self._unrecorded_trace_bytes
                + unrecorded_bytes
                > self.recording.trace_budget_mib * 1024 * 1024
            ):
                self._status, self._reason = "partial", "descriptor_budget"
                return self._report({"tick": self.tick}, {"scores": self.scores})
        # A failed full trace is flushed at the top of step(). Its world update is
        # already staged, so decode/advance/randomness never repeat on retry.
        if prepared["boundary_waiting"] and not self._pending_trace:
            prepared["boundary_done"] = True
        if not prepared["boundary_done"]:
            try:
                self._publishing = True
                accepted = self._boundary(
                    prepared["events"],
                    force=(
                        self.protocol.horizon is not None and self.tick + 1 >= self.protocol.horizon
                    ),
                )
            except BaseException:
                prepared["boundary_waiting"] = bool(self._pending_trace)
                raise
            finally:
                self._publishing = False
            prepared["boundary_waiting"] = bool(self._pending_trace)
            if not accepted:
                return self._report({"tick": self.tick}, {"scores": self.scores})
        self._world_state, self.channels = prepared["world"], prepared["channels"]
        self._codec_state, self._actions = prepared["codec"], prepared["actions"]
        self.reward_sum += prepared["reward"] * prepared["judged"]
        self.reward_count += prepared["judged"]
        self.tick += 1
        self._committed_chemistry = self.chemistry.payload()
        self._pending = None
        self._consume("world_ticks", self._world_work)
        self._unrecorded_trace.extend(unrecorded_rows)
        self._unrecorded_trace_bytes += unrecorded_bytes
        if descriptor is not None:
            self._descriptor.append(descriptor)
            self._descriptor_bytes += descriptor_bytes
        if self.protocol.horizon is not None and self.tick >= self.protocol.horizon:
            self._status = "complete"
        return self._report({"tick": self.tick}, {"scores": self.scores}, prepared["transition"])

    def _flush_world_trace(self) -> bool:
        projecting = self._pending is not None and "prepared" in self._pending
        try:
            self._publishing = projecting
            accepted = self._flush_pending_trace()
        finally:
            self._publishing = False
        if accepted and projecting:
            assert self._pending is not None
            self._pending["prepared"]["boundary_done"] = True
        return accepted

    def run(self, steps: int | None = None):
        if steps is not None and (type(steps) is not int or steps < 0):
            raise ValueError("steps must be a nonnegative integer or None")
        if steps is None and self.protocol.horizon is None and self.limits.max_world_ticks is None:
            raise ValueError("unbounded episodes require steps or a world-tick work limit")
        count = 0
        if self._pending_trace and not self._flush_world_trace():
            return self._report({"tick": self.tick}, {"scores": self.scores})
        if self.protocol.horizon is not None and self.tick >= self.protocol.horizon:
            self._status = "complete"
        report = self._report({"tick": self.tick}, {"scores": self.scores})
        while self._status != "complete" and (steps is None or count < steps):
            report = self.step()
            count += 1
            if self._status in {"partial", "failed"}:
                break
        return report

    def _checkpoint_payload(self) -> dict:
        payload = {
            "task": _plain(self.task),
            "protocol": _plain(self.protocol),
            "controller": _plain(self.controller),
            "genotypes": self._source_genotypes,
            "seed": self.seed,
            "panel": self.panel,
            "agent_axis": self.agent_axis,
            "identities": self.identities,
            "tick": self.tick,
            "world": self._world_state,
            "chemistry": self.chemistry.payload(),
            "committed_chemistry": self._committed_chemistry,
            "codec": self._codec_state,
            "pending": self._pending,
            "actions": self._actions,
            "reward_sum": self.reward_sum,
            "reward_count": self.reward_count,
            "channels": self.channels,
            "descriptor": self._descriptor,
            "descriptor_bytes": self._descriptor_bytes,
            "unrecorded_trace": self._unrecorded_trace,
            "unrecorded_trace_bytes": self._unrecorded_trace_bytes,
            "warmup_events": self._warmup_events,
            "warmup_event_bytes": self._warmup_event_bytes,
            "warmup_done": self._warmup_done,
            "reset_chemistry": self._reset_chemistry,
        }
        if self._publishing:
            assert self._pending is not None
            prepared = self._pending["prepared"]
            descriptor = list(self._descriptor)
            descriptor_bytes = self._descriptor_bytes
            if bool(self.task.config.get("record_descriptor", False)):
                message = np.asarray(prepared["channels"].get("message", []), dtype=np.int32)
                descriptor.append({"actions": prepared["actions"], "message": message})
                descriptor_bytes += prepared["actions"].nbytes + message.nbytes + 1024
            payload.update(
                {
                    "tick": self.tick + 1,
                    "world": prepared["world"],
                    "committed_chemistry": self.chemistry.payload(),
                    "codec": prepared["codec"],
                    "pending": None,
                    "actions": prepared["actions"],
                    "channels": prepared["channels"],
                    "reward_sum": self.reward_sum + prepared["reward"] * prepared["judged"],
                    "reward_count": self.reward_count + prepared["judged"],
                    "descriptor": descriptor,
                    "descriptor_bytes": descriptor_bytes,
                }
            )
        return payload

    def checkpoint(self) -> Checkpoint:
        if not self._publishing:
            return super().checkpoint()
        status, used, total = self._status, self._used, self._total
        try:
            self._used = {**used, "world_ticks": used["world_ticks"] + self._world_work}
            self._total = {**total, "world_ticks": total["world_ticks"] + self._world_work}
            self._status = (
                "complete"
                if self.protocol.horizon is not None and self.tick + 1 >= self.protocol.horizon
                else "ready"
            )
            return super().checkpoint()
        finally:
            self._status, self._used, self._total = status, used, total

    def _restore_payload(self, payload: dict) -> None:
        tick = payload["tick"]
        if (
            type(tick) is not int
            or tick < 0
            or (self.protocol.horizon is not None and tick > self.protocol.horizon)
        ):
            raise ValueError("invalid rollout world cursor")
        if (
            self._status == "complete"
            and self.protocol.horizon is not None
            and tick != self.protocol.horizon
        ):
            raise ValueError("completed checkpoint disagrees with episode horizon")
        reward_sum = np.asarray(payload["reward_sum"])
        reward_count = np.asarray(payload["reward_count"])
        actions = np.asarray(payload["actions"])
        if (
            reward_sum.shape != self.world_shape
            or reward_sum.dtype.kind not in "fiu"
            or not np.all(np.isfinite(reward_sum))
            or np.any(reward_sum < 0)
            or reward_count.shape != self.world_shape
            or reward_count.dtype.kind not in "iu"
            or np.any((reward_count < 0) | (reward_count > tick))
            or np.any(reward_sum > reward_count)
        ):
            raise ValueError("invalid rollout checkpoint reward arrays")
        if (
            actions.shape != self.controller_shape
            or actions.dtype.kind not in "biu"
            or np.any((actions < 0) | (actions > np.iinfo(np.int32).max))
        ):
            raise ValueError("invalid rollout checkpoint action arrays")
        pending = payload["pending"]
        if pending is not None and (
            pending.get("tick") != tick
            or type(pending.get("chemical_cursor")) is not int
            or not 0 <= pending["chemical_cursor"] <= self.protocol.chemical_updates
        ):
            raise ValueError("invalid staged chemical cursor")
        warmup = payload["warmup_done"]
        if type(warmup) is not int or not 0 <= warmup <= self.controller.warmup:
            raise ValueError("invalid warmup cursor")
        self.tick = tick
        self._world_state = numerical(payload["world"])
        self.chemistry.restore_payload(payload["chemistry"])
        self._committed_chemistry = numerical(payload["committed_chemistry"])
        self._codec_state = numerical(payload["codec"])
        self._pending = numerical(payload["pending"])
        self._actions = np.array(payload["actions"], copy=True)
        self.reward_sum = np.array(reward_sum, dtype=np.float64, copy=True)
        self.reward_count = np.array(reward_count, dtype=np.int64, copy=True)
        self.channels = numerical(payload["channels"])
        self._descriptor = numerical(payload["descriptor"])
        self._descriptor_bytes = int(payload.get("descriptor_bytes", 0))
        self._unrecorded_trace = numerical(payload.get("unrecorded_trace", []))
        self._unrecorded_trace_bytes = int(payload.get("unrecorded_trace_bytes", 0))
        self._warmup_events = numerical(payload.get("warmup_events", []))
        self._warmup_event_bytes = int(payload.get("warmup_event_bytes", 0))
        self._warmup_done = warmup
        self._reset_chemistry = numerical(payload.get("reset_chemistry"))
