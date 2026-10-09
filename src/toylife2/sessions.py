"""Shared session lifecycle, typed work accounting and numerical checkpoints."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .protocols import WorkLimits
from .specs import RecordingSpec


def owned(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, Mapping):
        return {str(k): owned(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(owned(x) for x in value)
    if isinstance(value, list):
        return [owned(x) for x in value]
    return copy.deepcopy(value)


def pack(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("checkpoints require numeric or string arrays, never object arrays")
        name = f"a{len(arrays):06d}"
        arrays[name] = owned(value)
        return {"__array__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("checkpoint mapping keys must be strings")
        result = {k: pack(v, arrays) for k, v in value.items()}
        if set(result) in ({"__array__"}, {"__tuple__"}, {"__mapping__"}):
            return {"__mapping__": result}
        return result
    if isinstance(value, tuple):
        return {"__tuple__": [pack(x, arrays) for x in value]}
    if isinstance(value, list):
        return [pack(x, arrays) for x in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported checkpoint value: {type(value).__name__}")


def unpack(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__mapping__"}:
            return {k: unpack(v, arrays) for k, v in value["__mapping__"].items()}
        if set(value) == {"__array__"}:
            return np.array(arrays[value["__array__"]], copy=True)
        if set(value) == {"__tuple__"}:
            return tuple(unpack(x, arrays) for x in value["__tuple__"])
        return {k: unpack(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(x, arrays) for x in value]
    return value


@dataclass(frozen=True)
class Checkpoint:
    kind: str
    identity: str
    state: dict[str, Any]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported session checkpoint schema")
        object.__setattr__(self, "state", owned(self.state))
        object.__setattr__(self, "arrays", owned(self.arrays))
        json.dumps(self.state, allow_nan=False)

    @property
    def payload(self) -> dict[str, Any]:
        return unpack(self.state, self.arrays)

    def record_state(self) -> dict[str, Any]:
        return {
            "session_schema": self.schema_version,
            "session_kind": self.kind,
            "session_identity": self.identity,
            "session_state": self.state,
        }

    @classmethod
    def from_record(cls, data: Mapping[str, Any]) -> Checkpoint:
        state = data["state"]
        session_state = owned(state["session_state"])
        if state.get("pending_trace"):
            pending = state["pending_trace"]
            session_state["base"]["pending_trace"] = owned(
                pending if isinstance(pending, list) else [pending]
            )
            session_state["base"]["status"] = "partial"
        return cls(
            state["session_kind"],
            state["session_identity"],
            session_state,
            dict(data.get("loaded_arrays", {})),
            state["session_schema"],
        )

    def write(self, directory: str | Path) -> Path:
        """Write an unrecorded portable checkpoint to a new directory."""
        from .records import _array, _write_json

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        for name, array in self.arrays.items():
            _array(path / f"{name}.npy", array)
        _write_json(
            path / "checkpoint.json",
            {"state": self.record_state(), "arrays": {name: f"{name}.npy" for name in self.arrays}},
        )
        return path

    @classmethod
    def read(cls, directory: str | Path) -> Checkpoint:
        from .records import read_checkpoint

        return cls.from_record(read_checkpoint(directory))


@dataclass(frozen=True)
class StepReport:
    status: str
    cursors: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None
    transition: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cursors", owned(self.cursors))
        object.__setattr__(self, "metrics", owned(self.metrics))
        object.__setattr__(self, "transition", owned(self.transition))

    @property
    def complete(self) -> bool:
        return self.status == "complete"


class SessionBase:
    """Subclasses own transitions; the base owns invocation limits and durability."""

    def __init__(
        self,
        kind: str,
        identity: str,
        limits: WorkLimits | None = None,
        recording: RecordingSpec | None = None,
        checkpoint: Checkpoint | None = None,
    ):
        self.kind, self.identity = kind, identity
        self.limits = limits or WorkLimits()
        self.recording = recording or RecordingSpec()
        self._status = "ready"
        self._reason: str | None = None
        self._used = {name: 0 for name in ("chemical_updates", "world_ticks", "evolution_updates")}
        self._total = dict(self._used)
        self._pending_trace: list[dict] = []
        self._record_callback: Callable[[Checkpoint, list[dict]], bool] | None = None
        self._boundaries = 0
        if checkpoint is not None:
            if checkpoint.kind != kind or checkpoint.identity != identity:
                raise ValueError("checkpoint kind or scientific identity mismatch")
            base = checkpoint.payload["base"]
            self._total = dict(base["total"])
            if any(type(v) is not int or v < 0 for v in self._total.values()):
                raise ValueError("invalid checkpoint work counters")
            self._pending_trace = list(base.get("pending_trace", []))
            self._boundaries = int(base.get("boundaries", 0))
            if base.get("status") == "complete":
                self._status = "complete"
            if self._pending_trace:
                self._status, self._reason = "partial", "trace_budget"

    def remaining(self, resource: str) -> int | None:
        if resource not in self._used:
            raise ValueError(f"unknown resource: {resource}")
        limit = getattr(self.limits, "max_" + resource)
        return None if limit is None else max(0, limit - self._used[resource])

    def _allow(self, resource: str, count: int = 1) -> bool:
        if type(count) is not int or count < 0:
            raise ValueError("work count must be a nonnegative integer")
        available = self.remaining(resource)
        if available is not None and count > available:
            self._status, self._reason = "partial", "max_" + resource
            return False
        return True

    def _consume(self, resource: str, count: int = 1) -> None:
        if not self._allow(resource, count):
            raise RuntimeError("work quota exhausted")
        self._used[resource] += count
        self._total[resource] += count

    def _checkpoint_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def checkpoint(self) -> Checkpoint:
        arrays: dict[str, np.ndarray] = {}
        state = pack(
            {
                "base": {
                    "status": self._status,
                    "reason": self._reason,
                    "used": self._used,
                    "total": self._total,
                    "boundaries": self._boundaries,
                    "pending_trace": self._pending_trace,
                },
                "payload": self._checkpoint_payload(),
            },
            arrays,
        )
        return Checkpoint(self.kind, self.identity, state, arrays)

    def _flush_pending_trace(self) -> bool:
        if not self._pending_trace:
            return True
        if self._record_callback is None:
            return False
        if self._record_callback(self.checkpoint(), self._pending_trace):
            self._pending_trace = []
            self._status, self._reason = "ready", None
            return True
        self._status, self._reason = "partial", "trace_budget"
        return False

    def _boundary(self, events: list[dict] | None = None, force: bool = False) -> bool:
        if self._pending_trace and not self._flush_pending_trace():
            return False
        self._boundaries += 1
        rows = list(events or []) if self.recording.mode == "full" else []
        if self._record_callback is None:
            return True
        due = force or bool(rows) or self._boundaries % self.recording.checkpoint_interval == 0
        if not due:
            return True
        self._pending_trace = rows
        try:
            accepted = self._record_callback(self.checkpoint(), rows)
        except BaseException:
            self._status, self._reason = "partial", "checkpoint_write"
            raise
        if not accepted:
            self._status, self._reason = "partial", "trace_budget"
            return False
        self._pending_trace = []
        return True

    @property
    def counters(self) -> dict[str, dict[str, int]]:
        return {"invocation": dict(self._used), "total": dict(self._total)}

    def iter_steps(self):
        while self._status not in {"complete", "partial", "failed"}:
            yield self.step()  # type: ignore[attr-defined]

    def _report(
        self,
        cursors: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        transition: Any = None,
    ) -> StepReport:
        return StepReport(self._status, cursors or {}, metrics or {}, self._reason, transition)
