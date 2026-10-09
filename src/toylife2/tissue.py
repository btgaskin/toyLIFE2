"""Incremental tissue evolution using the canonical six-inventory CA rules."""

from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType

import numpy as np

from .chemistry import _owned
from .genotypes import validate
from .sessions import SessionBase
from .spatial import ca_from_descriptors, response_descriptors
from .specs import RULESET, SpatialSpec, digest


class TissueSession(SessionBase):
    """Own one current tissue row per genotype and an exact time cursor.

    ``spec.steps`` defines the legacy scheduled injection window. ``run(0)`` and
    snapshots leave the initial boundary intact. Later ticks extend the tissue
    naturally, with continuous central injections or zero explicit injections.
    """

    def __init__(self, model, genotypes, spec=None, limits=None, recording=None, checkpoint=None):
        spec = spec or SpatialSpec()
        genes = validate(genotypes)
        if genes.shape[-1] != 2:
            raise ValueError("tissue requires exactly two genes")
        self.model, self.spec = model, spec
        self.genotypes = _owned(genes)
        self.leading = genes.shape[:-1]
        self.tick = 0
        identity = digest(
            {
                "kind": "tissue",
                "table_digest": str(model.tables.digest),
                "genotypes": genes.tolist(),
                "spatial": asdict(spec),
            }
        )
        super().__init__("tissue", identity, limits, recording, checkpoint)
        flat = genes.reshape(-1, 2)
        descriptors = response_descriptors(model.tables, flat, engine=model.engine)
        ca = ca_from_descriptors(descriptors, spec.diffusion, input_states=6)
        initial = np.zeros(spec.cells, np.int32)
        if spec.initial_state is not None:
            initial[:] = spec.initial_state
        self._host = np.broadcast_to(initial, (len(flat), spec.cells)).copy()
        device_bytes = ca.nbytes + self._host.nbytes * 3 + len(flat) * 4
        host_bytes = self._host.nbytes * 3 + self.genotypes.nbytes
        self._buffer_memory = {
            "device": device_bytes,
            "host": host_bytes,
            "total": device_bytes + host_bytes,
        }
        required = self._buffer_memory["total"]
        if (
            required
            + model.engine._table_bytes
            + getattr(model.engine, "_reserved_session_bytes", 0)
            > model.execution.memory_budget_mib * 1024 * 1024
        ):
            raise MemoryError("persistent tissue exceeds execution memory budget")
        self._ca = model.engine._i32(ca.shape, ca)
        self._dimer = model.engine._i32((len(flat),), descriptors[:, 6])
        self._source = model.engine._i32(self._host.shape, self._host)
        self._target = model.engine._i32(self._host.shape)
        self._injections = model.engine._i32(self._host.shape)
        if checkpoint is not None:
            self.restore_payload(checkpoint.payload["payload"])

    @property
    def buffer_bytes(self):
        return self._buffer_memory["total"]

    @property
    def buffer_memory(self):
        return MappingProxyType(dict(self._buffer_memory))

    @property
    def state(self):
        return MappingProxyType(
            {
                "tick": self.tick,
                "status": self._status,
                "reason": self._reason,
                "tissue": _owned(self._host.reshape((*self.leading, self.spec.cells))),
                "counters": self.counters,
            }
        )

    def observe(self):
        return self.state

    def _schedule(self):
        result = np.zeros(self._host.shape, np.int32)
        spec = self.spec
        if spec.morphogen == "explicit":
            for tick, cell, state in spec.injections:
                if tick == self.tick:
                    result[:, cell] |= state
        elif spec.morphogen != "none":
            if spec.morphogen.endswith("continuous") or self.tick == 0:
                species = 2 if "_A_" in spec.morphogen else 1
                cells = (
                    (spec.cells // 2,) if spec.cells % 2 else (spec.cells // 2 - 1, spec.cells // 2)
                )
                result[:, list(cells)] = species
        return result

    def step(self, injections=None):
        from ._chemistry_kernels import tissue_tick

        was_complete = self._status == "complete"
        if self._pending_trace:
            if not self._flush_pending_trace():
                return self._report({"tissue_tick": self.tick}, transition={"advanced": False})
            if was_complete:
                self._status = "complete"
        if self._status in {"complete", "partial", "failed"}:
            return self._report({"tissue_tick": self.tick}, transition={"advanced": False})
        units = len(self._host)
        if not self._allow("world_ticks", units):
            return self._report({"tissue_tick": self.tick}, transition={"advanced": False})
        if self._pending_trace and not self._flush_pending_trace():
            return self._report({"tissue_tick": self.tick}, transition={"advanced": False})
        schedule = self._schedule()
        if injections is not None:
            incoming = np.asarray(injections)
            if incoming.dtype.kind not in "iu" or np.any(incoming < 0) or np.any(incoming > 3):
                raise ValueError("injections must be integer states in [0,3]")
            try:
                schedule |= (
                    np.broadcast_to(incoming, (*self.leading, self.spec.cells))
                    .reshape(schedule.shape)
                    .astype(np.int32)
                )
            except ValueError as error:
                raise ValueError(
                    "injections shape must match tissue leading dimensions and cells"
                ) from error
        self._injections.from_numpy(schedule)
        tissue_tick(
            self._ca,
            self._injections,
            self._dimer,
            self._source,
            self._target,
            int(self.spec.boundary == "ring"),
        )
        host = self._target.to_numpy()
        self._source, self._target = self._target, self._source
        self._host = host
        self.tick += 1
        self._consume("world_ticks", units)
        self._boundary(
            [
                {
                    "type": "tissue_tick",
                    "tick": self.tick,
                    "tissue": self._host,
                    "injections": schedule,
                }
            ]
        )
        return self._report({"tissue_tick": self.tick}, transition={"advanced": True})

    def run(self, steps):
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        report = self._report({"tissue_tick": self.tick}, transition={"advanced": False})
        for _ in range(steps):
            report = self.step()
            if report.status in {"partial", "complete", "failed"}:
                break
        return report

    def snapshot(self, fields=None, include_pending=False):
        result = {
            "state": dict(self.state),
            "genotypes": _owned(self.genotypes),
            "tissue": _owned(self._host.reshape((*self.leading, self.spec.cells))),
        }
        if include_pending:
            from .sessions import owned

            result["pending_trace"] = owned(self._pending_trace)
        if fields is not None:
            result = {key: result[key] for key in fields}
        return MappingProxyType(result)

    def _checkpoint_payload(self):
        return self.payload()

    def payload(self):
        return {
            "identity": self.identity,
            "table_digest": str(self.model.tables.digest),
            "ruleset": RULESET,
            "tick": self.tick,
            "spec": asdict(self.spec),
            "genotypes": self.genotypes.copy(),
            "tissue": self._host.copy(),
        }

    def restore_payload(self, payload):
        if (
            payload.get("table_digest") != str(self.model.tables.digest)
            or payload.get("ruleset") != RULESET
        ):
            raise ValueError("tissue checkpoint table identity mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("tissue payload scientific identity mismatch")
        tissue = np.asarray(payload["tissue"])
        tick = payload["tick"]
        if (
            type(tick) is not int
            or tick < 0
            or tissue.shape != self._host.shape
            or tissue.dtype.kind not in "iu"
            or np.any(tissue < 0)
            or np.any(tissue > 3)
        ):
            raise ValueError("invalid tissue checkpoint")
        self._host = tissue.astype(np.int32, copy=True)
        self._source.from_numpy(self._host)
        self.tick = tick
