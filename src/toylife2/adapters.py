"""Transactional ask/tell sessions for numerical external optimizers."""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .evaluation import EvaluationResult, Evaluator
from .genotypes import validate
from .sessions import SessionBase, owned, pack
from .specs import digest, plain, scientific_identity


def _mutable(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_mutable(item) for item in value)
    if isinstance(value, list):
        return [_mutable(item) for item in value]
    return copy.deepcopy(value)


def _numeric_state(value):
    arrays = {}
    encoded = pack(value, arrays)
    json.dumps(encoded, allow_nan=False)
    for array in arrays.values():
        if array.dtype.kind not in "biufUS":
            raise ValueError("adapter state requires numerical or string arrays")
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise ValueError("adapter state arrays must be finite")
    return _mutable(value)


@dataclass(frozen=True)
class CandidateBatch:
    """One ordered optimizer request, with stable per-candidate identities."""

    request_id: str | int
    genotypes: np.ndarray
    candidate_ids: np.ndarray | None = None

    def __post_init__(self):
        if type(self.request_id) not in (str, int) or self.request_id == "":
            raise ValueError("request_id must be a nonempty string or integer")
        genes = validate(self.genotypes)
        if genes.ndim != 2 or not len(genes):
            raise ValueError("external optimizer requests require nonempty [candidate,gene]")
        ids = (
            np.arange(len(genes), dtype=np.uint64)
            if self.candidate_ids is None
            else np.asarray(self.candidate_ids)
        )
        if ids.shape != (len(genes),) or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("candidate_ids must be nonnegative integer IDs in request order")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("candidate_ids must be unique within an optimizer request")
        object.__setattr__(self, "genotypes", owned(genes))
        object.__setattr__(self, "candidate_ids", owned(ids))

    def payload(self):
        return {
            "request_id": self.request_id,
            "genotypes": self.genotypes,
            "candidate_ids": self.candidate_ids,
        }


def _result_payload(result):
    if result is None:
        return None
    return {
        field.name: _mutable(getattr(result, field.name))
        for field in dataclasses.fields(EvaluationResult)
    }


class ExternalSession(SessionBase):
    """Ask once, retain a pending evaluation, and commit each successful tell.

    The adapter implements ``ask() -> CandidateBatch``, ``tell(batch, result)``,
    ``state_dict()`` and ``load_state_dict(state)``. Its numerical state must
    contain every optimizer and random-generator cursor needed for continuation.
    """

    def __init__(
        self, model, spec, adapter, limits=None, recording=None, checkpoint=None, evaluator=None
    ):
        for name in ("ask", "tell", "state_dict", "load_state_dict"):
            if not callable(getattr(adapter, name, None)):
                raise TypeError(f"external adapter requires {name}()")
        self.model, self.spec, self.adapter = model, spec, adapter
        self.evaluator = evaluator or Evaluator(model, spec)
        self._adapter_source = spec.task.config.get("adapter_entrypoint")
        identity = digest(
            {
                "kind": "external",
                "scientific_identity": scientific_identity(spec),
                "evaluator_context": self.evaluator.context,
                "adapter": self._adapter_source
                or f"{type(adapter).__module__}:{type(adapter).__qualname__}",
            }
        )
        self.update = 0
        self.phase = "idle"
        self.pending = None
        self.result = None
        self._postask_state = None
        self._optimizer_state = _numeric_state(adapter.state_dict())
        self._last_request_id = None
        super().__init__(
            "external", identity, limits or spec.limits, recording or spec.recording, checkpoint
        )
        if checkpoint is not None:
            self.restore_payload(checkpoint.payload["payload"])

    @property
    def state(self):
        return MappingProxyType(
            {
                "update": self.update,
                "phase": self.phase,
                "status": self._status,
                "reason": self._reason,
                "request_id": None if self.pending is None else self.pending.request_id,
                "pending_candidates": 0 if self.pending is None else len(self.pending.genotypes),
                "counters": self.counters,
            }
        )

    def observe(self):
        return self.state

    def _cursor(self):
        return {
            "external_update": self.update,
            "phase": self.phase,
            "request_id": None if self.pending is None else self.pending.request_id,
        }

    def _stop(self, reason):
        self._status, self._reason = "partial", reason
        return self._report(self._cursor(), transition={"advanced": False})

    def _ask(self):
        self.phase = "preask"
        self._optimizer_state = _numeric_state(self.adapter.state_dict())
        preask = _mutable(self._optimizer_state)
        if not self._boundary([{"type": "external_preask", "update": self.update}], force=True):
            return False
        try:
            batch = self.adapter.ask()
            if not isinstance(batch, CandidateBatch):
                raise TypeError("adapter.ask must return CandidateBatch")
            if batch.genotypes.shape[-1] != self.spec.model.genes:
                raise ValueError("adapter candidate gene axis differs from ModelSpec")
            if batch.request_id == self._last_request_id:
                raise ValueError("adapter request IDs must change between requests")
            postask = _numeric_state(self.adapter.state_dict())
        except BaseException:
            self.adapter.load_state_dict(_mutable(preask))
            self._optimizer_state = preask
            self._status, self._reason = "partial", "adapter_ask"
            raise
        self.pending, self.result = batch, None
        self._optimizer_state = self._postask_state = postask
        self.phase = "postask"
        return self._boundary(
            [{"type": "external_postask", "request": batch.payload(), "update": self.update}],
            force=True,
        )

    def step(self):
        was_complete = self._status == "complete"
        if self._pending_trace:
            if not self._flush_pending_trace():
                return self._report(self._cursor(), transition={"advanced": False})
            if was_complete:
                self._status = "complete"
        if self._status in {"complete", "partial", "failed"}:
            return self._report(self._cursor(), transition={"advanced": False})
        if not self._allow("evolution_updates"):
            return self._report(self._cursor(), transition={"advanced": False})
        if self._pending_trace and not self._flush_pending_trace():
            return self._report(self._cursor(), transition={"advanced": False})
        if self.pending is None and not self._ask():
            return self._report(self._cursor(), transition={"advanced": False})
        request = self.pending
        assert request is not None and request.candidate_ids is not None
        if self.phase == "postask":
            self.phase = "evaluating"
        if self.result is None or not self.result.complete:
            limits = dataclasses.replace(
                self.limits,
                max_world_ticks=self.remaining("world_ticks"),
                max_chemical_updates=self.remaining("chemical_updates"),
            )
            self.evaluator.set_work_limits(limits)
            before = dict(self.evaluator.metrics)
            try:
                result = self.evaluator.evaluate(
                    request.genotypes, candidate_ids=request.candidate_ids
                )
            except BaseException:
                self._status, self._reason = "partial", "adapter_evaluation"
                raise
            finally:
                for resource in ("world_ticks", "chemical_updates"):
                    used = self.evaluator.metrics.get(resource, 0) - before.get(resource, 0)
                    if used:
                        self._consume(resource, int(used))
            if not isinstance(result, EvaluationResult):
                raise TypeError("external evaluation requires EvaluationResult")
            if result.scores.shape != (len(request.genotypes),):
                raise ValueError("evaluation must preserve ordered candidate axes")
            if result.candidate_ids is None or not np.array_equal(
                result.candidate_ids, request.candidate_ids
            ):
                raise ValueError("evaluation candidate IDs must preserve request order")
            self.result = result
            events = self.evaluator.take_events() if hasattr(self.evaluator, "take_events") else []
            if not result.complete:
                self._boundary(
                    events
                    + [
                        {
                            "type": "external_evaluation_partial",
                            "request_id": request.request_id,
                            "result": _result_payload(result),
                        }
                    ],
                    force=True,
                )
                return self._stop("evaluation_incomplete")
            if not self._boundary(
                events
                + [
                    {
                        "type": "external_evaluation_complete",
                        "request_id": request.request_id,
                        "result": _result_payload(result),
                    }
                ],
                force=True,
            ):
                return self._stop("trace_budget")
        self.phase = "tell"
        # Keep the post-ask optimizer and pending complete result as the stable
        # retry point until the tell checkpoint has been durably accepted.
        rollback = {
            "update": self.update,
            "pending": self.pending,
            "result": self.result,
            "used": dict(self._used),
            "total": dict(self._total),
            "boundaries": self._boundaries,
            "trace": _mutable(self._pending_trace),
            "last_request_id": self._last_request_id,
        }
        try:
            self.adapter.tell(self.pending, self.result)
            self._optimizer_state = _numeric_state(self.adapter.state_dict())
            request_id = request.request_id
            result_payload = _result_payload(self.result)
            self.update += 1
            self._consume("evolution_updates")
            self._last_request_id = request_id
            self.pending, self.result = None, None
            self.phase = "idle"
            accepted = self._boundary(
                [
                    {
                        "type": "external_tell",
                        "update": self.update,
                        "request_id": request_id,
                        "result": result_payload,
                    }
                ],
                force=True,
            )
            if not accepted:
                self._rollback_tell(rollback)
                return self._stop("adapter_commit")
        except BaseException:
            self._rollback_tell(rollback)
            self._status, self._reason = "partial", "adapter_tell_or_commit"
            raise
        self._postask_state = None
        return self._report(self._cursor(), transition={"advanced": True})

    def _rollback_tell(self, previous):
        self.adapter.load_state_dict(_mutable(self._postask_state))
        self._optimizer_state = _mutable(self._postask_state)
        self.update, self.pending, self.result = (
            previous["update"],
            previous["pending"],
            previous["result"],
        )
        self._used, self._total = previous["used"], previous["total"]
        self._boundaries, self._pending_trace = previous["boundaries"], previous["trace"]
        self._last_request_id = previous["last_request_id"]
        self.phase = "tell"

    def run(self, updates=None):
        if updates is not None and (type(updates) is not int or updates < 0):
            raise ValueError("updates must be a nonnegative integer or None")
        report = self._report(self._cursor(), transition={"advanced": False})
        count = 0
        while updates is None or count < updates:
            report = self.step()
            if report.status in {"partial", "complete", "failed"}:
                break
            count += 1
        return report

    def snapshot(self, fields=None, include_pending=False):
        result = {
            "state": dict(self.state),
            "optimizer": owned(self._optimizer_state),
            "request": None if self.pending is None else owned(self.pending.payload()),
            "result": owned(_result_payload(self.result)),
        }
        if include_pending:
            result["pending_trace"] = owned(self._pending_trace)
        if fields is not None:
            result = {key: result[key] for key in fields}
        return MappingProxyType(result)

    def _checkpoint_payload(self):
        return {
            "spec": plain(self.spec),
            "adapter_entrypoint": self._adapter_source,
            "update": self.update,
            "phase": self.phase,
            "optimizer": _numeric_state(self._optimizer_state),
            "postask_optimizer": _numeric_state(self._postask_state),
            "request": None if self.pending is None else self.pending.payload(),
            "result": _result_payload(self.result),
            "last_request_id": self._last_request_id,
            "evaluator": self.evaluator.checkpoint_payload(),
        }

    def restore_payload(self, payload):
        update = payload["update"]
        phase = payload["phase"]
        if (
            type(update) is not int
            or update < 0
            or phase not in {"idle", "preask", "postask", "evaluating", "tell"}
        ):
            raise ValueError("invalid external checkpoint cursor")
        optimizer = _numeric_state(payload["optimizer"])
        postask = _numeric_state(payload["postask_optimizer"])
        request = None if payload["request"] is None else CandidateBatch(**payload["request"])
        result = None if payload["result"] is None else EvaluationResult(**payload["result"])
        if (request is None) != (phase in {"idle", "preask"}):
            raise ValueError("external checkpoint pending request disagrees with phase")
        if result is not None and (
            request is None
            or request.candidate_ids is None
            or result.candidate_ids is None
            or not np.array_equal(result.candidate_ids, request.candidate_ids)
        ):
            raise ValueError("external checkpoint result/request order mismatch")
        self.adapter.load_state_dict(_mutable(optimizer))
        self._optimizer_state, self._postask_state = optimizer, postask
        self.update, self.phase = update, phase
        self.pending, self.result = request, result
        self._last_request_id = payload["last_request_id"]
        self.evaluator.restore_payload(payload["evaluator"])
