from __future__ import annotations

import numpy as np
import pytest

from toylife2.adapters import CandidateBatch, ExternalSession
from toylife2.evaluation import EvaluationResult
from toylife2.protocols import WorkLimits
from toylife2.specs import RecordingSpec, RunSpec


class Optimizer:
    def __init__(self):
        self.asks = 0
        self.tells = 0
        self.best = np.zeros(2)
        self.fail_ask = False
        self.fail_tell = False

    def state_dict(self):
        return {"asks": self.asks, "tells": self.tells, "best": self.best.copy()}

    def load_state_dict(self, state):
        self.asks, self.tells = state["asks"], state["tells"]
        self.best = state["best"].copy()

    def ask(self):
        request = self.asks
        self.asks += 1
        if self.fail_ask:
            raise RuntimeError("ask failed after mutation")
        return CandidateBatch(
            f"request-{request}",
            np.asarray([[0, 0], [1, 0]], np.uint32),
            np.asarray([44, 2], np.uint64),
        )

    def tell(self, request, result):
        np.testing.assert_array_equal(request.candidate_ids, [44, 2])
        np.testing.assert_array_equal(result.candidate_ids, request.candidate_ids)
        self.tells += 1
        self.best[:] = result.scores
        if self.fail_tell:
            raise RuntimeError("tell failed after mutation")


class Scores:
    context = "fixture-context"

    def __init__(self, complete=True):
        self.complete = complete
        self.calls = 0
        self.metrics = dict(world_ticks=0, chemical_updates=0)

    def set_work_limits(self, limits):
        self.limits = limits

    def evaluate(self, genes, candidate_ids):
        self.calls += 1
        return EvaluationResult(
            np.asarray([0.2, 0.9]),
            status=np.full(2, "complete" if self.complete else "incomplete"),
            candidate_ids=candidate_ids,
        )

    def checkpoint_payload(self):
        return {"context": self.context, "calls": self.calls}

    def restore_payload(self, data):
        assert data["context"] == self.context
        self.calls = data["calls"]


def spec():
    return RunSpec(
        operation="external", schema_version=2, seed=4, recording=RecordingSpec(mode="full")
    )


def make(adapter=None, evaluator=None, checkpoint=None, limits=None):
    return ExternalSession(
        object(),
        spec(),
        adapter or Optimizer(),
        evaluator=evaluator or Scores(),
        checkpoint=checkpoint,
        limits=limits,
    )


def test_pre_and_post_ask_durable_boundaries_and_exact_candidate_order():
    session = make()
    phases = []

    def commit(cp, rows):
        payload = cp.payload["payload"]
        phases.append((payload["phase"], payload["optimizer"]["asks"], rows))
        return True

    session._record_callback = commit
    report = session.step()
    assert report.transition["advanced"]
    assert [(phase, asks) for phase, asks, _ in phases] == [
        ("preask", 0),
        ("postask", 1),
        ("evaluating", 1),
        ("idle", 1),
    ]
    assert session.adapter.tells == 1
    assert session.update == 1 and session.pending is None
    assert session.counters["total"]["evolution_updates"] == 1
    with pytest.raises(ValueError):
        phases[1][2][0]["request"]["genotypes"][0, 0] = 8


def test_incomplete_has_one_pending_request_and_resume_does_not_ask_again():
    session = make(evaluator=Scores(complete=False))
    assert session.step().status == "partial"
    assert session.adapter.asks == 1 and session.adapter.tells == 0
    assert session.update == 0 and session.pending.request_id == "request-0"
    cp = session.checkpoint()
    resumed = make(checkpoint=cp)
    assert resumed.step().transition["advanced"]
    assert resumed.adapter.asks == 1 and resumed.adapter.tells == 1
    assert resumed.evaluator.calls == 2


def test_ask_exception_rolls_back_to_pre_ask_numerical_state():
    adapter = Optimizer()
    adapter.fail_ask = True
    session = make(adapter)
    with pytest.raises(RuntimeError, match="ask failed"):
        session.step()
    assert adapter.asks == 0
    assert session.phase == "preask" and session.pending is None
    resumed = make(checkpoint=session.checkpoint())
    resumed.step()
    assert resumed.adapter.asks == 1 and resumed.adapter.tells == 1


def test_tell_exception_rolls_back_post_ask_pending_complete_result():
    adapter = Optimizer()
    adapter.fail_tell = True
    session = make(adapter)
    with pytest.raises(RuntimeError, match="tell failed"):
        session.step()
    assert adapter.asks == 1 and adapter.tells == 0
    np.testing.assert_array_equal(adapter.best, [0, 0])
    assert session.update == 0 and session.result.complete
    resumed = make(checkpoint=session.checkpoint())
    resumed.step()
    assert resumed.adapter.asks == 1 and resumed.adapter.tells == 1
    assert resumed.evaluator.calls == 1


@pytest.mark.parametrize("raises", [False, True])
def test_tell_commit_failure_rolls_back_optimizer_request_and_cursors(raises):
    session = make()

    def commit(cp, rows):
        if cp.payload["payload"]["phase"] == "idle":
            if raises:
                raise OSError("durability failed")
            return False
        return True

    session._record_callback = commit
    if raises:
        with pytest.raises(OSError, match="durability failed"):
            session.step()
    else:
        assert session.step().reason == "adapter_commit"
    assert session.adapter.asks == 1 and session.adapter.tells == 0
    assert session.update == 0 and session.phase == "tell"
    assert session.counters["total"]["evolution_updates"] == 0
    assert session.pending is not None and session.result.complete
    resumed = make(checkpoint=session.checkpoint())
    assert resumed.step().transition["advanced"]
    assert resumed.adapter.asks == 1 and resumed.adapter.tells == 1
    assert resumed.update == 1


def test_preask_commit_rejection_never_calls_ask():
    session = make()
    session._record_callback = lambda cp, rows: False
    assert session.step().status == "partial"
    assert session.adapter.asks == 0 and session.pending is None


def test_postask_commit_rejection_never_evaluates_or_tells():
    session = make()
    session._record_callback = lambda cp, rows: cp.payload["payload"]["phase"] == "preask"
    assert session.step().status == "partial"
    assert session.adapter.asks == 1 and session.adapter.tells == 0
    assert session.evaluator.calls == 0 and session.pending is not None
    resumed = make(checkpoint=session.checkpoint())
    resumed._record_callback = lambda cp, rows: True
    assert resumed.step().transition["advanced"]
    assert resumed.adapter.asks == 1 and resumed.adapter.tells == 1


def test_evolution_quota_zero_steps_and_owned_snapshot():
    session = make(limits=WorkLimits(max_evolution_updates=1))
    assert session.run(0).cursors["external_update"] == 0
    assert session.adapter.asks == 0
    session.run(4)
    assert session.update == 1 and session.adapter.asks == 1
    assert session.state["status"] == "partial"
    snapshot = session.snapshot()
    with pytest.raises(ValueError):
        snapshot["optimizer"]["best"][0] = 9
    with pytest.raises(ValueError):
        CandidateBatch("request", np.zeros((2, 2), np.uint32), [2, 2])
    with pytest.raises(ValueError):
        session.run(-1)


def test_numerical_adapter_contract_excludes_opaque_state():
    class Opaque(Optimizer):
        def state_dict(self):
            return {"opaque": object()}

    with pytest.raises(ValueError, match="unsupported checkpoint"):
        make(Opaque())
