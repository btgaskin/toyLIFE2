from __future__ import annotations

import numpy as np
import pytest

from toylife2 import Model
from toylife2.protocols import ControllerSpec, EpisodeSpec, TaskSpec, WorkLimits
from toylife2.rollout import RolloutSession
from toylife2.sessions import Checkpoint, pack
from toylife2.specs import ExecutionSpec, PortSpec, RecordingSpec


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return Model(execution=ExecutionSpec(cache_dir=str(tmp_path_factory.mktemp("qd-rollout"))))


def test_empty_episode_differs_from_completed_and_computational_partial(model):
    empty = RolloutSession(model, [0, 0], protocol=EpisodeSpec(horizon=0))
    empty.run()
    assert empty.tick == 0 and empty.status[0] == "complete"
    assert empty.scores[0] == 0 and not empty.success[0]
    complete = RolloutSession(model, [0, 0], protocol=EpisodeSpec(horizon=1))
    complete.run()
    assert complete.tick == 1 and complete.status[0] == "complete"
    partial = RolloutSession(
        model, [0, 0], protocol=EpisodeSpec(horizon=1), limits=WorkLimits(max_world_ticks=0)
    )
    partial.run()
    assert partial.tick == 0 and partial.status[0] == "partial"


def test_pending_chemical_tick_resumes_exactly_and_snapshot_is_committed(model, tmp_path):
    task = TaskSpec(kind="run_tumble", config={"record_descriptor": True})
    episode = EpisodeSpec(horizon=3, chemical_updates=3, episodes=2)
    genes = np.array([[0, 0], [1048465, 907050]], np.int32)
    full = RolloutSession(model, genes, task, episode, seed=83)
    full.run()
    capped = RolloutSession(
        model, genes, task, episode, seed=83, limits=WorkLimits(max_chemical_updates=8)
    )
    before = capped.snapshot()
    capped.run()
    assert capped.tick == 0
    np.testing.assert_array_equal(
        capped.snapshot()["world"]["position"], before["world"]["position"]
    )
    assert capped.snapshot()["chemistry"]["update"] == 0
    staged = capped.snapshot(include_pending=True)["pending"]
    assert staged["chemical_cursor"] == 2 and staged["chemistry"]["update"] == 2
    with pytest.raises(ValueError):
        staged["chemistry"]["counts"].flat[0] = 999
    checkpoint = capped.checkpoint()
    path = checkpoint.write(tmp_path / "checkpoint")
    resumed = RolloutSession(model, genes, task, episode, seed=83, checkpoint=Checkpoint.read(path))
    resumed.run()
    assert resumed.tick == 3
    np.testing.assert_array_equal(resumed.scores, full.scores)
    np.testing.assert_array_equal(
        resumed.snapshot()["world"]["position"], full.snapshot()["world"]["position"]
    )
    np.testing.assert_array_equal(
        resumed.snapshot()["chemistry"]["counts"], full.snapshot()["chemistry"]["counts"]
    )
    assert resumed.counters["total"] == full.counters["total"]
    for a, b in zip(resumed.descriptor, full.descriptor, strict=True):
        np.testing.assert_array_equal(a["actions"], b["actions"])


def test_genome_episode_and_explicit_agent_shapes(model):
    independent = RolloutSession(
        model, np.zeros((2, 3, 2), np.int32), protocol=EpisodeSpec(horizon=1, episodes=4)
    )
    independent.run()
    assert independent.scores.shape == (2, 3, 4)
    assert independent.aggregate_scores().shape == (2, 3)
    assert independent.chemistry.genotypes.shape == (2, 3, 4, 2)
    with pytest.raises(ValueError, match="two-agent"):
        RolloutSession(model, np.zeros((2, 2), np.int32), TaskSpec(kind="signalling_game"))
    coupled = RolloutSession(
        model,
        np.zeros((2, 3, 2), np.int32),
        TaskSpec(kind="signalling_game", config={"rounds": 2}),
        EpisodeSpec(horizon=4, episodes=4),
        agent_axis=0,
    )
    coupled.run()
    assert coupled.scores.shape == (3, 4)
    assert coupled.chemistry.genotypes.shape == (3, 4, 2, 2)
    assert np.all(coupled.reward_count == 2)
    restored = RolloutSession(
        model,
        coupled.checkpoint().payload["payload"]["genotypes"],
        coupled.task,
        coupled.protocol,
        agent_axis=0,
        checkpoint=coupled.checkpoint(),
    )
    np.testing.assert_array_equal(restored.scores, coupled.scores)


def test_pulse_and_replenish_are_explicit_chemical_ports(model):
    for mode in ("pulse", "replenish"):
        controller = ControllerSpec(ports=PortSpec(metabolites=("1",), input_mode=mode))
        session = RolloutSession(
            model,
            [0, 0],
            TaskSpec(kind="temporal_probe", config={"sequence": [1]}),
            EpisodeSpec(horizon=2, chemical_updates=2),
            controller,
        )
        updates = []
        original = session.chemistry.step

        def capture(inputs, updates=updates, original=original):
            updates.append(dict(inputs))
            return original(inputs)

        session.chemistry.step = capture
        session.run()
        assert [bool(x) for x in updates] == (
            [True, False, True, False] if mode == "pulse" else [True, True, True, True]
        )
        assert session.chemistry.update == 4


def test_reset_per_decision_ablation_and_discrete_descriptor(model):
    controller = ControllerSpec(ports=PortSpec(metabolites=("1",)))
    task = TaskSpec(kind="temporal_probe", config={"sequence": [1], "record_descriptor": True})
    session = RolloutSession(
        model, [0, 0], task, EpisodeSpec(horizon=3, reset_per_decision=True), controller
    )
    session.run()
    assert session.chemistry.update == 1
    assert session.counters["total"]["chemical_updates"] == 3
    assert len(session.descriptor) == 3
    assert all(row["actions"].dtype.kind in "iu" for row in session.descriptor)
    assert "reward" not in session.descriptor[0]


def test_record_rejection_keeps_publication_old_and_retries_same_prepared_tick(model):
    session = RolloutSession(
        model,
        [0, 0],
        TaskSpec(kind="run_tumble"),
        EpisodeSpec(horizon=1),
        recording=RecordingSpec(mode="full"),
    )
    before = session.snapshot()
    advances = []
    original = session.world.advance

    def advance(state, actions):
        advances.append(1)
        return original(state, actions)

    session.world.advance = advance
    checkpoints = []

    def record(checkpoint, events):
        checkpoints.append(checkpoint)
        assert checkpoint.payload["payload"]["tick"] == 1
        assert checkpoint.payload["base"]["total"]["world_ticks"] == 1
        assert checkpoint.payload["base"]["status"] == "complete"
        return len(checkpoints) > 1

    session._record_callback = record
    report = session.step()
    assert report.status == "partial" and session.tick == 0
    np.testing.assert_array_equal(
        session.snapshot()["world"]["position"], before["world"]["position"]
    )
    assert session.snapshot()["chemistry"]["update"] == 0
    assert "prepared" in session.snapshot(include_pending=True)["pending"]
    session.run()
    assert session.tick == 1 and session.status[0] == "complete"
    assert len(advances) == 1 and len(checkpoints) == 2
    # The record callback removes pending rows after accepting their trace chunk.
    payload = checkpoints[-1].payload
    payload["base"]["pending_trace"] = []
    arrays = {}
    durable = Checkpoint("rollout", session.identity, pack(payload, arrays), arrays)
    restored = RolloutSession(
        model,
        [0, 0],
        session.task,
        session.protocol,
        recording=session.recording,
        checkpoint=durable,
    )
    restored.run()
    assert restored.tick == 1
    np.testing.assert_array_equal(restored.scores, session.scores)
    assert restored.counters["total"] == session.counters["total"]


def test_callback_exception_keeps_staged_tick_for_retry(model):
    session = RolloutSession(
        model, [0, 0], protocol=EpisodeSpec(horizon=1), recording=RecordingSpec(mode="full")
    )
    calls = []

    def record(checkpoint, events):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("synthetic write failure")
        return True

    session._record_callback = record
    with pytest.raises(OSError, match="synthetic"):
        session.step()
    assert session.tick == 0 and session.chemistry.update == 1
    session.run()
    assert session.tick == 1 and session.chemistry.update == 1


def test_mutable_python_codec_state_is_staged_once_on_quota_resume(model, monkeypatch):
    import sys
    import types

    module = types.ModuleType("rollout_test_codec")
    module.__file__ = __file__

    class Codec:
        def __init__(self, config, ports):
            pass

        def encode(self, observation, state):
            return {"1": np.ones_like(observation["dose"])}, {"calls": state["calls"] + 1}

        def decode(self, readouts, state):
            return np.zeros_like(readouts["motor"], dtype=np.int32), {"calls": state["calls"] + 1}

    module.Codec = Codec
    monkeypatch.setitem(sys.modules, module.__name__, module)
    task = TaskSpec(
        kind="temporal_probe",
        config={"codec_entrypoint": "rollout_test_codec:Codec", "codec_state": {"calls": 0}},
    )
    controller = ControllerSpec(ports=PortSpec(metabolites=("1",)))
    episode = EpisodeSpec(horizon=1, chemical_updates=3)
    session = RolloutSession(
        model, [0, 0], task, episode, controller, limits=WorkLimits(max_chemical_updates=1)
    )
    session.run()
    assert session.snapshot()["codec"]["calls"] == 0
    assert session.snapshot(include_pending=True)["pending"]["codec"]["calls"] == 1
    resumed = RolloutSession(
        model, [0, 0], task, episode, controller, checkpoint=session.checkpoint()
    )
    resumed.run()
    assert resumed.snapshot()["codec"]["calls"] == 2
    assert resumed.chemistry.update == 3


def test_logical_world_and_agent_work_is_charged_per_lane(model):
    genes = np.zeros((3, 2, 2), np.int32)
    session = RolloutSession(
        model,
        genes,
        TaskSpec(kind="signalling_game"),
        EpisodeSpec(horizon=2, episodes=4, chemical_updates=2),
        agent_axis=-2,
    )
    session.run()
    assert session.counters["total"]["world_ticks"] == 3 * 4 * 2
    assert session.counters["total"]["chemical_updates"] == 3 * 4 * 2 * 2 * 2
    cap = RolloutSession(
        model,
        genes,
        session.task,
        session.protocol,
        agent_axis=-2,
        limits=WorkLimits(max_chemical_updates=23),
    )
    cap.run()
    assert cap.tick == 0 and cap.chemistry.update == 0
    assert cap.counters["total"]["chemical_updates"] == 0


def test_warmup_checkpoint_and_reset_ablation_reuse_warmed_initial_state(model):
    controller = ControllerSpec(warmup=2)
    protocol = EpisodeSpec(horizon=2, chemical_updates=1, reset_per_decision=True)
    session = RolloutSession(
        model,
        [0, 0],
        protocol=protocol,
        controller=controller,
        limits=WorkLimits(max_chemical_updates=1),
    )
    session.run()
    assert session.tick == 0
    assert session.snapshot(include_pending=True)["pending"]["warmup_done"] == 1
    resumed = RolloutSession(
        model, [0, 0], protocol=protocol, controller=controller, checkpoint=session.checkpoint()
    )
    resumed.run()
    assert resumed.chemistry.update == 3
    assert resumed.counters["total"]["chemical_updates"] == 4


def test_observation_doses_outside_ports_are_rejected(model):
    session = RolloutSession(
        model,
        [0, 0],
        TaskSpec(kind="temporal_probe", config={"sequence": [9]}),
        EpisodeSpec(horizon=1),
    )
    with pytest.raises(ValueError, match="outside configured"):
        session.step()
    assert session.tick == 0 and session.chemistry.update == 0


def test_full_unrecorded_trace_contains_chemical_and_world_boundaries(model):
    session = RolloutSession(
        model,
        [0, 0],
        protocol=EpisodeSpec(horizon=2, chemical_updates=2),
        recording=RecordingSpec(mode="full"),
    )
    session.run()
    rows = session.drain_trace()
    assert [row["type"] for row in rows] == ["chemical_update", "chemical_update", "world_tick"] * 2
    assert session.drain_trace() == []
    chemical = rows[0]
    assert all(
        name in chemical for name in ("inputs", "before", "after_input", "after_output", "boundary")
    )


def test_full_trace_keeps_partial_chemical_rows_pending_until_world_commit(model):
    protocol = EpisodeSpec(horizon=1, chemical_updates=2)
    session = RolloutSession(
        model,
        [0, 0],
        protocol=protocol,
        recording=RecordingSpec(mode="full"),
        limits=WorkLimits(max_chemical_updates=1),
    )
    session.run()
    assert session.drain_trace() == []
    pending = session.checkpoint().payload["payload"]["pending"]
    assert len(pending["chemical_events"]) == 1
    resumed = RolloutSession(
        model,
        [0, 0],
        protocol=protocol,
        recording=session.recording,
        checkpoint=session.checkpoint(),
    )
    resumed.run()
    assert [row["type"] for row in resumed.drain_trace()] == [
        "chemical_update",
        "chemical_update",
        "world_tick",
    ]
