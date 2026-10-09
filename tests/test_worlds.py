from __future__ import annotations

import numpy as np
import pytest

from toylife2.protocols import TaskSpec
from toylife2.worlds import RunTumble, ScenarioRandom, SignallingGame, make_world, numerical


def random(shape=(4,), seed=123, panel=0):
    return ScenarioRandom(seed, panel, np.arange(np.prod(shape)).reshape(shape))


def test_signalling_private_mask_delayed_messages_and_saved_correctness():
    world = SignallingGame({}, (4,), 2, random(), 0.05, False)
    initial = world.reset()
    observation = world.observe(initial)
    np.testing.assert_array_equal(observation["mask"], [[1, 0]] * 4)
    np.testing.assert_array_equal(observation["dose"][:, 0], initial["private"])
    actions = np.stack((initial["private"], 1 - initial["private"]), axis=-1)
    sent, reward, channels = world.advance(initial, actions)
    assert not np.any(reward) and not np.any(channels["judged"])
    np.testing.assert_array_equal(initial["delivered"], 0)
    received = world.observe(sent)
    np.testing.assert_array_equal(received["mask"], [[0, 1]] * 4)
    np.testing.assert_array_equal(received["dose"][:, 1], initial["private"])
    final, reward, channels = world.advance(
        sent, np.stack((1 - initial["private"], initial["private"]), axis=-1)
    )
    np.testing.assert_array_equal(reward, 1)
    assert np.all(channels["judged"])
    np.testing.assert_array_equal(final["private"], 1 - initial["private"])


def test_signalling_balanced_rounds_and_common_candidate_randomness():
    scenarios = np.broadcast_to(np.arange(3, dtype=np.uint64), (5, 3))
    world = SignallingGame({}, (5, 3), 2, ScenarioRandom(3, 4, scenarios), 0.05, False)
    bits = np.stack([world._private(i) for i in range(256)])
    np.testing.assert_array_equal(bits.sum(axis=0), 128)
    for candidate in range(1, 5):
        np.testing.assert_array_equal(bits[:, candidate], bits[:, 0])
    different = SignallingGame({}, (5, 3), 2, ScenarioRandom(3, 5, scenarios), 0.05, False)
    assert np.any(bits != np.stack([different._private(i) for i in range(256)]))


def test_habitat_two_tick_tumble_and_reward_at_new_field_position():
    world = RunTumble({"low_motor_hazard": 1e6}, (4,), 1, random(), 0.05, False)
    initial = world.reset()
    first, reward, channels = world.advance(initial, np.zeros(4, np.int32))
    np.testing.assert_array_equal(first["position"], initial["position"])
    np.testing.assert_array_equal(first["tumble"], 1)
    np.testing.assert_array_equal(channels["tumbling"], 1)
    second, _, channels = world.advance(first, np.zeros(4, np.int32))
    np.testing.assert_array_equal(second["position"], initial["position"])
    np.testing.assert_array_equal(second["tumble"], 0)
    np.testing.assert_array_equal(channels["tumbling"], 1)
    assert np.any(second["theta"] != initial["theta"])
    np.testing.assert_allclose(reward, world.field(first))
    assert np.all((reward >= 0) & (reward <= 1))
    assert np.any(first["latent"] != initial["latent"])


def test_habitat_periodic_minimum_image_and_absolute_sensor():
    world = RunTumble({}, (1,), 1, random((1,)), 0.05, False)
    state = world.reset()
    state["position"][:] = [4, 4]
    state["latent"][:] = 0
    reference = world.field(state)
    state["position"] += 16
    np.testing.assert_allclose(world.field(state), reference)
    observation = world.observe(state)
    np.testing.assert_array_equal(observation["dose"], np.floor(8 * reference))


def test_numerical_callbacks_reject_opaque_nonfinite_state():
    with pytest.raises(ValueError, match="serialized numerical"):
        numerical({"opaque": object()})
    with pytest.raises(ValueError, match="finite"):
        numerical(np.array([np.inf]))
    with pytest.raises(ValueError, match="not a closed-loop"):
        make_world(TaskSpec(), (1,), 1, random((1,)), 0.05)


def test_quadrants_builtin_worlds_match_host_reference(tmp_path):
    from toylife2 import Model
    from toylife2.specs import ExecutionSpec

    Model(execution=ExecutionSpec(cache_dir=str(tmp_path / "qd")))
    for cls, agents in ((SignallingGame, 2), (RunTumble, 1)):
        host = cls({}, (8,), agents, random((8,)), 0.05, False)
        device = cls({}, (8,), agents, random((8,)), 0.05, True)
        a, b = host.reset(), device.reset()
        for tick in range(10):
            shape = (8, 2) if agents == 2 else (8,)
            actions = np.full(shape, tick % 2, np.int32)
            a, ra, ca = host.advance(a, actions)
            b, rb, cb = device.advance(b, actions)
            np.testing.assert_allclose(ra, rb, atol=2e-6, rtol=2e-5)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=2e-5)
            for key in ca:
                np.testing.assert_allclose(ca[key], cb[key], atol=2e-6, rtol=2e-5)


def test_importable_python_world_uses_serialized_state_and_host_callbacks(monkeypatch):
    import sys
    import types

    module = types.ModuleType("worlds_test_callbacks")

    class Callbacks:
        def __init__(self, **kwargs):
            self.shape = kwargs["shape"]

        def reset(self):
            return {"tick": 0, "sensor": np.zeros(self.shape, np.int32)}

        def observe(self, state):
            state["sensor"][:] = 1
            return {"dose": state["sensor"]}

        def advance(self, state, actions):
            state["tick"] += 1
            return state, np.ones(self.shape), {"message": actions}

    module.factory = Callbacks
    monkeypatch.setitem(sys.modules, module.__name__, module)
    world = make_world(
        TaskSpec(kind="python", entrypoint="worlds_test_callbacks:factory"),
        (2,),
        1,
        random((2,)),
        0.05,
    )
    initial = world.reset()
    observed = world.observe(initial)
    np.testing.assert_array_equal(observed["dose"], 1)
    np.testing.assert_array_equal(initial["sensor"], 0)
    result, reward, channels = world.advance(initial, np.array([0, 1], np.int32))
    assert result["tick"] == 1 and initial["tick"] == 0
    np.testing.assert_array_equal(reward, 1)
    np.testing.assert_array_equal(channels["message"], [0, 1])
