"""Addressing, dynamics and execution contracts for batched chemotaxis randomness."""

from dataclasses import replace

import numpy as np
import pytest

from toylife2 import Model
from toylife2._world_random import FLOAT32, DrawLayout, uniform32
from toylife2.molecular_worlds import Chemotaxis
from toylife2.protocols import BinaryOutputSpec, EpisodeSpec, StructuralInterface, TaskSpec
from toylife2.random import philox
from toylife2.resident_world import RandomBlockCache, WorldBuffers
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec, RecordingSpec
from toylife2.worlds import ScenarioRandom

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def model(canonical_tables):
    return Model(tables=canonical_tables, execution=ExecutionSpec())


def world(ids=None, *, math=FLOAT32, **config):
    ids = np.array([0, 9, (1 << 33) + 7, 9], np.uint64) if ids is None else ids
    task = TaskSpec(
        kind="chemotaxis",
        config=dict(
            world_engine="quadrants", world_precision="float32", world_random_math=math, **config
        ),
    )
    return Chemotaxis(task.config, ids.shape, 1, ScenarioRandom(123, 7, ids), 0.05, False)


def test_open_uniform_endpoints_and_component_overflow():
    words = np.array(
        [[0, 0], [0xFFFFFFFF, 0xFFFFFFFF], [0xFFFFF000, 1], [0xFFFFEFFF, 1]], np.uint32
    )
    values = uniform32(words)
    assert values.dtype == np.float32
    assert values[0] == np.float32(2**-53)
    assert values[1] == np.float32(1 - 2**-24)
    assert np.all((values > 0) & (values < 1))
    with pytest.raises(ValueError, match="component ID"):
        DrawLayout(world(np.array([2**64 - 1], np.uint64)))
    with pytest.raises(ValueError, match="world_precision"):
        TaskSpec(kind="chemotaxis", config={"world_random_math": FLOAT32})


@pytest.mark.parametrize("static", [True, False])
def test_layout_tiling_and_cache_preserve_addresses(static):
    w = world(static=static)
    plan = DrawLayout(w)
    full = plan.generate(w, 11, 8)
    tiled = plan.generate(w, 11, 8, scratch_bytes=512 * max(plan.requests))
    np.testing.assert_array_equal(full, tiled)
    assert np.all(np.isfinite(full))
    reversed_world = world(w.rng._unique[::-1], static=static)
    reverse = DrawLayout(reversed_world).generate(reversed_world, 11, 8)
    np.testing.assert_array_equal(reverse, full)  # unique scenarios are sorted canonically
    cache = RandomBlockCache(full.nbytes + 512)
    first = cache.block(w, 11, 8)
    assert not first.flags.writeable and cache.block(w, 11, 8) is first
    cache.block(w, 19, 8)
    assert cache.bytes <= cache.capacity and len(cache.entries) == 1
    assert not np.array_equal(first, cache.block(w, 19, 8))
    keys = cache.key_block(w, 19, 8)
    assert not keys.flags.writeable and cache.key_block(w, 19, 8) is keys
    cache.reclaim(0)
    assert cache.bytes == 0 and not cache.entries
    with pytest.raises(MemoryError, match="scratch"):
        plan.generate(w, 11, 8, scratch_bytes=1)


def test_device_philox_words_and_uniforms(model):
    import quadrants as qd

    from toylife2._output_kernels import philox_pair
    from toylife2._world_random_kernels import open_uniform

    @qd.kernel
    def evaluate(
        ids: qd.types.NDArray[qd.u32, 2],
        keys: qd.types.NDArray[qd.u32, 2],
        result: qd.types.NDArray[qd.u32, 2],
        uniforms: qd.types.NDArray[qd.f32, 2],
    ):
        for i in range(ids.shape[0]):
            a, b = philox_pair(ids[i, 0], ids[i, 1], keys[i, 0], keys[i, 1], qd.u32(0))
            result[i, 0], result[i, 1] = a, b
            uniforms[i, 0] = open_uniform(a, b)
            uniforms[i, 1] = open_uniform(ids[i, 0], ids[i, 1])

    ids = np.array([[0, 0], [0xFFFFFFFF, 0xFFFFFFFF], [123, 1], [2**31, 33]], np.uint32)
    keys = np.array([[0, 0], [0xFFFFFFFF, 0xFFFFFFFF], [93, 7], [2**31, 77]], np.uint32)
    counters = np.zeros((len(ids), 4), np.uint32)
    counters[:, :2] = ids
    expected = philox(counters, keys)[:, :2]
    a, b, output = (qd.ndarray(qd.u32, ids.shape) for _ in range(3))
    uniforms = qd.ndarray(qd.f32, (len(ids), 2))
    a.from_numpy(ids)
    b.from_numpy(keys)
    evaluate(a, b, output, uniforms)
    np.testing.assert_array_equal(output.to_numpy(), expected)
    np.testing.assert_array_equal(uniforms.to_numpy()[:, 0], uniform32(expected))
    np.testing.assert_array_equal(uniforms.to_numpy()[:, 1], uniform32(ids))


@pytest.mark.parametrize("static", [True, False])
def test_direct_draws_packing_and_short_blocks(model, monkeypatch, static):
    monkeypatch.setattr(model.engine, "_world_random_provider", "direct", raising=False)
    w = world(static=static)
    buffers = WorldBuffers(w, model.engine)
    buffers.prepare_draws(19, 8)
    expected = DrawLayout(w).generate(w, 19, 8)
    actual = buffers.draws.to_numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    for offset in range(8):
        buffers.prepare_draws(19 + offset, 1)
        one = buffers.draws.to_numpy()
        np.testing.assert_array_equal(one[0], actual[offset])
        assert np.all(one[1:] == 0)
    assert buffers.native and buffers.random_implementation == "quadrants_direct"


@pytest.mark.parametrize("provider", ["host", "direct"])
def test_new_math_rollout_blocks_reset_and_resume(model, monkeypatch, provider):
    monkeypatch.setattr(model.engine, "_world_random_provider", provider, raising=False)
    task = TaskSpec(
        kind="chemotaxis",
        config=dict(world_engine="quadrants", world_precision="float32", world_random_math=FLOAT32),
    )
    controller = StructuralInterface(inputs={"cue": "0001"}, targets={"motor": "1111"}).controller(
        output=BinaryOutputSpec(markers=("motor",))
    )
    genes = np.array([[431068, 265408], [1023394, 200150]], np.uint32)
    episode = EpisodeSpec(horizon=13, episodes=3)

    def session(**kwargs):
        return RolloutSession(model, genes, task, episode, controller, seed=123, **kwargs)

    scalar = session()
    while scalar.tick < 13:
        scalar.step()
    blocked = session()
    blocked.run()
    resumed = session()
    resumed.run(steps=5)
    resumed = session(checkpoint=resumed.checkpoint())
    resumed.run()
    for other in (blocked, resumed):
        np.testing.assert_array_equal(other.scores, scalar.scores)
        for name in ("orientation", "position", "tumble_remaining"):
            np.testing.assert_array_equal(
                other.snapshot()["world"][name], scalar.snapshot()["world"][name]
            )
    # The ordinary observation/motion route is exercised by per-decision resets.
    reset = RolloutSession(
        model,
        genes,
        task,
        replace(episode, reset_per_decision=True),
        controller,
        seed=123,
        recording=RecordingSpec(),
    )
    assert reset.run().status == "complete"
    detailed = session(recording=RecordingSpec(mode="full"))
    detailed._record_callback = lambda checkpoint, events: True
    assert detailed.run().status == "complete"
    np.testing.assert_array_equal(detailed.scores, scalar.scores)
    for name in ("orientation", "position", "tumble_remaining"):
        np.testing.assert_array_equal(
            detailed.snapshot()["world"][name], scalar.snapshot()["world"][name]
        )
    failing = session(recording=RecordingSpec(checkpoint_interval=8))

    def fail(checkpoint, events):
        raise OSError("injected publication failure")

    failing._record_callback = fail
    with pytest.raises(OSError, match="publication failure"):
        failing.run()
    assert failing.tick == 0
    failing._record_callback = lambda checkpoint, events: True
    assert failing.run().status == "complete"
    np.testing.assert_array_equal(failing.scores, scalar.scores)


@pytest.mark.integration
def test_recorded_draw_contract_and_execution(model, tmp_path):
    import json

    from toylife2 import BatchRunner
    from toylife2.genotypes import decode
    from toylife2.molecular_experiments import profile_spec
    from toylife2.records import verify_record

    genes = np.array([[431068, 265408]], np.uint32)
    spec = profile_spec("chemotaxis", backend=model.execution.backend)
    spec = replace(
        spec,
        operation="rollout",
        genotypes=tuple(decode(genes)),
        execution=model.execution,
        episode=replace(spec.episode, horizon=3),
    )
    root = tmp_path / "draw-record"
    with BatchRunner(model=model) as runner:
        session = runner.rollout(genes, spec=spec, destination=root)
        assert session.run().status == "complete"
        runner.finish(session)
    assert verify_record(root)["ok"]
    manifest = json.loads((root / "manifest.json").read_text())
    execution = manifest["execution"]["world"]
    assert execution["world_random_math"] == FLOAT32
    expected = "quadrants_direct" if model.execution.backend == "cuda" else "batched_numpy"
    assert execution["world_random_implementation"] == expected
    assert FLOAT32 in (root / "methods.md").read_text()


@pytest.mark.qualification
def test_directional_diffusion_and_zero_noise(model):
    n = 8192
    for substep in (0.02, 0.01, 0.005):
        w = world(
            np.arange(n, dtype=np.uint64),
            hazard_zero=0.0,
            hazard_one=0.0,
            motion_max_substep=substep,
            rotational_diffusion=1.0,
        )
        w._engine = model.engine
        state = w.reset()
        state["orientation"][:] = [1, 0, 0]
        for _ in range(4):
            state, _, _ = w.advance(state, np.zeros(n, np.int32))
        orientation = state["orientation"]
        np.testing.assert_allclose(np.linalg.norm(orientation, axis=-1), 1, atol=2e-6)
        # E[n(t).n(0)] = exp(-2Dt); allow sampling uncertainty and O(h) integration error.
        uncertainty = 6 * orientation[:, 0].std() / np.sqrt(n)
        assert abs(orientation[:, 0].mean() - np.exp(-0.4)) < uncertainty + 2 * 0.2 * substep
        assert abs(orientation[:, 1].mean()) < 6 / np.sqrt(n)
        assert abs(orientation[:, 2].mean()) < 6 / np.sqrt(n)
    w = world(
        np.arange(8, dtype=np.uint64), hazard_zero=0.0, hazard_one=0.0, rotational_diffusion=0.0
    )
    state = w.reset()
    start = state["position"].copy()
    state["orientation"][:] = [1, 0, 0]
    for _ in range(4):
        state, _, channels = w.advance(state, np.zeros(8, np.int32))
        assert not channels["tumbling"].any()
    expected = start.copy()
    expected[:, 0] = (expected[:, 0] + 0.2) % 16
    # Twenty float32 position additions plus rounding of the analytic endpoint.
    rounding_bound = 21 * np.spacing(np.float32(16)) / 2
    np.testing.assert_allclose(state["position"], expected, rtol=0, atol=rounding_bound)


@pytest.mark.qualification
def test_noise_moments_tails_and_independence(model, monkeypatch):
    monkeypatch.setattr(model.engine, "_world_random_provider", "direct", raising=False)
    n = 65536
    for panel in range(4):
        w = world(np.arange(n, dtype=np.uint64), motion_event_capacity=1, motion_max_substep=0.05)
        w.rng = ScenarioRandom(123, panel, w.rng.scenarios)
        buffers = WorldBuffers(w, model.engine)
        buffers.prepare_draws(0, 1)
        draws = buffers.draws.to_numpy()[0]
        u, v = draws[:, 8], draws[:, 9]
        z = draws[:, 10:13].astype(np.float64)
        assert np.all((u > 0) & (u < 1)) and np.isfinite(draws).all()
        assert abs(u.mean() - 0.5) < 6 / np.sqrt(12 * n)
        assert abs(np.var(u) - 1 / 12) < 6 / np.sqrt(180 * n)
        assert np.all(np.abs(z.mean(axis=0)) < 6 / np.sqrt(n))
        assert np.all(np.abs(z.var(axis=0) - 1) < 6 * np.sqrt(2 / n))
        assert np.all(np.abs((np.abs(z) > 3).mean(axis=0) - 0.002699796) < 6 * np.sqrt(0.0027 / n))
        samples = np.column_stack((u, v, z))
        correlations = np.corrcoef(samples, rowvar=False) - np.eye(5)
        assert np.max(np.abs(correlations)) < 6 / np.sqrt(n)
        waits = -np.log(u.astype(np.float64))
        assert abs(waits.mean() - 1) < 6 / np.sqrt(n)
        assert abs(waits.var() - 1) < 6 * np.sqrt(8 / n)
