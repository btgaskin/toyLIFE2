"""Scientific and publication contracts of resident chemotaxis decisions."""

from dataclasses import fields

import numpy as np
import pytest

from toylife2 import Model
from toylife2.molecular_worlds import Chemotaxis
from toylife2.protocols import EpisodeSpec, StructuralInterface, TaskSpec, WorkLimits
from toylife2.resident_world import RandomBlockCache
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec, RecordingSpec
from toylife2.worlds import ScenarioRandom

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def model(canonical_tables):
    return Model(tables=canonical_tables, execution=ExecutionSpec())


def task(field="fluctuating", static=True, precision="float32", engine="quadrants"):
    return TaskSpec(
        kind="chemotaxis",
        config=dict(field=field, static=static, world_engine=engine, world_precision=precision),
    )


def controller(**kwargs):
    return StructuralInterface(inputs={"cue": "0001"}, targets={"motor": "1111"}).controller(
        **kwargs
    )


def rollout(model, *, config=None, updates=1, horizon=13, genes=None, reset=False, **kwargs):
    if genes is None:
        genes = np.random.default_rng(323).integers(0, 1 << 20, (3, 4), dtype=np.int32)
    return RolloutSession(
        model,
        genes,
        config or task(),
        EpisodeSpec(
            horizon=horizon, episodes=3, chemical_updates=updates, reset_per_decision=reset
        ),
        kwargs.pop("controller", controller()),
        seed=483,
        **kwargs,
    )


def assert_equal(a, b):
    np.testing.assert_array_equal(a.scores, b.scores)
    aa, bb = a.snapshot(), b.snapshot()
    for name in ("position", "theta", "phases", "source"):
        np.testing.assert_array_equal(aa["world"][name], bb["world"][name])
    for name in ("tumble_remaining", "orientation"):
        np.testing.assert_array_equal(aa["world"][name], bb["world"][name])
    for name in ("counts", "bits", "events", "first_events", "after_input", "after_output"):
        np.testing.assert_array_equal(aa["chemistry"][name], bb["chemistry"][name])
    assert a.counters["total"] == b.counters["total"]


@pytest.mark.parametrize(
    "field,static", [("source", True), ("fluctuating", True), ("fluctuating", False)]
)
@pytest.mark.parametrize("updates", [1, 3])
def test_blocks_steps_and_cohort_width_agree(model, field, static, updates):
    config = task(field, static)
    scalar = rollout(model, config=config, updates=updates)
    while scalar.tick < 13:
        report = scalar.step()
        assert report.transition["tick"] == scalar.tick
        assert report.transition["observation"]["inputs"]["cue"].shape == scalar.world_shape
    for size in (1, 2, 8):
        model.engine._world_block_size = size
        batched = rollout(model, config=config, updates=updates)
        assert batched.run().status == "complete"
        assert_equal(scalar, batched)
    model.engine._world_block_size = 8
    for i, row in enumerate(scalar.genotypes):
        single = rollout(model, config=config, updates=updates, genes=row[None, :])
        single.run()
        np.testing.assert_array_equal(single.scores[0], scalar.scores[i])
        np.testing.assert_array_equal(
            single.snapshot()["world"]["position"][0], scalar.snapshot()["world"]["position"][i]
        )


@pytest.mark.parametrize(
    "field,static", [("source", True), ("fluctuating", True), ("fluctuating", False)]
)
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_fixed_actions_match_reference(model, field, static, precision):
    if precision == "float64" and model.execution.backend != "cpu":
        pytest.skip("Quadrants float64 worlds require CPU")
    config = task(field, static, precision)
    shape = (8,)
    host_config = dict(config.config, world_engine="numpy")
    host = Chemotaxis(host_config, shape, 1, ScenarioRandom(3, 4, np.arange(8)), 0.05, False)
    device = Chemotaxis(config.config, shape, 1, ScenarioRandom(3, 4, np.arange(8)), 0.05, True)
    a, b = host.reset(), device.reset()
    atol, rtol = (2e-6, 2e-5) if precision == "float32" else (1e-12, 1e-10)
    for tick in range(8):
        actions = np.full(shape, tick % 2, np.int32)
        a, ra, ca = host.advance(a, actions)
        b, rb, cb = device.advance(b, actions)
        np.testing.assert_allclose(ra, rb, atol=atol, rtol=rtol)
        for name in ("position", "theta", "phases", "source"):
            np.testing.assert_allclose(a[name], b[name], atol=atol, rtol=rtol)
        np.testing.assert_array_equal(ca["tumbling"], cb["tumbling"])
        for name in ("tumble_remaining", "orientation"):
            np.testing.assert_allclose(a[name], b[name], atol=atol, rtol=rtol)


def test_random_blocks_preserve_addresses_and_are_bounded(model):
    shape = (2, 3)
    scenarios = np.asarray([[9, 2, 9], [2, 9, 2]], np.uint64)
    world = Chemotaxis(
        task(static=False, precision="float64", engine="numpy").config,
        shape,
        1,
        ScenarioRandom(15, 7, scenarios),
        0.05,
        False,
    )
    cache = RandomBlockCache(32768)
    block = cache.block(world, 11, 2)
    for i in range(2):
        expanded = block[i][world.rng._inverse]
        np.testing.assert_array_equal(expanded[..., 0], world.rng.uniform(11 + i, "chem-sensor"))
        np.testing.assert_array_equal(
            expanded[..., 4:8], world.rng.normal(11 + i, "chem-phase-update", 4)
        )
        for event in range(world.event_capacity):
            start = 8 + event * world.motion_stride
            for step in range(world.motion_steps):
                np.testing.assert_array_equal(
                    expanded[..., start + 2 + 3 * step : start + 5 + 3 * step],
                    world.rng.normal(11 + i, f"chem-motion-rotation:{event}:{step}", 3),
                )
    assert not block.flags.writeable
    assert cache.block(world, 11, 2) is block and cache.hits == 1
    for tick in range(12, 24):
        cache.block(world, tick, 2)
        assert cache.bytes <= cache.capacity
    cache.reclaim(0)
    assert not cache.entries and cache.bytes == 0


@pytest.mark.parametrize("engine", ["numpy", "quadrants"])
def test_finite_tumble_pauses_and_resumes_at_continuous_endpoint(model, engine):
    config = dict(
        task(engine=engine).config,
        hazard_zero=0.0,
        hazard_one=0.0,
        rotational_diffusion=0.0,
        tumble_rotational_diffusion=0.0,
    )
    world = Chemotaxis(config, (1,), 1, ScenarioRandom(3, 4, np.arange(1)), 0.05, True)
    state = world.reset()
    state["position"][:] = 0
    state["orientation"][:] = [1, 0, 0]
    state["theta"][:] = 0
    state["tumble_remaining"][:] = 0.12
    for tick in range(3):
        state, _, channels = world.advance(state, np.zeros(1, np.int32))
        assert channels["tumbling"][0] == 1
        np.testing.assert_allclose(state["position"], [[0 if tick < 2 else 0.03, 0]], atol=1e-7)
    assert state["tumble_remaining"][0] == 0


@pytest.mark.parametrize("engine", ["numpy", "quadrants"])
def test_projected_swimming_preserves_full_orientation(model, engine):
    config = dict(
        task(engine=engine).config, hazard_zero=0.0, hazard_one=0.0, rotational_diffusion=0.0
    )
    world = Chemotaxis(config, (1,), 1, ScenarioRandom(3, 4, np.arange(1)), 0.05, True)
    state = world.reset()
    state["position"][:] = 0
    state["orientation"][:] = [0.6, 0, 0.8]
    state, _, _ = world.advance(state, np.zeros(1, np.int32))
    np.testing.assert_allclose(state["position"], [[0.03, 0]], atol=1e-7)
    np.testing.assert_allclose(state["orientation"], [[0.6, 0, 0.8]], atol=1e-7)


def test_spherical_tumble_diffusion_retains_directional_persistence():
    count = 4096
    config = dict(task(engine="numpy", precision="float64").config)
    world = Chemotaxis(config, (count,), 1, ScenarioRandom(31, 4, np.arange(count)), 0.01, False)
    state = world.reset()
    state["orientation"][:] = [1, 0, 0]
    state["tumble_remaining"][:] = 1
    state, _, _ = world.advance(state, np.zeros(count, np.int32))
    np.testing.assert_allclose(np.linalg.norm(state["orientation"], axis=-1), 1, atol=1e-12)
    expected = np.exp(-2 * config["tumble_rotational_diffusion"] * world.dt)
    assert abs(state["orientation"][:, 0].mean() - expected) < 0.007


@pytest.mark.parametrize("engine", ["numpy", "quadrants"])
def test_event_capacity_exhaustion_rejects_motion(model, engine):
    config = dict(task(engine=engine).config, motion_event_capacity=1)
    world = Chemotaxis(config, (1,), 1, ScenarioRandom(3, 4, np.arange(1)), 0.05, True)
    state = world.reset()
    state["tumble_remaining"][:] = 0.001
    before = state["position"].copy()
    with pytest.raises(RuntimeError, match="event capacity exhausted"):
        world.advance(state, np.zeros(1, np.int32))
    np.testing.assert_array_equal(state["position"], before)


@pytest.mark.parametrize("cap", [0, 9, 18, 27, 36, 81])
def test_arbitrary_chemical_quota_and_resume(model, cap):
    config = task()
    direct = rollout(model, updates=3, config=config)
    direct.run()
    partial = rollout(model, updates=3, config=config, limits=WorkLimits(max_chemical_updates=cap))
    partial.run()
    resumed = rollout(model, updates=3, config=config, checkpoint=partial.checkpoint())
    resumed.run()
    assert_equal(direct, resumed)


def test_block_checkpoint_failure_retains_prepared_result(model):
    s = rollout(model, horizon=8, recording=RecordingSpec(checkpoint_interval=8))
    before = s.snapshot()
    calls = []

    def fail(checkpoint, events):
        calls.append(checkpoint)
        raise OSError("injected block checkpoint failure")

    s._record_callback = fail
    with pytest.raises(OSError):
        s.run()
    assert s.tick == 0 and s._pending["prepared"]["advanced"] == 8
    np.testing.assert_array_equal(s.snapshot()["world"]["position"], before["world"]["position"])
    assert calls[0].payload["payload"]["tick"] == 8
    chemical = s.chemistry.update
    s._record_callback = lambda checkpoint, events: True
    assert s.run().status == "complete"
    assert s.chemistry.update == chemical
    direct = rollout(model, horizon=8)
    direct.run()
    assert_equal(s, direct)
    assert s._boundaries == 8


def test_numerical_declarations_legacy_and_retirement(model):
    from toylife2.molecular_experiments import profile_spec

    legacy = TaskSpec(kind="chemotaxis")
    assert "world_engine" not in legacy.config and "world_precision" not in legacy.config
    s = rollout(model, config=legacy)
    assert s._resident is None and s.world.precision == "float64"
    assert profile_spec("chemotaxis").task.config["world_precision"] == "float32"
    with pytest.raises(ValueError, match="earlier release"):
        TaskSpec(kind="run_tumble")
    with pytest.raises(ValueError, match="conflicts"):
        TaskSpec(kind="chemotaxis", config=dict(world_engine="quadrants", reference_world=True))
    if model.execution.backend != "cpu":
        with pytest.raises(ValueError, match="CPU backend"):
            rollout(model, config=task(precision="float64"))


def test_warmup_reset_and_full_trace_paths(model):
    for reset in (False, True):
        c = controller(warmup=3)
        a = rollout(model, controller=c, horizon=4, reset=reset)
        b = rollout(
            model, controller=c, horizon=4, reset=reset, recording=RecordingSpec(mode="full")
        )
        a.run()
        b.run()
        assert_equal(a, b)
        assert b.drain_trace()


def test_gpu_is_declared_default():
    assert next(f for f in fields(ExecutionSpec) if f.name == "backend").default == "auto"
    from toylife2.cli import build_parser

    assert build_parser().parse_args(["molecular", "chemotaxis"]).backend == "auto"


def test_checkpoint_failure_after_lazy_commit(model):
    s = rollout(model, horizon=16, recording=RecordingSpec(checkpoint_interval=16))
    s.run(steps=8)
    assert s.tick == 8 and s._resident.dirty
    direct = rollout(model, horizon=16)
    direct.run(steps=8)
    old = direct.snapshot()

    def fail(checkpoint, events):
        raise OSError("injected lazy checkpoint failure")

    s._record_callback = fail
    with pytest.raises(OSError):
        s.run()
    failed = s.snapshot()
    for name in ("world", "channels"):
        for key, value in old[name].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(failed[name][key], value)
    np.testing.assert_array_equal(failed["actions"], old["actions"])
    s._record_callback = lambda checkpoint, events: True
    s.run()
    direct.run()
    assert_equal(s, direct)


@pytest.mark.parametrize(
    "system,available,expected",
    [
        ("Darwin", {"metal", "cuda"}, "metal"),
        ("Linux", {"metal", "cuda"}, "cuda"),
        ("Linux", {"cuda"}, "cuda"),
        ("Darwin", set(), "cpu"),
        ("Linux", set(), "cpu"),
    ],
)
def test_automatic_backend_priority(system, available, expected):
    from toylife2.specs import _preferred_backend

    assert _preferred_backend(available.__contains__, system) == expected


def test_auto_backend_is_captured_as_concrete_choice(monkeypatch):
    # Test the original specification outside the numerical-backend harness.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import toylife2.specs as specs
specs._default_backend = lambda: 'cuda'
assert specs.ExecutionSpec().backend == 'cuda'
specs._default_backend = lambda: 'cpu'
assert specs.ExecutionSpec(backend='auto').backend == 'cpu'
assert specs.ExecutionSpec(backend='metal').backend == 'metal'
""",
        ],
        check=True,
    )
