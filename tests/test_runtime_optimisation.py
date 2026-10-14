"""Composition contracts for shared plans, residency and safe update blocks."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from toylife2 import Model
from toylife2.chemistry import ChemistrySession, _BatchChemistry
from toylife2.protocols import ControllerSpec, EpisodeSpec, PortSpec, ReadoutSpec, WorkLimits
from toylife2.random import Stream
from toylife2.reference import CellState
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec, RecordingSpec
from toylife2.worlds import ScenarioRandom

pytestmark = pytest.mark.device


@pytest.fixture
def model(canonical_tables):
    return Model(tables=canonical_tables, execution=ExecutionSpec())


def assert_payload_equal(a, b):
    for key in ("counts", "bits", "events", "first_events", "after_input", "after_output"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["update"] == b["update"]


def test_repeated_episode_graphs_are_shared_across_sessions(model, monkeypatch):
    import toylife2.chemistry as chemistry

    original = chemistry._graph
    calls = []

    def compile_graph(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chemistry, "_graph", compile_graph)
    genes = np.repeat(np.asarray([[0, 0], [12345, 54321]], np.int32), 8, axis=0)
    first = ChemistrySession(model, genes)
    second = ChemistrySession(model, genes)
    assert len(calls) == 2
    assert first._batch.graphs[0] is second._batch.graphs[0]
    assert first._batch.graphs[0] is first._batch.graphs[7]
    assert not first._batch.graphs[0][4].flags.writeable
    before = second.payload()
    first.step({"00000001": 1})
    assert_payload_equal(second.payload(), before)


def test_extra_state_vocabulary_and_caller_capacity_are_cache_keys(model):
    genes = np.asarray([[0]], np.int32)
    a = _BatchChemistry(model, genes, ControllerSpec(), [CellState(proteins=Counter({47: 1}))])
    b = _BatchChemistry(model, genes, ControllerSpec(), [CellState(proteins=Counter({105: 1}))])
    assert ("P", 47) in a.graphs[0][1]
    assert ("P", 105) in b.graphs[0][1]
    assert ("P", 105) not in a.graphs[0][1]
    with pytest.raises(OverflowError):
        _BatchChemistry(
            model, genes, ControllerSpec(capacity=1), [CellState(proteins=Counter({47: 1}))]
        )


@pytest.mark.parametrize("width", [None, 1, 7])
def test_unique_scenario_draws_match_full_addressed_stream(width):
    scenarios = np.asarray([[9, 2, 9], [2, 9, 2]], np.uint64)
    rng = ScenarioRandom(123, 4, scenarios)
    ids = (
        scenarios
        if width is None
        else (scenarios[..., None] * np.uint64(width) + np.arange(width, dtype=np.uint64))
    )
    expected = Stream(123).uniform(ids, role="world:probe", trial=4, event=11)
    actual = rng.uniform(11, "probe", width)
    np.testing.assert_array_equal(actual, expected)
    u = Stream(123).uniform(ids, role="world:probe:normal-radius", trial=4, event=11)
    v = Stream(123).uniform(ids, role="world:probe:normal-phase", trial=4, event=11)
    np.testing.assert_array_equal(
        rng.normal(11, "probe", width), np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)
    )
    actual[...] = 0
    scenarios[...] = 0
    np.testing.assert_array_equal(rng.uniform(11, "probe", width), expected)


@pytest.mark.parametrize("mode", ["canonical_cell", "regulatory"])
@pytest.mark.parametrize("input_mode", ["pulse", "replenish"])
def test_blocks_match_single_steps_including_stage_readouts(model, mode, input_mode):
    genes = np.random.default_rng(678).integers(0, 1 << 20, (12, 4), dtype=np.int32)
    readouts = (
        ReadoutSpec("expression", "expression"),
        ReadoutSpec("functional", "functional_expression", gene=1),
        ReadoutSpec("protein", "protein", gene=1),
        ReadoutSpec("bound", "bound", stage="after_output"),
        ReadoutSpec("first", "fission", stage="after_input"),
        ReadoutSpec("all", "fission"),
    )
    spec = ControllerSpec(
        mode=mode, initial_expression=5, ports=PortSpec(input_mode=input_mode, readouts=readouts)
    )
    scalar = ChemistrySession(model, genes, spec)
    block = ChemistrySession(model, genes, spec)
    for tick in range(8):
        scalar.step({"00000001": 3} if input_mode == "replenish" or tick == 0 else {})
    model.engine._chemistry_block_size = 8
    report = block._step_block({"00000001": 3}, 8, pulse=input_mode == "pulse")
    assert report.transition["updates"] == 8
    assert_payload_equal(scalar.payload(), block.payload())
    assert scalar.counters == block.counters
    for name in block.last_readouts:
        np.testing.assert_array_equal(block.last_readouts[name], scalar.last_readouts[name])


def test_block_fallback_keeps_earliest_capacity_failure(model):
    model.engine._chemistry_block_size = 16
    spec = ControllerSpec(mode="regulatory", ports=PortSpec(max_dose=2**31 - 1))
    session = ChemistrySession(model, [0], spec)
    copies = ((2**31 - 1) // 16) // 3
    for _ in range(3):
        report = session._step_block({"00000001": copies}, 16)
        assert report.transition["advanced"]
        assert session.update <= 3
    before = session.payload()
    report = session._step_block({"00000001": copies}, 16)
    assert report.reason == "chemistry_capacity"
    assert not report.transition["advanced"]
    assert_payload_equal(before, session.payload())


def test_partial_block_keeps_committed_snapshot_and_resumes(model):
    model.engine._chemistry_block_size = 8
    genes = np.asarray([[12345, 54321]], np.int32)
    protocol = EpisodeSpec(horizon=2, chemical_updates=5)
    partial = RolloutSession(
        model, genes, protocol=protocol, limits=WorkLimits(max_chemical_updates=3)
    )
    initial = partial.snapshot(fields=("chemistry",))["chemistry"]
    assert partial.step().status == "partial"
    assert partial._pending["chemical_cursor"] == 3
    assert_payload_equal(initial, partial.snapshot(fields=("chemistry",))["chemistry"])
    resumed = RolloutSession(model, genes, protocol=protocol, checkpoint=partial.checkpoint())
    assert resumed.run().status == "complete"
    expected = RolloutSession(model, genes, protocol=protocol)
    expected.run()
    assert_payload_equal(resumed.chemistry.payload(), expected.chemistry.payload())
    np.testing.assert_array_equal(resumed.scores, expected.scores)


def test_full_recording_and_wide_expression_keep_single_step_behavior(model):
    model.engine._chemistry_block_size = 16
    session = ChemistrySession(
        model,
        np.zeros(65, np.int32),
        ControllerSpec(initial_expression=(1 << 65) - 1),
        recording=RecordingSpec(mode="full"),
    )
    assert session.state["expression"].dtype == object
    assert session.state["expression"] == (1 << 65) - 1
    report = session._step_block(None, 16)
    assert report.transition.get("updates", 1) == 1
    assert len(session.drain_trace()) == 1


def test_reset_reuses_buffers_and_warmed_state(model):
    protocol = EpisodeSpec(horizon=3, chemical_updates=4, reset_per_decision=True)
    session = RolloutSession(
        model, [12345, 54321], protocol=protocol, controller=ControllerSpec(warmup=2)
    )
    child = session.chemistry
    arrays = tuple(map(id, child._batch.arrays))
    for _ in range(3):
        session.step()
        assert session.chemistry is child
        assert tuple(map(id, child._batch.arrays)) == arrays
        assert child.update == 6
    restored = RolloutSession(
        model,
        [12345, 54321],
        protocol=replace(protocol, horizon=4),
        controller=ControllerSpec(warmup=2),
    )
    # Identity-preserving resume is exercised separately; reset storage survives
    # ordinary checkpoints without becoming part of the scientific interface.
    assert restored._reset_storage.buffer_bytes > 0


@pytest.mark.parametrize("steps", [1, 8])
def test_readout_failure_preserves_all_completed_state(model, monkeypatch, steps):
    import toylife2._chemistry_kernels as kernels

    model.engine._chemistry_block_size = steps
    spec = ControllerSpec(mode="regulatory", initial_expression=3)
    genes = np.asarray([12345, 54321], np.int32)
    session = ChemistrySession(model, genes, spec)
    session.run(3)
    before = session.payload()
    counters = session.counters
    original = kernels.compact_readouts

    def fail_after_dispatch(*args):
        original(*args)
        raise RuntimeError("injected readout transfer failure")

    with monkeypatch.context() as patch:
        patch.setattr(kernels, "compact_readouts", fail_after_dispatch)
        with pytest.raises(RuntimeError, match="readout transfer"):
            session._step_block(None, steps)
    assert_payload_equal(before, session.payload())
    assert session.counters == counters
    expected = ChemistrySession(model, genes, spec)
    expected.restore_payload(before)
    for _ in range(steps):
        expected.step()
    session._step_block(None, steps)
    assert_payload_equal(expected.payload(), session.payload())


def test_selected_snapshot_never_materialises_unselected_pending_state(model, monkeypatch):
    session = RolloutSession(
        model,
        [12345, 54321],
        protocol=EpisodeSpec(horizon=1, chemical_updates=3),
        limits=WorkLimits(max_chemical_updates=1),
    )
    session.step()

    def reject_materialisation():
        raise AssertionError("unselected working chemistry materialised")

    monkeypatch.setattr(session.chemistry, "_payload_view", reject_materialisation)
    assert session.snapshot(fields=("tick",), include_pending=True) == {"tick": 0}


def test_novel_graph_construction_admitted_before_device_allocations(model, monkeypatch):
    available = model.engine._available_bytes()
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 47)[0])

    def reject_allocation(*args, **kwargs):
        raise AssertionError("allocation before graph construction admission")

    monkeypatch.setattr(model.engine, "_i32", reject_allocation)
    with model.engine._temporary_bytes(available - 4096, message="test scratch"):
        with pytest.raises(MemoryError, match="graph construction"):
            ChemistrySession(model, [coding, coding])


def test_retention_failure_precedes_recording_publication(model, monkeypatch):
    session = RolloutSession(
        model,
        [12345, 54321],
        protocol=EpisodeSpec(horizon=1),
        recording=RecordingSpec(mode="full"),
    )
    calls = []
    session._record_callback = lambda checkpoint, events: calls.append(checkpoint) or True
    original = session._commit_storage.capture

    def fail_capture():
        raise RuntimeError("retention failed")

    monkeypatch.setattr(session._commit_storage, "capture", fail_capture)
    with pytest.raises(RuntimeError, match="retention failed"):
        session.step()
    assert session.tick == 0 and calls == []
    monkeypatch.setattr(session._commit_storage, "capture", original)
    assert session.run().status == "complete"
    assert len(calls) == 1 and session.tick == 1


def test_empty_graph_still_admits_gene_metadata(model):
    from toylife2.chemistry import _graph

    estimates = []
    _graph(
        model.tables,
        np.zeros(128, np.int32),
        ControllerSpec(ports=PortSpec(metabolites=())),
        admit=estimates.append,
    )
    assert estimates and max(estimates) >= 128 * 128


def test_readout_plan_admits_before_dense_allocation(model, monkeypatch):
    import toylife2.chemistry as chemistry

    session = ChemistrySession(model, [12345, 54321])
    batch = session._batch
    available = model.engine._available_bytes()
    with model.engine._temporary_bytes(available - 1, message="test scratch"):
        with monkeypatch.context() as patch:
            patch.setattr(
                chemistry.np,
                "full",
                lambda *a, **kw: pytest.fail("dense allocation before admission"),
            )
            with pytest.raises(MemoryError, match="readout construction"):
                batch._compile_readouts(batch.graphs, np.asarray([[12345, 54321]]), np.asarray([0]))


def test_idle_plan_finalizers_do_not_keep_engine_cache_alive():
    import gc
    import weakref

    from toylife2._chemistry_cache import GraphCache

    cache = GraphCache(1024 * 1024)
    plan = cache.put(("test",), (), ())
    reference = weakref.ref(cache)
    del plan, cache
    gc.collect()
    assert reference() is None
