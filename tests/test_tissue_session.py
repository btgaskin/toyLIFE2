from __future__ import annotations

import numpy as np
import pytest

from toylife2 import Model
from toylife2.protocols import WorkLimits
from toylife2.specs import ExecutionSpec, SpatialSpec
from toylife2.tissue import TissueSession


@pytest.fixture(scope="module")
def model():
    return Model(execution=ExecutionSpec(backend="cpu"))


@pytest.mark.parametrize("boundary", ["open", "ring"])
@pytest.mark.parametrize("diffusion", ["none", "A", "B", "both", "all"])
def test_incremental_exact_legacy_trajectory(model, boundary, diffusion):
    genes = np.asarray([[[33177, 950375], [907050, 1048465]]], np.int32)
    spec = SpatialSpec(
        cells=5,
        steps=6,
        boundary=boundary,
        diffusion=diffusion,
        morphogen="explicit",
        initial_state=(3, 0, 2, 1, 3),
        injections=((0, 0, 2), (0, 0, 1), (1, 4, 1), (3, 0, 2)),
    )
    expected = model.patterns(genes, spec).values
    session = TissueSession(model, genes, spec)
    assert session.run(0).cursors["tissue_tick"] == 0
    np.testing.assert_array_equal(session.state["tissue"][0, 0], spec.initial_state)
    for tick in range(spec.steps):
        assert session.step().transition["advanced"]
        np.testing.assert_array_equal(session.state["tissue"], expected[..., tick, :])


def test_tissue_resume_limits_and_owned_snapshots(model):
    genes = np.asarray([[33177, 950375]], np.int32)
    spec = SpatialSpec(cells=7, steps=7, morphogen="central_A_continuous", diffusion="all")
    uninterrupted = TissueSession(model, genes, spec)
    uninterrupted.run(7)
    split = TissueSession(model, genes, spec, WorkLimits(max_world_ticks=3))
    split.run(10)
    assert split.tick == 3 and split.state["status"] == "partial"
    resumed = TissueSession(model, genes, spec, checkpoint=split.checkpoint())
    resumed.run(4)
    np.testing.assert_array_equal(uninterrupted.state["tissue"], resumed.state["tissue"])
    assert resumed.counters["total"]["world_ticks"] == 7
    with pytest.raises(ValueError):
        resumed.state["tissue"][0, 0] = 1


def test_dynamic_injections_and_validation(model):
    session = TissueSession(model, [[33177, 950375]], SpatialSpec(cells=3, morphogen="none"))
    assert session.step([0, 2, 0]).transition["advanced"]
    assert session.tick == 1
    for invalid in ([0, -1, 0], [0.0, 1.0, 0.0], [0, 4, 0]):
        with pytest.raises(ValueError):
            session.step(invalid)
    assert session.tick == 1


def test_tissue_quota_counts_each_logical_genotype(model):
    genes = np.asarray([[[33177, 950375], [907050, 1048465]]], np.int32)
    session = TissueSession(model, genes, limits=WorkLimits(max_world_ticks=3))
    assert session.step().transition["advanced"]
    assert session.counters["invocation"]["world_ticks"] == 2
    before = session.state["tissue"]
    assert session.step().status == "partial"
    assert session.tick == 1
    np.testing.assert_array_equal(session.state["tissue"], before)
    assert session.buffer_bytes == session.buffer_memory["host"] + session.buffer_memory["device"]
    payload = session.checkpoint().payload["payload"]
    assert payload["table_digest"] == model.tables.digest
    assert payload["genotypes"].shape == genes.shape
