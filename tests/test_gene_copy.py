"""Genetic kernel, mixed-length continuation, and recorded inspection contracts."""

import copy
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from toylife2 import BatchRunner, GeneCopySpec, GenomeBatch, Model
from toylife2.config import apply_overrides, dump_spec, load_spec, write_spec
from toylife2.evaluation import ObjectiveOutput
from toylife2.genotypes import load_selected, save_selected
from toylife2.molecular_experiments import profile_spec, run_experiment
from toylife2.population import PopulationSession, wright_fisher_step
from toylife2.protocols import ObjectiveSpec, WorkLimits
from toylife2.random import Stream
from toylife2.records import verify_record
from toylife2.sessions import Checkpoint, pack
from toylife2.specs import EvolutionSpec, plain, scientific_identity
from toylife2.variation import gene_copy_step


@pytest.fixture(scope="module")
def model(canonical_tables):
    return Model(tables=canonical_tables)


def spec(**kwargs):
    source = profile_spec("cue", genes=2)
    return replace(
        source,
        evolution=replace(
            source.evolution,
            population_size=6,
            trials=2,
            generations=3,
            mutation_probability=0.1,
            gene_copy=GeneCopySpec(0.4, 0.3),
            **kwargs,
        ),
    )


def input_reward_objective(genes, rollout, config):
    scale = int(np.load(config["inputs"]["scale"], allow_pickle=False)[0])
    scores = rollout.aggregate_scores()
    if config.get("omit_weights_for_gene") is not None and np.any(
        genes[:, 0] == config["omit_weights_for_gene"]
    ):
        return ObjectiveOutput(scores=scores)
    return ObjectiveOutput(
        scores=scores,
        weights=np.maximum(1, np.floor(scores * scale + 0.5).astype(np.int64)),
        denominator=scale,
        success=scores >= 0.9,
        descriptor=(genes[:, :1] > 0).astype(np.int64),
        metrics={"active_genes": np.full(len(genes), genes.shape[-1], np.int64)},
    )


@pytest.mark.parametrize("cap", [0, 16, 32])
@pytest.mark.device
def test_custom_inputs_and_partial_objective_fields_resume(model, tmp_path, cap):
    path = tmp_path / "scale.npy"
    np.save(path, [1000], allow_pickle=False)
    source = replace(
        spec(),
        execution=replace(spec().execution, batch_size=1),
        objective=ObjectiveSpec(
            kind="python",
            entrypoint="test_gene_copy:input_reward_objective",
            config={"inputs": {"scale": str(path)}},
        ),
    )
    batch = GenomeBatch(np.array([[1, 0], [2, 0], [3, 4]], np.uint32), [1, 1, 2])
    with BatchRunner(model=model) as runner:
        expected = runner.evaluator(source).evaluate(batch, panel=7)
        partial = runner.evaluator(source)
        partial.set_work_limits(WorkLimits(max_world_ticks=cap))
        result = partial.evaluate(batch, panel=7)
        assert not result.complete
        assert np.count_nonzero(result.status == "complete") == cap // 16
        if cap:
            assert result.denominator == 1000
            assert result.weights is not None
        state = partial.checkpoint_payload()
        assert runner.metrics["live_session_buffer_bytes"] > 0
        partial.close()
        assert runner.metrics["live_session_buffer_bytes"] == 0
        assert partial.cache.bytes == 0
        resumed = runner.evaluator(source)
        resumed.restore_payload(state)
        actual = resumed.evaluate(batch, panel=7)
        assert actual.complete
        for name in ("scores", "weights", "success", "descriptor"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        np.testing.assert_array_equal(actual.metrics["active_genes"], [1, 1, 2])


@pytest.mark.parametrize("mixed_lengths", [False, True])
@pytest.mark.device
def test_complete_custom_candidates_require_consistent_weights(model, tmp_path, mixed_lengths):
    path = tmp_path / "scale.npy"
    np.save(path, [1000], allow_pickle=False)
    source = replace(
        spec(),
        execution=replace(spec().execution, batch_size=1),
        objective=ObjectiveSpec(
            kind="python",
            entrypoint="test_gene_copy:input_reward_objective",
            config={"inputs": {"scale": str(path)}, "omit_weights_for_gene": 2},
        ),
    )
    batch = GenomeBatch(
        np.array([[1, 0], [2, 3 if mixed_lengths else 0]], np.uint32),
        [1, 2 if mixed_lengths else 1],
    )
    with BatchRunner(model=model) as runner:
        with pytest.raises(ValueError, match="complete candidates.*weights"):
            runner.evaluator(source).evaluate(batch)


@pytest.mark.device
def test_gene_copy_checkpoint_rejects_malformed_metadata(model):
    source = spec()
    with BatchRunner(model=model) as runner:
        session = runner.evolution(source)
        session.observe()
        checkpoint = session.checkpoint()
        for case in ("missing", "keys", "partition", "coordinate", "shape", "total", "duplicate"):
            data = copy.deepcopy(checkpoint.payload)
            payload = data["payload"]
            if case == "missing":
                payload["copy_statistics"] = {}
            elif case == "keys":
                payload["copy_statistics"]["extra"] = np.zeros((2, 4), np.int64)
            elif case == "partition":
                payload["copy_statistics"]["applied"][0, 0] += 1
            elif case == "coordinate":
                payload["gene_count_history"][0][0][0] = 999
            elif case == "shape":
                payload["gene_count_history"][0][1] = np.zeros(2, np.int64)
            elif case == "total":
                payload["gene_count_history"][0][1][2] += 1
            else:
                payload["gene_count_history"].append(
                    copy.deepcopy(payload["gene_count_history"][0])
                )
            arrays = {}
            altered = Checkpoint(checkpoint.kind, checkpoint.identity, pack(data, arrays), arrays)
            with pytest.raises(ValueError, match="gene-copy|gene-count"):
                runner.evolution(source, checkpoint=altered)


@pytest.mark.device
def test_gene_count_history_uses_pinned_observation_retention(model):
    source = spec()
    with BatchRunner(model=model) as runner:
        session = runner.evolution(source)
        session.history_capacity = 2
        assert session.run().complete
        assert list(session.gene_count_history) == list(session.history)
        restored = runner.evolution(source, checkpoint=session.checkpoint())
        assert list(restored.gene_count_history) == list(session.history)


def test_merge_admission_covers_uncompressed_history_probes(tmp_path):
    from toylife2._molecular_gene_copy import _inspection_merge_bytes

    group = tmp_path / "group"
    group.mkdir()
    arrays = {str(i): np.zeros((32, 8, 4), np.uint64) for i in range(12)}
    np.savez_compressed(group / "history_probes.npz", **arrays)
    payload = sum(value.nbytes for value in arrays.values())
    assert (group / "history_probes.npz").stat().st_size < payload
    assert _inspection_merge_bytes(tmp_path, [{"path": "group"}]) >= 3 * payload


@pytest.mark.integration
def test_structural_resume_releases_temporary_pending_cohorts(tmp_path, monkeypatch):
    from toylife2._variable_evaluation import VariableGeneEvaluator

    source = profile_spec("cue", genes=2)
    source = replace(
        source,
        execution=replace(source.execution, cpu_threads=None),
        episode=replace(source.episode, horizon=2),
        evolution=replace(
            source.evolution,
            trials=1,
            population_size=2,
            generations=1,
            mutation_probability=0,
            gene_copy=GeneCopySpec(1, 0),
        ),
    )
    closed = []
    original = VariableGeneEvaluator.close

    def close(evaluator):
        pending = {
            id(session)
            for child in evaluator.children.values()
            for session in child.pending.values()
        }
        original(evaluator)
        if pending and evaluator.runner is not None:
            assert not pending.intersection(map(id, evaluator.runner._sessions))
            closed.append(len(pending))

    monkeypatch.setattr(VariableGeneEvaluator, "close", close)
    parent = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="completed population"):
        run_experiment(replace(source, limits=WorkLimits(max_chemical_updates=2)), parent)
    destination = run_experiment(source, tmp_path / "continued", resume_from=parent)
    assert len(closed) >= 2  # Interrupted runner, then temporary restored evaluator.
    assert load_selected(destination).gene_counts.tolist() == [3]
    assert verify_record(destination / "evolution")["ok"]


def test_serialization_and_validation(tmp_path):
    baseline = profile_spec("cue")
    assert "gene_copy" not in plain(baseline)["evolution"]
    assert "gene_copy" not in dump_spec(baseline)
    modified = spec()
    path = tmp_path / "profile.toml"
    path.write_text(dump_spec(modified))
    assert load_spec(path) == modified
    assert scientific_identity(modified) != scientific_identity(baseline)
    overridden = apply_overrides(baseline, ["evolution.gene_copy.duplication_probability=0.02"])
    assert overridden.evolution.gene_copy == GeneCopySpec(0.02, 0)
    assert apply_overrides(baseline, []) == baseline
    for configuration in (
        dict(duplication_probability=-0.1),
        dict(max_genes=21),
        dict(min_genes=0),
    ):
        with pytest.raises(ValueError):
            GeneCopySpec(**configuration)
    with pytest.raises(ValueError, match="sum"):
        EvolutionSpec(mutation_probability=0.5, gene_copy=GeneCopySpec(0.5, 0.1))
    with pytest.raises(ValueError, match="expression"):
        replace(modified, controller=replace(modified.controller, initial_expression=1))
    with pytest.raises(ValueError):
        replace(modified, operation="sswm")


def test_active_zero_is_a_gene_and_padding_is_checked(tmp_path):
    batch = GenomeBatch(np.array([[1, 0], [1, 0]], np.uint32), np.array([1, 2]))
    assert len(set(batch.keys())) == 2
    assert list(map(len, batch.strings())) == [20, 40]
    save_selected(tmp_path, batch)
    restored = load_selected(tmp_path)
    assert restored.keys() == batch.keys()
    (tmp_path / "selected_gene_counts.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_selected(tmp_path)
    with pytest.raises(ValueError, match="padding"):
        GenomeBatch(np.array([[1, 2]], np.uint32), np.array([1]))
    for counts in (np.array([0]), np.array([3]), np.array([1.5])):
        with pytest.raises(ValueError):
            GenomeBatch(np.array([[1, 0]], np.uint32), counts)


def test_gene_targets_and_blocked_events():
    batch = GenomeBatch(np.array([[7, 7, 0], [7, 7, 7]], np.uint32), np.array([2, 3]))
    weights = np.array([1, 0], np.int64)
    child, parents, events = gene_copy_step(
        batch,
        weights,
        stream=Stream(12),
        replicate=0,
        generation=0,
        mutation_probability=0,
        gene_copy=GeneCopySpec(1, 0, 1, 3),
    )
    assert parents.tolist() == [0, 0]
    assert child.keys() == [(7, 7, 7), (7, 7, 7)]
    assert events["operations"].tolist() == [2, 2]
    assert not events["blocked"].any()
    unchanged, _, events = gene_copy_step(
        child,
        np.ones(2, np.int64),
        stream=Stream(12),
        replicate=0,
        generation=1,
        mutation_probability=0,
        gene_copy=GeneCopySpec(1, 0, 1, 3),
    )
    assert unchanged.keys() == child.keys()
    assert events["blocked"].all()
    only = GenomeBatch(np.array([[0, 0, 0]], np.uint32), np.array([1]))
    unchanged, _, events = gene_copy_step(
        only,
        np.ones(1, np.int64),
        stream=Stream(12),
        replicate=0,
        generation=1,
        mutation_probability=0,
        gene_copy=GeneCopySpec(0, 1, 1, 3),
    )
    assert unchanged.keys() == [(0,)] and events["blocked"].all()
    deleted, _, _ = gene_copy_step(
        batch,
        weights,
        stream=Stream(12),
        replicate=0,
        generation=0,
        mutation_probability=0,
        gene_copy=GeneCopySpec(0, 1, 1, 3),
    )
    assert deleted.keys() == [(7,), (7,)]


def test_zero_structural_rates_preserve_exact_point_kernel():
    population = np.random.default_rng(41).integers(0, 1 << 20, (32, 3), dtype=np.uint32)
    weights = np.arange(32, dtype=np.int64)
    legacy, parents, _, _ = wright_fisher_step(
        population, weights, stream=Stream(122), replicate=4, generation=8, mutation_probability=0.7
    )
    mixed, mixed_parents, _ = gene_copy_step(
        GenomeBatch.from_fixed(population, 20),
        weights,
        stream=Stream(122),
        replicate=4,
        generation=8,
        mutation_probability=0.7,
        gene_copy=GeneCopySpec(),
    )
    np.testing.assert_array_equal(parents, mixed_parents)
    np.testing.assert_array_equal(legacy, mixed.genes[:, :3])
    assert np.all(mixed.gene_counts == 3)
    assert not mixed.genes[:, 3:].any()


def test_single_event_distribution_and_copy_target_mass():
    size = 12000
    population = GenomeBatch.from_fixed(np.tile(np.array([7, 7, 9], np.uint32), (size, 1)), 4)
    child, _, events = gene_copy_step(
        population,
        np.ones(size, np.int64),
        stream=Stream(805),
        replicate=2,
        generation=1,
        mutation_probability=0.2,
        gene_copy=GeneCopySpec(0.3, 0.1, 1, 4),
    )
    frequencies = np.bincount(events["operations"], minlength=4) / size
    np.testing.assert_allclose(frequencies, [0.4, 0.2, 0.3, 0.1], atol=0.02)
    assert not events["blocked"].any()
    for code, positions in ((1, 60), (2, 3), (3, 3)):
        selected = events["operations"] == code
        observed = np.bincount(events["positions"][selected], minlength=positions)
        expected = selected.sum() / positions
        assert np.all(abs(observed - expected) < 6 * np.sqrt(expected))
    duplicates = child.genes[events["operations"] == 2]
    # Either identical copy leads to the same child, so their masses add.
    fraction = np.mean(np.all(duplicates[:, :4] == [7, 7, 7, 9], axis=1))
    assert abs(fraction - 2 / 3) < 0.04


@pytest.mark.device
def test_zero_rates_preserve_population_trajectory(model):
    structural = replace(spec(), evolution=replace(spec().evolution, gene_copy=GeneCopySpec()))
    legacy = replace(structural, evolution=replace(structural.evolution, gene_copy=None))
    a, b = PopulationSession(model, legacy), PopulationSession(model, structural)
    assert a.run().status == b.run().status == "complete"
    np.testing.assert_array_equal(a.populations, b.populations[..., :2])
    np.testing.assert_array_equal(
        a.snapshot()["arrays"]["history"], b.snapshot()["arrays"]["history"]
    )
    np.testing.assert_array_equal(a.evaluations[0].scores, b.evaluations[0].scores)
    b.evaluator.close()


@pytest.mark.device
def test_mixed_evaluation_matches_fixed_width_and_scatter(model):
    source = spec()
    batch = GenomeBatch(
        np.array([[50, 70, 0], [80, 0, 0], [50, 70, 90]], np.uint32), np.array([2, 1, 3])
    )
    with BatchRunner(model=model) as runner:
        evaluator = runner.evaluator(source)
        result = evaluator.evaluate(batch, np.array([17, 8, 29]), panel=7)
        for i, count in enumerate(batch.gene_counts):
            child = replace(
                source,
                model=replace(source.model, genes=int(count)),
                genotypes=(),
                evolution=replace(source.evolution, gene_copy=None),
            )
            expected = runner.evaluator(child).evaluate(batch.active(i)[None, :], panel=7)
            np.testing.assert_array_equal(result.scores[i], expected.scores[0])
        assert result.candidate_ids.tolist() == [17, 8, 29]
        reversed_result = evaluator.evaluate(batch[[2, 0, 1]], panel=7)
        np.testing.assert_array_equal(result.scores[[2, 0, 1]], reversed_result.scores)
        assert evaluator.cache.bytes <= evaluator.cache.capacity
        assert len(evaluator.cache) == 3
        evaluator.close()
        assert evaluator.cache.bytes == 0


@pytest.mark.parametrize("resource", ["chemical_updates", "world_ticks"])
@pytest.mark.parametrize("cap", [0, 2, 9, 25, 70])
@pytest.mark.device
def test_aggregate_quota_checkpoint_and_resume(model, cap, resource, tmp_path):
    source = spec()
    with BatchRunner(model=model) as runner:
        direct = runner.evolution(source)
        direct.run()
        partial = runner.evolution(source, limits=WorkLimits(**{"max_" + resource: cap}))
        partial.run()
        assert partial.counters["total"].get(resource, 0) <= cap
        partial.checkpoint().write(tmp_path / "checkpoint")
        continued = runner.evolution(
            source, checkpoint=Checkpoint.read(tmp_path / "checkpoint"), limits=WorkLimits()
        )
        assert continued.run().status == "complete"
        np.testing.assert_array_equal(continued.populations, direct.populations)
        np.testing.assert_array_equal(continued.gene_counts, direct.gene_counts)
        assert continued.counters["total"] == direct.counters["total"]
        assert 1 <= continued.gene_counts.min() <= continued.gene_counts.max() <= 20
        for name in direct.copy_statistics:
            np.testing.assert_array_equal(
                continued.copy_statistics[name], direct.copy_statistics[name]
            )


@pytest.mark.device
def test_structural_trajectory_is_independent_of_batch_width(model):
    source = spec()
    with BatchRunner(model=model) as runner:
        wide = runner.evolution(source)
        narrow = runner.evolution(
            replace(source, execution=replace(source.execution, batch_size=1))
        )
        assert wide.run().status == narrow.run().status == "complete"
        np.testing.assert_array_equal(wide.populations, narrow.populations)
        np.testing.assert_array_equal(wide.gene_counts, narrow.gene_counts)
        np.testing.assert_array_equal(
            wide.snapshot()["arrays"]["history"], narrow.snapshot()["arrays"]["history"]
        )


def test_structural_selection_playback_smoke(tmp_path, model):
    source = profile_spec("chemotaxis", genes=2)
    source = replace(
        source,
        execution=model.execution,
        episode=replace(source.episode, episodes=1, horizon=1),
        evolution=replace(
            source.evolution,
            trials=1,
            population_size=1,
            generations=1,
            mutation_probability=0,
            gene_copy=GeneCopySpec(1, 0),
        ),
    )
    parent = tmp_path / "selected"
    parent.mkdir()
    write_spec(source, parent / "profile.toml")
    with BatchRunner(model=model) as runner:
        session = runner.evolution(source, destination=parent / "evolution")
        session.run()
        selected = GenomeBatch(session.populations[0, :1], session.gene_counts[0, :1])
        score = float(session.evaluations[0].scores[0])
        runner.finish(session)
    save_selected(parent, selected)
    (parent / "experiment.json").write_text(
        json.dumps({"mean_scores": [score], "heldout_seed": 1234, "heldout_panel": 7})
    )
    assert load_selected(parent).gene_counts.tolist() == [3]
    assert verify_record(parent / "evolution")["ok"]
    replay = tmp_path / "replay"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "examples/replay_molecular_genome.py"),
            str(parent),
            str(replay),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert load_spec(replay / "profile.toml").model.genes == 3
    np.testing.assert_array_equal(
        np.load(replay / "selected_genomes.npy"), selected.active(0)[None]
    )
    assert verify_record(replay / "rollout")["ok"]
    assert (replay / "animation.html").is_file()


@pytest.mark.integration
def test_structural_records_inspection_and_playback(tmp_path):
    source = profile_spec("chemotaxis", genes=2)
    source = replace(
        source,
        execution=replace(source.execution, cpu_threads=None),
        episode=replace(source.episode, episodes=2, horizon=4),
        evolution=replace(
            source.evolution,
            trials=1,
            population_size=3,
            generations=1,
            mutation_probability=0,
            gene_copy=GeneCopySpec(1, 0),
        ),
    )
    destination = tmp_path / "experiment"
    run_experiment(source, destination, animate=True, frame_stride=1)
    selected = load_selected(destination)
    assert selected.gene_counts.tolist() == [3]
    assert load_selected(destination, initial=True).gene_counts.tolist() == [2]
    summary = json.loads((destination / "experiment.json").read_text())
    assert summary["genome_format"] == "gene-counts-v1"
    assert len(summary["selected_genomes"][0]) == 3
    assert (destination / "animation.html").exists()
    child = destination / summary["inspection_groups"][0]["path"]
    assert verify_record(destination / "evolution")["ok"]
    assert verify_record(child / "heldout")["ok"]
    assert verify_record(child / "initial_heldout")["ok"]
    with np.load(destination / "behaviour.npz", allow_pickle=False) as data:
        assert data["gene_counts"].tolist() == [3]
    replay = tmp_path / "replay"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "examples/replay_molecular_genome.py"),
            str(destination),
            str(replay),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert load_spec(replay / "profile.toml").model.genes == 3
    assert verify_record(replay / "rollout")["ok"]
    assert (replay / "animation.html").exists()


@pytest.mark.integration
def test_multiple_inspection_lengths_preserve_order_and_expression_masks(tmp_path, model):
    from toylife2._molecular_gene_copy import inspect_variable_selected

    source = spec()
    selected = GenomeBatch(np.array([[7, 0, 0], [7, 7, 9], [7, 9, 0]], np.uint32), [1, 3, 2])
    initial = GenomeBatch.from_fixed(np.tile(np.array([7, 9], np.uint32), (3, 1)), 3)
    write_spec(source, tmp_path / "profile.toml")
    save_selected(tmp_path, selected)
    save_selected(tmp_path, initial, initial=True)
    history = np.array([[i, 0, 0.5, 0.5, 1] for i in range(3)])
    np.save(tmp_path / "fitness_history.npy", history)
    with BatchRunner(model=model) as runner:
        inspect_variable_selected(
            runner,
            source,
            selected,
            initial,
            tmp_path,
            history=history,
            started=time.perf_counter(),
            evolution_status="complete",
            counters={},
            parent=None,
            animate=False,
            frame_stride=1,
        )
    summary = json.loads((tmp_path / "experiment.json").read_text())
    assert summary["selected_gene_counts"] == [1, 3, 2]
    assert [g["selection_ids"] for g in summary["inspection_groups"]] == [[0], [2], [1]]
    for filename in ("behaviour.npz", "initial_behaviour.npz", "history_probes.npz"):
        with np.load(tmp_path / filename) as merged:
            assert merged["expression"].shape[1] == 3
            assert merged["expression"].shape[-1] == (4 if filename == "history_probes.npz" else 2)
            for group in summary["inspection_groups"]:
                with np.load(tmp_path / group["path"] / filename) as child:
                    np.testing.assert_array_equal(
                        merged["expression"][:, group["selection_ids"]], child["expression"]
                    )
                assert verify_record(tmp_path / group["path"] / "heldout")["ok"]
    for name, values in summary["production"].items():
        assert len(values) == 3
        for group in summary["inspection_groups"]:
            child = json.loads((tmp_path / group["path"] / "experiment.json").read_text())
            np.testing.assert_array_equal(
                np.asarray(values)[group["selection_ids"]], child["production"][name]
            )
    with np.load(tmp_path / "behaviour.npz") as merged:
        assert merged["expected_scores"].shape == (3, 2)
