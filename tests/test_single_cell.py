"""Contracts of the explicit molecular observation → action → reproduction path."""

from dataclasses import replace

import numpy as np
import pytest

from toylife2 import BinaryOutputSpec, Model
from toylife2.config import mapping_spec
from toylife2.molecular_experiments import profile_spec
from toylife2.output import ROLE, decode
from toylife2.protocols import WorkLimits
from toylife2.random import Stream
from toylife2.reference import CellState, ReferenceModel
from toylife2.rollout import RolloutSession
from toylife2.single_cell import methods_summary, require_current
from toylife2.specs import ExecutionSpec, RecordingSpec, plain
from toylife2.worlds import ScenarioRandom


@pytest.fixture(scope="module")
def model(canonical_tables):
    return Model(tables=canonical_tables, execution=ExecutionSpec())


@pytest.mark.parametrize("rule", ["majority", "proportional"])
@pytest.mark.parametrize("representation", ["bound_free", "bound_pair"])
def test_count_rules_and_founding_assignment(rule, representation):
    markers = ("a",) if representation == "bound_free" else ("a", "b")
    spec = BinaryOutputSpec(representation, markers, rule)
    first, second = np.array([0, 0, 1, 3, 2, 1]), np.array([0, 1, 0, 1, 2, 3])
    rng = ScenarioRandom(17, 4, np.arange(6))
    readouts = dict(zip(spec.readouts, (first, second), strict=True))
    actions, observed = decode(spec, readouts, rng, 7)
    expected = [0.5, 1, 0, 0, 0.5, 1] if rule == "majority" else [0.5, 1, 0, 0.25, 0.5, 0.75]
    np.testing.assert_array_equal(observed["action_probability_one"], expected)
    swapped, swapped_observed = decode(replace(spec, permutation=(1, 0)), readouts, rng, 7)
    np.testing.assert_array_equal(swapped, 1 - actions)
    np.testing.assert_array_equal(
        swapped_observed["action_probability_one"], 1 - np.array(expected)
    )
    assert actions[1] == 1 and actions[2] == 0


def test_count_sampling_law_and_one_count_equivalence():
    rng = ScenarioRandom(19, 7, np.arange(8192))
    spec = BinaryOutputSpec()
    actions, _ = decode(
        spec, {"target_free:act": np.ones(8192, int), "act": np.full(8192, 3)}, rng, 2
    )
    assert abs(actions.mean() - 0.75) < 0.025
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        readouts = {"target_free:act": np.full(8192, a), "act": np.full(8192, b)}
        majority, _ = decode(replace(spec, rule="majority"), readouts, rng, 3)
        proportional, _ = decode(spec, readouts, rng, 3)
        np.testing.assert_array_equal(majority, proportional)
    with pytest.raises(ValueError, match="integer"):
        decode(spec, {"target_free:act": np.array([1.0]), "act": np.array([0])}, rng, 0)
    with pytest.raises(OverflowError):
        decode(spec, {"target_free:act": np.array([2**31 - 1]), "act": np.array([1])}, rng, 0)


@pytest.mark.device
@pytest.mark.parametrize("rule", ["majority", "proportional"])
def test_device_philox_rejection_and_actions_agree(model, rule):
    import quadrants as qd

    from toylife2._output_kernels import sample_actions

    # Large odd bounds exercise rejection; high episode words exercise both ID limbs.
    count = 129
    a = np.resize([0, 0, 1, 3, 2**30, 1, 11], count).astype(np.int32)
    b = np.resize([0, 1, 0, 1, 1, 2**30, 17], count).astype(np.int32)
    ids = np.arange(count, dtype=np.uint64) + np.uint64(2**32)
    rng = ScenarioRandom(173, 5, ids)
    spec = BinaryOutputSpec(rule=rule, permutation=(1, 0))
    expected, _ = decode(spec, {"target_free:act": a, "act": b}, rng, 19)
    counts = model.engine._i32((count, 2), np.stack((a, b), axis=-1))
    keys = qd.ndarray(qd.u32, (count, 2))
    keys.from_numpy(np.tile(rng.stream.key(role=ROLE, trial=5, event=19), (count, 1)))
    indices = qd.ndarray(qd.u32, (count, 2))
    indices.from_numpy(
        np.stack((ids & np.uint64(0xFFFFFFFF), ids >> np.uint64(32)), axis=-1).astype(np.uint32)
    )
    output = model.engine._i32((count,))
    sample_actions(counts, keys, indices, output, 2 if rule == "majority" else 1, 1)
    np.testing.assert_array_equal(output.to_numpy(), expected)


def test_current_contract_and_independent_task_cardinality():
    spec = profile_spec("cue_alternatives", backend="cpu", output_representation="bound_pair")
    assert len(spec.controller.output.markers) == 2 and spec.episode.episodes == 2
    assert spec.task.config["cue_doses"] == ((1, 0), (0, 1))
    assert "action_outputs" not in spec.task.config
    assert mapping_spec(plain(spec)) == spec
    require_current(spec)
    text = methods_summary(spec)
    assert "viable cellular background" in text and "translation event" in text
    old = replace(
        spec,
        task=replace(
            spec.task,
            config={k: v for k, v in spec.task.config.items() if k != "experiment_contract"},
        ),
    )
    with pytest.raises(ValueError, match="deprecated"):
        require_current(old)
    with pytest.raises(ValueError, match="permutation"):
        BinaryOutputSpec(permutation=(1, 1))


@pytest.mark.device
@pytest.mark.parametrize("representation", ["bound_free", "bound_pair"])
def test_cue_sampling_expected_accuracy_resume_and_cache(model, representation):
    from toylife2 import BatchRunner

    spec = profile_spec(
        "cue",
        backend=model.execution.backend,
        response_updates=3,
        output_representation=representation,
    )
    spec = replace(
        spec,
        episode=replace(spec.episode, chemical_updates=2),
        controller=replace(spec.controller, warmup=2),
    )
    genes = np.array([[1007704, 0], [0, 0]], np.uint32)

    def session(**kwargs):
        return RolloutSession(
            model, genes, spec.task, spec.episode, spec.controller, seed=13, **kwargs
        )

    full = session()
    full.run()
    partial = session(limits=WorkLimits(max_chemical_updates=4 * 3))
    assert partial.run().status == "partial"
    resumed = session(checkpoint=partial.checkpoint())
    assert resumed.run().status == "complete"
    np.testing.assert_array_equal(full.scores, resumed.scores)
    np.testing.assert_array_equal(full.expected_scores, resumed.expected_scores)
    for name, value in full.production_metrics().items():
        np.testing.assert_array_equal(value, resumed.production_metrics()[name])
    assert np.all(full.chemical_updates_warmup == 2)
    assert np.all(full.chemical_updates_task == 6)
    assert np.all(np.isin(full.snapshot()["actions"], (0, 1)))
    with BatchRunner(model=model) as runner:
        evaluator = runner.evaluator(spec)
        assert evaluator.stochastic
        first = evaluator.evaluate(genes)
        cached = evaluator.evaluate(genes[::-1])
        for name, value in first.metrics.items():
            np.testing.assert_array_equal(value[::-1], cached.metrics[name])
        np.testing.assert_array_equal(first.scores[::-1], cached.scores)
        assert evaluator.metrics["cache_hits"] >= 2


@pytest.mark.device
def test_translation_counts_reference_unfolded_and_blocking(model):
    from toylife2.chemistry import ChemistrySession
    from toylife2.protocols import ControllerSpec

    reference = ReferenceModel(model.tables)
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 0)[0])
    gene = next(
        (p << 16) | coding
        for p in range(16)
        if reference.expression(np.array([(p << 16) | coding]), CellState()) == 1
    )
    genes = np.array([[gene, gene], [1007704, gene]], np.uint32)
    expected = []
    for row in genes:
        state, total = CellState(), 0
        for _ in range(7):
            result = reference.step(row, state)
            total += int(result.expression).bit_count()
            state = result.state
        expected.append(total)
    blocked = ChemistrySession(model, genes, ControllerSpec())
    model.engine._chemistry_block_size = 4
    blocked.run(7)
    scalar = ChemistrySession(model, genes, ControllerSpec())
    for _ in range(7):
        scalar.step()
    np.testing.assert_array_equal(blocked.translation_events, expected)
    np.testing.assert_array_equal(scalar.translation_events, expected)
    assert expected[0] == 14  # Both nonfolding gene copies still imply translation.
    restored = ChemistrySession(model, genes, ControllerSpec(), checkpoint=blocked.checkpoint())
    np.testing.assert_array_equal(restored.translation_events, expected)


@pytest.mark.device
@pytest.mark.parametrize("rule", ["proportional", "majority"])
def test_resident_output_and_production_match_ordinary(model, rule):
    spec = profile_spec(
        "chemotaxis",
        backend=model.execution.backend,
        output_rule=rule,
        output_representation="bound_pair",
        marker_copies=2,
    )
    spec = replace(
        spec,
        episode=replace(spec.episode, episodes=2, horizon=4, chemical_updates=2),
        controller=replace(spec.controller, warmup=1),
    )
    genes = Stream(2).genotypes(2, 2)
    ordinary = RolloutSession(
        model,
        genes,
        spec.task,
        spec.episode,
        spec.controller,
        recording=RecordingSpec(mode="full"),
        seed=9,
    )
    ordinary.run()
    resident = RolloutSession(model, genes, spec.task, spec.episode, spec.controller, seed=9)
    assert resident._resident is not None
    resident.run()
    np.testing.assert_array_equal(resident.scores, ordinary.scores)
    np.testing.assert_array_equal(resident.snapshot()["actions"], ordinary.snapshot()["actions"])
    np.testing.assert_array_equal(
        resident.snapshot()["world"]["position"], ordinary.snapshot()["world"]["position"]
    )
    for name, value in ordinary.production_metrics().items():
        np.testing.assert_array_equal(value, resident.production_metrics()[name])
    assert np.all(resident.chemical_updates_task == 8)


@pytest.mark.device
def test_prepared_action_and_production_survive_publication_failure(model):
    spec = profile_spec(
        "cue_alternatives",
        backend=model.execution.backend,
        response_updates=2,
        output_representation="bound_pair",
    )
    genes = Stream(11).genotypes(2, 2)

    def session():
        return RolloutSession(
            model,
            genes,
            spec.task,
            spec.episode,
            spec.controller,
            seed=7,
            recording=RecordingSpec(checkpoint_interval=1),
        )

    direct = session()
    direct.run()
    interrupted = session()

    def fail(checkpoint, events):
        raise OSError("injected publication failure")

    interrupted._record_callback = fail
    with pytest.raises(OSError, match="injected"):
        interrupted.step()
    action = interrupted._pending["prepared"]["actions"].copy()
    production = interrupted.translation_events_task.copy()
    interrupted._record_callback = lambda checkpoint, events: True
    interrupted.step()
    np.testing.assert_array_equal(interrupted.snapshot()["actions"], action)
    np.testing.assert_array_equal(interrupted.translation_events_task, production)
    interrupted.run()
    np.testing.assert_array_equal(interrupted.scores, direct.scores)
    np.testing.assert_array_equal(interrupted.expected_scores, direct.expected_scores)
    np.testing.assert_array_equal(
        interrupted.translation_events_task, direct.translation_events_task
    )


def test_campaign_matrix_is_separate_and_matched(tmp_path):
    import runpy
    from pathlib import Path

    from toylife2.evolution import _initial

    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/single_cell_factorial.py")
    )
    args = example["parser"]().parse_args(["--destination", str(tmp_path), "--backend", "cpu"])
    matrix = list(example["conditions"](args))
    assert len(matrix) == 80 and not args.run
    assert {s.model.genes for _, _, s in matrix} == {1, 2, 3, 4, 5}
    matched = {}
    for _, _, spec in matrix:
        founders = _initial(spec, Stream(spec.seed), 0, spec.evolution.population_size)
        block = (spec.model.genes, spec.evolution.initial)
        previous = matched.setdefault(block, founders)
        np.testing.assert_array_equal(previous, founders)
    for genes in range(1, 6):
        np.testing.assert_array_equal(
            matched[(genes, "independent_random")][0], matched[(genes, "monomorphic_random")][0]
        )


@pytest.mark.device
@pytest.mark.integration
def test_small_campaign_records_current_methods_and_diagnostics(model, tmp_path):
    import runpy
    from pathlib import Path

    from toylife2.records import verify_record

    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/single_cell_factorial.py")
    )
    args = example["parser"]().parse_args(
        [
            "--destination",
            str(tmp_path),
            "--backend",
            model.execution.backend,
            "--genes",
            "1",
            "--initial",
            "monomorphic_random",
            "--population",
            "2",
            "--generations",
            "1",
            "--trials",
            "1",
            "--response-updates",
            "2",
        ]
    )
    _, switches, spec = next(example["conditions"](args))
    assert switches == (0, 0, 0)
    result = example["run_condition"](model, spec, tmp_path)
    assert len(result["rows"]) == 2
    assert verify_record(tmp_path / "evolution")["ok"]
    assert (tmp_path / "evolution" / "methods.md").exists()
    assert "translation_events_task" in result["rows"][-1]["production"]
    assert result["first_novel_solution"] == [None]


def test_methods_describe_executed_operation_and_objective():
    from toylife2.protocols import ObjectiveSpec

    spec = profile_spec("cue", backend="cpu")
    assert "Sampled task reward supplies reproductive weights" in methods_summary(spec)
    rollout = replace(spec, operation="rollout")
    text = methods_summary(rollout)
    assert (
        "No parent sampling" in text
        and "Sampled task reward supplies reproductive weights" not in text
    )
    custom = replace(
        spec,
        objective=ObjectiveSpec(kind="python", entrypoint="test_evaluation:minimizing_objective"),
    )
    text = methods_summary(custom)
    assert "Reproductive mapping uses objective `python`" in text
    assert "Sampled task reward supplies reproductive weights" not in text


@pytest.mark.device
def test_resident_candidate_readout_failure_retries_without_advancing(model, monkeypatch):
    spec = profile_spec("chemotaxis", backend=model.execution.backend)
    spec = replace(spec, episode=replace(spec.episode, horizon=2, chemical_updates=2))
    genes = Stream(31).genotypes(2, 2)

    def session():
        return RolloutSession(model, genes, spec.task, spec.episode, spec.controller, seed=17)

    direct = session()
    direct.step()
    retry = session()
    before = retry.chemistry.payload()
    readouts = retry.chemistry._batch.readouts

    def fail(devices=None):
        raise OSError("injected candidate readout failure")

    monkeypatch.setattr(retry.chemistry._batch, "readouts", fail)
    with pytest.raises(OSError, match="candidate"):
        retry.step()
    assert retry.tick == 0 and retry.chemistry.update == 0
    assert np.all(retry.translation_events_task == 0) and np.all(retry.chemical_updates_task == 0)
    monkeypatch.setattr(retry.chemistry._batch, "readouts", readouts)
    np.testing.assert_array_equal(retry.chemistry.payload()["counts"], before["counts"])
    retry.step()
    np.testing.assert_array_equal(retry.snapshot()["actions"], direct.snapshot()["actions"])
    np.testing.assert_array_equal(
        retry.snapshot()["world"]["position"], direct.snapshot()["world"]["position"]
    )
    np.testing.assert_array_equal(
        retry.chemistry.payload()["counts"], direct.chemistry.payload()["counts"]
    )
    np.testing.assert_array_equal(retry.translation_events_task, direct.translation_events_task)


@pytest.mark.device
def test_reset_keeps_cumulative_production_and_counts_warmup_once(model):
    spec = profile_spec("cue", backend=model.execution.backend, response_updates=3)
    spec = replace(
        spec,
        episode=replace(spec.episode, reset_per_decision=True, chemical_updates=2),
        controller=replace(spec.controller, warmup=2),
    )
    cell = RolloutSession(
        model, np.array([[1007704, 1007704]], np.uint32), spec.task, spec.episode, spec.controller
    )
    cell.run()
    assert np.all(cell.chemical_updates_warmup == 2)
    assert np.all(cell.translation_events_warmup == 4)
    assert np.all(cell.chemical_updates_task == 6)
    assert np.all(cell.translation_events_task == 12)
    assert cell.chemistry.update == 4  # Local reset clock differs from cumulative production.


@pytest.mark.device
def test_population_metrics_keep_distinct_trial_axes(model):
    from toylife2 import BatchRunner, GeneCopySpec, GenomeBatch, StructuralInterface

    spec = profile_spec(
        "cue",
        genes=1,
        backend=model.execution.backend,
        response_updates=3,
        interface=StructuralInterface(inputs={"cue": "00101110"}, targets={"act": "0111"}),
    )
    spec = replace(
        spec,
        evolution=replace(spec.evolution, population_size=2, trials=2, gene_copy=GeneCopySpec()),
    )
    with BatchRunner(model=model) as runner:
        population = runner.evolution(spec)
        population.populations[:] = 0
        population.populations[0, :, 0] = 1007704
        population.observe()
        measured = []
        for i, value in enumerate(population.evaluations):
            direct = runner.evaluator(spec).evaluate(
                GenomeBatch(population.populations[i], population.gene_counts[i])
            )
            for name in ("expected_accuracy", "translation_events_task", "chemical_updates_task"):
                assert value.metrics[name].shape == (2,)
                np.testing.assert_array_equal(value.metrics[name], direct.metrics[name])
            measured.append(value.metrics)
        assert measured[0]["expected_accuracy"].mean() != measured[1]["expected_accuracy"].mean()
        assert (
            measured[0]["translation_events_task"].mean()
            != measured[1]["translation_events_task"].mean()
        )
