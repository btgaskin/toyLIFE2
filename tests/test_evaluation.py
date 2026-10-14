import dataclasses

import numpy as np
import pytest

from toylife2.evaluation import EvaluationResult, Evaluator, ObjectiveOutput
from toylife2.protocols import EpisodeSpec, ObjectiveSpec, TaskSpec
from toylife2.results import ArrayResult
from toylife2.specs import AssaySpec, ExecutionSpec, RunSpec, SpatialSpec


class CountingModel:
    table_digest = "test-tables"

    def __init__(self):
        self.calls = []
        self.incomplete = False

    def patterns(self, genes, spec):
        self.calls.append(genes.copy())
        bits = (genes[..., 0] % 2).astype(np.uint8)
        values = np.broadcast_to(bits[..., None, None], (*genes.shape[:-1], spec.steps, spec.cells))
        return ArrayResult(
            values, np.full(genes.shape[:-1], "incomplete" if self.incomplete else "complete")
        )

    def regulatory(self, genes):
        return ArrayResult(genes[..., :1] % 2, np.full(genes.shape[:-1], "complete"))

    def metabolic(self, genes, environment):
        self.calls.append(genes.copy())
        values = np.broadcast_to(
            np.arange(len(environment)) % 2, (*genes.shape[:-1], len(environment))
        )
        status = np.full(values.shape, "complete")
        if self.incomplete:
            status[..., -1] = "incomplete"
        return ArrayResult(values, status)


def spec(**kwargs):
    return RunSpec(
        schema_version=2,
        seed=7,
        operation="wright_fisher",
        spatial=SpatialSpec(steps=2, cells=3),
        **kwargs,
    )


def test_exact_pattern_floor_duplicate_scatter_and_owned_outputs():
    model = CountingModel()
    run = spec(
        objective=ObjectiveSpec(kind="pattern_match", floor_numerator=1, floor_denominator=7),
        execution=ExecutionSpec(batch_size=1),
    )
    evaluator = Evaluator(model, run, target=np.ones((2, 3), np.uint8))
    genes = np.asarray([[[0, 0], [1, 0]], [[1, 0], [0, 0]]], np.uint32)
    ids = np.asarray([[9, 2], [7, 4]], np.uint64)
    out = evaluator.evaluate(genes, ids)
    assert out.complete and out.denominator == 42
    np.testing.assert_array_equal(out.weights, [[6, 42], [42, 6]])
    np.testing.assert_array_equal(out.scores, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(out.candidate_ids, ids)
    assert len(model.calls) == 2
    assert evaluator.metrics["duplicate_requests"] == 2
    with pytest.raises(ValueError):
        out.scores[0, 0] = 3
    ids[:] = 1
    assert out.candidate_ids[0, 0] == 9
    evaluator.evaluate(genes[::-1])
    assert len(model.calls) == 2


def test_incomplete_assays_are_invalid_and_never_cached():
    model = CountingModel()
    model.incomplete = True
    evaluator = Evaluator(model, spec(), target=np.ones((2, 3), np.uint8))
    genes = np.zeros((3, 2), np.uint32)
    result = evaluator.evaluate(genes)
    assert not result.complete and np.all(np.isnan(result.scores))
    with pytest.raises(RuntimeError, match="incomplete"):
        result.require_complete()
    assert not evaluator.cache
    model.incomplete = False
    assert evaluator.evaluate(genes).complete
    assert len(model.calls) == 2


def test_metabolic_coverage_counts_supplied_repetitions_and_all_validity():
    model = CountingModel()
    run = spec(
        objective=ObjectiveSpec(kind="metabolic_coverage"),
        assay=AssaySpec(environment=("0000", "0000", "1111")),
    )
    result = Evaluator(model, run).evaluate(np.zeros((2, 2), np.uint32))
    assert result.denominator == 3
    np.testing.assert_array_equal(result.weights, [1, 1])
    model.incomplete = True
    result = Evaluator(model, run).evaluate(np.zeros((2, 2), np.uint32))
    assert not result.complete


def minimizing_objective(genes, phenotype, config):
    assert not genes.flags.writeable
    with pytest.raises(TypeError):
        config["new"] = 2
    scores = genes[..., 0].astype(float) - config["shift"]
    return ObjectiveOutput(
        scores,
        weights=(10 - genes[..., 0]).astype(np.int64),
        denominator=10,
        selection_fitness=np.maximum(10 - scores, 0),
        metrics={"cost": scores},
    )


def test_custom_minimization_has_explicit_independent_selection_mapping():
    run = spec(
        objective=ObjectiveSpec(
            kind="python",
            direction="minimize",
            entrypoint="test_evaluation:minimizing_objective",
            config={"shift": 2},
        ),
        assay=AssaySpec(phenotype="regulatory"),
    )
    genes = np.asarray([[2, 0], [1, 0], [2, 0]], np.uint32)
    result = Evaluator(CountingModel(), run).evaluate(genes)
    np.testing.assert_array_equal(result.scores, [0, -1, 0])
    np.testing.assert_array_equal(result.weights, [8, 9, 8])
    np.testing.assert_array_equal(result.metrics["cost"], result.scores)
    assert result.direction == "minimize"


def test_cache_identity_includes_objective_target_and_panel():
    model = CountingModel()
    evaluator = Evaluator(model, spec(), target=np.ones((2, 3), np.uint8))
    genes = np.zeros((1, 2), np.uint32)
    evaluator.evaluate(genes, panel=0)
    evaluator.evaluate(genes, panel=1)
    assert len(model.calls) == 2
    other = Evaluator(
        model,
        spec(objective=ObjectiveSpec(kind="pattern_match")),
        target=np.zeros((2, 3), np.uint8),
    )
    assert evaluator.context != other.context
    other.evaluate(genes)
    assert len(model.calls) == 3


def test_invalid_exact_selection_and_result_shape_fail_early():
    with pytest.raises(ValueError, match="nonnegative"):
        EvaluationResult(
            np.asarray([1.0]),
            weights=np.asarray([-1]),
            denominator=3,
            status=np.asarray(["complete"]),
        )
    with pytest.raises(ValueError, match="status"):
        EvaluationResult(np.zeros(3))


class FakeRollout:
    def __init__(self, model, genes, task, protocol, controller, **kwargs):
        self.genes = genes
        self.status = np.full((*genes.shape[:-1], protocol.episodes), "complete")
        self.descriptor = ()

    def run(self):
        pass

    def aggregate_scores(self):
        return np.full(self.genes.shape[:-1], 0.125)


def test_bounded_reward_quantization_and_floor_are_exact():
    run = spec(
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(episodes=2),
        objective=ObjectiveSpec(
            kind="bounded_reward", quantization=16, floor_numerator=1, floor_denominator=3
        ),
    )
    result = Evaluator(CountingModel(), run, rollout_factory=FakeRollout).evaluate(
        np.zeros((2, 2), np.uint32)
    )
    assert result.denominator == 48
    np.testing.assert_array_equal(result.weights, [16, 16])
    np.testing.assert_array_equal(result.scores, [0.125, 0.125])
    np.testing.assert_array_equal(result.selection_fitness, [1 / 3, 1 / 3])


@pytest.fixture(scope="module")
def molecular_model(tmp_path_factory, canonical_tables):
    from toylife2 import Model

    return Model(
        tables=canonical_tables,
        execution=ExecutionSpec(cache_dir=str(tmp_path_factory.mktemp("eval-cpu"))),
    )


@pytest.mark.device
def test_rollout_discrete_descriptor_retains_candidate_axes(molecular_model):
    run = dataclasses.replace(
        spec(),
        operation="neutral_walk",
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=2, episodes=2),
        objective=ObjectiveSpec(kind="descriptor"),
    )
    out = Evaluator(molecular_model, run).evaluate(np.zeros((2, 3, 2), np.uint32))
    assert out.complete and out.descriptor.shape[:2] == (2, 3)
    assert out.descriptor.dtype.kind in "biuUS"


@pytest.mark.device
def test_contextual_roster_balanced_roles_and_exact_denominator(molecular_model):
    run = spec(
        task=TaskSpec(kind="signalling_game", config={"rounds": 2}),
        episode=EpisodeSpec(horizon=4, episodes=2),
        objective=ObjectiveSpec(kind="bounded_reward", scope="population"),
    )
    evaluator = Evaluator(molecular_model, run)
    genes = np.zeros((3, 2), np.uint32)
    out = evaluator.evaluate_population(genes, replicate=2, generation=0)
    assert out.complete and out.scores.shape == (3,)
    assert out.denominator == 16  # 2 scenarios * 2 rounds * 2 opponents * 2 roles
    assert out.metrics["opponents"].shape == (3, 2)
    for focal, opponents in enumerate(out.metrics["opponents"]):
        assert focal not in opponents and len(np.unique(opponents)) == 2
    assert not evaluator.cache and not evaluator.pending


@pytest.mark.device
def test_nested_world_chemical_caps_and_checkpoint_resume(molecular_model):
    from toylife2.population import PopulationSession
    from toylife2.protocols import WorkLimits
    from toylife2.specs import EvolutionSpec

    run = spec(
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=2, chemical_updates=2),
        objective=ObjectiveSpec(kind="bounded_reward"),
        evolution=EvolutionSpec(trials=2, population_size=2, generations=0),
    )
    full = PopulationSession(molecular_model, run)
    assert full.run().complete
    partial = PopulationSession(molecular_model, run, limits=WorkLimits(max_chemical_updates=2))
    assert partial.run().status == "partial"
    assert partial.counters["invocation"]["chemical_updates"] == 2
    assert partial.counters["invocation"]["world_ticks"] == 0
    assert partial.evaluator.pending
    cp = partial.checkpoint()
    resumed = PopulationSession(molecular_model, run, checkpoint=cp)
    assert resumed.run().complete
    assert (
        resumed.counters["total"]["chemical_updates"] == full.counters["total"]["chemical_updates"]
    )
    np.testing.assert_array_equal(
        resumed.snapshot()["arrays"]["history"], full.snapshot()["arrays"]["history"]
    )


@pytest.mark.device
def test_contextual_half_barrier_checkpoint_keeps_completed_sender_roles(molecular_model):
    from toylife2.population import PopulationSession
    from toylife2.protocols import WorkLimits
    from toylife2.specs import EvolutionSpec

    run = spec(
        task=TaskSpec(kind="signalling_game", config={"rounds": 2}),
        episode=EpisodeSpec(horizon=4, episodes=2),
        objective=ObjectiveSpec(kind="bounded_reward", scope="population"),
        evolution=EvolutionSpec(trials=1, population_size=3, generations=0),
    )
    full = PopulationSession(molecular_model, run)
    assert full.run().complete
    capped = PopulationSession(molecular_model, run, limits=WorkLimits(max_world_ticks=48))
    assert capped.run().status == "partial"
    assert capped.observed[0] == -1
    assert capped.counters["invocation"]["world_ticks"] == 48
    assert (
        sum(np.all(session.status == "complete") for session in capped.evaluator.pending.values())
        == 1
    )
    resumed = PopulationSession(molecular_model, run, checkpoint=capped.checkpoint())
    assert resumed.run().complete
    assert resumed.counters["total"] == full.counters["total"]
    np.testing.assert_array_equal(resumed.evaluations[0].weights, full.evaluations[0].weights)


@pytest.mark.device
def test_pending_candidate_chunks_survive_permutation_and_retain_batching(molecular_model):
    from toylife2.protocols import WorkLimits

    run = spec(
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=3),
        objective=ObjectiveSpec(kind="bounded_reward"),
        execution=ExecutionSpec(batch_size=2),
        limits=WorkLimits(max_world_ticks=2),
    )
    genes = np.asarray([[2, 0], [0, 0], [1, 0]], np.uint32)
    evaluator = Evaluator(molecular_model, run)
    out = evaluator.evaluate(genes, [8, 4, 2])
    assert not out.complete
    assert [args["genes"].shape[0] for args in evaluator.pending_arguments.values()] == [2, 1]
    data = evaluator.checkpoint_payload()
    restored = Evaluator(molecular_model, run)
    restored.restore_payload(data)
    restored.set_work_limits(WorkLimits())
    result = restored.evaluate(genes[::-1], [2, 4, 8])
    assert result.complete
    np.testing.assert_array_equal(result.candidate_ids, [2, 4, 8])
    assert not restored.pending
    assert restored.metrics["chunks"] == 2


@pytest.mark.device
def test_floor_applied_after_contextual_role_mean(molecular_model):
    run = spec(
        task=TaskSpec(kind="signalling_game", config={"rounds": 2}),
        episode=EpisodeSpec(horizon=4, episodes=1),
        objective=ObjectiveSpec(
            kind="bounded_reward", scope="population", floor_numerator=3, floor_denominator=4
        ),
    )
    out = Evaluator(molecular_model, run).evaluate_population(
        np.zeros((2, 2), np.uint32), replicate=5, generation=0
    )
    assert out.complete
    assert out.denominator == 8
    np.testing.assert_array_equal(out.weights, np.maximum(out.scores * 8, 6).astype(np.int64))


@pytest.mark.device
def test_small_logical_quota_admits_narrow_candidate_chunks_and_eventually_finishes(
    molecular_model,
):
    from toylife2.population import PopulationSession
    from toylife2.protocols import WorkLimits
    from toylife2.specs import EvolutionSpec

    run = spec(
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=2),
        objective=ObjectiveSpec(kind="bounded_reward"),
        evolution=EvolutionSpec(trials=2, population_size=2, generations=0),
    )
    full = PopulationSession(molecular_model, run)
    assert full.run().complete
    checkpoint = None
    for _ in range(8):
        current = PopulationSession(
            molecular_model, run, checkpoint=checkpoint, limits=WorkLimits(max_world_ticks=1)
        )
        report = current.run()
        assert current.counters["invocation"]["world_ticks"] <= 1
        checkpoint = current.checkpoint()
        if report.complete:
            break
    assert report.complete
    assert current.counters["total"] == full.counters["total"]


def test_assay_limits_bind_copies_without_mutating_shared_model():
    from types import SimpleNamespace

    from toylife2.protocols import WorkLimits

    model = CountingModel()
    model.execution = ExecutionSpec(max_steps=999)
    model.engine = SimpleNamespace(execution=model.execution)
    original = model.patterns
    seen = []

    class BoundedModel(CountingModel):
        def patterns(self, genes, spatial):
            seen.append((self.execution.max_steps, self.engine.execution.max_steps))
            return original(genes, spatial)

    model.__class__ = BoundedModel
    run = spec(limits=WorkLimits(max_assay_steps=2))
    out = Evaluator(model, run, target=np.zeros((2, 3), np.uint8)).evaluate(
        np.zeros((1, 2), np.uint32)
    )
    assert out.complete and seen == [(2, 2)]
    assert model.execution.max_steps == model.engine.execution.max_steps == 999


def test_component_input_copy_paths_retain_context_and_population_checkpoint_identity(tmp_path):
    from toylife2.population import PopulationSession
    from toylife2.specs import EvolutionSpec

    left = tmp_path / "original.txt"
    right = tmp_path / "copied.txt"
    left.write_text("same input")
    right.write_text("same input")
    base = spec(
        assay=AssaySpec(phenotype="regulatory"),
        objective=ObjectiveSpec(
            kind="python",
            entrypoint="test_evaluation:minimizing_objective",
            config={"shift": 2, "inputs": {"data": str(left)}},
        ),
        evolution=EvolutionSpec(trials=1, population_size=1, generations=0),
        genotypes=("0" * 40,),
    )
    copied = dataclasses.replace(
        base,
        objective=dataclasses.replace(
            base.objective, config={"shift": 2, "inputs": {"data": str(right)}}
        ),
    )
    model = CountingModel()
    first = PopulationSession(model, base)
    assert first.run().complete
    second = PopulationSession(model, copied, checkpoint=first.checkpoint())
    assert first.identity == second.identity
    assert first.evaluator.context == second.evaluator.context
    right.write_text("changed input")
    assert Evaluator(model, copied).context != first.evaluator.context


@pytest.mark.device
def test_full_nested_trace_retains_world_chemical_rows_and_cache_requests(molecular_model):
    import json

    from toylife2.population import PopulationSession
    from toylife2.specs import EvolutionSpec, RecordingSpec

    run = spec(
        genotypes=("0" * 40,),
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(horizon=3, chemical_updates=2),
        objective=ObjectiveSpec(kind="bounded_reward"),
        evolution=EvolutionSpec(trials=1, population_size=2, generations=1, mutation_probability=0),
        recording=RecordingSpec(mode="full"),
    )
    session = PopulationSession(molecular_model, run)
    events = []

    def accept(checkpoint, rows):
        json.dumps(rows, allow_nan=False)
        events.extend(rows)
        return True

    session._record_callback = accept
    assert session.run().complete
    nested = [row["event"] for row in events if row["type"] == "rollout_transition"]
    assert sum(row["type"] == "world_tick" for row in nested) == 3
    assert sum(row["type"] == "chemical_update" for row in nested) == 6
    requests = [row for row in events if row["type"] == "evaluation_request"]
    assert len(requests) == 2
    assert session.counters["total"]["world_ticks"] == 3


def test_memory_admission_splits_without_changing_candidate_results():
    attempts = []

    class LimitedRollout(FakeRollout):
        def __init__(self, model, genes, *args, **kwargs):
            attempts.append(len(genes))
            if len(genes) > 2:
                raise MemoryError("cohort exceeds budget")
            super().__init__(model, genes, *args, **kwargs)

        def aggregate_scores(self):
            return self.genes[:, 0] / 16

    run = spec(
        task=TaskSpec(kind="temporal_probe"),
        episode=EpisodeSpec(episodes=2),
        objective=ObjectiveSpec(kind="bounded_reward"),
        execution=ExecutionSpec(batch_size=8),
    )
    genes = np.asarray([[i, 0] for i in [3, 1, 5, 2, 1, 4]], np.uint32)
    evaluator = Evaluator(CountingModel(), run, rollout_factory=LimitedRollout)
    result = evaluator.evaluate(genes)
    np.testing.assert_array_equal(result.scores, genes[:, 0] / 16)
    assert result.complete
    assert evaluator.metrics["new_genotype_evaluations"] == 5
    assert evaluator.metrics["memory_batch_retries"] == 2
    assert attempts == [5, 3, 2, 1, 2]
    count = len(attempts)
    evaluator.evaluate(genes)
    assert len(attempts) == count


def test_memory_failure_during_simulation_is_not_retried():
    attempts = []

    class FailedRollout(FakeRollout):
        def run(self):
            attempts.append(len(self.genes))
            raise MemoryError("runtime allocation")

    run = spec(task=TaskSpec(kind="temporal_probe"), objective=ObjectiveSpec(kind="bounded_reward"))
    evaluator = Evaluator(CountingModel(), run, rollout_factory=FailedRollout)
    with pytest.raises(MemoryError, match="runtime allocation"):
        evaluator.evaluate(np.asarray([[0, 0], [1, 0]], np.uint32))
    assert attempts == [2]


def test_single_candidate_admission_failure_is_reported():
    def failed_factory(*args, **kwargs):
        raise MemoryError("single cell exceeds budget")

    run = spec(task=TaskSpec(kind="temporal_probe"), objective=ObjectiveSpec(kind="bounded_reward"))
    evaluator = Evaluator(CountingModel(), run, rollout_factory=failed_factory)
    with pytest.raises(MemoryError, match="single cell exceeds budget"):
        evaluator.evaluate(np.zeros((1, 2), np.uint32))


def test_panel_cache_pruning_preserves_current_and_pending_panels():
    evaluator = Evaluator(
        CountingModel(),
        spec(task=TaskSpec(kind="temporal_probe"), objective=ObjectiveSpec(kind="bounded_reward")),
        rollout_factory=FakeRollout,
    )
    genes = np.zeros((1, 2), np.uint32)
    for panel in [0, 1, 2]:
        evaluator.evaluate(genes, panel=panel)
    evaluator.pending_arguments["pending"] = {"panel": 1, "agent_axis": None, "genes": genes}
    evaluator.retain_cache_panels([2])
    assert len(evaluator.cache) == 2
    before = evaluator.metrics["new_genotype_evaluations"]
    evaluator.evaluate(genes, panel=2)
    assert evaluator.metrics["new_genotype_evaluations"] == before
    evaluator.evaluate(genes, panel=0)
    assert evaluator.metrics["new_genotype_evaluations"] == before + 1


@pytest.mark.device
def test_memory_fallback_matches_real_chemotaxis_evaluation(molecular_model):
    from toylife2.protocols import StructuralInterface
    from toylife2.rollout import RolloutSession

    run = spec(
        task=TaskSpec(kind="chemotaxis"),
        episode=EpisodeSpec(horizon=8, episodes=3),
        controller=StructuralInterface(
            inputs={"cue": "0001"}, targets={"motor": "1111"}
        ).controller(),
        objective=ObjectiveSpec(kind="bounded_reward"),
    )
    genes = np.asarray([[907050, 1048465], [1005689, 644498], [907050, 1048465]], np.uint32)
    expected = Evaluator(molecular_model, run).evaluate(genes, panel=7)

    def constrained_factory(model, batch, *args, **kwargs):
        if len(batch) > 1:
            raise MemoryError("test admission cap")
        return RolloutSession(model, batch, *args, **kwargs)

    evaluator = Evaluator(molecular_model, run, rollout_factory=constrained_factory)
    actual = evaluator.evaluate(genes, panel=7)
    for name in ["scores", "weights", "selection_fitness", "status", "candidate_ids"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.denominator == expected.denominator
    assert evaluator.metrics["memory_batch_retries"] == 1


def test_shared_panel_cache_retains_sibling_scores():
    evaluator = Evaluator(
        CountingModel(),
        spec(task=TaskSpec(kind="temporal_probe"), objective=ObjectiveSpec(kind="bounded_reward")),
        rollout_factory=FakeRollout,
    )
    genes = np.zeros((1, 2), np.uint32)
    for panel in [0, 1]:
        evaluator.evaluate(genes, panel=panel)
    evaluator._shared_cache = True
    evaluator.retain_cache_panels([1])
    assert len(evaluator.cache) == 2


@pytest.mark.device
def test_memory_fallback_checkpoint_resumes_interrupted_cohorts(molecular_model):
    from toylife2.protocols import StructuralInterface, WorkLimits
    from toylife2.rollout import RolloutSession

    run = spec(
        task=TaskSpec(kind="chemotaxis"),
        episode=EpisodeSpec(horizon=4, episodes=3),
        controller=StructuralInterface(
            inputs={"cue": "0001"}, targets={"motor": "1111"}
        ).controller(),
        objective=ObjectiveSpec(kind="bounded_reward"),
    )
    genes = np.asarray([[907050, 1048465], [1005689, 644498]], np.uint32)
    expected = Evaluator(molecular_model, run).evaluate(genes, panel=7)

    def constrained_factory(model, batch, *args, **kwargs):
        if len(batch) > 1:
            raise MemoryError("test admission cap")
        return RolloutSession(model, batch, *args, **kwargs)

    partial = Evaluator(molecular_model, run, rollout_factory=constrained_factory)
    partial.set_work_limits(WorkLimits(max_chemical_updates=4))
    assert not partial.evaluate(genes, panel=7).complete
    assert partial.pending
    resumed = Evaluator(molecular_model, run, rollout_factory=constrained_factory)
    resumed.restore_payload(partial.checkpoint_payload())
    actual = resumed.evaluate(genes, panel=7)
    assert actual.complete
    np.testing.assert_array_equal(actual.scores, expected.scores)
    np.testing.assert_array_equal(actual.weights, expected.weights)
