from dataclasses import replace

import numpy as np
import pytest

from toylife2 import Model, StructuralInterface
from toylife2.chemistry import ChemistrySession
from toylife2.config import apply_overrides, dump_spec, mapping_spec
from toylife2.protocols import EpisodeSpec, TaskSpec, WorkLimits
from toylife2.reference import CellState, ReferenceModel
from toylife2.rollout import RolloutSession
from toylife2.specs import ExecutionSpec, RecordingSpec, RunSpec, plain

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def model(canonical_tables):
    return Model(tables=canonical_tables, execution=ExecutionSpec(backend="cpu"))


def interface(two=False):
    return StructuralInterface(
        inputs={"cue_a": "0001", "cue_b": "0110"} if two else {"cue": "0001"},
        targets={"act_a": "1111", "act_b": "1001"} if two else {"act": "1111"},
    )


def test_spec_roundtrip_and_target_override():
    import tomllib

    spec = RunSpec(schema_version=2, seed=1, controller=interface().controller())
    assert mapping_spec(tomllib.loads(dump_spec(spec))) == spec
    assert "ports" not in plain(spec)["controller"]
    assert "interface" not in plain(RunSpec(schema_version=2, seed=1))["controller"]
    changed = apply_overrides(spec, ["controller.interface.target_copies.act=2"])
    assert changed.controller.interface.target_copies["act"] == 2
    with pytest.raises(ValueError, match="distinct"):
        StructuralInterface(inputs={"a": "0001", "b": "1000"}, targets={"o": "1111"})


def test_counts_and_two_channels_match_reference(model):
    ports = interface(True)
    genes = np.random.default_rng(14).integers(0, 1 << 20, size=(4, 2), dtype=np.int32)
    session = ChemistrySession(model, genes, ports.controller())
    oracle = ReferenceModel(model.tables)
    states = [CellState() for _ in genes]
    for tick in range(8):
        counts = ports.encode({"cue_a": np.arange(4) >> 1, "cue_b": np.arange(4) & 1})
        if tick == 4:
            counts = ports.encode({"cue_a": 0, "cue_b": 0})
        session.step(counts)
        for lane, row in enumerate(genes):
            for molecule, value in counts.items():
                states[lane].metabolites[molecule] = int(np.broadcast_to(value, (4,))[lane])
            result = oracle.step(row, states[lane])
            states[lane] = result.state
            assert session.inventories[lane].key() == result.state.key()
        assert set(ports.outputs(session.last_readouts)) == {"act_a", "act_b"}
    for bad in ({"cue_a": 1}, {"cue_a": -1, "cue_b": 0}, {"cue_a": 1.0, "cue_b": 0}):
        with pytest.raises(ValueError):
            ports.encode(bad)


def test_bath_warmup_block_and_full_recording(model):
    ports = interface()
    genes = np.array([907050, 1048465], np.int32)
    warmed = ChemistrySession(model, genes, ports.controller(warmup=8))
    full = ChemistrySession(model, genes, ports.controller(), recording=RecordingSpec(mode="full"))
    for _ in range(8):
        full.step()
    assert warmed.inventories[0].key() == full.inventories[0].key()
    for name, value in warmed.last_readouts.items():
        np.testing.assert_array_equal(value, full.last_readouts[name])
    assert full.last_readouts["target_free:act"] + full.last_readouts["act"] <= 1


def test_exact_structural_readouts_and_aliases(model):
    ports = interface()
    binder = ChemistrySession(model, [0], ports.controller(initial_proteins=((47, 1),)))
    binder.step(ports.encode({"cue": 0}))
    assert ports.diagnostics(binder.last_readouts)["act"] == {"bound": 1, "free": 0, "fission": 0}
    # Admitted preassembled reaction, independent of empty-start reachability.
    cleavage = ChemistrySession(
        model,
        [0],
        ports.controller(initial_proteins=((470, 1),), initial_bound=(("D", 345, 351, "1111", 1),)),
    )
    cleavage.step(ports.encode({"cue": 0}))
    assert ports.diagnostics(cleavage.last_readouts)["act"] == {"bound": 0, "free": 1, "fission": 1}
    assert cleavage.inventories[0].metabolites["11"] == 2
    canonical = ChemistrySession(model, [0], ports.controller())
    alias = ChemistrySession(model, [0], ports.controller())
    canonical.step({"0001": 1})
    alias.step({("M", "1000"): 1})
    assert canonical.inventories[0].key() == alias.inventories[0].key()
    for bad in ({"0001": 1, "1000": 1}, {"1111": 0}, {"0001": 1.5}):
        with pytest.raises(ValueError):
            alias.step(bad)
    similar = StructuralInterface(inputs={"cue": "0001"}, targets={"act": "0010"})
    info = similar.endpoint_information(model.tables)["similarities"][0]
    assert info["identical_protein_contacts"] and info["identical_dimer_contacts"]


@pytest.mark.parametrize(
    "profile,field",
    [
        ("basic", "fluctuating"),
        ("cue_alternatives", "fluctuating"),
        ("chemotaxis", "fluctuating"),
        ("chemotaxis", "source"),
    ],
)
@pytest.mark.qualification
def test_recorded_molecular_experiment(tmp_path, profile, field):
    import hashlib
    import json

    from toylife2.molecular_experiments import profile_spec, run_experiment
    from toylife2.records import verify_record

    spec = profile_spec(profile, field=field)
    spec = replace(
        spec,
        execution=replace(spec.execution, cpu_threads=None),
        evolution=replace(spec.evolution, generations=1, population_size=2),
        recording=replace(spec.recording, trace_budget_mib=16),
        episode=replace(spec.episode, horizon=3 if profile == "chemotaxis" else 1),
    )
    root = run_experiment(
        spec, tmp_path / "experiment", animate=profile == "chemotaxis", frame_stride=2
    )
    summary = json.loads((root / "experiment.json").read_text())
    assert summary["behaviour_status"] == summary["evolution_status"] == "complete"
    assert (root / "report.html").is_file()
    for name, digest in summary["artifacts"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert verify_record(root / "evolution")["ok"]
    assert verify_record(root / "heldout")["ok"]
    from toylife2.config import load_spec

    held_record = load_spec(root / "heldout" / "spec.toml")
    assert held_record.recording.trace_budget_mib == 16
    assert held_record.recording.mode == spec.recording.mode
    with np.load(root / "history_probes.npz", allow_pickle=False) as saved:
        assert saved["inputs"].shape == (32, 4)
    if profile == "chemotaxis":
        with np.load(root / "behaviour.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(
                summary["mean_scores"], np.mean(saved["reward"], axis=(0, 2))
            )
            assert saved["sensed_exposure"].shape == (3, 2, 32)
        assert summary["reset_status"] == "complete"
        assert (root / "animation.html").is_file()
        assert verify_record(root / "initial_heldout")["ok"]
        with (
            np.load(root / "initial_behaviour.npz", allow_pickle=False) as initial,
            np.load(root / "behaviour.npz", allow_pickle=False) as final,
        ):
            np.testing.assert_array_equal(initial["initial_position"], final["initial_position"])
            np.testing.assert_array_equal(initial["initial_theta"], final["initial_theta"])


@pytest.mark.integration
def test_interrupted_experiment_retains_checkpoint(tmp_path):
    from toylife2.molecular_experiments import profile_spec, run_experiment
    from toylife2.records import verify_record

    spec = profile_spec()
    spec = replace(
        spec,
        execution=replace(spec.execution, cpu_threads=None),
        limits=WorkLimits(max_evolution_updates=0),
    )
    root = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="completed population"):
        run_experiment(spec, root)
    assert verify_record(root / "evolution")["ok"]
    assert not (root / "selected_genomes.npy").exists()


@pytest.mark.integration
def test_extended_evolution_matches_uninterrupted(model, tmp_path):
    from toylife2.molecular_experiments import profile_spec, run_experiment
    from toylife2.population import PopulationSession
    from toylife2.records import inspect_record
    from toylife2.workspaces import check_workspace, init_workspace, list_records

    spec = profile_spec()
    spec = replace(
        spec,
        execution=replace(spec.execution, cpu_threads=None),
        evolution=replace(spec.evolution, generations=1, population_size=3),
    )
    workspace = init_workspace(tmp_path / "workspace")
    first = run_experiment(spec, workspace=workspace, name="first")
    extended_spec = replace(spec, evolution=replace(spec.evolution, generations=3))
    extended = run_experiment(
        extended_spec, workspace=workspace, name="extended", resume_from=first
    )
    full = PopulationSession(model, extended_spec)
    full.run()
    manifest = inspect_record(extended / "evolution")
    saved = manifest["artifacts"]["arrays"]["arrays_populations"]["path"]
    np.testing.assert_array_equal(np.load(extended / "evolution" / saved), full.populations)
    assert len(list_records(workspace)) == 2
    assert check_workspace(workspace)["ok"]
    changed = replace(
        extended_spec, evolution=replace(extended_spec.evolution, mutation_probability=0.2)
    )
    with pytest.raises(ValueError, match="preserves"):
        run_experiment(changed, tmp_path / "invalid", resume_from=first)


def test_fluctuating_field_correlation_and_sensor_law():
    from toylife2.molecular_worlds import Chemotaxis
    from toylife2.worlds import ScenarioRandom

    config = TaskSpec(
        kind="chemotaxis",
        config={
            "modes": [[1, 0]],
            "speed": 0,
            "static": False,
            "hazard_zero": 0,
            "hazard_one": 0,
            "rotational_diffusion": 0,
            "motion_max_substep": 3 * np.log(2),
        },
    ).config
    size = 20000
    rng = ScenarioRandom(19, 0, np.arange(size, dtype=np.uint64))
    world = Chemotaxis(config, (size,), 1, rng, 3 * np.log(2), False)
    state = world.reset()
    current = world.field(state) - config["mean"]
    next_state, _, _ = world.advance(state, np.zeros(size, np.int32))
    later = world.field(next_state) - config["mean"]
    # Motion is frozen: independent phase diffusion gives exp(-dt/tau)
    # covariance at a fixed point, including over this long observation interval.
    assert np.mean(current * later) / np.mean(current * current) == pytest.approx(0.5, abs=0.025)
    cue = world.observe(state)["inputs"]["cue"]
    concentration = world.field(state)
    assert np.mean(cue) == pytest.approx(
        np.mean(concentration / (config["sensor_scale"] + concentration)), abs=0.015
    )


def test_exhaustive_two_channel_scoring_and_resume(model):
    task = TaskSpec(
        kind="cue_reading",
        config=dict(
            fixed_map=[0, 1, 3, 2],
            input_channels=["cue_a", "cue_b"],
            action_outputs=["act_a", "act_b"],
        ),
    )
    protocol = EpisodeSpec(horizon=1, episodes=4, chemical_updates=8)
    genes = np.array([[907050, 1048465]], np.int32)
    spec = interface(True).controller()
    uninterrupted = RolloutSession(model, genes, task, protocol, spec)
    uninterrupted.run()
    partial = RolloutSession(
        model, genes, task, protocol, spec, limits=WorkLimits(max_chemical_updates=12)
    )
    partial.step()
    resumed = RolloutSession(model, genes, task, protocol, spec, checkpoint=partial.checkpoint())
    resumed.run()
    np.testing.assert_array_equal(uninterrupted.scores, resumed.scores)
    for name, value in uninterrupted.chemistry.last_readouts.items():
        np.testing.assert_array_equal(value, resumed.chemistry.last_readouts[name])
    world = uninterrupted.world
    state = world.reset()
    np.testing.assert_array_equal(world.observe(state)["inputs"]["cue_a"], [[0, 0, 1, 1]])
    np.testing.assert_array_equal(world.observe(state)["inputs"]["cue_b"], [[0, 1, 1, 0]])
    _, reward, _ = world.advance(state, np.array([[0, 1, 2, 3]]))
    assert np.all(reward == 1)
    _, reward, channels = world.advance(state, np.array([[0, 1, 3, 2]]))
    np.testing.assert_array_equal(reward, [[1, 1, 0, 0]])
    assert np.all(channels["component_correct"]["act_a"])


@pytest.mark.parametrize("field", ["fluctuating", "source"])
def test_chemotaxis_fields_axes_and_scenario_batching(model, field):
    ports = StructuralInterface(inputs={"cue": "0001"}, targets={"motor": "1111"})
    task = TaskSpec(kind="chemotaxis", config={"field": field})
    assert task.config["static"] is True
    protocol = EpisodeSpec(horizon=4, episodes=3, chemical_updates=2)
    genes = np.array([[907050, 1048465], [1005689, 644498]], np.int32)
    combined = RolloutSession(model, genes, task, protocol, ports.controller(), seed=42)
    combined.run()
    for lane, row in enumerate(genes):
        single = RolloutSession(model, row, task, protocol, ports.controller(), seed=42)
        single.run()
        np.testing.assert_array_equal(combined.scores[lane], single.scores)
    world = combined.world
    state = world.reset()
    original = world.field(state)
    shifted = {**state, "position": state["position"] + task.config["domain"]}
    np.testing.assert_allclose(world.field(shifted), original, atol=1e-14)
    assert np.all((original > 0) & (original <= 1))
    assert np.all((combined.scores >= 0) & (combined.scores <= 1))
    frozen = replace(task, config={**task.config, "static": True})
    fixed = RolloutSession(model, genes, frozen, protocol, ports.controller())
    before = fixed.world.reset()
    after, _, _ = fixed.world.advance(before, np.zeros(fixed.world_shape, np.int32))
    np.testing.assert_array_equal(before["phases"], after["phases"])


def test_seeded_port_pools(model):
    from toylife2.molecular_experiments import sample_interface

    first, metadata = sample_interface(model.tables, ["cue"], ["act"])
    again, repeat = sample_interface(model.tables, ["cue"], ["act"])
    assert first == again and metadata == repeat
    assert len(metadata["candidates"]) == 9
    assert "0000" not in metadata["candidates"]
    assert all(len(m) == 4 for m in metadata["candidates"])
    long, broader = sample_interface(
        model.tables, ["a", "b"], ["x", "y"], pool="all_lengths_active"
    )
    assert any(len(m) == 8 for m in broader["candidates"])
    assert len(set([*long.inputs.values(), *long.targets.values()])) == 4


def test_repeated_binary_positive_control(model):
    ports = StructuralInterface(inputs={"cue": "00000011"}, targets={"act": "1111"})
    genes = np.array([360861, 503167], np.int32)
    task = TaskSpec(kind="cue_reading", config={"fixed_map": [1, 0]})
    cell = RolloutSession(
        model,
        genes,
        task=task,
        protocol=EpisodeSpec(horizon=8, episodes=2),
        controller=ports.controller(),
    )
    actions = []
    for _ in range(8):
        report = cell.step()
        actions.append(report.transition["actions"].tolist())
    np.testing.assert_array_equal(np.asarray(actions), [[0, 0], [0, 1]] * 4)
    np.testing.assert_allclose(cell.scores, [1, 0.5])
    settled = RolloutSession(
        model,
        genes,
        task=replace(task, config={**task.config, "settling_steps": 1}),
        protocol=EpisodeSpec(horizon=8, episodes=2),
        controller=ports.controller(),
    )
    settled.run()
    np.testing.assert_allclose(settled.scores, [1, 4 / 7])
    legacy = RolloutSession(
        model,
        genes,
        task=task,
        protocol=EpisodeSpec(horizon=1, episodes=2, chemical_updates=8),
        controller=ports.controller(),
    )
    legacy.run()
    np.testing.assert_array_equal(legacy.scores, [1, 1])


def test_repeated_binary_constant_and_joint_scores():
    from toylife2.molecular_worlds import CueReading
    from toylife2.worlds import ScenarioRandom

    for channels in (1, 2):
        states = 1 << channels
        config = {
            "fixed_map": list(range(states)),
            "input_channels": [f"in{i}" for i in range(channels)],
            "action_outputs": [f"out{i}" for i in range(channels)],
            "settling_steps": 2,
        }
        world = CueReading(
            config,
            (states,),
            1,
            ScenarioRandom(1, 0, np.arange(states, dtype=np.uint64)),
            0.05,
            False,
        )
        state = world.reset()
        judged_rewards = []
        for tick in range(6):
            state, reward, extra = world.advance(state, np.zeros(states, np.int32))
            assert extra["judged"] == (tick >= 2)
            if extra["judged"]:
                judged_rewards.append(reward)
        assert np.mean(judged_rewards) == 1 / states


def test_interface_seed_independent_of_evolution_and_explicit_override():
    from toylife2.molecular_experiments import profile_spec

    first = profile_spec(seed=1)
    second = profile_spec(seed=987)
    assert first.controller.interface == second.controller.interface
    explicit = StructuralInterface(inputs={"cue": "00000011"}, targets={"act": "1111"})
    selected = profile_spec("cue", interface=explicit, response_updates=16, settling_updates=2)
    assert selected.controller.interface == explicit
    assert selected.episode.horizon == 18
    assert selected.episode.chemical_updates == 1
    assert selected.task.config["settling_steps"] == 2
    assert "port_sampling" not in selected.task.config


@pytest.mark.parametrize("profile", ["basic", "cue", "cue4", "chemotaxis", "signalling_game"])
def test_fixed_eight_unit_cues_preserve_other_ports(profile):
    from toylife2.molecular_experiments import profile_spec

    current = profile_spec(profile)
    previous = profile_spec(profile, cue_length=4)
    interface = current.controller.interface
    assert interface.targets == previous.controller.interface.targets
    for name, molecule in interface.inputs.items():
        if name == "active":
            assert molecule == previous.controller.interface.inputs[name]
        else:
            assert len(molecule) == 8
    assert current.episode == previous.episode
    assert current.evolution == previous.evolution
    assert current.controller.interface == profile_spec(profile, seed=987).controller.interface
    metadata = current.task.config["port_sampling"]
    assert metadata["resolved"] == plain(interface)
    assert metadata["input_sampling"]["8"]["candidate_count"] == 135


def test_cue_identity_matches_across_single_and_two_cell_profiles():
    from toylife2.cli import build_parser
    from toylife2.molecular_experiments import profile_spec

    assert profile_spec("cue").controller.interface.inputs["cue"] == (
        profile_spec("signalling_game").controller.interface.inputs["cue"]
    )
    args = build_parser().parse_args(["molecular", "cue", "--cue-length", "8"])
    assert args.cue_length == 8
