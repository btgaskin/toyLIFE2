"""Recorded evolution of molecular response machinery in a viable background."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .batch import BatchRunner
from .config import load_spec, write_spec
from .genotypes import GenomeBatch, decode, load_selected, save_selected
from .molecular_worlds import Chemotaxis
from .protocols import (
    SINGLE_CELL_CONTRACT,
    BinaryOutputSpec,
    EpisodeSpec,
    ObjectiveSpec,
    StructuralInterface,
    TaskSpec,
)
from .specs import EvolutionSpec, ExecutionSpec, ModelSpec, RecordingSpec, RunSpec, plain
from .worlds import ScenarioRandom

PROFILES = ("basic", "cue", "cue_alternatives", "chemotaxis")
_DEPRECATED_PROFILES = ("cue4", "signalling_game")


def result_destination(spec, *, workspace=None, name=None):
    """Use results/ by default, or a named workspace with the existing index."""
    if workspace is not None:
        from .workspaces import new_record_path

        path = Path(workspace).expanduser()
        if not path.is_absolute() and len(path.parts) == 1:
            path = Path("results") / path
        return new_record_path(path, spec, name)
    if name is not None and (not name or not all(c.isalnum() or c in "_-" for c in name)):
        raise ValueError("run name must contain only letters, numbers, '_' or '-'")
    stem = name or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    return Path("results") / stem


PORT_POOLS = ("four_site_active", "all_lengths_active")


def sample_interface(
    tables,
    inputs,
    targets,
    *,
    pool="four_site_active",
    seed=0,
    input_lengths=None,
    target_lengths=None,
):
    """Sample fixed canonical ports; contact eligibility does not imply reachability."""
    if pool not in PORT_POOLS or type(seed) is not int or not 0 <= seed < 1 << 64:
        raise ValueError("invalid molecular port pool or interface seed")
    perimeters = np.unique(tables.protein_perimeter[1:])
    eligible = np.any(tables.protein_met_energy[:, perimeters] < 0, axis=1)
    eligible |= np.any(tables.dimer_met_energy < 0, axis=1)
    candidates = sorted(
        m
        for i, m in enumerate(tables.met_strings)
        if eligible[i] and (pool == "all_lengths_active" or len(m) == 4)
    )
    input_names, target_names = sorted(inputs), sorted(targets)
    input_lengths, target_lengths = dict(input_lengths or {}), dict(target_lengths or {})
    for lengths, names in ((input_lengths, input_names), (target_lengths, target_names)):
        if set(lengths) - set(names) or any(
            type(length) is not int or not 4 <= length <= 8 for length in lengths.values()
        ):
            raise ValueError("molecular lengths must name declared channels and lie in 4..8")
    occupied = set()
    resolved_inputs, resolved_targets = {}, {}
    input_sampling, target_sampling = {}, {}
    # Fixed marker machinery is sampled first, so changing cue length can hold
    # it constant. Each draw is uniform over its declared eligible repertoire.
    for names, lengths, resolved, sampling, role in (
        (target_names, target_lengths, resolved_targets, target_sampling, 0x4D41524B),
        (input_names, input_lengths, resolved_inputs, input_sampling, None),
    ):
        groups = {}
        for name in names:
            groups.setdefault(lengths.get(name), []).append(name)
        for length, channels in sorted(groups.items(), key=lambda row: row[0] or 0):
            available = sorted(
                molecule
                for index, molecule in enumerate(tables.met_strings)
                if eligible[index]
                and molecule not in occupied
                and (len(molecule) == length if length is not None else molecule in candidates)
            )
            address = [seed, length or 0] if role is None else [seed, role, length or 0]
            selected = (
                np.random.default_rng(np.random.SeedSequence(address))
                .choice(available, len(channels), replace=False)
                .tolist()
            )
            resolved.update(zip(channels, selected, strict=True))
            occupied.update(selected)
            sampling[str(length) if length is not None else "pool"] = dict(
                channels=channels,
                seed_sequence=address,
                candidate_count=len(available),
                candidates_digest=hashlib.sha256("\n".join(available).encode()).hexdigest(),
            )
    interface = StructuralInterface(inputs=resolved_inputs, targets=resolved_targets)
    metadata = dict(
        pool=pool,
        seed=seed,
        algorithm="numpy_pcg64_marker_then_cue_v1",
        eligibility="negative_protein_or_dimer_contact",
        candidates=candidates,
        tables_digest=tables.digest,
        resolved=plain(interface),
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        input_sampling=input_sampling,
        target_sampling=target_sampling,
    )
    return interface, metadata


def profile_spec(
    profile="basic",
    *,
    preset="smoke",
    field="fluctuating",
    seed=123,
    genes=2,
    backend="auto",
    world_precision="float32",
    port_pool="four_site_active",
    interface_seed=0,
    cue_length=8,
    response_updates=8,
    settling_updates=0,
    interface=None,
    marker_length=4,
    output_representation="bound_free",
    output_rule="proportional",
    marker_copies=1,
    initial="independent_random",
) -> RunSpec:
    """Construct independently evolved task profiles; defaults are fully resolved."""
    if profile not in (*PROFILES, *_DEPRECATED_PROFILES) or preset not in {"smoke", "pilot"}:
        raise ValueError("unknown molecular profile or preset")
    if type(marker_length) is not int or not 4 <= marker_length <= 8:
        raise ValueError("marker length must lie in 4..8")
    if cue_length is not None and (type(cue_length) is not int or not 4 <= cue_length <= 8):
        raise ValueError("cue length must be None or an integer in 4..8")
    if (
        type(response_updates) is not int
        or response_updates < 1
        or type(settling_updates) is not int
        or settling_updates < 0
    ):
        raise ValueError("response updates must be positive and settling updates nonnegative")
    smoke = preset == "smoke"
    inputs, targets = {"cue": "0001"}, {"act": "1111"}
    if profile == "cue4":
        inputs, targets = {"cue_a": "0001", "cue_b": "0110"}, {"act_a": "1111", "act_b": "1001"}
    if profile == "signalling_game":
        if settling_updates:
            raise ValueError("clonal signalling uses persistent chemistry between decisions")
        inputs = {"cue": "0001", "active": "0110"}
        task = TaskSpec(kind="signalling_game", config={"pairing": "clonal", "rounds": 16})
        episode = EpisodeSpec(
            horizon=32, episodes=2, chemical_updates=response_updates, panel="fixed"
        )
    elif profile == "chemotaxis":
        targets = {"motor": "1111"}
        task = TaskSpec(
            kind="chemotaxis",
            config={
                "field": field,
                "world_engine": "quadrants",
                "world_precision": world_precision,
                "world_random_math": "box_muller_f32_v1"
                if world_precision == "float32"
                else "numpy_f64_v1",
                "motion_model": "ecoli_spherical_v1",
            },
        )
        episode = EpisodeSpec(
            horizon=20 if smoke else 1000, episodes=2 if smoke else 16, panel="generation"
        )
    else:
        config: dict[str, Any] = dict(
            profile=profile,
            fixed_map=[1, 0] if profile == "cue" else [0, 1],
            source_model="CUE:B4" if profile == "cue" else "binary_matching",
        )
        if profile == "cue4":
            config.update(
                fixed_map=[0, 1, 3, 2],
                input_channels=["cue_a", "cue_b"],
                action_outputs=["act_a", "act_b"],
                source_model="CUE:B4+COD:B3",
            )
        config["settling_steps"] = settling_updates
        task = TaskSpec(kind="cue_reading", config=config)
        episode = EpisodeSpec(
            horizon=settling_updates + response_updates,
            episodes=4 if profile == "cue4" else 2,
            chemical_updates=1,
            panel="fixed",
        )
    current = profile in PROFILES
    if current:
        if profile == "cue_alternatives":
            inputs = {"cue_a": "0001", "cue_b": "0110"}
        stem = "motor" if profile == "chemotaxis" else "act"
        targets = (
            {stem: "1111"}
            if output_representation == "bound_free"
            else {f"{stem}_a": "1111", f"{stem}_b": "1001"}
        )
    if interface is None:
        from .tables import Tables

        interface, sampling = sample_interface(
            Tables.load(cache_dir=".cache/toylife2"),
            inputs,
            targets,
            pool=port_pool,
            seed=interface_seed,
            input_lengths={name: cue_length for name in inputs if name != "active"}
            if cue_length is not None
            else None,
            target_lengths={name: marker_length for name in targets} if current else None,
        )
        task = replace(task, config={**task.config, "port_sampling": sampling})
    output = None
    if current:
        from .random import Stream

        interface = StructuralInterface(
            inputs=interface.inputs,
            targets=interface.targets,
            target_copies={name: marker_copies for name in interface.targets},
            max_dose=interface.max_dose,
        )
        permutation = (
            (1, 0)
            if int(
                Stream(interface_seed).integers(
                    2, np.array(0, np.uint64), role="interface:output-permutation-v1"
                )
            )
            else (0, 1)
        )
        output = BinaryOutputSpec(
            output_representation, tuple(sorted(interface.targets)), output_rule, permutation
        )
        config = {
            key: value
            for key, value in task.config.items()
            if key not in {"fixed_map", "action_outputs"}
        }
        config["experiment_contract"] = SINGLE_CELL_CONTRACT
        config["source_model"] = "single_cell_response"
        if "port_sampling" in config:
            config["port_sampling"] = {**config["port_sampling"], "resolved": plain(interface)}
        if profile == "chemotaxis":
            config["input_channel"] = next(iter(interface.inputs))
        else:
            config["input_channels"] = list(interface.inputs)
            config["cue_doses"] = (
                [[1, 0], [0, 1]]
                if profile == "cue_alternatives"
                else [[1], [0]]
                if profile == "cue"
                else [[0], [1]]
            )
        task = replace(task, config=config)
        episode = replace(episode, episodes=episode.episodes if profile == "chemotaxis" else 2)
    return RunSpec(
        schema_version=2,
        operation="wright_fisher",
        seed=seed,
        model=ModelSpec(genes=genes),
        task=task,
        episode=episode,
        controller=interface.controller(output=output),
        objective=ObjectiveSpec(
            kind="bounded_reward", floor_numerator=1, floor_denominator=100_000_000
        ),
        evolution=EvolutionSpec(
            population_size=4 if smoke else 64,
            generations=2 if smoke else 100,
            trials=2,
            initial=initial,
            mutation_probability=0.01,
        ),
        execution=ExecutionSpec(backend=backend, cpu_threads=1),
        recording=RecordingSpec(
            trace_budget_mib=512 if profile == "chemotaxis" and not smoke else 128
        ),
    )


def baseline_scores(task, episode, seed, panel):
    """Paired controls using the same sensor draws, field and motion randomness."""
    world_seed = int.from_bytes(
        hashlib.sha256(json.dumps([seed, episode.seed]).encode()).digest()[:8], "little"
    )
    scenarios = np.arange(episode.episodes, dtype=np.uint64)
    rng = ScenarioRandom(world_seed, panel, scenarios)
    scores = {}
    for policy in ("constant_zero", "constant_one", "cue", "inverse_cue", "random", "temporal"):
        world = Chemotaxis(task.config, scenarios.shape, 1, rng, episode.dt, False)
        state = world.reset()
        total = np.zeros(scenarios.shape)
        fast = slow = np.full(scenarios.shape, 0.5)
        for tick in range(episode.horizon):
            cue = world.observe(state)["inputs"][task.config["input_channel"]]
            if policy == "constant_zero":
                action = np.zeros_like(cue)
            elif policy == "constant_one":
                action = np.ones_like(cue)
            elif policy == "cue":
                action = cue
            elif policy == "inverse_cue":
                action = 1 - cue
            elif policy == "random":
                action = (rng.uniform(tick, "control-action") < 0.5).astype(np.int32)
            else:
                fast = fast + (-np.expm1(-episode.dt / 0.2)) * (cue - fast)
                slow = slow + (-np.expm1(-episode.dt / 1.0)) * (cue - slow)
                action = (fast > slow).astype(np.int32)
            state, reward, _ = world.advance(state, action)
            total += reward
        scores[policy] = (total / episode.horizon).tolist()
    return scores


def run_experiment(
    spec: RunSpec,
    destination: str | Path | None = None,
    *,
    workspace=None,
    name=None,
    resume_from=None,
    animate=False,
    frame_stride=5,
) -> Path:
    """Evolve, record selected cells and save a held-out behavioural inspection."""
    if (
        spec.task.kind not in {"cue_reading", "chemotaxis"}
        or spec.operation != "wright_fisher"
        or spec.controller.interface is None
    ):
        raise ValueError("molecular experiments require an interfaced Wright–Fisher task")
    if destination is not None and workspace is not None:
        raise ValueError("choose destination or workspace")
    from .single_cell import require_current

    require_current(spec)
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer")
    root = (
        Path(destination)
        if destination
        else result_destination(spec, workspace=workspace, name=name)
    )
    if spec.seed is None or spec.episode.horizon is None or spec.episode.horizon < 1:
        raise ValueError("molecular experiments require a seed and a positive finite horizon")
    root.mkdir(parents=True, exist_ok=False)
    write_spec(spec, root / "profile.toml")
    started = time.perf_counter()
    with BatchRunner(spec.execution) as runner:
        parent = None
        if resume_from is not None:
            from .records import inspect_record, verify_record
            from .runner import _check_source
            from .sessions import Checkpoint
            from .specs import scientific_identity

            source = Path(resume_from).resolve()
            if not verify_record(source / "evolution")["ok"]:
                raise ValueError("parent evolution record failed integrity verification")
            _check_source(source / "evolution")
            original = load_spec(source / "profile.toml")
            if scientific_identity(
                replace(
                    spec,
                    evolution=replace(spec.evolution, generations=original.evolution.generations),
                )
            ) != scientific_identity(original):
                raise ValueError(
                    "continuation preserves the molecular task, genome and mutation protocol"
                )
            restored = runner.resume(Checkpoint.read(source / "evolution"))
            if spec.evolution.generations > original.evolution.generations:
                _, checkpoint = restored.extension(spec.evolution.generations)
                relationship = "extension"
            elif spec.evolution.generations == original.evolution.generations:
                checkpoint = restored.checkpoint()
                relationship = "resume"
            else:
                raise ValueError("continuation cannot reduce the generation target")
            runner.release(restored)
            if original.evolution.gene_copy is not None:
                restored.evaluator.close()
            parent = dict(
                relationship=relationship,
                path=str(source),
                scientific_identity=inspect_record(source / "evolution")["scientific_identity"],
                manifest_sha256=hashlib.sha256(
                    (source / "evolution" / "manifest.json").read_bytes()
                ).hexdigest(),
            )
            evolution = runner.evolution(
                spec, checkpoint=checkpoint, destination=root / "evolution", parent=parent
            )
            initial_path = source / "initial_selected_genomes.npy"
            initial_selected = (
                load_selected(source, initial=True) if initial_path.exists() else None
            )
        else:
            evolution = runner.evolution(spec, destination=root / "evolution")
            evolution.observe()
            initial_selected = (
                np.stack(
                    [
                        evolution.populations[i, int(np.argmax(value.scores))]
                        for i, value in enumerate(evolution.evaluations)
                    ]
                )
                if all(value is not None for value in evolution.evaluations)
                else None
            )
        if initial_selected is not None:
            if spec.evolution.gene_copy is not None and not isinstance(
                initial_selected, GenomeBatch
            ):
                initial_selected = GenomeBatch.from_fixed(
                    initial_selected[..., : spec.model.genes], spec.evolution.gene_copy.max_genes
                )
            save_selected(root, initial_selected, initial=True)
        evolution.run()
        if evolution.state["status"] != "complete" or any(
            value is None for value in evolution.evaluations
        ):
            runner.finish(evolution)
            raise RuntimeError("evolution produced no completed population evaluations")
        selected = np.stack(
            [
                evolution.populations[i, int(np.argmax(value.scores))]
                for i, value in enumerate(evolution.evaluations)
            ]
        )
        if spec.evolution.gene_copy is not None:
            selected_counts = np.asarray(
                [
                    evolution.gene_counts[i, int(np.argmax(v.scores))]
                    for i, v in enumerate(evolution.evaluations)
                ]
            )
            selected = GenomeBatch(selected, selected_counts)
        history = evolution.snapshot(fields=["history"])["arrays"]["history"]
        evolution_status = evolution.state["status"]
        counters = evolution.counters
        runner.finish(evolution)
        runner.release(evolution)
        if spec.evolution.gene_copy is not None:
            snapshot = evolution.snapshot()
            np.save(
                root / "gene_count_history.npy",
                snapshot["arrays"]["gene_count_history"],
                allow_pickle=False,
            )
            (root / "gene_copy_statistics.json").write_text(
                json.dumps(plain(snapshot["summary"]), indent=2) + "\n"
            )
            evolution.evaluator.close()
        save_selected(root, selected)
        np.save(root / "fitness_history.npy", history, allow_pickle=False)
        arguments = dict(
            started=started,
            evolution_status=evolution_status,
            counters=counters,
            parent=parent,
            animate=animate,
            frame_stride=frame_stride,
        )
        if spec.evolution.gene_copy is not None:
            from ._molecular_gene_copy import inspect_variable_selected

            return inspect_variable_selected(
                runner, spec, selected, initial_selected, root, history=history, **arguments
            )
        return _inspect_selected(runner, spec, selected, initial_selected, root, **arguments)


def _inspect_selected(
    runner,
    spec,
    selected,
    initial_selected,
    root,
    *,
    started,
    evolution_status,
    counters,
    parent,
    animate,
    frame_stride,
):
    """Record fixed-width selected cells; structural orchestration calls this per length."""
    reservation = 0
    try:
        chemotaxis = spec.task.kind == "chemotaxis"
        episode = replace(
            spec.episode, panel="fixed", episodes=32 if chemotaxis else spec.episode.episodes
        )
        held = replace(
            spec,
            operation="rollout",
            genotypes=tuple(np.asarray(decode(selected)).tolist()),
            episode=episode,
            seed=(spec.seed + 1) % (1 << 64),
            recording=spec.recording,
        )
        panel = 1_000_000 if chemotaxis else 0
        assert episode.horizon is not None
        horizon = episode.horizon
        # Reserve the saved inspection arrays, copies and stack scratch alongside
        # live runtime sessions. Detailed recording covers only selected cells.
        modes = len(spec.task.config["modes"]) if chemotaxis else 0
        inspection_bytes = (
            len(selected)
            * episode.episodes
            * episode.horizon
            * (2 if animate and chemotaxis else 1)
            * (
                512
                + 128 * modes
                + 64 * spec.model.genes
                + 64 * len(spec.controller.ports.metabolites)
            )
            + len(selected) * 4 * 32 * (128 + 64 * spec.model.genes)
            + 2 * 128 * 128 * 8
        )
        runner.model.engine._check_memory(
            inspection_bytes,
            message="saved behavioural inspection exceeds the execution memory budget",
        )
        runner.model.engine._reserved_assay_bytes += inspection_bytes
        reservation = inspection_bytes
        rollout = runner.rollout(selected, spec=held, panel=panel, destination=root / "heldout")
        rows = {}
        row_counts = {}

        def append(name, value):
            array = np.asarray(value)
            if name not in row_counts:
                rows[name] = np.empty((horizon, *array.shape), dtype=array.dtype)
                row_counts[name] = 0
            rows[name][row_counts[name]] = array
            row_counts[name] += 1

        if chemotaxis:
            initial = rollout.snapshot(fields=["world"])["world"]
            for name in (
                "position",
                "phases",
                "source",
                "theta",
                "orientation",
                "tumble_remaining",
            ):
                rows[f"initial_{name}"] = [initial[name]]
            axis = np.linspace(0, spec.task.config["domain"], 128, endpoint=False)
            grid = np.stack(np.meshgrid(axis, axis), -1)
            rows["field_initial"] = [
                rollout.world.field(
                    dict(
                        position=grid,
                        phases=initial["phases"][0, 0],
                        source=initial["source"][0, 0],
                    )
                )
            ]
        while str(rollout.status.flat[0]) not in {"complete", "partial", "failed"}:
            report = rollout.step()
            if report.transition is None or "actions" not in report.transition:
                break
            transition = report.transition
            append("actions", transition["actions"])
            append("reward", transition["reward"])
            if chemotaxis:
                append("sensed_exposure", transition["observation"]["exposure"])
            for name, counts in transition["observation"]["inputs"].items():
                append("input_" + name, counts)
            for name, value in rollout.chemistry.last_readouts.items():
                append("readout_" + name, value)
            append("expression", rollout.chemistry.state["expression"])
            for name, value in transition["channels"].items():
                if isinstance(value, dict):
                    for component, counts in value.items():
                        append(name + "_" + component, counts)
                else:
                    append(name, value)
        if str(rollout.status.flat[0]) != "complete":
            runner.finish(rollout)
            raise RuntimeError("held-out assessment interrupted; inspect its saved checkpoint")
        arrays = {
            name: values[: row_counts[name]] if name in row_counts else np.stack(values)
            for name, values in rows.items()
        }
        arrays["scores"] = rollout.scores.copy()
        if not chemotaxis and spec.controller.output is not None:
            arrays["expected_scores"] = rollout.expected_scores.copy()
        if chemotaxis:
            final = rollout.snapshot(fields=["world"])["world"]
            arrays["field_final"] = rollout.world.field(
                dict(position=grid, phases=final["phases"][0, 0], source=final["source"][0, 0])
            )
        np.savez_compressed(root / "behaviour.npz", allow_pickle=False, **arrays)
        summary = dict(
            profile=spec.task.config.get("profile", "chemotaxis"),
            field=spec.task.config.get("field"),
            evolution_status=evolution_status,
            behaviour_status=str(rollout.status.flat[0]),
            behaviour_reason=rollout.state["reason"],
            heldout_seed=held.seed,
            heldout_panel=panel,
            counters=counters,
            mean_scores=np.mean(rollout.scores, axis=-1).tolist(),
            selected_genomes=selected.tolist(),
            elapsed_seconds=time.perf_counter() - started,
            resolved_interface=plain(spec.controller.interface),
            resolved_ports=plain(spec.controller.ports),
            parent=parent,
            initial_selected_genomes=None
            if initial_selected is None
            else initial_selected.tolist(),
            endpoint_information=spec.controller.interface.endpoint_information(
                runner.model.tables
            ),
            task=plain(spec.task),
            episode=plain(episode),
            future_work="Environmental variation, gene count and emergent modularity.",
        )
        summary["output"] = plain(spec.controller.output)
        summary["experiment_contract"] = spec.task.config.get("experiment_contract")
        summary["production"] = {k: v.tolist() for k, v in rollout.production_metrics().items()}
        if not chemotaxis and spec.controller.output is not None:
            summary["expected_accuracy"] = np.mean(rollout.expected_scores, axis=-1).tolist()
        summary["target_fission_totals"] = {
            name: arrays["readout_target_fission:" + name].sum(axis=0).tolist()
            for name in spec.controller.interface.targets
        }
        summary["fission_coverage"] = (
            "every_chemical_update"
            if episode.chemical_updates == 1
            else "last_chemical_update_per_decision"
        )
        summary["port_sampling"] = plain(spec.task.config.get("port_sampling"))
        if summary["port_sampling"] is not None:
            summary["port_sampling"]["draw_matches_resolved_interface"] = (
                summary["port_sampling"]["resolved"] == summary["resolved_interface"]
            )
        runner.finish(rollout)
        runner.release(rollout)
        if not chemotaxis and initial_selected is not None:
            initial_spec = replace(
                held,
                model=replace(held.model, genes=initial_selected.shape[-1]),
                genotypes=tuple(np.asarray(decode(initial_selected)).tolist()),
            )
            initial_cell = runner.rollout(
                initial_selected,
                spec=initial_spec,
                panel=panel,
                destination=root / "initial_heldout",
            )
            initial_rows = {}
            while str(initial_cell.status.flat[0]) not in {"complete", "partial", "failed"}:
                report = initial_cell.step()
                if report.transition is None:
                    break
                transition = report.transition
                values = {
                    "actions": transition["actions"],
                    "reward": transition["reward"],
                    "judged": transition["channels"].get("judged", True),
                    "expression": initial_cell.chemistry.state["expression"],
                    **{"input_" + k: v for k, v in transition["observation"]["inputs"].items()},
                    **{"readout_" + k: v for k, v in initial_cell.chemistry.last_readouts.items()},
                }
                for key, value in values.items():
                    initial_rows.setdefault(key, []).append(np.array(value, copy=True))
            if str(initial_cell.status.flat[0]) != "complete":
                runner.finish(initial_cell)
                raise RuntimeError("initial cue assessment interrupted; inspect its checkpoint")
            np.savez_compressed(
                root / "initial_behaviour.npz",
                allow_pickle=False,
                **{key: np.stack(value) for key, value in initial_rows.items()},
            )
            summary["initial_mean_scores"] = np.mean(initial_cell.scores, axis=-1).tolist()
            if spec.controller.output is not None:
                summary["initial_expected_accuracy"] = np.mean(
                    initial_cell.expected_scores, axis=-1
                ).tolist()
            runner.finish(initial_cell)
            runner.release(initial_cell)
        # Independent empty-start histories, ending with the same current cue.
        histories = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]])
        probe = runner.chemistry(
            np.broadcast_to(
                selected[:, None, :], (len(selected), len(histories), selected.shape[-1])
            ),
            spec.controller,
        )
        probe_rows = {"inputs": [], "expression": []}
        for cues in np.repeat(histories, 8, axis=1).T:
            inputs = {
                name: np.broadcast_to(cues if index == 0 else 1 - cues, (len(selected), len(cues)))
                for index, name in enumerate(spec.controller.interface.inputs)
            }
            probe.step(spec.controller.interface.encode(inputs))
            probe_rows["inputs"].append(cues.copy())
            probe_rows["expression"].append(probe.state["expression"].copy())
            for name, value in probe.last_readouts.items():
                probe_rows.setdefault(name, []).append(value.copy())
        np.savez_compressed(
            root / "history_probes.npz",
            allow_pickle=False,
            **{name: np.stack(values) for name, values in probe_rows.items()},
        )
        runner.release(probe)
        if chemotaxis:
            summary["temporal_baseline"] = dict(
                fast_time=0.2, slow_time=1.0, initial=0.5, action="fast > slow"
            )
            summary["baselines"] = baseline_scores(spec.task, episode, held.seed, panel)
            reset_spec = replace(
                held, episode=replace(episode, reset_per_decision=True), recording=RecordingSpec()
            )
            reset = runner.rollout(
                selected, spec=reset_spec, panel=panel, destination=root / "reset_control"
            )
            reset.run()
            if str(reset.status.flat[0]) != "complete":
                runner.finish(reset)
                raise RuntimeError("reset control interrupted; inspect its saved checkpoint")
            summary["reset_mean_scores"] = np.mean(reset.scores, axis=-1).tolist()
            summary["reset_status"] = str(reset.status.flat[0])
            runner.finish(reset)
            runner.release(reset)
            if animate and initial_selected is not None:
                initial_spec = replace(
                    held,
                    model=replace(held.model, genes=initial_selected.shape[-1]),
                    genotypes=tuple(np.asarray(decode(initial_selected)).tolist()),
                )
                initial_rollout = runner.rollout(
                    initial_selected,
                    spec=initial_spec,
                    panel=panel,
                    destination=root / "initial_heldout",
                )
                initial_state = initial_rollout.snapshot(fields=["world"])["world"]
                initial_rows = {
                    "initial_" + key: [initial_state[key]]
                    for key in (
                        "position",
                        "phases",
                        "source",
                        "theta",
                        "orientation",
                        "tumble_remaining",
                    )
                }
                initial_rows["field_initial"] = rows["field_initial"]
                while str(initial_rollout.status.flat[0]) not in {"complete", "partial", "failed"}:
                    report = initial_rollout.step()
                    if report.transition is None or "actions" not in report.transition:
                        break
                    transition = report.transition
                    values = {
                        **transition["channels"],
                        "actions": transition["actions"],
                        "sensed_exposure": transition["observation"]["exposure"],
                        **{"input_" + k: v for k, v in transition["observation"]["inputs"].items()},
                        **{
                            "readout_" + k: v
                            for k, v in initial_rollout.chemistry.last_readouts.items()
                        },
                    }
                    for key, value in values.items():
                        initial_rows.setdefault(key, []).append(np.array(value, copy=True))
                if str(initial_rollout.status.flat[0]) != "complete":
                    runner.finish(initial_rollout)
                    raise RuntimeError(
                        "initial-cell assessment interrupted; inspect its checkpoint"
                    )
                np.savez_compressed(
                    root / "initial_behaviour.npz",
                    allow_pickle=False,
                    **{key: np.stack(values) for key, values in initial_rows.items()},
                )
                runner.finish(initial_rollout)
                runner.release(initial_rollout)
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["artifacts"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.iterdir()
            if path.suffix in {".npy", ".npz", ".toml"}
        }
        root.joinpath("experiment.json").write_text(json.dumps(summary, indent=2) + "\n")
        from .reports import render_molecular_report

        if animate and spec.task.kind == "chemotaxis":
            from .molecular_animation import render_molecular_animation

            render_molecular_animation(root, frame_stride=frame_stride)
        render_molecular_report(root)
        return root

    finally:
        runner.model.engine._reserved_assay_bytes -= reservation
