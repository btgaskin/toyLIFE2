"""The explicit single-cell experiment boundary and its recorded explanation."""

from __future__ import annotations

from .protocols import SINGLE_CELL_CONTRACT


def require_current(spec):
    """Validate the current realization without interpreting deprecated experiments."""
    if spec.task.config.get("experiment_contract") != SINGLE_CELL_CONTRACT:
        raise ValueError(
            "deprecated or incompatible molecular experiment; "
            f"create a {SINGLE_CELL_CONTRACT} profile"
        )
    interface, output = spec.controller.interface, spec.controller.output
    if output is None or interface is None or spec.controller.mode != "canonical_cell":
        raise ValueError("single-cell experiments require canonical supplies and binary output")
    if spec.operation not in {"wright_fisher", "rollout"}:
        raise ValueError("single-cell contract supports Wright-Fisher evolution or rollout")
    if spec.task.kind not in {"cue_reading", "chemotaxis"} or spec.episode.panel == "auto":
        raise ValueError("single-cell experiments require a supported task and explicit panel")
    obsolete = {
        "codec_entrypoint",
        "codec_state",
        "action_outputs",
        "action_readout",
        "action_threshold",
        "fixed_map",
    }
    if obsolete.intersection(spec.task.config):
        raise ValueError(
            "use explicit cue_doses and ControllerSpec.output in the single-cell contract"
        )
    if spec.task.kind == "cue_reading":
        names = tuple(spec.task.config["input_channels"])
        rows = spec.task.config["cue_doses"]
        if (
            set(names) != set(interface.inputs)
            or not 1 <= len(names) <= 2
            or len(set(names)) != len(names)
            or len(rows) != 2
            or any(len(row) != len(names) for row in rows)
            or any(
                type(x) is not int or not 0 <= x <= interface.max_dose for row in rows for x in row
            )
        ):
            raise ValueError("binary cue task requires two explicit dose rows for declared inputs")
        if len(names) == 2:
            active = [tuple(i for i, value in enumerate(row) if value) for row in rows]
            if any(len(row) != 1 for row in active) or active[0] == active[1]:
                raise ValueError("two cue identities must be supplied mutually exclusively")
        if spec.episode.episodes != 2:
            raise ValueError("binary cue evaluation requires one episode per prescribed condition")
    elif set(interface.inputs) != {spec.task.config["input_channel"]}:
        raise ValueError("chemotaxis requires its one declared cue channel")


def methods_summary(spec):
    """Explain applied settings; spec.json remains the complete numerical configuration."""
    require_current(spec)
    interface, output = spec.controller.interface, spec.controller.output
    assert interface is not None and output is not None
    episode, evolution, objective = spec.episode, spec.evolution, spec.objective
    rows = [
        "# Single-cell molecular response experiment",
        "",
        f"Contract: `{SINGLE_CELL_CONTRACT}`. Molecular rules: `{spec.model.rules}`.",
        f"Operation executed in this record: `{spec.operation}`.",
        "",
        "The evolving genome specifies the simulated promoters and protein sequences within an "
        "assumed viable cellular background. Background physiology and downstream action apparatus "
        "are outside this evolving module. "
        "The full resolved settings are in [spec.json](spec.json).",
        "",
        "## Molecular interface",
        "",
        f"Fixed cues: `{dict(interface.inputs)}`.",
        f"Fixed markers: `{dict(interface.targets)}`; "
        f"supplied copies: `{dict(interface.target_copies)}`.",
        "Replenishment sets the free supplies each chemical update. Canonical binding, cleavage "
        "and boundary release determine the observed intact pool.",
        "",
        "## Assumed output apparatus",
        "",
        f"Representation: `{output.representation}`; ordered markers: `{output.markers}`.",
        f"Rule: `{output.rule}`; "
        f"founding assignment of alternatives to actions: `{output.permutation}`.",
        "Observations are at after_output. bound_free compares free and bound intact marker; "
        "bound_pair compares two occupied marker counts. Majority chooses the larger count and "
        "fairly resolves ties. Proportional sampling draws according to counts and fairly "
        "resolves zero totals. This is an assumed observation-to-action coupling.",
        "",
        "## Task and evaluation",
        "",
        f"Task: `{spec.task.kind}`; profile: `{spec.task.config.get('profile', 'chemotaxis')}`.",
        f"Chemical updates per decision: `{episode.chemical_updates}`; "
        f"decisions: `{episode.horizon}`.",
        f"Episodes: `{episode.episodes}`; panel policy: `{episode.panel}`.",
        f"Warmup updates: `{spec.controller.warmup}`; "
        f"reset per decision: `{episode.reset_per_decision}`.",
        f"Experiment seed: `{spec.seed}`; episode seed: `{episode.seed}`.",
    ]
    if spec.task.kind == "cue_reading":
        rows.append(
            f"Cue channels: `{tuple(spec.task.config['input_channels'])}`; "
            f"prescribed doses: `{spec.task.config['cue_doses']}`."
        )
        rows.append(f"Settling decisions: `{spec.task.config['settling_steps']}`.")
        rows.append(
            "Expected cue accuracy integrates out decoder randomness because actions "
            "do not change the prescribed cue schedule or chemical trajectory."
        )
    else:
        rows.append(
            f"Field: `{spec.task.config['field']}`; "
            f"world decision duration: `{episode.dt}` seconds."
        )
        rows.append(
            "Probabilities along a sampled trajectory are observations of that trajectory; "
            "expected total performance requires evaluation over trajectories."
        )
    rows.extend(
        [
            "Action draws use Philox panel/episode/decision coordinates shared across matched "
            "candidates. Identical genotypes may share an evaluated estimate. Episodes sample "
            "performance; evolutionary trials are separate population replicates. Chemical updates "
            "are discrete model steps; world duration does not calibrate molecular kinetics.",
            "",
            "## Reproduction",
            "",
        ]
    )
    if spec.operation == "rollout":
        rows.append(
            "This record evaluates molecular responses. No parent sampling or population "
            "replacement occurs in this record; configured evolution settings are inactive."
        )
    else:
        rows.extend(
            [
                "Wright-Fisher reproduction uses global competition, discrete nonoverlapping "
                "generations and fixed population size. "
                "Zero total weight invokes uniform parent sampling.",
                f"Initialization: `{evolution.initial}`; "
                f"population size: `{evolution.population_size}`.",
                f"Generations: `{evolution.generations}`; "
                f"evolutionary trials: `{evolution.trials}`.",
                f"Point mutation probability per offspring: `{evolution.mutation_probability}`.",
            ]
        )
        if evolution.gene_copy is not None:
            copy = evolution.gene_copy
            rows.append(
                f"Single-event duplication/deletion probabilities: "
                f"`{copy.duplication_probability}` / `{copy.deletion_probability}`; "
                f"active gene bounds: `{copy.min_genes}`–`{copy.max_genes}`."
            )
        if objective.kind in {"auto", "bounded_reward"}:
            rows.append(
                "Sampled task reward supplies reproductive weights. Expected cue "
                "accuracy is a separate diagnostic."
            )
            rows.append(
                f"Reward quantization: `{objective.quantization}`; weight floor: "
                f"`{objective.floor_numerator}/{objective.floor_denominator}`."
            )
        else:
            rows.append(
                f"Reproductive mapping uses objective `{objective.kind}`; "
                f"entrypoint: `{objective.entrypoint}`. Its archived implementation and "
                "resolved configuration define the mapping from observations to weights."
            )
    rows.extend(
        [
            "",
            "## Production measurements",
            "",
            "Each expressed gene copy contributes one translation event per canonical update, "
            "including products omitted from the folded inventory. Warmup and task totals are "
            "separate; task totals include settling. "
            "Sequence-unit production is 16 times the event "
            "count. Counts describe the simulated cell, so cache reuse and retries preserve them. "
            "These counters consume no molecular resources. Reproductive effects are determined "
            "by the explicit objective; the default uses task reward alone.",
            "Shared expression resources are a future chemistry extension, motivated by "
            "[Qian et al. (2017)](https://doi.org/10.1021/acssynbio.6b00361). Production counts "
            "and folding/contact energies provide no calibrated synthesis cost.",
            "",
        ]
    )
    return "\n".join(rows)
