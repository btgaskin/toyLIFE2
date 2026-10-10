"""Two-gene cellular automata and batched tissue trajectories."""

# Quadrants kernel decorators have dynamic call signatures in Python.
# pyright: reportCallIssue=false

from __future__ import annotations

import numpy as np

from .genotypes import validate
from .results import ArrayResult
from .runtime import Engine
from .specs import ExecutionSpec, SpatialSpec
from .tables import Tables


def pair_promoter_matrix(
    tables: Tables,
    protein_pairs: np.ndarray,
    *,
    engine: Engine,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate six fixed molecule inventories against all promoter sequences."""
    from ._kernels import pair_promoter_responses

    pairs = np.asarray(protein_pairs, dtype=np.int32)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("protein_pairs must be nonempty [pair,2]")
    if np.any(pairs < 0) or np.any(pairs >= len(tables.protein_perimeter)):
        raise ValueError("protein ID outside table range")
    engine._available_bytes(
        table_names=engine._REGULATORY_TABLES,
        minimum_bytes=len(pairs) * (16 + 2 * (16 * 6 * 4 + 4)),
        message="pair response matrix exceeds the execution memory budget",
    )
    table_buffers = engine._regulatory_tables()[1:]
    device_pairs = engine._i32(pairs.shape, pairs)
    responses = engine._i32((len(pairs), 16, 6))
    present = engine._i32((len(pairs),))
    pair_promoter_responses(device_pairs, *table_buffers, responses, present)
    return responses.to_numpy(), present.to_numpy()


def response_descriptors(
    tables: Tables,
    genotypes: np.ndarray,
    execution: ExecutionSpec | None = None,
    *,
    engine: Engine | None = None,
) -> np.ndarray:
    """Six input-state outputs plus the ordered-pair dimer presence bit.

    Duplicating A or B as a third gene invokes the same Quadrants chemistry for
    the two-copy input states. Only the original two promoter outputs are read.
    """
    genes = validate(genotypes)
    if genes.shape[-1] != 2:
        raise ValueError("spatial phenotypes require exactly two genes")
    leading = genes.shape[:-1]
    flat = genes.reshape(-1, 2)
    work = engine or Engine(tables, execution)
    # Validated inputs, duplicated genes, both truth arrays, descriptors and
    # vectorized protein/dimer indexing scratch remain live across nested assays.
    with work._temporary_bytes(len(flat) * 256 + len(tables.protein_perimeter) * 4,
                               message="spatial descriptors exceed the execution memory budget"):
        return _response_descriptors(tables, flat, leading, work)


def _response_descriptors(tables, flat, leading, work):
    with_extra_a = np.concatenate((flat, flat[:, :1]), axis=1)
    with_extra_b = np.concatenate((flat, flat[:, 1:2]), axis=1)
    a = work.regulatory(with_extra_a).values
    b = work.regulatory(with_extra_b).values
    answer = np.stack((a[:, 0], a[:, 2], a[:, 4], a[:, 6], a[:, 7], b[:, 7]), axis=1)
    answer = (answer >> 1).astype(np.int32, copy=False)
    proteins = np.asarray(tables.gene_to_protein, dtype=np.int32)[flat & 65535]
    answer[:, :] &= np.where(proteins[:, 0, None] == 0, 1, 3)
    answer[:, :] &= np.where(proteins[:, 1, None] == 0, 2, 3)
    left = np.minimum(proteins[:, 0], proteins[:, 1])
    right = np.maximum(proteins[:, 0], proteins[:, 1])
    perimeters = np.asarray(tables.protein_perimeter, dtype=np.int32)
    pa, pb = perimeters[left], perimeters[right]
    present = np.zeros(len(flat), dtype=np.int32)
    valid = (pa >= 0) & (pb >= 0)
    present[valid] = (np.asarray(tables.dimer_id)[pa[valid], pb[valid]] >= 0).astype(np.int32)
    if np.any(present):
        from ._kernels import preformed_dimer_response

        selected = np.flatnonzero(present)
        device_genes = work._i32((len(selected), 2), flat[selected])
        fixed = work._i32((len(selected), 2))
        preformed_dimer_response(device_genes, *work._regulatory_tables(), fixed)
        answer[selected, 4:6] = fixed.to_numpy()
        answer[:, :] &= np.where(proteins[:, 0, None] == 0, 1, 3)
        answer[:, :] &= np.where(proteins[:, 1, None] == 0, 2, 3)
    return np.concatenate((answer, present[:, None]), axis=1).reshape((*leading, 7))


def _transported_state(left: int, centre: int, right: int, dimer: bool, mode: str) -> int:
    if mode == "none":
        return centre
    if mode not in ("A", "B", "both", "all"):
        raise ValueError(f"unknown diffusion mode: {mode}")
    # States 4/5 arise when a morphogen enters a cell holding preformed AB.
    # The dimer moves as a unit only under 'all'; 4/5 also carry free A/B.
    free_a = centre in (2, 4) or (centre == 3 and not dimer)
    free_b = centre in (1, 5) or (centre == 3 and not dimer)
    fixed_ab = centre in (3, 4, 5) and dimer
    for neighbour in (left, right):
        if mode in ("A", "both", "all"):
            free_a |= neighbour in (2, 4) or (neighbour == 3 and not dimer)
        if mode in ("B", "both", "all"):
            free_b |= neighbour in (1, 5) or (neighbour == 3 and not dimer)
        if mode == "all" and dimer:
            fixed_ab |= neighbour in (3, 4, 5)
    if fixed_ab:
        if free_a and free_b:
            return 3
        if free_a:
            return 4
        if free_b:
            return 5
        return 3
    return int(free_b) | (int(free_a) << 1)


def compile_ca(
    tables: Tables,
    genotypes: np.ndarray,
    diffusion: str = "A",
    execution: ExecutionSpec | None = None,
    *,
    engine: Engine | None = None,
    input_states: int = 4,
) -> np.ndarray:
    """Compile one 64- or 216-entry local transition table per genotype."""
    if diffusion not in ("none", "A", "B", "both", "all"):
        raise ValueError("unknown diffusion mode")
    if input_states not in (4, 6):
        raise ValueError("input_states must be 4 or 6")
    work = engine or Engine(tables, execution)
    descriptors = response_descriptors(tables, genotypes, execution, engine=work)
    work._available_bytes(
        fixed_bytes=descriptors.nbytes,
        minimum_bytes=descriptors.size // 7 * input_states**3 * 8,
        message="cellular automata exceed the execution memory budget",
    )
    return ca_from_descriptors(descriptors, diffusion, input_states=input_states)


def ca_from_descriptors(
    descriptors: np.ndarray,
    diffusion: str = "A",
    *,
    input_states: int = 4,
) -> np.ndarray:
    """Map six chemical responses to the local three-cell transition table."""
    if diffusion not in ("none", "A", "B", "both", "all"):
        raise ValueError("unknown diffusion mode")
    if input_states not in (4, 6):
        raise ValueError("input_states must be 4 or 6")
    leading = descriptors.shape[:-1]
    flat = descriptors.reshape(-1, 7)
    size = input_states**3
    ca = np.empty((len(flat), size), dtype=np.int32)
    for present in (0, 1):
        selected = np.flatnonzero(flat[:, 6] == present)
        if len(selected):
            states = np.asarray(
                [
                    _transported_state(
                        code // (input_states * input_states),
                        code // input_states % input_states,
                        code % input_states,
                        bool(present),
                        diffusion,
                    )
                    for code in range(size)
                ],
                dtype=np.intp,
            )
            ca[selected] = flat[selected[:, None], states[None, :]]
    return ca.reshape((*leading, size))


def _requires_extended_inputs(spec: SpatialSpec) -> bool:
    return spec.morphogen in (
        "explicit",
        "central_A_continuous",
        "central_B_continuous",
    ) or (spec.initial_state is not None and 3 in spec.initial_state and spec.morphogen != "none")


def _morphogen(spec: SpatialSpec) -> np.ndarray:
    result = np.zeros((spec.steps, spec.cells), dtype=np.int32)
    if spec.morphogen == "explicit":
        for tick, cell, state in spec.injections:
            result[tick, cell] |= state
        return result
    if spec.morphogen == "none":
        return result
    species = 2 if "_A_" in spec.morphogen else 1
    cells = (spec.cells // 2,) if spec.cells % 2 else (spec.cells // 2 - 1, spec.cells // 2)
    if spec.morphogen.endswith("continuous"):
        result[:, list(cells)] = species
    else:
        result[0, list(cells)] = species
    return result


def _trajectories_from_ca(
    ca: np.ndarray,
    config: SpatialSpec,
    work: Engine,
    dimer_flags: np.ndarray | None = None,
    *,
    owned_result: bool = False,
) -> np.ndarray:
    """Advance distinct local rules in Quadrants without recompiling chemistry."""
    from ._kernels import spatial_trajectory

    output_bytes = (2 if owned_result else 1) * len(ca) * config.steps * config.cells * 4
    morphogen_bytes = config.steps * config.cells * 4
    input_states = 4 if ca.shape[1] == 64 else 6
    if ca.shape[1] not in (64, 216):
        raise ValueError("CA rows require 64 or 216 entries")
    if dimer_flags is None:
        if input_states == 6:
            raise ValueError("extended CA trajectories require dimer flags")
        dimer_flags = np.zeros(len(ca), dtype=np.int32)
    if len(dimer_flags) != len(ca):
        raise ValueError("one dimer flag is required per CA")
    # Device trajectory and its host readback overlap; morphogen lives on both.
    per_sample = 4 * (ca.shape[1] + 2 + config.cells * (2 * config.steps + 3))
    fixed_bytes = (
        2 * morphogen_bytes + output_bytes + ca.nbytes + dimer_flags.nbytes + config.cells * 4
    )
    available = work._available_bytes(
        fixed_bytes=fixed_bytes, minimum_bytes=per_sample,
        message="one tissue trajectory exceeds the execution memory budget",
    )
    morphogen = _morphogen(config)
    batch = min(work.execution.batch_size, available // per_sample)
    output = np.empty((len(ca), config.steps, config.cells), dtype=np.int32)
    device_morphogen = work._i32(morphogen.shape, morphogen)
    for start in range(0, len(ca), batch):
        part = ca[start : start + batch]
        device_ca = work._i32(part.shape, part)
        device_dimer = work._i32((len(part),), dimer_flags[start : start + batch])
        initial = np.zeros(config.cells, np.int32)
        if config.initial_state is not None:
            initial[:] = config.initial_state
        state = work._i32(
            (len(part), config.cells), np.broadcast_to(initial, (len(part), config.cells)).copy()
        )
        next_state = work._i32((len(part), config.cells))
        trajectory = work._i32((len(part), config.steps, config.cells))
        spatial_trajectory(
            device_ca,
            device_morphogen,
            device_dimer,
            state,
            next_state,
            trajectory,
            int(config.boundary == "ring"),
            input_states,
        )
        output[start : start + len(part)] = trajectory.to_numpy()
        del device_ca, device_dimer, initial, state, next_state, trajectory
    return output


def patterns(
    tables: Tables,
    genotypes: np.ndarray,
    spec: SpatialSpec | None = None,
    execution: ExecutionSpec | None = None,
    *,
    engine: Engine | None = None,
) -> ArrayResult:
    config = spec or SpatialSpec()
    work = engine or Engine(tables, execution)
    genes = validate(genotypes)
    leading = genes.shape[:-1]
    if genes.shape[-1] != 2:
        raise ValueError("spatial phenotypes require exactly two genes")
    flat = genes.reshape(-1, 2)
    rule_size = 216 if _requires_extended_inputs(config) else 64
    work._available_bytes(
        table_names=work._REGULATORY_TABLES,
        minimum_bytes=len(flat) * (config.steps * config.cells * 8 + 160 + 8 + 28 + rule_size * 8),
        message="tissue output exceeds the execution memory budget",
    )
    with work._temporary_bytes(genes.nbytes + len(flat) * 160,
                               message="tissue inputs exceed the execution memory budget"):
        return _patterns(tables, flat, leading, config, work)


def _patterns(tables, flat, leading, config, work):
    descriptors = response_descriptors(tables, flat, engine=work)
    input_states = 6 if _requires_extended_inputs(config) else 4
    ca = ca_from_descriptors(descriptors, config.diffusion, input_states=input_states)
    with work._temporary_bytes(descriptors.nbytes,
                               message="tissue descriptors exceed the execution memory budget"):
        output = _trajectories_from_ca(
            ca, config, work, descriptors[:, 6], owned_result=True
        )
    return ArrayResult(
        output.reshape((*leading, config.steps, config.cells)),
        np.full(leading or (), "complete", dtype="<U20"),
        {
            "backend": work.execution.backend,
            "diffusion": config.diffusion,
            "table_digest": str(tables.digest),
        },
    )
