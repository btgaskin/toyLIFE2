"""Reachable regulatory cycles and bounded Quadrants metabolic assays.

The number of regulatory launches follows the trajectory from all-off rather
than the size of the complete Boolean truth table. Expression is represented
as one integer per gene, so no signed-state-width limit is involved.
"""

# Quadrants kernel specializations do not retain Python's static call signature.
# pyright: reportCallIssue=false

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np

from .genotypes import validate
from .results import ArrayResult
from .specs import RULESET
from .tables import canonical_metabolite


def _single(engine, genes: np.ndarray, bits: np.ndarray,
            tables: tuple[object, ...]) -> tuple[np.ndarray, ...]:
    from ._trajectory_kernels import regulatory_single

    lanes, n_genes = genes.shape
    capacity = n_genes * (n_genes + 1) // 2
    device_genes = engine._i32(genes.shape, genes)
    device_bits = engine._i32(bits.shape, bits)
    pids = engine._i32((lanes, n_genes))
    pcounts = engine._i32((lanes, n_genes))
    candidates = [engine._i32((lanes, capacity)) for _ in range(6)]
    next_bits = engine._i32((lanes, n_genes))
    regulatory_single(device_genes, device_bits, *tables, pids, pcounts,
                      *candidates, next_bits)
    return (next_bits.to_numpy(), pids.to_numpy(), pcounts.to_numpy(),
            candidates[0].to_numpy(), candidates[1].to_numpy(),
            candidates[2].to_numpy(), candidates[5].to_numpy())


def _key(bits: np.ndarray) -> bytes:
    return np.packbits(bits, bitorder="big").tobytes()


def _unpack(key: bytes, n_genes: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(key, dtype=np.uint8),
                         count=n_genes, bitorder="big").astype(np.int32)


def _regulatory_chunk(engine, n_genes: int) -> int:
    capacity = n_genes * (n_genes + 1) // 2
    per_lane = 4 * (5 * n_genes + 6 * capacity)
    available = engine.execution.memory_budget_mib * 1024 * 1024 - engine._table_bytes
    if available < per_lane:
        raise MemoryError("one reachable regulatory lane exceeds the execution memory budget")
    return min(engine.execution.batch_size, max(1, available // per_lane))


def _reachable_cycles(engine, genes: np.ndarray,
                      tables: tuple[object, ...],
                      reserved_bytes: int = 0) -> tuple[list[list[bytes]], np.ndarray]:
    count, n_genes = genes.shape
    current = np.zeros((count, n_genes), dtype=np.int32)
    seen: list[dict[bytes, int]] = [{} for _ in range(count)]
    visited: list[list[bytes]] = [[] for _ in range(count)]
    cycles: list[list[bytes]] = [[] for _ in range(count)]
    incomplete = np.zeros(count, dtype=bool)
    active = np.arange(count, dtype=np.int32)
    chunk_size = _regulatory_chunk(engine, n_genes)
    seen_bytes = 0
    budget = engine.execution.memory_budget_mib * 1024 * 1024
    for _ in range(engine.execution.max_steps):
        pending: list[int] = []
        for index in active:
            row = int(index)
            key = _key(current[row])
            if key in seen[row]:
                cycles[row] = visited[row][seen[row][key]:]
            else:
                seen_bytes += len(key) + 128  # dict entry, list pointer, and key payload
                if engine._table_bytes + reserved_bytes + seen_bytes > budget:
                    raise MemoryError(
                        "reachable regulatory history exceeds the execution memory budget")
                seen[row][key] = len(visited[row])
                visited[row].append(key)
                pending.append(row)
        if not pending:
            break
        active = np.asarray(pending, dtype=np.int32)
        for start in range(0, len(active), chunk_size):
            chunk = active[start:start + chunk_size]
            next_bits = _single(engine, genes[chunk], current[chunk], tables)[0]
            current[chunk] = next_bits
    else:
        # A cycle reached exactly after the final transition still requires
        # one more boundary check, matching the scalar reference cap.
        incomplete[active] = True
    return cycles, incomplete


def _source_inventories(engine, genotype: np.ndarray, cycle: Sequence[bytes],
                        tables: tuple[object, ...]) -> tuple[np.ndarray, ...]:
    n_genes = len(genotype)
    capacity = n_genes * (n_genes + 1) // 2
    count = len(cycle) + 1  # the final lane supplies all possible species
    bits = np.empty((count, n_genes), dtype=np.int32)
    for index, state in enumerate(cycle):
        bits[index] = _unpack(state, n_genes)
    bits[-1] = 1
    pids = np.full((count, n_genes), -1, dtype=np.int32)
    pcounts = np.zeros((count, n_genes), dtype=np.int32)
    did = np.full((count, capacity), -1, dtype=np.int32)
    dleft = np.full((count, capacity), -1, dtype=np.int32)
    dright = np.full((count, capacity), -1, dtype=np.int32)
    dcount = np.zeros((count, capacity), dtype=np.int32)
    chunk_size = _regulatory_chunk(engine, n_genes)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        batch = np.broadcast_to(genotype, (end - start, n_genes))
        _, a, b, c, d, e, f = _single(engine, batch, bits[start:end], tables)
        pids[start:end] = a
        pcounts[start:end] = b
        did[start:end] = c
        dleft[start:end] = d
        dright[start:end] = e
        dcount[start:end] = f
    return pids, pcounts, did, dleft, dright, dcount


def metabolic_reachable(engine, genotypes: np.ndarray,
                        environment: tuple[str, ...] | list[str] | None = None) -> ArrayResult:
    """Assay only phases on the reachable attractor with exact bounded history."""
    from ._kernels import metabolic_assay

    genes = validate(genotypes)
    leading = genes.shape[:-1]
    n_genes = genes.shape[-1]
    flat = genes.reshape((-1, n_genes))
    env = (tuple(environment) if environment is not None else tuple(
        met for met, unbreakable in zip(engine.tables.met_strings,
                                        engine.tables.met_unbreakable, strict=True)
        if not unbreakable))
    if not env:
        raise ValueError("metabolic environment must contain at least one metabolite")
    for met in env:
        if not isinstance(met, str) or not 4 <= len(met) <= 8 or set(met) - {"0", "1"}:
            raise ValueError("metabolites must be binary strings of length four through eight")
    met_indices = np.asarray([engine.tables.met_index[canonical_metabolite(met)]
                              for met in env], dtype=np.int32)
    met_lengths = np.asarray([len(met) for met in env], dtype=np.int32)
    budget = engine.execution.memory_budget_mib * 1024 * 1024
    result_bytes = len(flat) * len(env) * (1 + 80)  # uint8 values and <U20 statuses
    if result_bytes >= budget:
        raise MemoryError("metabolic result arrays exceed the execution memory budget")
    regulatory_tables = engine._regulatory_tables()
    met_tables = tuple(engine._table(name) for name in (
        "protein_met_energy", "dimer_met_energy", "dimer_met_seq", "dimer_met_pos",
        "dimer_met_breakable", "breaking_seq_to_index", "breaking_energy"))
    cycles, unfinished = _reachable_cycles(engine, flat, regulatory_tables, result_bytes)
    values = np.zeros((len(flat), len(env)), dtype=np.uint8)
    statuses = np.full((len(flat), len(env)), "incomplete", dtype="<U20")
    dim_capacity = n_genes * (n_genes + 1) // 2
    candidate_capacity = 2 * dim_capacity + 3 * n_genes
    for sample, cycle in enumerate(cycles):
        if unfinished[sample] or not cycle:
            continue
        count = len(cycle) + 1
        source_bytes = 4 * count * (2 * n_genes + 4 * dim_capacity)
        if budget - engine._table_bytes - result_bytes <= source_bytes:
            raise MemoryError("reachable cycle inventories exceed the execution memory budget")
        source = _source_inventories(engine, flat[sample], cycle, regulatory_tables)
        # A boundary has one free metabolite and at most g protein-derived
        # molecules. Only protein/dimer species admitted by this genotype can
        # occur. The finite composition bound preserves exact repeat checks
        # while avoiding a large history for sparse all-off genotypes.
        protein_species = int(np.count_nonzero(source[0][-1] > 0))
        dimer_species = int(np.count_nonzero(source[2][-1] >= 0))
        finite_bound = min(engine.execution.max_steps,
                           comb(n_genes + protein_species + dimer_species, n_genes))
        # One slot selects the constant-memory Brent proof in metabolic_assay.
        # Small state spaces retain exact first-repeat behavior.
        history_capacity = finite_bound if finite_bound <= 1024 else 1
        lane_bytes = 4 * (4 * n_genes + 7 * dim_capacity + 6 * candidate_capacity
                          + history_capacity * (n_genes + dim_capacity))
        fixed_bytes = source_bytes + 4 * (n_genes + count + 1)
        available = budget - engine._table_bytes - result_bytes - fixed_bytes
        if available < lane_bytes + 16:
            raise MemoryError("one reachable metabolic lane exceeds the execution memory budget")
        met_chunk = min(engine.execution.batch_size, len(env),
                        max(1, (available - 16) // (lane_bytes + 8)))
        device_genes = engine._i32((1, n_genes), flat[sample:sample + 1])
        source_device = tuple(engine._i32(a.shape, a) for a in source)
        cycle_indices = np.arange(count, dtype=np.int32)[None, :]
        device_cycles = engine._i32(cycle_indices.shape, cycle_indices)
        device_lengths = engine._i32((1,), np.asarray([len(cycle)], dtype=np.int32))
        for start in range(0, len(env), met_chunk):
            end = min(start + met_chunk, len(env))
            lanes = end - start
            device_met_ids = engine._i32((lanes,), met_indices[start:end])
            device_met_lengths = engine._i32((lanes,), met_lengths[start:end])
            pids = engine._i32((lanes, n_genes))
            pcounts = engine._i32((lanes, n_genes))
            previous_pcounts = engine._i32((lanes, n_genes))
            dims = [engine._i32((lanes, dim_capacity)) for _ in range(6)]
            candidates = [engine._i32((lanes, candidate_capacity)) for _ in range(6)]
            protein_demand = engine._i32((lanes, n_genes))
            dimer_demand = engine._i32((lanes, dim_capacity))
            history_p = engine._i32((lanes, history_capacity, n_genes))
            history_d = engine._i32((lanes, history_capacity, dim_capacity))
            output = engine._i32((1, lanes))
            status = engine._i32((1, lanes))
            metabolic_assay(
                device_genes, *regulatory_tables[:3], *regulatory_tables[4:5],
                *regulatory_tables[5:], *met_tables, device_met_ids,
                device_met_lengths, device_cycles, device_lengths,
                *source_device, pids, pcounts, previous_pcounts,
                *dims, *candidates, protein_demand, dimer_demand,
                history_p, history_d, output, status, engine.execution.max_steps)
            values[sample, start:end] = output.to_numpy()[0].astype(np.uint8)
            statuses[sample, start:end] = np.where(
                status.to_numpy()[0] != 0, "incomplete", "complete")
    return ArrayResult(values.reshape((*leading, len(env))),
                       statuses.reshape((*leading, len(env))),
                       {"ruleset": RULESET, "backend": engine.execution.backend,
                        "table_digest": str(engine.tables.digest), "environment": env,
                        "regulatory_mode": "reachable_multiword"})
