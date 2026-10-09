"""Experimental, unvalidated reachable-metabolism batching prototype.

Engine does not call this module. Correctness and performance verification are
deferred to a future optimization task; no numerical claims accompany it.
"""
from math import comb

import numpy as np

from toylife2._trajectory import _reachable_cycles, _single, _unpack
from toylife2.genotypes import validate
from toylife2.results import ArrayResult
from toylife2.specs import RULESET
from toylife2.tables import canonical_metabolite


def metabolic_reachable_batched(engine, genotypes, environment=None):
    from toylife2._kernels import metabolic_assay

    genes = validate(genotypes)
    leading, g = genes.shape[:-1], genes.shape[-1]
    flat = genes.reshape((-1, g))
    env = tuple(environment) if environment is not None else tuple(
        met for met, flag in zip(engine.tables.met_strings,
                                engine.tables.met_unbreakable, strict=True) if not flag)
    if not env:
        raise ValueError("metabolic environment must contain at least one metabolite")
    for met in env:
        if not isinstance(met, str) or not 4 <= len(met) <= 8 or set(met) - {"0", "1"}:
            raise ValueError("metabolites must be binary strings of length four through eight")
    met_ids = np.asarray([engine.tables.met_index[canonical_metabolite(m)] for m in env], np.int32)
    met_lengths = np.asarray([len(m) for m in env], np.int32)
    budget = engine.execution.memory_budget_mib * 1024 * 1024
    result_bytes = len(flat) * len(env) * 81
    if result_bytes >= budget:
        raise MemoryError("metabolic result arrays exceed the execution memory budget")
    regulatory_names = ("gene_to_protein", "protein_perimeter", "protein_energy",
                        "dimer_id", "dimer_energy", "protein_promoter_energy",
                        "protein_promoter_pol", "dimer_promoter_energy",
                        "dimer_promoter_pol", "polymerase_energy")
    metabolic_names = (
        "protein_met_energy", "dimer_met_energy", "dimer_met_seq", "dimer_met_pos",
        "dimer_met_breakable", "breaking_seq_to_index", "breaking_energy")
    new_table_bytes = sum(np.asarray(getattr(engine.tables, name)).size * 4
                          for name in regulatory_names + metabolic_names
                          if name not in engine._buffers)
    d = g * (g + 1) // 2
    regulatory_lane = 4 * (5 * g + 6 * d)
    if engine._table_bytes + new_table_bytes + result_bytes + regulatory_lane > budget:
        raise MemoryError(
            "reachable tables and one regulatory lane exceed the execution memory budget")
    regulatory = engine._regulatory_tables()
    metabolic = tuple(engine._table(name) for name in metabolic_names)
    cycles, unfinished = _reachable_cycles(engine, flat, regulatory, result_bytes)
    values = np.zeros((len(flat), len(env)), np.uint8)
    statuses = np.full(values.shape, "incomplete", dtype="<U20")
    d = g * (g + 1) // 2
    c = 2 * d + 3 * g
    inventory_row = 4 * (2 * g + 4 * d)
    regulatory_lane = 4 * (5 * g + 6 * d)
    available = budget - engine._table_bytes - result_bytes
    valid = [i for i, cycle in enumerate(cycles) if cycle and not unfinished[i]]
    sample_launches = source_launches = 0
    cursor = 0
    while cursor < len(valid):
        # Admission precedes source uploads. Reserve the largest exact history,
        # one environment lane per sample, and both host/device inventories.
        stop = min(len(valid), cursor + engine.execution.batch_size)
        while True:
            indices = valid[cursor:stop]
            ns = len(indices)
            width = max(len(cycles[i]) for i in indices) + 1
            source_bytes = ns * width * inventory_row
            fixed = 4 * ns * (g + width + 1)
            worst_lane = 4 * (4 * g + 7 * d + 6 * c + 1024 * (g + d))
            # Include phase genes/bits on host and one source construction lane.
            source_peak = (source_bytes + 8 * ns * width * g
                           + regulatory_lane + inventory_row + 4 * g)
            assay_peak = 2 * source_bytes + fixed + ns * (worst_lane + 8) + 16
            if max(source_peak, assay_peak) <= available:
                break
            if ns == 1:
                # Exact capacity is learned from the all-on inventory. Admit
                # source generation first; assay admission below uses its bound.
                if source_peak > available:
                    raise MemoryError(
                        "reachable cycle inventories exceed the execution memory budget")
                break
            stop = cursor + max(1, ns // 2)
        phase_genes = np.repeat(flat[indices], width, axis=0)
        bits = np.zeros((ns * width, g), np.int32)
        for row, index in enumerate(indices):
            for phase, key in enumerate(cycles[index]):
                bits[row * width + phase] = _unpack(key, g)
            bits[(row + 1) * width - 1] = 1
        shapes = [(ns * width, g)] * 2 + [(ns * width, d)] * 4
        source = [np.empty(shape, np.int32) for shape in shapes]
        source_chunk = min(engine.execution.batch_size,
                           (available - source_bytes - phase_genes.nbytes - bits.nbytes)
                           // (regulatory_lane + inventory_row + 4 * g))
        if source_chunk < 1:
            raise MemoryError("one reachable regulatory lane exceeds the execution memory budget")
        for start in range(0, ns * width, source_chunk):
            end = min(ns * width, start + source_chunk)
            outputs = _single(engine, phase_genes[start:end], bits[start:end], regulatory)[1:]
            source_launches += 1
            for target, output in zip(source, outputs, strict=True):
                target[start:end] = output
            del outputs, target, output
        del phase_genes, bits
        capacities = []
        for row in range(ns):
            all_on = (row + 1) * width - 1
            bound = min(engine.execution.max_steps, comb(
                g + int(np.count_nonzero(source[0][all_on] > 0))
                + int(np.count_nonzero(source[2][all_on] >= 0)), g))
            capacities.append(bound if bound <= 1024 else 1)
        # Mode 1 uses Brent. Exact-history lanes retain their own mode and may
        # share a larger exact allocation without changing first-repeat checks.
        for brent in (True, False):
            rows = [r for r, h in enumerate(capacities) if (h == 1) == brent]
            if not rows:
                continue
            history = max(capacities[r] for r in rows)
            n = len(rows)
            packed = [a.reshape(ns, width, -1)[rows].reshape(n * width, -1) for a in source]
            packed_bytes = sum(a.nbytes for a in packed)
            lane_bytes = 4 * (4 * g + 7 * d + 6 * c + history * (g + d))
            group_fixed = 4 * n * (g + width + 1)
            remaining = available - source_bytes - 2 * packed_bytes - group_fixed
            met_chunk = min(len(env), engine.execution.batch_size,
                            (remaining - 16) // (n * (lane_bytes + 8) + 8))
            if met_chunk < 1:
                raise MemoryError(
                    "one reachable metabolic lane exceeds the execution memory budget")
            selected = [indices[r] for r in rows]
            device_genes = engine._i32((n, g), flat[selected])
            source_device = tuple(engine._i32(a.shape, a) for a in packed)
            cycle_indices = np.broadcast_to(np.arange(width, dtype=np.int32), (n, width)).copy()
            device_cycles = engine._i32(cycle_indices.shape, cycle_indices)
            device_lengths = engine._i32(
                (n,), np.asarray([len(cycles[i]) for i in selected], np.int32))
            for start in range(0, len(env), met_chunk):
                end = min(len(env), start + met_chunk)
                targets, lanes = end - start, n * (end - start)
                ids = engine._i32((targets,), met_ids[start:end])
                lengths = engine._i32((targets,), met_lengths[start:end])
                p = [engine._i32((lanes, g)) for _ in range(3)]
                dims = [engine._i32((lanes, d)) for _ in range(6)]
                candidates = [engine._i32((lanes, c)) for _ in range(6)]
                demand_p, demand_d = engine._i32((lanes, g)), engine._i32((lanes, d))
                hp = engine._i32((lanes, history, g))
                hd = engine._i32((lanes, history, d))
                output, status = engine._i32((n, targets)), engine._i32((n, targets))
                metabolic_assay(device_genes, *regulatory[:3], *regulatory[4:5],
                                *regulatory[5:], *metabolic, ids, lengths,
                                device_cycles, device_lengths, *source_device,
                                *p, *dims, *candidates, demand_p, demand_d,
                                hp, hd, output, status, engine.execution.max_steps)
                sample_launches += 1
                values[selected, start:end] = output.to_numpy().astype(np.uint8)
                statuses[selected, start:end] = np.where(status.to_numpy() != 0,
                                                          "incomplete", "complete")
                del p, dims, candidates, demand_p, demand_d, hp, hd, output, status, ids, lengths
            del device_genes, source_device, device_cycles, device_lengths
            del packed
        del source
        cursor = stop
    return ArrayResult(values.reshape((*leading, len(env))), statuses.reshape((*leading, len(env))),
                       {"ruleset": RULESET, "backend": engine.execution.backend,
                        "table_digest": str(engine.tables.digest), "environment": env,
                        "regulatory_mode": "reachable_multiword",
                        "batch_scope": "reachable_sample_environment",
                        "metabolic_launches": sample_launches,
                        "source_launches": source_launches})
