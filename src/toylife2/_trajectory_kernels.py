"""Reachable-state regulatory transition with one Boolean word per gene."""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false

from __future__ import annotations

import quadrants as qd


@qd.kernel(fastcache=True)
def regulatory_single(
    genes: qd.types.NDArray[qd.i32, 2],
    input_bits: qd.types.NDArray[qd.i32, 2],
    gene_to_protein: qd.types.NDArray[qd.i32, 1],
    protein_perimeter: qd.types.NDArray[qd.i32, 1],
    protein_energy: qd.types.NDArray[qd.i32, 1],
    dimer_id: qd.types.NDArray[qd.i32, 2],
    dimer_energy: qd.types.NDArray[qd.i32, 1],
    protein_promoter_energy: qd.types.NDArray[qd.i32, 2],
    protein_promoter_pol: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_energy: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_pol: qd.types.NDArray[qd.i32, 2],
    polymerase_energy: qd.types.NDArray[qd.i32, 1],
    protein_ids: qd.types.NDArray[qd.i32, 2],
    protein_counts: qd.types.NDArray[qd.i32, 2],
    candidate_id: qd.types.NDArray[qd.i32, 2],
    candidate_left: qd.types.NDArray[qd.i32, 2],
    candidate_right: qd.types.NDArray[qd.i32, 2],
    candidate_energy: qd.types.NDArray[qd.i32, 2],
    candidate_capacity: qd.types.NDArray[qd.i32, 2],
    candidate_accepted: qd.types.NDArray[qd.i32, 2],
    next_bits: qd.types.NDArray[qd.i32, 2],
) -> None:
    for lane in range(genes.shape[0]):
        n_genes = genes.shape[1]
        n_proteins = 0
        for c in range(candidate_id.shape[1]):
            candidate_id[lane, c] = -1
            candidate_left[lane, c] = -1
            candidate_right[lane, c] = -1
            candidate_accepted[lane, c] = 0
        for g in range(n_genes):
            protein_ids[lane, g] = -1
            protein_counts[lane, g] = 0
            if input_bits[lane, g] != 0:
                protein = gene_to_protein[genes[lane, g] & 65535]
                if protein > 0:
                    slot = -1
                    for j in range(n_proteins):
                        if protein_ids[lane, j] == protein:
                            slot = j
                    if slot < 0:
                        slot = n_proteins
                        protein_ids[lane, slot] = protein
                        n_proteins += 1
                    protein_counts[lane, slot] += 1
        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                if protein_ids[lane, j] < protein_ids[lane, i]:
                    old_id = protein_ids[lane, i]
                    old_count = protein_counts[lane, i]
                    protein_ids[lane, i] = protein_ids[lane, j]
                    protein_counts[lane, i] = protein_counts[lane, j]
                    protein_ids[lane, j] = old_id
                    protein_counts[lane, j] = old_count
        n_candidates = 0
        for i in range(n_proteins):
            for j in range(i, n_proteins):
                left = protein_ids[lane, i]
                right = protein_ids[lane, j]
                per_left = protein_perimeter[left]
                per_right = protein_perimeter[right]
                d = dimer_id[per_left, per_right]
                if d >= 0:
                    capacity = protein_counts[lane, i]
                    if i == j:
                        capacity = capacity // 2
                    elif protein_counts[lane, j] < capacity:
                        capacity = protein_counts[lane, j]
                    if capacity > 0:
                        candidate_id[lane, n_candidates] = d
                        candidate_left[lane, n_candidates] = i
                        candidate_right[lane, n_candidates] = j
                        candidate_energy[lane, n_candidates] = (
                            dimer_energy[d] + protein_energy[left] + protein_energy[right]
                        )
                        candidate_capacity[lane, n_candidates] = capacity
                        n_candidates += 1
        processed = 0
        last_energy = -2147483647
        while processed < n_candidates:
            energy = 2147483647
            for c in range(n_candidates):
                e = candidate_energy[lane, c]
                if e > last_energy and e < energy:
                    energy = e
            for c in range(n_candidates):
                if candidate_energy[lane, c] == energy:
                    i = candidate_left[lane, c]
                    j = candidate_right[lane, c]
                    capacity = protein_counts[lane, i]
                    if i == j:
                        capacity = capacity // 2
                    elif protein_counts[lane, j] < capacity:
                        capacity = protein_counts[lane, j]
                    candidate_capacity[lane, c] = capacity
            for c in range(n_candidates):
                if candidate_energy[lane, c] == energy:
                    i = candidate_left[lane, c]
                    j = candidate_right[lane, c]
                    demand_i = 0
                    demand_j = 0
                    for k in range(n_candidates):
                        if candidate_energy[lane, k] == energy:
                            amount = candidate_capacity[lane, k]
                            if candidate_left[lane, k] == i:
                                demand_i += amount
                            if candidate_right[lane, k] == i:
                                demand_i += amount
                            if candidate_left[lane, k] == j:
                                demand_j += amount
                            if candidate_right[lane, k] == j:
                                demand_j += amount
                    if demand_i <= protein_counts[lane, i] and demand_j <= protein_counts[lane, j]:
                        candidate_accepted[lane, c] = candidate_capacity[lane, c]
                    processed += 1
            for c in range(n_candidates):
                if candidate_energy[lane, c] == energy:
                    amount = candidate_accepted[lane, c]
                    if amount > 0:
                        protein_counts[lane, candidate_left[lane, c]] -= amount
                        protein_counts[lane, candidate_right[lane, c]] -= amount
            last_energy = energy
        for g in range(n_genes):
            promoter = genes[lane, g] >> 16
            previous = -2147483647
            keep_searching = 1
            winner = 0
            while keep_searching != 0:
                best = 2147483647
                winner = 0
                ties = 0
                polymerase = polymerase_energy[promoter]
                if polymerase < 0 and polymerase > previous:
                    best = polymerase
                    winner = 1
                    ties = 1
                for i in range(n_proteins):
                    if protein_counts[lane, i] > 0:
                        protein = protein_ids[lane, i]
                        perimeter = protein_perimeter[protein]
                        binding = protein_promoter_energy[promoter, perimeter]
                        e = binding + protein_energy[protein]
                        if binding < 0 and e > previous:
                            if e < best:
                                best = e
                                winner = protein_promoter_pol[promoter, perimeter]
                                ties = 1
                            elif e == best:
                                ties += 1
                for c in range(n_candidates):
                    if candidate_accepted[lane, c] > 0:
                        d = candidate_id[lane, c]
                        binding = dimer_promoter_energy[promoter, d]
                        left = protein_ids[lane, candidate_left[lane, c]]
                        right = protein_ids[lane, candidate_right[lane, c]]
                        e = binding + dimer_energy[d] + protein_energy[left] + protein_energy[right]
                        if binding < 0 and e > previous:
                            if e < best:
                                best = e
                                winner = dimer_promoter_pol[promoter, d]
                                ties = 1
                            elif e == best:
                                ties += 1
                if ties <= 1:
                    keep_searching = 0
                else:
                    previous = best
            next_bits[lane, g] = winner
