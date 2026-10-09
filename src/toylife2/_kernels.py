"""Quadrants kernels for the canonical integer-energy toyLIFE rules.

Every parallel lane owns its scratch region. Chemistry is ordered inside a lane;
there are no floating-point reductions or atomics.
"""

# Quadrants' Python stubs do not model NDArray.shape inside its compiled DSL.
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false

from __future__ import annotations

import quadrants as qd


@qd.kernel(fastcache=True)
def regulatory_step(
    genes: qd.types.NDArray[qd.i32, 2],
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
    truth: qd.types.NDArray[qd.i32, 2],
) -> None:
    for lane in range(protein_ids.shape[0]):
        n_genes = genes.shape[1]
        n_states = truth.shape[1]
        sample = lane // n_states
        state = lane % n_states
        n_proteins = 0
        for c in range(candidate_id.shape[1]):
            candidate_id[lane, c] = -1
            candidate_accepted[lane, c] = 0
        for g in range(n_genes):
            protein_ids[lane, g] = -1
            protein_counts[lane, g] = 0
            if (state & (1 << (n_genes - g - 1))) != 0:
                protein = gene_to_protein[genes[sample, g] & 65535]
                slot = -1
                for j in range(n_proteins):
                    if protein_ids[lane, j] == protein:
                        slot = j
                if slot < 0:
                    slot = n_proteins
                    protein_ids[lane, slot] = protein
                    n_proteins += 1
                protein_counts[lane, slot] += 1

        # The reference orders protein species by ID before looking up a
        # perimeter pair; dimer geometry can depend on that orientation.
        for i in range(n_proteins):
            for j in range(i + 1, n_proteins):
                if protein_ids[lane, j] < protein_ids[lane, i]:
                    old_id = protein_ids[lane, i]
                    old_count = protein_counts[lane, i]
                    protein_ids[lane, i] = protein_ids[lane, j]
                    protein_counts[lane, i] = protein_counts[lane, j]
                    protein_ids[lane, j] = old_id
                    protein_counts[lane, j] = old_count

        # Enumerate distinct protein species, including an admissible
        # homodimer with floor(n/2) copies.
        n_candidates = 0
        for i in range(n_proteins):
            for j in range(i, n_proteins):
                left = protein_ids[lane, i]
                right = protein_ids[lane, j]
                per_left = protein_perimeter[left]
                per_right = protein_perimeter[right]
                if per_left >= 0 and per_right >= 0:
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
                            candidate_accepted[lane, n_candidates] = 0
                            n_candidates += 1

        # Process energy groups simultaneously. A candidate whose same-energy
        # group overclaims either constituent is discarded.
        processed = 0
        last_energy = -2147483647
        while processed < n_candidates:
            energy = 2147483647
            for c in range(n_candidates):
                e = candidate_energy[lane, c]
                if e > last_energy and e < energy:
                    energy = e
            # A stronger group may already have consumed either constituent.
            # Recompute the available copies for this group from the current
            # inventory before testing simultaneous same-energy demand.
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

        next_state = 0
        for g in range(n_genes):
            promoter = genes[sample, g] >> 16
            best = 2147483647
            winner = 0
            # A tie annihilates all binders at its energy. The next weaker
            # distinct energy is considered before polymerase acts.
            previous = -2147483647
            keep_searching = 1
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
                        if perimeter >= 0:
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
                if ties == 0:
                    winner = 0
                    keep_searching = 0
                elif ties == 1:
                    keep_searching = 0
                else:
                    previous = best
            next_state += winner << (n_genes - g - 1)
        truth[sample, state] = next_state


@qd.kernel(fastcache=True)
def spatial_trajectory(
    ca: qd.types.NDArray[qd.i32, 2],
    morphogen: qd.types.NDArray[qd.i32, 2],
    dimer_present: qd.types.NDArray[qd.i32, 1],
    state: qd.types.NDArray[qd.i32, 2],
    next_state: qd.types.NDArray[qd.i32, 2],
    trajectory: qd.types.NDArray[qd.i32, 3],
    ring: qd.i32,
    input_states: qd.i32,
) -> None:
    for sample in range(ca.shape[0]):
        cells = state.shape[1]
        for t in range(morphogen.shape[0]):
            for c in range(cells):
                left = c - 1
                right = c + 1
                if left < 0:
                    left = cells - 1
                if right == cells:
                    right = 0
                a = state[sample, left]
                b = state[sample, c]
                d = state[sample, right]
                injection = morphogen[t, left]
                if dimer_present[sample] != 0 and a == 3:
                    if injection == 2:
                        a = 4
                    elif injection == 1:
                        a = 5
                else:
                    a = a | injection
                injection = morphogen[t, c]
                if dimer_present[sample] != 0 and b == 3:
                    if injection == 2:
                        b = 4
                    elif injection == 1:
                        b = 5
                else:
                    b = b | injection
                injection = morphogen[t, right]
                if dimer_present[sample] != 0 and d == 3:
                    if injection == 2:
                        d = 4
                    elif injection == 1:
                        d = 5
                else:
                    d = d | injection
                if ring == 0:
                    if c == 0:
                        a = 0
                    if c == cells - 1:
                        d = 0
                next_state[sample, c] = ca[
                    sample, a * input_states * input_states + b * input_states + d
                ]
            for c in range(cells):
                value = next_state[sample, c]
                state[sample, c] = value
                trajectory[sample, t, c] = value


@qd.kernel(fastcache=True)
def preformed_dimer_response(
    genes: qd.types.NDArray[qd.i32, 2],
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
    output: qd.types.NDArray[qd.i32, 2],
) -> None:
    for sample in range(genes.shape[0]):
        a = gene_to_protein[genes[sample, 0] & 65535]
        b = gene_to_protein[genes[sample, 1] & 65535]
        first = a
        second = b
        if first > second:
            first = b
            second = a
        pa = protein_perimeter[first]
        pb = protein_perimeter[second]
        dimer = -1
        if pa >= 0 and pb >= 0:
            dimer = dimer_id[pa, pb]
        for extra in range(2):
            response = 0
            free = a
            if extra == 1:
                free = b
            for gene in range(2):
                promoter = genes[sample, gene] >> 16
                polymerase = polymerase_energy[promoter]
                free_perimeter = protein_perimeter[free]
                e_free = 0
                free_eligible = 0
                if free_perimeter >= 0:
                    bind = protein_promoter_energy[promoter, free_perimeter]
                    e_free = bind + protein_energy[free]
                    if bind < 0:
                        free_eligible = 1
                e_dimer = 0
                dimer_eligible = 0
                if dimer >= 0:
                    bind = dimer_promoter_energy[promoter, dimer]
                    e_dimer = (
                        bind + dimer_energy[dimer] + protein_energy[first] + protein_energy[second]
                    )
                    if bind < 0:
                        dimer_eligible = 1
                on = 0
                previous = -2147483647
                searching = 1
                while searching != 0:
                    best = 2147483647
                    ties = 0
                    on = 0
                    if polymerase < 0 and polymerase > previous:
                        best = polymerase
                        on = 1
                        ties = 1
                    if free_eligible != 0 and e_free > previous:
                        if e_free < best:
                            best = e_free
                            on = protein_promoter_pol[promoter, free_perimeter]
                            ties = 1
                        elif e_free == best:
                            ties += 1
                    if dimer_eligible != 0 and e_dimer > previous:
                        if e_dimer < best:
                            best = e_dimer
                            on = dimer_promoter_pol[promoter, dimer]
                            ties = 1
                        elif e_dimer == best:
                            ties += 1
                    if ties == 0:
                        on = 0
                        searching = 0
                    elif ties == 1:
                        searching = 0
                    else:
                        previous = best
                response += on << (1 - gene)
            output[sample, extra] = response


@qd.kernel(fastcache=True)
def pair_promoter_responses(
    protein_pairs: qd.types.NDArray[qd.i32, 2],
    protein_perimeter: qd.types.NDArray[qd.i32, 1],
    protein_energy: qd.types.NDArray[qd.i32, 1],
    dimer_id: qd.types.NDArray[qd.i32, 2],
    dimer_energy: qd.types.NDArray[qd.i32, 1],
    protein_promoter_energy: qd.types.NDArray[qd.i32, 2],
    protein_promoter_pol: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_energy: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_pol: qd.types.NDArray[qd.i32, 2],
    polymerase_energy: qd.types.NDArray[qd.i32, 1],
    responses: qd.types.NDArray[qd.i32, 3],
    dimer_present: qd.types.NDArray[qd.i32, 1],
) -> None:
    """Six chemical inputs once per protein pair, then all sixteen promoters."""
    for pair in range(protein_pairs.shape[0]):
        a = protein_pairs[pair, 0]
        b = protein_pairs[pair, 1]
        pa = -1
        pb = -1
        if a > 0:
            pa = protein_perimeter[a]
        if b > 0:
            pb = protein_perimeter[b]
        ab = -1
        if pa >= 0 and pb >= 0:
            lo = pa
            hi = pb
            if a > b:
                lo = pb
                hi = pa
            ab = dimer_id[lo, hi]
        dimer_present[pair] = 0
        if ab >= 0:
            dimer_present[pair] = 1
        for state in range(6):
            free_a = 0
            free_b = 0
            if state == 1:
                free_b = 1
            elif state == 2:
                free_a = 1
            elif state == 3:
                free_a = 1
                free_b = 1
            elif state == 4:
                free_a = 2
                free_b = 1
            elif state == 5:
                free_a = 1
                free_b = 2
            did = -1
            dim_total = 0
            if ab >= 0 and state >= 3:
                # AB has formed before spatial propagation. The extra molecule
                # in state 4/5 cannot displace this preformed dimer.
                did = ab
                dim_total = dimer_energy[did] + protein_energy[a] + protein_energy[b]
                free_a -= 1
                free_b -= 1
            elif state == 4 and pa >= 0:
                aa = dimer_id[pa, pa]
                if aa >= 0:
                    did = aa
                    dim_total = dimer_energy[did] + 2 * protein_energy[a]
                    free_a -= 2
            elif state == 5 and pb >= 0:
                bb = dimer_id[pb, pb]
                if bb >= 0:
                    did = bb
                    dim_total = dimer_energy[did] + 2 * protein_energy[b]
                    free_b -= 2
            if a == b:
                free_a += free_b
                free_b = 0
            for promoter in range(16):
                previous = -2147483647
                activated = 0
                searching = 1
                while searching != 0:
                    best = 2147483647
                    ties = 0
                    activated = 0
                    poly = polymerase_energy[promoter]
                    if poly < 0 and poly > previous:
                        best = poly
                        activated = 1
                        ties = 1
                    if free_a > 0 and pa >= 0:
                        binding = protein_promoter_energy[promoter, pa]
                        energy = binding + protein_energy[a]
                        if binding < 0 and energy > previous:
                            if energy < best:
                                best = energy
                                activated = protein_promoter_pol[promoter, pa]
                                ties = 1
                            elif energy == best:
                                ties += 1
                    if free_b > 0 and pb >= 0:
                        binding = protein_promoter_energy[promoter, pb]
                        energy = binding + protein_energy[b]
                        if binding < 0 and energy > previous:
                            if energy < best:
                                best = energy
                                activated = protein_promoter_pol[promoter, pb]
                                ties = 1
                            elif energy == best:
                                ties += 1
                    if did >= 0:
                        binding = dimer_promoter_energy[promoter, did]
                        energy = binding + dim_total
                        if binding < 0 and energy > previous:
                            if energy < best:
                                best = energy
                                activated = dimer_promoter_pol[promoter, did]
                                ties = 1
                            elif energy == best:
                                ties += 1
                    if ties <= 1:
                        searching = 0
                    else:
                        previous = best
                responses[pair, promoter, state] = activated


@qd.kernel(fastcache=True)
def metabolic_assay(
    genes: qd.types.NDArray[qd.i32, 2],
    gene_to_protein: qd.types.NDArray[qd.i32, 1],
    protein_perimeter: qd.types.NDArray[qd.i32, 1],
    protein_energy: qd.types.NDArray[qd.i32, 1],
    dimer_energy: qd.types.NDArray[qd.i32, 1],
    protein_promoter_energy: qd.types.NDArray[qd.i32, 2],
    protein_promoter_pol: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_energy: qd.types.NDArray[qd.i32, 2],
    dimer_promoter_pol: qd.types.NDArray[qd.i32, 2],
    polymerase_energy: qd.types.NDArray[qd.i32, 1],
    protein_met_energy: qd.types.NDArray[qd.i32, 2],
    dimer_met_energy: qd.types.NDArray[qd.i32, 2],
    dimer_met_seq: qd.types.NDArray[qd.i32, 2],
    dimer_met_pos: qd.types.NDArray[qd.i32, 2],
    dimer_met_breakable: qd.types.NDArray[qd.i32, 2],
    breaking_seq_to_index: qd.types.NDArray[qd.i32, 1],
    breaking_energy: qd.types.NDArray[qd.i32, 3],
    met_ids: qd.types.NDArray[qd.i32, 1],
    met_lengths: qd.types.NDArray[qd.i32, 1],
    cycle_states: qd.types.NDArray[qd.i32, 2],
    cycle_lengths: qd.types.NDArray[qd.i32, 1],
    source_pids: qd.types.NDArray[qd.i32, 2],
    source_pcounts: qd.types.NDArray[qd.i32, 2],
    source_dimid: qd.types.NDArray[qd.i32, 2],
    source_dimleft: qd.types.NDArray[qd.i32, 2],
    source_dimright: qd.types.NDArray[qd.i32, 2],
    source_dimcount: qd.types.NDArray[qd.i32, 2],
    pids: qd.types.NDArray[qd.i32, 2],
    pcounts: qd.types.NDArray[qd.i32, 2],
    prev_pcounts: qd.types.NDArray[qd.i32, 2],
    dimids: qd.types.NDArray[qd.i32, 2],
    dimleft: qd.types.NDArray[qd.i32, 2],
    dimright: qd.types.NDArray[qd.i32, 2],
    dimenergies: qd.types.NDArray[qd.i32, 2],
    dimcounts: qd.types.NDArray[qd.i32, 2],
    prev_dimcounts: qd.types.NDArray[qd.i32, 2],
    candidate_kind: qd.types.NDArray[qd.i32, 2],
    candidate_left: qd.types.NDArray[qd.i32, 2],
    candidate_right: qd.types.NDArray[qd.i32, 2],
    candidate_energy: qd.types.NDArray[qd.i32, 2],
    candidate_capacity: qd.types.NDArray[qd.i32, 2],
    candidate_accepted: qd.types.NDArray[qd.i32, 2],
    protein_demand: qd.types.NDArray[qd.i32, 2],
    dimer_demand: qd.types.NDArray[qd.i32, 2],
    history_proteins: qd.types.NDArray[qd.i32, 3],
    history_dimers: qd.types.NDArray[qd.i32, 3],
    values: qd.types.NDArray[qd.i32, 2],
    statuses: qd.types.NDArray[qd.i32, 2],
    max_steps: qd.i32,
) -> None:
    for lane in range(pids.shape[0]):
        n_genes = genes.shape[1]
        met_count = met_ids.shape[0]
        sample = lane // met_count
        target = lane % met_count
        met = met_ids[target]
        met_length = met_lengths[target]
        n_states = cycle_states.shape[1]
        all_on_lane = sample * n_states + n_states - 1
        n_proteins = 0
        for i in range(n_genes):
            pids[lane, i] = -1
            pcounts[lane, i] = 0
            protein = source_pids[all_on_lane, i]
            if protein > 0:
                pids[lane, n_proteins] = protein
                n_proteins += 1
        n_dimers = 0
        for c in range(dimids.shape[1]):
            dimids[lane, c] = -1
            dimcounts[lane, c] = 0
            did = source_dimid[all_on_lane, c]
            if did >= 0:
                left_id = source_pids[all_on_lane, source_dimleft[all_on_lane, c]]
                right_id = source_pids[all_on_lane, source_dimright[all_on_lane, c]]
                li = -1
                ri = -1
                for i in range(n_proteins):
                    if pids[lane, i] == left_id:
                        li = i
                    if pids[lane, i] == right_id:
                        ri = i
                if li >= 0 and ri >= 0:
                    dimids[lane, n_dimers] = did
                    dimleft[lane, n_dimers] = li
                    dimright[lane, n_dimers] = ri
                    dimenergies[lane, n_dimers] = (
                        dimer_energy[did] + protein_energy[left_id] + protein_energy[right_id]
                    )
                    n_dimers += 1

        result = 0
        incomplete = 0
        for phase in range(cycle_lengths[sample]):
            if result == 0:
                state = cycle_states[sample, phase]
                source_lane = sample * n_states + state
                for i in range(n_proteins):
                    pcounts[lane, i] = 0
                for d in range(n_dimers):
                    dimcounts[lane, d] = 0
                for i in range(n_genes):
                    protein = source_pids[source_lane, i]
                    if protein > 0 and source_pcounts[source_lane, i] > 0:
                        for j in range(n_proteins):
                            if pids[lane, j] == protein:
                                pcounts[lane, j] = source_pcounts[source_lane, i]
                for c in range(dimids.shape[1]):
                    amount = source_dimcount[source_lane, c]
                    if amount > 0:
                        left_id = source_pids[source_lane, source_dimleft[source_lane, c]]
                        right_id = source_pids[source_lane, source_dimright[source_lane, c]]
                        for d in range(n_dimers):
                            if (
                                dimids[lane, d] == source_dimid[source_lane, c]
                                and pids[lane, dimleft[lane, d]] == left_id
                                and pids[lane, dimright[lane, d]] == right_id
                            ):
                                dimcounts[lane, d] = amount

                met_mode = 1  # 1 free, 2 protein-bound, 3 dimer-bound, 0 cleaved/absent
                bound_dimer = -1
                finished = 0
                brent_power = 1
                brent_lam = 0
                for i in range(n_proteins):
                    history_proteins[lane, 0, i] = pcounts[lane, i]
                for d in range(n_dimers):
                    history_dimers[lane, 0, d] = dimcounts[lane, d]
                step = 0
                while step < max_steps and finished == 0 and result == 0:
                    for reaction_round in range(2):
                        if reaction_round == 1:
                            # Evaluate every promoter against the same bound
                            # inventory, then translate into separate counts.
                            # This keeps arbitrarily many gene outputs distinct.
                            for i in range(n_proteins):
                                prev_pcounts[lane, i] = 0
                            for gene in range(n_genes):
                                promoter = genes[sample, gene] >> 16
                                previous = -2147483647
                                on = 0
                                searching = 1
                                while searching != 0:
                                    best = 2147483647
                                    ties = 0
                                    on = 0
                                    poly = polymerase_energy[promoter]
                                    if poly < 0 and poly > previous:
                                        best = poly
                                        on = 1
                                        ties = 1
                                    for i in range(n_proteins):
                                        if pcounts[lane, i] > 0:
                                            protein = pids[lane, i]
                                            perimeter = protein_perimeter[protein]
                                            binding = protein_promoter_energy[promoter, perimeter]
                                            e = binding + protein_energy[protein]
                                            if binding < 0 and e > previous:
                                                if e < best:
                                                    best = e
                                                    on = protein_promoter_pol[promoter, perimeter]
                                                    ties = 1
                                                elif e == best:
                                                    ties += 1
                                    for d in range(n_dimers):
                                        if dimcounts[lane, d] > 0:
                                            did = dimids[lane, d]
                                            binding = dimer_promoter_energy[promoter, did]
                                            e = binding + dimenergies[lane, d]
                                            if binding < 0 and e > previous:
                                                if e < best:
                                                    best = e
                                                    on = dimer_promoter_pol[promoter, did]
                                                    ties = 1
                                                elif e == best:
                                                    ties += 1
                                    if ties <= 1:
                                        searching = 0
                                    else:
                                        previous = best

                                if on != 0:
                                    protein = gene_to_protein[genes[sample, gene] & 65535]
                                    if protein > 0:
                                        for i in range(n_proteins):
                                            if pids[lane, i] == protein:
                                                prev_pcounts[lane, i] += 1
                            for i in range(n_proteins):
                                pcounts[lane, i] = prev_pcounts[lane, i]
                            for d in range(n_dimers):
                                dimcounts[lane, d] = 0

                        n_candidates = 0
                        for d in range(n_dimers):
                            i = dimleft[lane, d]
                            j = dimright[lane, d]
                            if pcounts[lane, i] > 0 and pcounts[lane, j] > 0:
                                candidate_kind[lane, n_candidates] = 0
                                candidate_left[lane, n_candidates] = i
                                candidate_right[lane, n_candidates] = d
                                candidate_energy[lane, n_candidates] = dimenergies[lane, d]
                                n_candidates += 1
                        if met_mode == 1:
                            for i in range(n_proteins):
                                if pcounts[lane, i] > 0:
                                    protein = pids[lane, i]
                                    contact = protein_met_energy[met, protein_perimeter[protein]]
                                    if contact < 0:
                                        candidate_kind[lane, n_candidates] = 1
                                        candidate_left[lane, n_candidates] = i
                                        candidate_right[lane, n_candidates] = -1
                                        candidate_energy[lane, n_candidates] = (
                                            contact + protein_energy[protein]
                                        )
                                        n_candidates += 1
                            for d in range(n_dimers):
                                if dimcounts[lane, d] > 0:
                                    did = dimids[lane, d]
                                    contact = dimer_met_energy[met, did]
                                    if contact < 0:
                                        candidate_kind[lane, n_candidates] = 2
                                        candidate_left[lane, n_candidates] = d
                                        candidate_right[lane, n_candidates] = -1
                                        candidate_energy[lane, n_candidates] = (
                                            contact + dimenergies[lane, d]
                                        )
                                        n_candidates += 1
                        elif met_mode == 3:
                            did = dimids[lane, bound_dimer]
                            if dimer_met_breakable[met, did] != 0:
                                position = dimer_met_pos[met, did]
                                if position > 0 and position < met_length:
                                    sequence = dimer_met_seq[met, did]
                                    seq_index = breaking_seq_to_index[sequence]
                                    if seq_index >= 0:
                                        left_protein = pids[lane, dimleft[lane, bound_dimer]]
                                        right_protein = pids[lane, dimright[lane, bound_dimer]]
                                        old_energy = dimenergies[lane, bound_dimer]
                                        for i in range(n_proteins):
                                            if pcounts[lane, i] > 0:
                                                breaker = pids[lane, i]
                                                perimeter = protein_perimeter[breaker]
                                                for side in range(2):
                                                    contact = breaking_energy[
                                                        seq_index, perimeter, side
                                                    ]
                                                    attached = left_protein
                                                    if side == 1:
                                                        attached = right_protein
                                                    e = (
                                                        contact
                                                        + protein_energy[breaker]
                                                        + protein_energy[attached]
                                                    )
                                                    if contact < 0 and e < old_energy:
                                                        candidate_kind[lane, n_candidates] = 3
                                                        candidate_left[lane, n_candidates] = i
                                                        candidate_right[lane, n_candidates] = side
                                                        candidate_energy[lane, n_candidates] = e
                                                        n_candidates += 1

                        previous_energy = -2147483647
                        processed = 0
                        while processed < n_candidates:
                            energy = 2147483647
                            for c in range(n_candidates):
                                e = candidate_energy[lane, c]
                                if e > previous_energy and e < energy:
                                    energy = e
                            for i in range(n_proteins):
                                protein_demand[lane, i] = 0
                            for d in range(n_dimers):
                                dimer_demand[lane, d] = 0
                            free_demand = 0
                            bound_demand = 0
                            for c in range(n_candidates):
                                if candidate_energy[lane, c] == energy:
                                    kind = candidate_kind[lane, c]
                                    i = candidate_left[lane, c]
                                    amount = 0
                                    if kind == 0:
                                        d = candidate_right[lane, c]
                                        j = dimright[lane, d]
                                        amount = pcounts[lane, i]
                                        if i == j:
                                            amount = amount // 2
                                        elif pcounts[lane, j] < amount:
                                            amount = pcounts[lane, j]
                                        protein_demand[lane, i] += amount
                                        protein_demand[lane, j] += amount
                                    elif kind == 1:
                                        if pcounts[lane, i] > 0 and met_mode == 1:
                                            amount = 1
                                        protein_demand[lane, i] += amount
                                        free_demand += amount
                                    elif kind == 2:
                                        if dimcounts[lane, i] > 0 and met_mode == 1:
                                            amount = 1
                                        dimer_demand[lane, i] += amount
                                        free_demand += amount
                                    else:
                                        if pcounts[lane, i] > 0 and met_mode == 3:
                                            amount = 1
                                        protein_demand[lane, i] += amount
                                        bound_demand += amount
                                    candidate_capacity[lane, c] = amount
                                    candidate_accepted[lane, c] = 0
                                    processed += 1
                            for c in range(n_candidates):
                                if candidate_energy[lane, c] == energy:
                                    kind = candidate_kind[lane, c]
                                    i = candidate_left[lane, c]
                                    amount = candidate_capacity[lane, c]
                                    acceptable = amount > 0
                                    if kind == 0:
                                        d = candidate_right[lane, c]
                                        j = dimright[lane, d]
                                        if (
                                            protein_demand[lane, i] > pcounts[lane, i]
                                            or protein_demand[lane, j] > pcounts[lane, j]
                                        ):
                                            acceptable = False
                                    elif kind == 1:
                                        if (
                                            protein_demand[lane, i] > pcounts[lane, i]
                                            or free_demand > 1
                                        ):
                                            acceptable = False
                                    elif kind == 2:
                                        if (
                                            dimer_demand[lane, i] > dimcounts[lane, i]
                                            or free_demand > 1
                                        ):
                                            acceptable = False
                                    elif (
                                        protein_demand[lane, i] > pcounts[lane, i]
                                        or bound_demand > 1
                                    ):
                                        acceptable = False
                                    if acceptable:
                                        candidate_accepted[lane, c] = amount
                            for c in range(n_candidates):
                                if candidate_energy[lane, c] == energy:
                                    amount = candidate_accepted[lane, c]
                                    if amount > 0:
                                        kind = candidate_kind[lane, c]
                                        i = candidate_left[lane, c]
                                        if kind == 0:
                                            d = candidate_right[lane, c]
                                            j = dimright[lane, d]
                                            pcounts[lane, i] -= amount
                                            pcounts[lane, j] -= amount
                                            dimcounts[lane, d] += amount
                                        elif kind == 1:
                                            pcounts[lane, i] -= amount
                                            met_mode = 2
                                        elif kind == 2:
                                            dimcounts[lane, i] -= amount
                                            met_mode = 3
                                            bound_dimer = i
                                        else:
                                            pcounts[lane, i] -= amount
                                            met_mode = 0
                                            result = 1
                            previous_energy = energy
                    if result == 0:
                        if met_mode == 2 or met_mode == 3:
                            met_mode = 1
                            if history_proteins.shape[1] == 1:
                                same = 1
                                for i in range(n_proteins):
                                    if pcounts[lane, i] != history_proteins[lane, 0, i]:
                                        same = 0
                                for d in range(n_dimers):
                                    if dimcounts[lane, d] != history_dimers[lane, 0, d]:
                                        same = 0
                                if same != 0:
                                    finished = 1
                                else:
                                    brent_lam += 1
                                    if brent_lam == brent_power:
                                        for i in range(n_proteins):
                                            history_proteins[lane, 0, i] = pcounts[lane, i]
                                        for d in range(n_dimers):
                                            history_dimers[lane, 0, d] = dimcounts[lane, d]
                                        if brent_power <= max_steps // 2:
                                            brent_power *= 2
                                        else:
                                            brent_power = max_steps
                                        brent_lam = 0
                            else:
                                for prior in range(step + 1):
                                    same = 1
                                    for i in range(n_proteins):
                                        if pcounts[lane, i] != history_proteins[lane, prior, i]:
                                            same = 0
                                    for d in range(n_dimers):
                                        if dimcounts[lane, d] != history_dimers[lane, prior, d]:
                                            same = 0
                                    if same != 0:
                                        finished = 1
                                if (
                                    finished == 0
                                    and step + 1 < max_steps
                                    and step + 1 < history_proteins.shape[1]
                                ):
                                    for i in range(n_proteins):
                                        history_proteins[lane, step + 1, i] = pcounts[lane, i]
                                    for d in range(n_dimers):
                                        history_dimers[lane, step + 1, d] = dimcounts[lane, d]
                        else:
                            finished = 1
                    step += 1
                if result == 0 and finished == 0:
                    incomplete = 1
        values[sample, target] = result
        statuses[sample, target] = incomplete if result == 0 else 0
