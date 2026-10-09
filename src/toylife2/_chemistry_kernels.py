"""General molecule reaction graphs executed in independent Quadrants lanes.

Candidate eligibility is captured once at the start of each chemistry phase.
The graph is ordered by exact integer energy; each energy group proposes demand
simultaneously before any accepted reaction changes the molecule inventory.
"""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false
from __future__ import annotations

import quadrants as qd


@qd.func
def reaction_phase(
    lane: qd.i32,
    counts: qd.template(),
    reactions: qd.template(),
    eligible: qd.template(),
    capacity: qd.template(),
    demand: qd.template(),
    events: qd.template(),
    generated: qd.template(),
):
    # reaction columns: energy, a, a_units, b, b_units, product, left, right
    for c in range(reactions.shape[1]):
        a = reactions[lane, c, 1]
        b = reactions[lane, c, 3]
        eligible[lane, c] = 0
        if a >= 0:
            if counts[lane, a] >= reactions[lane, c, 2]:
                if b < 0 or counts[lane, b] >= reactions[lane, c, 4]:
                    eligible[lane, c] = 1
    start = 0
    while start < reactions.shape[1]:
        end = start + 1
        while end < reactions.shape[1] and reactions[lane, end, 0] == reactions[lane, start, 0]:
            end += 1
        for s in range(counts.shape[1]):
            demand[lane, s] = 0
        for c in range(start, end):
            capacity[lane, c] = 0
            if eligible[lane, c] != 0:
                a = reactions[lane, c, 1]
                b = reactions[lane, c, 3]
                n = counts[lane, a] // reactions[lane, c, 2]
                if b >= 0:
                    n = qd.min(n, counts[lane, b] // reactions[lane, c, 4])
                capacity[lane, c] = n
                demand[lane, a] += n * reactions[lane, c, 2]
                if b >= 0:
                    demand[lane, b] += n * reactions[lane, c, 4]
        # All acceptance flags must be computed before applying this group.
        for c in range(start, end):
            if capacity[lane, c] > 0:
                a = reactions[lane, c, 1]
                b = reactions[lane, c, 3]
                if demand[lane, a] > counts[lane, a]:
                    capacity[lane, c] = 0
                elif b >= 0 and demand[lane, b] > counts[lane, b]:
                    capacity[lane, c] = 0
        for c in range(start, end):
            n = capacity[lane, c]
            if n > 0:
                a = reactions[lane, c, 1]
                b = reactions[lane, c, 3]
                counts[lane, a] -= n * reactions[lane, c, 2]
                if b >= 0:
                    counts[lane, b] -= n * reactions[lane, c, 4]
                product = reactions[lane, c, 5]
                if product >= 0:
                    counts[lane, product] += n
                left = reactions[lane, c, 6]
                right = reactions[lane, c, 7]
                if left >= 0:
                    counts[lane, left] += n
                    counts[lane, right] += n
                    generated[lane, left] += n
                    generated[lane, right] += n
                    events[lane, c] += n
        start = end


@qd.kernel(fastcache=True)
def canonical_step(
    source: qd.types.NDArray[qd.i32, 2],
    inputs: qd.types.NDArray[qd.i32, 2],
    target: qd.types.NDArray[qd.i32, 2],
    kinds: qd.types.NDArray[qd.i32, 2],
    release: qd.types.NDArray[qd.i32, 2],
    reactions: qd.types.NDArray[qd.i32, 3],
    products: qd.types.NDArray[qd.i32, 2],
    promoters: qd.types.NDArray[qd.i32, 4],
    polymerase: qd.types.NDArray[qd.i32, 2],
    expression: qd.types.NDArray[qd.i32, 2],
    after_input: qd.types.NDArray[qd.i32, 2],
    after_output: qd.types.NDArray[qd.i32, 2],
    eligible: qd.types.NDArray[qd.i32, 2],
    capacity: qd.types.NDArray[qd.i32, 2],
    demand: qd.types.NDArray[qd.i32, 2],
    events: qd.types.NDArray[qd.i32, 2],
    generated: qd.types.NDArray[qd.i32, 2],
    first_events: qd.types.NDArray[qd.i32, 2],
    replenish: qd.i32,
    regulatory: qd.i32,
) -> None:
    for lane in range(source.shape[0]):
        for s in range(source.shape[1]):
            target[lane, s] = source[lane, s]
            if regulatory != 0 and (kinds[lane, s] == 1 or kinds[lane, s] == 2):
                target[lane, s] = 0
            if inputs[lane, s] >= 0:
                if replenish != 0:
                    target[lane, s] = inputs[lane, s]
                else:
                    target[lane, s] += inputs[lane, s]
            generated[lane, s] = 0
        for c in range(reactions.shape[1]):
            events[lane, c] = 0
        if regulatory != 0:
            # Explicit expression-clamp adapter: reconstruct requested expression
            # each update rather than enumerating the Boolean state space.
            for g in range(products.shape[1]):
                if expression[lane, g] != 0 and products[lane, g] >= 0:
                    target[lane, products[lane, g]] += 1
        reaction_phase(lane, target, reactions, eligible, capacity, demand, events, generated)
        for c in range(reactions.shape[1]):
            first_events[lane, c] = events[lane, c]
        for s in range(source.shape[1]):
            after_input[lane, s] = target[lane, s]
        for g in range(products.shape[1]):
            best_energy = 2147483647
            active = 0
            found = 0
            # Search unique energy groups, including polymerase. A tied best
            # group is skipped and the next unique group can activate.
            lower = -2147483647
            searching = 1
            while searching != 0:
                best_energy = 2147483647
                hits = 0
                active = 0
                poly = polymerase[lane, g]
                if poly < 0 and poly > lower:
                    best_energy = poly
                    hits = 1
                    active = 1
                for s in range(source.shape[1]):
                    energy = promoters[lane, g, s, 0]
                    if target[lane, s] > 0 and energy != 2147483647 and energy > lower:
                        if energy < best_energy:
                            best_energy = energy
                            hits = 1
                            active = promoters[lane, g, s, 1]
                        elif energy == best_energy:
                            hits += 1
                if hits == 1:
                    found = active
                    searching = 0
                elif hits == 0:
                    searching = 0
                else:
                    lower = best_energy
            expression[lane, g] = found
        if regulatory == 0:
            for s in range(source.shape[1]):
                if kinds[lane, s] == 1 or kinds[lane, s] == 2:
                    target[lane, s] = 0
            for g in range(products.shape[1]):
                if expression[lane, g] != 0 and products[lane, g] >= 0:
                    target[lane, products[lane, g]] += 1
            reaction_phase(lane, target, reactions, eligible, capacity, demand, events, generated)
        for s in range(source.shape[1]):
            after_output[lane, s] = target[lane, s]
        if regulatory == 0:
            # Canonical release: bound metabolites plus surviving generated
            # fragments; free environment metabolites are discarded.
            for s in range(source.shape[1]):
                if kinds[lane, s] == 3:
                    target[lane, s] = qd.min(generated[lane, s], target[lane, s])
            for s in range(source.shape[1]):
                if kinds[lane, s] == 4:
                    target[lane, release[lane, s]] += target[lane, s]
                    target[lane, s] = 0


@qd.kernel(fastcache=True)
def tissue_tick(
    ca: qd.types.NDArray[qd.i32, 2],
    injections: qd.types.NDArray[qd.i32, 2],
    dimers: qd.types.NDArray[qd.i32, 1],
    state: qd.types.NDArray[qd.i32, 2],
    target: qd.types.NDArray[qd.i32, 2],
    ring: qd.i32,
) -> None:
    for lane, cell in qd.ndrange(state.shape[0], state.shape[1]):
        cells = state.shape[1]
        left = (cell + cells - 1) % cells
        right = (cell + 1) % cells
        a = state[lane, left]
        b = state[lane, cell]
        c = state[lane, right]
        ia = injections[lane, left]
        ib = injections[lane, cell]
        ic = injections[lane, right]
        if dimers[lane] != 0 and a == 3 and ia == 2:
            a = 4
        elif dimers[lane] != 0 and a == 3 and ia == 1:
            a = 5
        else:
            a = a | ia
        if dimers[lane] != 0 and b == 3 and ib == 2:
            b = 4
        elif dimers[lane] != 0 and b == 3 and ib == 1:
            b = 5
        else:
            b = b | ib
        if dimers[lane] != 0 and c == 3 and ic == 2:
            c = 4
        elif dimers[lane] != 0 and c == 3 and ic == 1:
            c = 5
        else:
            c = c | ic
        if ring == 0:
            if cell == 0:
                a = 0
            if cell == cells - 1:
                c = 0
        target[lane, cell] = ca[lane, a * 36 + b * 6 + c]
