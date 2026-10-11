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
    lengths: qd.template(),
    ends: qd.template(),
    graph: qd.i32,
):
    # reaction columns: energy, a, a_units, b, b_units, product, left, right
    for c in range(lengths[graph, 1]):
        a = reactions[graph, c, 1]
        b = reactions[graph, c, 3]
        eligible[lane, c] = 0
        if a >= 0:
            if counts[lane, a] >= reactions[graph, c, 2]:
                if b < 0 or counts[lane, b] >= reactions[graph, c, 4]:
                    eligible[lane, c] = 1
    start = 0
    while start < lengths[graph, 1]:
        end = ends[graph, start]
        for s in range(lengths[graph, 0]):
            demand[lane, s] = 0
        for c in range(start, end):
            capacity[lane, c] = 0
            if eligible[lane, c] != 0:
                a = reactions[graph, c, 1]
                b = reactions[graph, c, 3]
                n = counts[lane, a] // reactions[graph, c, 2]
                if b >= 0:
                    n = qd.min(n, counts[lane, b] // reactions[graph, c, 4])
                capacity[lane, c] = n
                demand[lane, a] += n * reactions[graph, c, 2]
                if b >= 0:
                    demand[lane, b] += n * reactions[graph, c, 4]
        # All acceptance flags must be computed before applying this group.
        for c in range(start, end):
            if capacity[lane, c] > 0:
                a = reactions[graph, c, 1]
                b = reactions[graph, c, 3]
                if demand[lane, a] > counts[lane, a]:
                    capacity[lane, c] = 0
                elif b >= 0 and demand[lane, b] > counts[lane, b]:
                    capacity[lane, c] = 0
        for c in range(start, end):
            n = capacity[lane, c]
            if n > 0:
                a = reactions[graph, c, 1]
                b = reactions[graph, c, 3]
                counts[lane, a] -= n * reactions[graph, c, 2]
                if b >= 0:
                    counts[lane, b] -= n * reactions[graph, c, 4]
                product = reactions[graph, c, 5]
                if product >= 0:
                    counts[lane, product] += n
                left = reactions[graph, c, 6]
                right = reactions[graph, c, 7]
                if left >= 0:
                    counts[lane, left] += n
                    counts[lane, right] += n
                    generated[lane, left] += n
                    generated[lane, right] += n
                    events[lane, c] += n
        start = end


@qd.func
def canonical_lane(
    lane: qd.i32,
    graph: qd.i32,
    source: qd.template(),
    inputs: qd.template(),
    target: qd.template(),
    kinds: qd.template(),
    release: qd.template(),
    reactions: qd.template(),
    products: qd.template(),
    previous_expression: qd.template(),
    expression: qd.template(),
    after_input: qd.template(),
    after_output: qd.template(),
    eligible: qd.template(),
    capacity: qd.template(),
    demand: qd.template(),
    events: qd.template(),
    generated: qd.template(),
    first_events: qd.template(),
    lengths: qd.template(),
    ends: qd.template(),
    competitors: qd.template(),
    promoter_ends: qd.template(),
    promoter_lengths: qd.template(),
    replenish: qd.i32,
    regulatory: qd.i32,
):
    for s in range(source.shape[1]):
        target[lane, s] = source[lane, s]
    for s in range(lengths[graph, 0]):
        if regulatory != 0 and (kinds[graph, s] == 1 or kinds[graph, s] == 2):
            target[lane, s] = 0
        if inputs[lane, s] >= 0:
            if replenish != 0:
                target[lane, s] = inputs[lane, s]
            else:
                target[lane, s] += inputs[lane, s]
        generated[lane, s] = 0
    for c in range(events.shape[1]):
        events[lane, c] = 0
    if regulatory != 0:
        # Explicit expression-clamp adapter: reconstruct requested expression
        # each update rather than enumerating the Boolean state space.
        for g in range(products.shape[1]):
            if previous_expression[lane, g] != 0 and products[graph, g] >= 0:
                target[lane, products[graph, g]] += 1
    reaction_phase(
        lane, target, reactions, eligible, capacity, demand, events, generated, lengths, ends, graph
    )
    for c in range(events.shape[1]):
        first_events[lane, c] = events[lane, c]
    for s in range(source.shape[1]):
        after_input[lane, s] = target[lane, s]
    for g in range(products.shape[1]):
        cursor = 0
        found = 0
        searching = 1
        while cursor < promoter_lengths[graph, g] and searching != 0:
            end = promoter_ends[graph, g, cursor]
            hits = 0
            active = 0
            for c in range(cursor, end):
                species = competitors[graph, g, c, 0]
                if species < 0 or target[lane, species] > 0:
                    hits += 1
                    active = competitors[graph, g, c, 1]
            if hits == 1:
                found = active
                searching = 0
            cursor = end
        expression[lane, g] = found
    if regulatory == 0:
        for s in range(lengths[graph, 0]):
            if kinds[graph, s] == 1 or kinds[graph, s] == 2:
                target[lane, s] = 0
        for g in range(products.shape[1]):
            if expression[lane, g] != 0 and products[graph, g] >= 0:
                target[lane, products[graph, g]] += 1
        reaction_phase(
            lane,
            target,
            reactions,
            eligible,
            capacity,
            demand,
            events,
            generated,
            lengths,
            ends,
            graph,
        )
    for s in range(source.shape[1]):
        after_output[lane, s] = target[lane, s]
    if regulatory == 0:
        # Canonical release: bound metabolites plus surviving generated
        # fragments; free environment metabolites are discarded.
        for s in range(lengths[graph, 0]):
            if kinds[graph, s] == 3:
                target[lane, s] = qd.min(generated[lane, s], target[lane, s])
        for s in range(lengths[graph, 0]):
            if kinds[graph, s] == 4:
                target[lane, release[graph, s]] += target[lane, s]
                target[lane, s] = 0


@qd.kernel(fastcache=True)
def canonical_step(
    source: qd.types.NDArray[qd.i32, 2],
    inputs: qd.types.NDArray[qd.i32, 2],
    target: qd.types.NDArray[qd.i32, 2],
    kinds: qd.types.NDArray[qd.i32, 2],
    release: qd.types.NDArray[qd.i32, 2],
    reactions: qd.types.NDArray[qd.i32, 3],
    products: qd.types.NDArray[qd.i32, 2],
    previous_expression: qd.types.NDArray[qd.i32, 2],
    expression: qd.types.NDArray[qd.i32, 2],
    after_input: qd.types.NDArray[qd.i32, 2],
    after_output: qd.types.NDArray[qd.i32, 2],
    eligible: qd.types.NDArray[qd.i32, 2],
    capacity: qd.types.NDArray[qd.i32, 2],
    demand: qd.types.NDArray[qd.i32, 2],
    events: qd.types.NDArray[qd.i32, 2],
    generated: qd.types.NDArray[qd.i32, 2],
    first_events: qd.types.NDArray[qd.i32, 2],
    lengths: qd.types.NDArray[qd.i32, 2],
    ends: qd.types.NDArray[qd.i32, 2],
    competitors: qd.types.NDArray[qd.i32, 4],
    promoter_ends: qd.types.NDArray[qd.i32, 3],
    promoter_lengths: qd.types.NDArray[qd.i32, 2],
    lane_graph: qd.types.NDArray[qd.i32, 1],
    steps: qd.i32,
    pulse: qd.i32,
    replenish: qd.i32,
    regulatory: qd.i32,
) -> None:
    for lane in range(source.shape[0]):
        for tick in range(steps):
            if tick == 0:
                canonical_lane(
                    lane,
                    lane_graph[lane],
                    source,
                    inputs,
                    target,
                    kinds,
                    release,
                    reactions,
                    products,
                    previous_expression,
                    expression,
                    after_input,
                    after_output,
                    eligible,
                    capacity,
                    demand,
                    events,
                    generated,
                    first_events,
                    lengths,
                    ends,
                    competitors,
                    promoter_ends,
                    promoter_lengths,
                    replenish,
                    regulatory,
                )
            else:
                if pulse != 0:
                    for s in range(lengths[lane_graph[lane], 0]):
                        inputs[lane, s] = -1
                canonical_lane(
                    lane,
                    lane_graph[lane],
                    target,
                    inputs,
                    target,
                    kinds,
                    release,
                    reactions,
                    products,
                    expression,
                    expression,
                    after_input,
                    after_output,
                    eligible,
                    capacity,
                    demand,
                    events,
                    generated,
                    first_events,
                    lengths,
                    ends,
                    competitors,
                    promoter_ends,
                    promoter_lengths,
                    replenish,
                    regulatory,
                )


@qd.kernel(fastcache=True)
def copy_chemistry_state(
    counts: qd.types.NDArray[qd.i32, 2],
    bits: qd.types.NDArray[qd.i32, 2],
    events: qd.types.NDArray[qd.i32, 2],
    first: qd.types.NDArray[qd.i32, 2],
    before: qd.types.NDArray[qd.i32, 2],
    after: qd.types.NDArray[qd.i32, 2],
    saved_counts: qd.types.NDArray[qd.i32, 2],
    saved_bits: qd.types.NDArray[qd.i32, 2],
    saved_events: qd.types.NDArray[qd.i32, 2],
    saved_first: qd.types.NDArray[qd.i32, 2],
    saved_before: qd.types.NDArray[qd.i32, 2],
    saved_after: qd.types.NDArray[qd.i32, 2],
) -> None:
    for lane in range(counts.shape[0]):
        for s in range(counts.shape[1]):
            saved_counts[lane, s] = counts[lane, s]
            saved_before[lane, s] = before[lane, s]
            saved_after[lane, s] = after[lane, s]
        for g in range(bits.shape[1]):
            saved_bits[lane, g] = bits[lane, g]
        for c in range(events.shape[1]):
            saved_events[lane, c] = events[lane, c]
            saved_first[lane, c] = first[lane, c]


@qd.kernel(fastcache=True)
def compact_readouts(
    boundary: qd.types.NDArray[qd.i32, 2],
    after_input: qd.types.NDArray[qd.i32, 2],
    after_output: qd.types.NDArray[qd.i32, 2],
    expression: qd.types.NDArray[qd.i32, 2],
    events: qd.types.NDArray[qd.i32, 2],
    first_events: qd.types.NDArray[qd.i32, 2],
    lane_graph: qd.types.NDArray[qd.i32, 1],
    indices: qd.types.NDArray[qd.i32, 3],
    sizes: qd.types.NDArray[qd.i32, 2],
    descriptors: qd.types.NDArray[qd.i32, 2],
    output: qd.types.NDArray[qd.i32, 2],
) -> None:
    for lane, readout in qd.ndrange(output.shape[0], output.shape[1]):
        graph = lane_graph[lane]
        total = 0
        mode = descriptors[readout, 0]
        stage = descriptors[readout, 1]
        for c in range(sizes[graph, readout]):
            index = indices[graph, readout, c]
            if mode == 0:
                total += expression[lane, index]
            elif mode == 2:
                if stage == 1:
                    total += first_events[lane, index]
                else:
                    total += events[lane, index]
            elif stage == 1:
                total += after_input[lane, index]
            elif stage == 2:
                total += after_output[lane, index]
            else:
                total += boundary[lane, index]
        output[lane, readout] = total


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
