"""Resident independent chemotaxis lanes on Quadrants CPU, Metal and CUDA."""

# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false
from __future__ import annotations

import quadrants as qd

from ._chemistry_kernels import canonical_lane, readout_lane


@qd.func
def distance_lane(lane, world: qd.template(), source: qd.template(), p: qd.template(), scenario):
    squared = world[lane, 0] * 0
    for axis in range(2):
        delta = world[lane, axis] - source[scenario, axis]
        delta -= p[0] * qd.floor(delta / p[0] + 0.5)
        squared += delta * delta
    return qd.sqrt(squared)


@qd.func
def field_lane(
    lane,
    world: qd.template(),
    source: qd.template(),
    waves: qd.template(),
    p: qd.template(),
    scenario,
    kind,
):
    value = world[lane, 0] * 0
    if kind == 1:
        value = p[8] * qd.exp(-distance_lane(lane, world, source, p, scenario) / p[9])
    else:
        for mode in range(waves.shape[0]):
            angle = world[lane, 0] * waves[mode, 0] + world[lane, 1] * waves[mode, 1]
            angle += world[lane, 4 + mode]
            value += qd.cos(angle)
        value = p[6] + p[7] * (value / waves.shape[0])
    return value


@qd.func
def move_lane(
    lane,
    world: qd.template(),
    source: qd.template(),
    waves: qd.template(),
    p: qd.template(),
    draws: qd.template(),
    tick,
    scenario,
    action,
    kind,
    dynamic,
):
    base = 4 + waves.shape[0]
    remaining = world[lane, base + 3]
    time_left = p[13]
    tumble = 0
    hazard = p[3]
    if action != 0:
        hazard = p[4]
    orientation = qd.Vector([world[lane, base], world[lane, base + 1], world[lane, base + 2]])
    steps = qd.cast(p[18], qd.i32)
    for event in range(qd.cast(p[19], qd.i32)):  # pyright: ignore[reportArgumentType]
        if time_left > 0:
            start = base + event * (2 + 3 * steps)
            is_tumble = remaining > 0
            waiting = time_left
            if hazard > 0:
                waiting = -qd.log(draws[tick, scenario, start]) / hazard
            duration = qd.min(time_left, waiting)
            if is_tumble:
                duration = qd.min(time_left, remaining)
            new_tumble = not is_tumble and waiting < time_left
            diffusion = p[12]
            velocity = p[1] / p[13]
            if is_tumble:
                diffusion = p[16]
                velocity *= p[15]
                tumble = 1
            segment_left = duration
            for step in range(steps):  # pyright: ignore[reportArgumentType]
                h = qd.min(segment_left, p[17])
                if step == steps - 1:
                    h = segment_left
                if h > 0:
                    offset = start + 2 + 3 * step
                    noise = qd.Vector(
                        [
                            draws[tick, scenario, offset],
                            draws[tick, scenario, offset + 1],
                            draws[tick, scenario, offset + 2],
                        ]
                    )
                    tangent = noise - noise.dot(orientation) * orientation
                    w = qd.sqrt(2 * diffusion * h) * tangent
                    angle = w.norm()
                    sinc = angle * 0 + 1
                    if angle > 0:
                        sinc = qd.sin(angle) / angle
                    orientation = qd.cos(angle) * orientation + sinc * w
                    orientation /= orientation.norm()
                    world[lane, 0] += velocity * h * orientation[0]
                    world[lane, 1] += velocity * h * orientation[1]
                    segment_left = qd.max(segment_left - h, 0)
            if is_tumble:
                remaining = qd.max(remaining - duration, 0)
            elif new_tumble:
                remaining = -p[14] * qd.log(draws[tick, scenario, start + 1])
            time_left = qd.max(time_left - duration, 0)
    for axis in range(3):
        world[lane, base + axis] = orientation[axis]
    world[lane, base + 3] = remaining
    theta = qd.atan2(orientation[1], orientation[0])
    world[lane, 2] = theta - p[11] * qd.floor(theta / p[11])
    for axis in range(2):
        world[lane, axis] -= p[0] * qd.floor(world[lane, axis] / p[0])
    if dynamic != 0:
        for mode in range(waves.shape[0]):
            phase = world[lane, 4 + mode] + p[10] * draws[tick, scenario, 4 + mode]
            world[lane, 4 + mode] = phase - p[11] * qd.floor(phase / p[11])
    world[lane, 3] = field_lane(lane, world, source, waves, p, scenario, kind)
    if time_left > 0:
        # Rejected before publishing the candidate chemistry/world transaction.
        world[lane, base + 3] = -1
        world[lane, 3] = -1
    return qd.cast(tumble, qd.i32)


@qd.kernel(fastcache=True)
def world_advance(
    world: qd.types.ndarray(ndim=2),
    source: qd.types.ndarray(ndim=2),
    scenarios: qd.types.NDArray[qd.i32, 1],
    waves: qd.types.ndarray(ndim=2),
    p: qd.types.ndarray(ndim=1),
    draws: qd.types.ndarray(ndim=3),
    actions: qd.types.NDArray[qd.i32, 1],
    tumble: qd.types.NDArray[qd.i32, 1],
    kind: qd.i32,
    dynamic: qd.i32,
):
    for lane in range(world.shape[0]):
        tumble[lane] = move_lane(
            lane, world, source, waves, p, draws, 0, scenarios[lane], actions[lane], kind, dynamic
        )


@qd.kernel(fastcache=True)
def decision_block(
    current: qd.types.ndarray(ndim=2),
    world: qd.types.ndarray(ndim=2),
    source_world: qd.types.ndarray(ndim=2),
    scenarios: qd.types.NDArray[qd.i32, 1],
    waves: qd.types.ndarray(ndim=2),
    p: qd.types.ndarray(ndim=1),
    draws: qd.types.ndarray(ndim=3),
    rewards: qd.types.ndarray(ndim=2),
    observations: qd.types.NDArray[qd.i32, 1],
    actions: qd.types.NDArray[qd.i32, 1],
    tumble: qd.types.NDArray[qd.i32, 1],
    supplies: qd.types.NDArray[qd.i32, 2],
    input_plan: qd.types.NDArray[qd.i32, 2],
    cue_slots: qd.types.NDArray[qd.i32, 1],
    counts: qd.types.NDArray[qd.i32, 2],
    target: qd.types.NDArray[qd.i32, 2],
    kinds: qd.types.NDArray[qd.i32, 2],
    release: qd.types.NDArray[qd.i32, 2],
    reactions: qd.types.NDArray[qd.i32, 3],
    products: qd.types.NDArray[qd.i32, 2],
    previous: qd.types.NDArray[qd.i32, 2],
    expression: qd.types.NDArray[qd.i32, 2],
    before: qd.types.NDArray[qd.i32, 2],
    after: qd.types.NDArray[qd.i32, 2],
    eligible: qd.types.NDArray[qd.i32, 2],
    capacity: qd.types.NDArray[qd.i32, 2],
    demand: qd.types.NDArray[qd.i32, 2],
    events: qd.types.NDArray[qd.i32, 2],
    generated: qd.types.NDArray[qd.i32, 2],
    first: qd.types.NDArray[qd.i32, 2],
    lengths: qd.types.NDArray[qd.i32, 2],
    ends: qd.types.NDArray[qd.i32, 2],
    competitors: qd.types.NDArray[qd.i32, 4],
    promoter_ends: qd.types.NDArray[qd.i32, 3],
    promoter_lengths: qd.types.NDArray[qd.i32, 2],
    lane_graph: qd.types.NDArray[qd.i32, 1],
    indices: qd.types.NDArray[qd.i32, 3],
    sizes: qd.types.NDArray[qd.i32, 2],
    descriptors: qd.types.NDArray[qd.i32, 2],
    readouts: qd.types.NDArray[qd.i32, 2],
    decisions: qd.i32,
    updates: qd.i32,
    action_readout: qd.i32,
    kind: qd.i32,
    dynamic: qd.i32,
    regulatory: qd.i32,
):
    for lane in range(world.shape[0]):
        graph = lane_graph[lane]
        scenario = scenarios[lane]
        for column in range(world.shape[1]):
            world[lane, column] = current[lane, column]
        for tick in range(decisions):
            cue = qd.cast(
                draws[tick, scenario, 0] < world[lane, 3] / (p[5] + world[lane, 3]), qd.i32
            )
            for species in range(supplies.shape[1]):
                supplies[lane, species] = input_plan[lane, species]
            supplies[lane, cue_slots[lane]] = cue
            for _chemical in range(updates):
                canonical_lane(
                    lane,
                    graph,
                    counts,
                    supplies,
                    target,
                    kinds,
                    release,
                    reactions,
                    products,
                    previous,
                    expression,
                    before,
                    after,
                    eligible,
                    capacity,
                    demand,
                    events,
                    generated,
                    first,
                    lengths,
                    ends,
                    competitors,
                    promoter_ends,
                    promoter_lengths,
                    1,
                    regulatory,
                )
                for species in range(counts.shape[1]):
                    counts[lane, species] = target[lane, species]
                for gene in range(previous.shape[1]):
                    previous[lane, gene] = expression[lane, gene]
            for readout in range(readouts.shape[1]):
                readouts[lane, readout] = readout_lane(
                    lane,
                    readout,
                    counts,
                    before,
                    after,
                    previous,
                    events,
                    first,
                    lane_graph,
                    indices,
                    sizes,
                    descriptors,
                )
            action = qd.cast(readouts[lane, action_readout] > 0, qd.i32)
            observations[lane] = cue
            actions[lane] = action
            tumble[lane] = move_lane(
                lane, world, source_world, waves, p, draws, tick, scenario, action, kind, dynamic
            )
            rewards[tick, lane] = world[lane, 3]
