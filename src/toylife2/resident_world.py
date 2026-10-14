"""Private resident world storage and exact scenario-block preparation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, cast

import numpy as np


class RandomBlockCache:
    """Engine-owned, bounded cache of immutable scenario draws shared by cohorts."""

    def __init__(self, capacity, engine=None):
        self.engine = engine
        self.capacity = capacity
        self.bytes = 0
        self.hits = 0
        self.entries = OrderedDict()

    def reclaim(self, maximum):
        while self.entries and self.bytes > max(0, maximum):
            _, value = self.entries.popitem(last=False)
            self.bytes -= value.nbytes + 512

    def block(self, world, tick, count):
        rng = world.rng
        dynamic = world.kind == "fluctuating" and not world.config["static"]
        modes = len(world.wavevectors)
        columns = world.draw_columns
        key = (
            rng.stream.seed,
            rng.panel,
            tuple(map(int, rng._unique)),
            tick,
            count,
            dynamic,
            modes,
            world.precision,
            world.motion_steps,
            world.event_capacity,
        )
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        size = count * len(rng._unique) * columns * world.dtype.itemsize + 512
        retention = self.capacity
        if self.engine is not None:
            available = self.engine._available_bytes(minimum_bytes=size)
            retention = min(retention, self.bytes + available)
        self.reclaim(retention - size)
        result = np.zeros((count, len(rng._unique), columns), world.dtype)
        ids = rng._unique
        for offset in range(count):

            def uniform(role, event=tick + offset):
                return rng.stream.uniform(ids, role=f"world:{role}", trial=rng.panel, event=event)

            result[offset, :, 0] = uniform("chem-sensor")
            if dynamic:
                expanded = ids[:, None] * np.uint64(modes) + np.arange(modes, dtype=np.uint64)
                u = rng.stream.uniform(
                    expanded,
                    role="world:chem-phase-update:normal-radius",
                    trial=rng.panel,
                    event=tick + offset,
                )
                v = rng.stream.uniform(
                    expanded,
                    role="world:chem-phase-update:normal-phase",
                    trial=rng.panel,
                    event=tick + offset,
                )
                result[offset, :, 4 : 4 + modes] = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
            finfo = np.finfo(world.dtype)
            for event in range(world.event_capacity):
                start = 4 + modes + event * world.motion_stride
                for column, role in enumerate(
                    (f"chem-motion-wait:{event}", f"chem-motion-duration:{event}")
                ):
                    result[offset, :, start + column] = np.clip(
                        uniform(role).astype(world.dtype),
                        finfo.tiny,
                        np.nextafter(world.dtype.type(1), world.dtype.type(0)),
                    )
                expanded = ids[:, None] * np.uint64(3) + np.arange(3, dtype=np.uint64)
                for step in range(world.motion_steps):
                    role = f"world:chem-motion-rotation:{event}:{step}"
                    u = rng.stream.uniform(
                        expanded, role=role + ":normal-radius", trial=rng.panel, event=tick + offset
                    )
                    v = rng.stream.uniform(
                        expanded, role=role + ":normal-phase", trial=rng.panel, event=tick + offset
                    )
                    result[offset, :, start + 2 + step * 3 : start + 5 + step * 3] = np.sqrt(
                        -2 * np.log(u)
                    ) * np.cos(2 * np.pi * v)
        result.flags.writeable = False
        size = result.nbytes + 512
        self.reclaim(retention - size)
        if size <= retention:
            self.entries[key] = result
            self.bytes += size
        return result


class WorldBuffers:
    """Packed world arrays, reused random staging and last-decision diagnostics."""

    def __init__(self, world, engine=None):
        import quadrants as qd

        self.world, self.engine = world, engine
        if world.precision == "float64" and cast(Any, qd).lang.impl.current_cfg().arch != qd.cpu:
            raise ValueError("Quadrants float64 chemotaxis requires the CPU backend")
        dtype = qd.f32 if world.precision == "float32" else qd.f64
        self.lanes = int(np.prod(world.shape))
        self.modes = len(world.wavevectors)
        self.state_columns = 8 + self.modes
        self.draw_columns = world.draw_columns
        self.maximum = 8
        scenarios = len(world.rng._unique)
        self.current = qd.ndarray(dtype, (self.lanes, self.state_columns))
        self.candidate = qd.ndarray(dtype, (self.lanes, self.state_columns))
        self.source = qd.ndarray(dtype, (scenarios, 2))
        self.scenarios = qd.ndarray(qd.i32, (self.lanes,))
        self.scenarios.from_numpy(world.rng._inverse.reshape(-1).astype(np.int32))
        self.waves = qd.ndarray(dtype, world.wavevectors.shape)
        self.waves.from_numpy(world.wavevectors)
        self.parameters = qd.ndarray(dtype, world.constants.shape)
        self.parameters.from_numpy(world.constants)
        self.draw_shape = (self.maximum, scenarios, self.draw_columns)
        self.draws = qd.ndarray(dtype, self.draw_shape)
        self.rewards = qd.ndarray(dtype, (self.maximum, self.lanes))
        self.actions = qd.ndarray(qd.i32, (self.lanes,))
        self.tumble = qd.ndarray(qd.i32, (self.lanes,))
        self.candidate_actions = qd.ndarray(qd.i32, (self.lanes,))
        self.candidate_tumble = qd.ndarray(qd.i32, (self.lanes,))
        self.observations = qd.ndarray(qd.i32, (self.lanes,))
        self._host_actions = np.zeros(self.lanes, np.int32)
        self.cache = engine._world_random_cache if engine is not None else RandomBlockCache(0)
        self.last_state = None
        self.bytes = (
            2 * self.lanes * self.state_columns
            + scenarios * 2
            + 2 * self.modes
            + len(world.constants)
            + self.maximum * scenarios * self.draw_columns
            + self.maximum * self.lanes
        ) * world.dtype.itemsize + self.lanes * 24
        self.uploads = self.blocks = self.readbacks = 0

    def upload(self, state):
        if self.last_state is state:
            return
        packed = np.empty((self.lanes, self.state_columns), self.world.dtype)
        packed[:, :2] = state["position"].reshape(-1, 2)
        packed[:, 2] = state["theta"].reshape(-1)
        packed[:, 3] = self.world.field(state).reshape(-1)
        packed[:, 4 : 4 + self.modes] = state["phases"].reshape(-1, self.modes)
        packed[:, 4 + self.modes : 7 + self.modes] = state["orientation"].reshape(-1, 3)
        packed[:, 7 + self.modes] = state["tumble_remaining"].reshape(-1)
        self.current.from_numpy(packed)
        inverse = self.world.rng._inverse.reshape(-1)
        _, first = np.unique(inverse, return_index=True)
        self._source_host = state["source"].reshape(-1, 2)[first].astype(self.world.dtype)
        self.source.from_numpy(self._source_host)
        self.last_state = state
        self.uploads += 1

    def prepare_draws(self, tick, count):
        draws = self.cache.block(self.world, tick, count)
        padded = np.zeros(self.draw_shape, self.world.dtype)
        padded[:count] = draws
        self.draws.from_numpy(padded)
        self.uploads += 1

    def advance(self, actions, tick):
        from ._chemotaxis_kernels import world_advance

        self.prepare_draws(tick, 1)
        self._host_actions[:] = np.asarray(actions).reshape(-1)
        self.actions.from_numpy(self._host_actions)
        cast(Any, world_advance)(
            self.current,
            self.source,
            self.scenarios,
            self.waves,
            self.parameters,
            self.draws,
            self.actions,
            self.tumble,
            int(self.world.kind == "source"),
            int(not self.world.config["static"] and self.world.kind == "fluctuating"),
        )
        self.blocks += 1

    def result(self, tick, *, candidate=False):
        packed = (self.candidate if candidate else self.current).to_numpy()
        shape = self.world.shape
        source = self._source_host[self.world.rng._inverse]
        state = dict(
            tick=tick,
            position=packed[:, :2].reshape((*shape, 2)).copy(),
            theta=packed[:, 2].reshape(shape).copy(),
            phases=packed[:, 4 : 4 + self.modes].reshape((*shape, self.modes)).copy(),
            source=source,
        )
        state["orientation"] = (
            packed[:, 4 + self.modes : 7 + self.modes].reshape((*shape, 3)).copy()
        )
        state["tumble_remaining"] = packed[:, 7 + self.modes].reshape(shape).copy()
        if np.any(state["tumble_remaining"] < 0):
            self.last_state = None
            raise RuntimeError(
                "chemotaxis motion event capacity exhausted; increase motion_event_capacity"
            )
        exposure = packed[:, 3].reshape(shape).copy()
        channels = {name: state[name].copy() for name in ("position", "theta", "phases", "source")}
        channels.update(
            exposure=exposure.copy(),
            tumbling=(self.candidate_tumble if candidate else self.tumble)
            .to_numpy()
            .reshape(shape),
        )
        channels["tumble_remaining"] = state["tumble_remaining"].copy()
        channels["orientation"] = state["orientation"].copy()
        if self.world.kind == "source":
            distance = self.world.distance(state["position"], state["source"])
            channels.update(
                source_distance=distance, arrived=distance <= self.world.config["arrival_radius"]
            )
        self.readbacks += 2
        if not candidate:
            self.last_state = state
        return state, exposure, channels


class ResidentDecisions:
    """Cohort-owned fused chemistry/world execution, admitted before construction."""

    def __init__(self, session):
        self.session = session
        self.dirty = False
        self.frames_prepared = False
        self.world = WorldBuffers(session.world, session.model.engine)
        self.world.upload(session._world_state)
        batch = session.chemistry._batch
        interface = session.controller.interface
        assert interface is not None
        molecule = interface.inputs[session.task.config["input_channel"]]
        plan = batch._input_counts(interface.encode({session.task.config["input_channel"]: 0}))
        self.plan = session.model.engine._i32(plan.shape, plan)
        slots = np.asarray([g[1][("M", molecule)] for g in batch.graphs], np.int32)
        self.slots = session.model.engine._i32(slots.shape, slots)
        self.worst = plan.copy()
        self.worst[np.arange(len(slots)), slots] = 1
        self.action = [r.name for r in session.controller.ports.readouts].index(
            session.task.config["action_outputs"][0]
        )
        self.bytes = self.world.bytes * 3 + plan.nbytes * 4 + slots.nbytes * 3

    @staticmethod
    def estimate(session):
        lanes = session._world_work
        modes = len(session.world.wavevectors)
        episodes = session.protocol.episodes
        size = session.world.dtype.itemsize
        world = (
            2 * lanes * (8 + modes)
            + episodes * 2
            + 2 * modes
            + len(session.world.constants)
            + 8 * episodes * session.world.draw_columns
            + 8 * lanes
        ) * size + lanes * 24
        return world * 3 + session.chemistry._batch._shape[0] * (
            session.chemistry._batch.species * 16 + 12
        )

    @staticmethod
    def eligible(session):
        interface = session.controller.interface
        return (
            session.task.kind == "chemotaxis"
            and session.world.engine == "quadrants"
            and session.recording.mode == "replayable"
            and session._codec is None
            and interface is not None
            and len(interface.inputs) == 1
            and session.task.config["input_channel"] in interface.inputs
            and len(session.task.config["action_outputs"]) == 1
            and session.task.config["action_outputs"][0] in interface.targets
            and not session.protocol.reset_per_decision
            and not session.task.config.get("record_descriptor", False)
        )

    def count(self, maximum):
        s = self.session
        if s._pending is not None or s._warmup_done < s.controller.warmup or s._pending_trace:
            return 0
        count = min(maximum, getattr(s.model.engine, "_world_block_size", 8), 8)
        if s.protocol.horizon is not None:
            count = min(count, s.protocol.horizon - s.tick)
        for name, units in (
            ("world_ticks", s._world_work),
            ("chemical_updates", s._chemical_work * s.protocol.chemical_updates),
        ):
            available = s.remaining(name)
            if available is not None:
                count = min(count, available // units)
        if s._record_callback is not None:
            count = min(
                count,
                s.recording.checkpoint_interval - s._boundaries % s.recording.checkpoint_interval,
            )
        steps, bound = s.chemistry._batch._safe_steps(
            self.worst, count * s.protocol.chemical_updates, False
        )
        count = min(count, steps // s.protocol.chemical_updates)
        if count:
            _, self.bound = s.chemistry._batch._safe_steps(
                self.worst, count * s.protocol.chemical_updates, False
            )
        return count

    def prepare(self, count, *, transition=False):
        from ._chemistry_kernels import copy_chemistry_state
        from ._chemotaxis_kernels import decision_block

        s = self.session
        b = s.chemistry._batch
        w = self.world
        if not self.dirty:
            w.upload(s._world_state)
        exposure_before = (
            w.current.to_numpy()[:, 3].reshape(s.world_shape).copy() if transition else None
        )
        w.prepare_draws(s.tick, count)
        working = s._commit_storage
        cast(Any, copy_chemistry_state)(*working._devices(b), *working.buffers)
        counts, previous, events, first, before, after = working.buffers
        expression = b._candidate_records[0]
        cast(Any, decision_block)(
            w.current,
            w.candidate,
            w.source,
            w.scenarios,
            w.waves,
            w.parameters,
            w.draws,
            w.rewards,
            w.observations,
            w.candidate_actions,
            w.candidate_tumble,
            b.inputs,
            self.plan,
            self.slots,
            counts,
            b.target,
            *b.arrays[:4],
            previous,
            expression,
            before,
            after,
            b.eligible,
            b.capacity,
            b.demand,
            events,
            b.generated,
            first,
            *b.arrays[4:],
            *b._readout_arrays,
            b._readout_values,
            count,
            s.protocol.chemical_updates,
            self.action,
            int(s.world.kind == "source"),
            int(s.world.kind == "fluctuating" and not s.world.config["static"]),
            int(s.controller.mode == "regulatory"),
        )
        rewards = w.rewards.to_numpy()[:count].reshape((count, *s.world_shape)).astype(np.float64)
        if np.any(rewards < 0):
            raise RuntimeError(
                "chemotaxis motion event capacity exhausted; increase motion_event_capacity"
            )
        if not np.all(np.isfinite(rewards)) or np.any((rewards < 0) | (rewards > 1)):
            raise ValueError("resident world reward must be finite and bounded in [0,1]")
        w.readbacks += 1
        state = channels = actions = None
        reward = rewards[-1]
        if transition:
            state, _, channels = w.result(s.tick + count, candidate=True)
            actions = w.candidate_actions.to_numpy().reshape(s.controller_shape)
            cue = w.observations.to_numpy().reshape(s.world_shape)
        reward_sum = s.reward_sum.copy()
        for row in rewards:
            reward_sum += row
        # Commit chemistry only after all candidate transfers have succeeded.
        cast(Any, copy_chemistry_state)(*working.buffers, *working._devices(b))
        b._mass_bound = self.bound
        b._dirty.update(("host", "bits", "last_events", "first_event_counts", "stages"))
        s.chemistry._readouts_dirty = True
        updates = count * s.protocol.chemical_updates
        s.chemistry.update += updates
        s.chemistry._consume("chemical_updates", s._chemical_work * updates)
        s.chemistry._boundaries += updates
        s._consume("chemical_updates", s._chemical_work * updates)
        s._pending = dict(
            tick=s.tick,
            chemical_cursor=s.protocol.chemical_updates,
            observation={},
            inputs={},
            codec=s._codec_state,
            chemical_events=[],
            chemical_bytes=0,
        )
        s._pending["prepared"] = dict(
            world=state,
            reward=reward,
            judged=True,
            channels=channels,
            codec=s._codec_state,
            actions=actions,
            transition=None,
            events=[],
            advanced=count,
            reward_sum=reward_sum,
            reward_count=s.reward_count + count,
            boundary_waiting=False,
            boundary_done=False,
            resident=not transition,
            device_block=True,
        )
        if transition:
            s._pending["prepared"]["transition"] = {
                "type": "world_tick",
                "tick": s.tick + 1,
                "observation": {
                    "inputs": {s.task.config["input_channel"]: cue},
                    "exposure": exposure_before,
                },
                "actions": actions,
                "reward": reward,
                "channels": channels,
            }
        w.blocks += 1
        w.current, w.candidate = w.candidate, w.current
        w.actions, w.candidate_actions = w.candidate_actions, w.actions
        w.tumble, w.candidate_tumble = w.candidate_tumble, w.tumble
        self.frames_prepared = True
        if state is not None:
            w.last_state = state
        if s._reset_chemistry is None:
            # With zero warmup this remains the initialized inventory.
            s._reset_chemistry = s._reset_storage
        return s._finish_prepared()

    def materialise(self):
        s, w = self.session, self.world
        state, _, channels = w.result(s.tick, candidate=self.frames_prepared)
        s._host_world_state = state
        s._host_channels = channels
        s._host_actions = (
            (w.candidate_actions if self.frames_prepared else w.actions)
            .to_numpy()
            .reshape(s.controller_shape)
        )
        self.dirty = False

    def materialise_prepared(self):
        s = self.session
        if s._pending is None:
            return
        prepared = s._pending.get("prepared")
        if prepared is not None and prepared.get("resident", False):
            state, _, channels = self.world.result(s.tick + prepared["advanced"])
            prepared.update(
                world=state,
                channels=channels,
                resident=False,
                actions=self.world.actions.to_numpy().reshape(s.controller_shape),
            )
