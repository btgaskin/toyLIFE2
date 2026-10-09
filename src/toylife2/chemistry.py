"""Persistent batched canonical cells, with exact compiled reaction graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .genotypes import validate
from .protocols import ControllerSpec
from .reference import CellState
from .specs import RULESET, digest
from .tables import canonical_metabolite

_MAX = np.iinfo(np.int32).max


def _owned(array):
    result = np.array(array, copy=True)
    result.flags.writeable = False
    return result


def _initial(tables, genes, spec):
    state = CellState()
    for collection in (
        spec.initial_proteins,
        spec.initial_dimers,
        spec.initial_metabolites,
        spec.initial_bound,
    ):
        for entry in collection:
            if type(entry[-1]) is not int or entry[-1] < 0:
                raise ValueError("initial molecule counts must be nonnegative integers")
    if spec.initial_expression >= 1 << len(genes):
        raise ValueError("initial_expression outside genotype")
    for g, gene in enumerate(genes):
        if spec.initial_expression & (1 << (len(genes) - g - 1)):
            protein = int(tables.gene_to_protein[int(gene) & 65535])
            if protein > 0:
                state.proteins[protein] += 1
    for p, n in spec.initial_proteins:
        state.proteins[p] += n
    for a, b, n in spec.initial_dimers:
        state.dimers[tuple(sorted((a, b)))] += n
    for m, n in spec.initial_metabolites:
        state.metabolites[canonical_metabolite(m)] += n
    for kind, a, b, m, n in spec.initial_bound:
        if kind == "D":
            a, b = sorted((a, b))
        state.bound[(kind, a, b, canonical_metabolite(m))] += n
    return _checked_state(tables, state)


def _checked_state(tables, state):
    state = state.copy()
    state.__post_init__()
    for p in state.proteins:
        if type(p) is not int or not 0 < p < len(tables.protein_energy):
            raise ValueError("protein ID outside functional table range")
    for a, b in state.dimers:
        if a > b or not 0 < a <= b < len(tables.protein_energy):
            raise ValueError("dimer requires sorted functional protein IDs")
        if tables.dimer_for(a, b) < 0:
            raise ValueError("initial or input dimer has no canonical bond")
    for kind, a, b, _ in state.bound:
        if kind == "P":
            if b != -1 or not 0 < a < len(tables.protein_energy):
                raise ValueError("protein bound complex requires p2=-1")
        elif kind == "D":
            if not 0 < a <= b < len(tables.protein_energy) or tables.dimer_for(a, b) < 0:
                raise ValueError("bound dimer has no canonical bond")
        else:
            raise ValueError("bound complex kind must be P or D")
    for met in (*state.metabolites, *(b[3] for b in state.bound)):
        if len(met) > 8:
            raise ValueError("metabolites have maximum length eight")
    return state


def _items(state):
    for kind, pool in (
        ("P", state.proteins),
        ("D", state.dimers),
        ("M", state.metabolites),
        ("B", state.bound),
    ):
        for key, count in pool.items():
            yield (kind, key), count


def _state_payload(state):
    return [
        [kind, list(key) if isinstance(key, tuple) else key, int(count)]
        for (kind, key), count in _items(state)
        if count
    ]


def _state_restore(payload):
    state = CellState()
    for kind, key, count in payload:
        if kind in ("D", "B"):
            key = tuple(key)
        {"P": state.proteins, "D": state.dimers, "M": state.metabolites, "B": state.bound}[kind][
            key
        ] += count
    return state


def _graph(t, genes, spec, extra_states=()):
    """Close the admitted vocabulary under binding and all fragment splits.

    Chemistry decisions remain on device. Host compilation uses immutable table
    energies and geometry to provide the generalized reaction graph.
    """
    initial = _initial(t, genes, spec)
    proteins = {int(t.gene_to_protein[int(g) & 65535]) for g in genes} - {0}
    proteins.update(spec.ports.proteins)
    pairs = {tuple(sorted(pair)) for pair in spec.ports.dimers}
    mets = {canonical_metabolite(m) for m in spec.ports.metabolites}
    for state in (initial, *extra_states):
        _checked_state(t, state)
        proteins.update(state.proteins)
        pairs.update(state.dimers)
        mets.update(state.metabolites)
        for kind, a, b, m in state.bound:
            proteins.add(a)
            if kind == "D":
                proteins.add(b)
                pairs.add((a, b))
                if len(pairs) + len(proteins) > spec.capacity:
                    raise OverflowError("closed dimer vocabulary exceeds controller.capacity")
            mets.add(m)
    for a, b in pairs:
        proteins.update((a, b))
    for p in proteins:
        if type(p) is not int or not 0 < p < len(t.protein_energy):
            raise ValueError("port protein outside functional table range")
    if len(proteins) > spec.capacity or len(pairs) > spec.capacity:
        raise OverflowError("input chemistry vocabulary exceeds controller.capacity")
    ps = sorted(proteins)
    for i, a in enumerate(ps):
        for b in ps[i:]:
            if t.dimer_for(a, b) >= 0:
                pairs.add((a, b))
                if len(pairs) + len(proteins) > spec.capacity:
                    raise OverflowError("closed dimer vocabulary exceeds controller.capacity")
    for pair in pairs:
        if t.dimer_for(*pair) < 0:
            raise ValueError("port dimer has no canonical bond")
    # Any split of a declared metabolite can produce a shorter fragment. This
    # finite closure also allows a generated fragment to bind in phase two.
    for m in tuple(mets):
        for left in range(len(m)):
            for right in range(left + 1, len(m) + 1):
                mets.add(canonical_metabolite(m[left:right]))
    keys: list[tuple[str, Any]] = []
    keys.extend(("P", p) for p in ps)
    keys.extend(("D", pair) for pair in sorted(pairs))
    keys += [("M", m) for m in sorted(mets)]
    bound = set(initial.bound)
    for state in extra_states:
        bound.update(state.bound)
    for p in ps:
        perimeter = int(t.protein_perimeter[p])
        for m in sorted(mets):
            mi = t.met_index.get(m)
            if mi is not None and t.protein_met_energy[mi, perimeter] < 0:
                bound.add(("P", p, -1, m))
    for a, b in sorted(pairs):
        di = t.dimer_for(a, b)
        for m in sorted(mets):
            mi = t.met_index.get(m)
            if mi is not None and t.dimer_met_energy[mi, di] < 0:
                bound.add(("D", a, b, m))
    keys += [("B", b) for b in sorted(bound)]
    if len(keys) > spec.capacity:
        raise OverflowError("closed species vocabulary exceeds controller.capacity")
    index = {key: i for i, key in enumerate(keys)}
    reactions, fissions = [], []

    def add(energy, a, au, b=None, bu=1, product=None, fragments=None, event=None):
        reactions.append(
            (
                int(energy),
                index[a],
                au,
                index[b] if b else -1,
                bu,
                index[product] if product else -1,
                index[("M", fragments[0])] if fragments else -1,
                index[("M", fragments[1])] if fragments else -1,
            )
        )
        fissions.append(event)
        if len(reactions) > spec.capacity:
            raise OverflowError("reaction candidate capacity exceeds controller.capacity")

    for pair in sorted(pairs):
        a, b = pair
        e = (
            int(t.protein_energy[a])
            + int(t.protein_energy[b])
            + int(t.dimer_energy[t.dimer_for(a, b)])
        )
        add(e, ("P", a), 2 if a == b else 1, ("P", b) if a != b else None, product=("D", pair))
    for complex_ in sorted(bound):
        kind, a, b, m = complex_
        mi = t.met_index.get(m)
        if mi is None:
            continue
        if kind == "P":
            contact = int(t.protein_met_energy[mi, int(t.protein_perimeter[a])])
            if contact < 0:
                add(
                    int(t.protein_energy[a]) + contact,
                    ("P", a),
                    1,
                    ("M", m),
                    product=("B", complex_),
                )
        else:
            di = t.dimer_for(a, b)
            base = int(t.protein_energy[a]) + int(t.protein_energy[b]) + int(t.dimer_energy[di])
            contact = int(t.dimer_met_energy[mi, di])
            if contact < 0:
                add(base + contact, ("D", (a, b)), 1, ("M", m), product=("B", complex_))
            split = int(t.dimer_met_pos[mi, di])
            if t.dimer_met_breakable[mi, di] and 0 < split < len(m):
                for breaker in ps:
                    for subunit, breaking in t.breaking_options(
                        int(t.dimer_met_seq[mi, di]), int(t.protein_perimeter[breaker])
                    ):
                        energy = (
                            int(t.protein_energy[breaker])
                            + int(t.protein_energy[(a, b)[subunit]])
                            + breaking
                        )
                        if energy < base:
                            add(
                                energy,
                                ("P", breaker),
                                1,
                                ("B", complex_),
                                fragments=(
                                    canonical_metabolite(m[:split]),
                                    canonical_metabolite(m[split:]),
                                ),
                                event=(m, m[:split], m[split:]),
                            )
    order = sorted(range(len(reactions)), key=lambda i: reactions[i][0])
    reactions = [reactions[i] for i in order]
    fissions = [fissions[i] for i in order]
    if max(len(keys), len(reactions)) > spec.capacity:
        raise OverflowError("closed chemistry vocabulary or candidates exceed controller.capacity")
    promoters = np.full((len(genes), len(keys), 2), _MAX, dtype=np.int32)
    polymers = np.empty(len(genes), dtype=np.int32)
    for g, gene in enumerate(genes):
        prom = int(gene) >> 16
        polymers[g] = t.polymerase_energy[prom]
        for s, (kind, key) in enumerate(keys):
            if kind == "P":
                p = int(key)
                perimeter = int(t.protein_perimeter[p])
                contact = int(t.protein_promoter_energy[prom, perimeter])
                if contact < 0:
                    promoters[g, s] = (
                        int(t.protein_energy[p]) + contact,
                        int(t.protein_promoter_pol[prom, perimeter]),
                    )
            elif kind == "D":
                a, b = key
                di = t.dimer_for(a, b)
                contact = int(t.dimer_promoter_energy[prom, di])
                if contact < 0:
                    promoters[g, s] = (
                        int(t.protein_energy[a])
                        + int(t.protein_energy[b])
                        + int(t.dimer_energy[di])
                        + contact,
                        int(t.dimer_promoter_pol[prom, di]),
                    )
    return keys, index, reactions, fissions, promoters, polymers, initial


class _BatchChemistry:
    def __init__(self, model, genotypes, spec, states=None):
        self.model, self.spec = model, spec
        self.genotypes = _owned(validate(genotypes))
        self.leading = self.genotypes.shape[:-1]
        genes = self.genotypes.reshape(-1, self.genotypes.shape[-1])
        if states is not None and len(states) != len(genes):
            raise ValueError("one CellState is required per genotype row")
        self.graphs = [
            _graph(model.tables, row, spec, (() if states is None else (states[i],)))
            for i, row in enumerate(genes)
        ]
        self.species = max(1, *(len(g[0]) for g in self.graphs))
        self.candidates = max(1, *(len(g[2]) for g in self.graphs))
        lanes, ng = genes.shape
        kinds = np.zeros((lanes, self.species), np.int32)
        release = np.zeros_like(kinds)
        rx = np.full((lanes, self.candidates, 8), -1, np.int32)
        rx[:, :, 0] = _MAX
        promoters = np.full((lanes, ng, self.species, 2), _MAX, np.int32)
        polymerase = np.empty((lanes, ng), np.int32)
        products = np.full((lanes, ng), -1, np.int32)
        initial = np.zeros_like(kinds)
        for lane, (keys, idx, reactions, _, prom, poly, state) in enumerate(self.graphs):
            for s, (kind, key) in enumerate(keys):
                kinds[lane, s] = {"P": 1, "D": 2, "M": 3, "B": 4}[kind]
                if kind == "B":
                    release[lane, s] = idx[("M", key[3])]
            if reactions:
                rx[lane, : len(reactions)] = reactions
            promoters[lane, :, : len(keys)] = prom
            polymerase[lane] = poly
            for g, gene in enumerate(genes[lane]):
                protein = int(model.tables.gene_to_protein[int(gene) & 65535])
                if protein:
                    products[lane, g] = idx[("P", protein)]
            for key, count in _items(state if states is None else states[lane]):
                if count > _MAX:
                    raise OverflowError("molecule counts exceed signed device representation")
                initial[lane, idx[key]] = count
        # Include all arrays and two state buffers in execution budget before allocation.
        device_bytes = sum(a.nbytes for a in (kinds, release, rx, promoters, polymerase, products))
        device_bytes += initial.nbytes * 7 + lanes * self.candidates * 16 + lanes * ng * 4
        # Numerical host buffers plus peak input/working/readout transfer arrays.
        # Python graph metadata and user-requested snapshots are separate.
        host_bytes = initial.nbytes * 8 + lanes * self.candidates * 8 + lanes * ng * 4
        host_bytes += self.genotypes.nbytes * 2 + lanes * ng * 4
        host_bytes += lanes * len(spec.ports.readouts) * 16
        self.buffer_memory = {
            "device": device_bytes,
            "host": host_bytes,
            "total": device_bytes + host_bytes,
        }
        needed = self.buffer_memory["total"]
        engine = model.engine
        if (
            needed + engine._table_bytes + getattr(engine, "_reserved_session_bytes", 0)
            > engine.execution.memory_budget_mib * 1024 * 1024
        ):
            raise MemoryError("persistent chemistry exceeds execution memory budget")
        self._admit(initial)
        self.arrays = [
            engine._i32(a.shape, a) for a in (kinds, release, rx, products, promoters, polymerase)
        ]
        self.source = engine._i32(initial.shape, initial)
        self.target = engine._i32(initial.shape)
        self.inputs = engine._i32(initial.shape)
        bits = np.zeros((lanes, ng), np.int32)
        for g in range(ng):
            bits[:, g] = bool(spec.initial_expression & (1 << (ng - g - 1)))
        self.expression = engine._i32(bits.shape, bits)
        self.after_input = engine._i32(initial.shape)
        self.after_output = engine._i32(initial.shape)
        self.eligible = engine._i32((lanes, self.candidates))
        self.capacity = engine._i32((lanes, self.candidates))
        self.demand = engine._i32(initial.shape)
        self.events = engine._i32((lanes, self.candidates))
        self.generated = engine._i32(initial.shape)
        self.first_events = engine._i32((lanes, self.candidates))
        self.host = initial
        self.bits = bits
        self.last_events = np.zeros((lanes, self.candidates), np.int32)
        self.first_event_counts = self.last_events.copy()
        self.stages = {"boundary": initial.copy()}

    def _admit(self, counts):
        # Host int64 admission gives a provable i32 bound for molecule creation,
        # both phases' fragment generation, and every simultaneous demand sum.
        bound = counts.astype(np.int64).sum(axis=1) + self.genotypes.shape[-1]
        factor = max(16, self.candidates * 16)
        if np.any(bound > _MAX // factor):
            raise OverflowError("molecule inventory exceeds checked device count/demand capacity")

    def _input_counts(self, inputs):
        lanes = len(self.graphs)
        result = np.full(self.host.shape, -1, np.int64)
        if inputs is None:
            return result
        if isinstance(inputs, Mapping):
            for molecule, count in inputs.items():
                # Dict inputs default to metabolites; explicit ('P',id),
                # ('D',(a,b)) and ('B',complex) also use the full vocabulary.
                key = (
                    molecule
                    if isinstance(molecule, tuple)
                    else ("M", canonical_metabolite(molecule))
                )
                if key[0] == "M":
                    key = ("M", canonical_metabolite(key[1]))
                values = np.asarray(count)
                if values.dtype.kind not in "iu" or values.dtype.kind == "b":
                    raise ValueError("input counts must be nonnegative integers")
                try:
                    values = np.broadcast_to(values, self.leading or ()).reshape(lanes)
                except ValueError as error:
                    raise ValueError(
                        "input count shape must match genotype leading dimensions"
                    ) from error
                if np.any(values < 0) or np.any(values > self.spec.ports.max_dose):
                    raise ValueError("input count outside configured max_dose")
                for lane, graph in enumerate(self.graphs):
                    if key not in graph[1]:
                        raise ValueError("input molecule outside declared chemistry vocabulary")
                    result[lane, graph[1][key]] = int(values[lane])
        else:
            states = [inputs] if isinstance(inputs, CellState) else list(inputs)
            if len(states) != lanes:
                raise ValueError("one input CellState is required per genotype row")
            for lane, state in enumerate(states):
                state = _checked_state(self.model.tables, state)
                for key, count in _items(state):
                    if count > self.spec.ports.max_dose:
                        raise ValueError("input count outside configured max_dose")
                    if key not in self.graphs[lane][1]:
                        raise ValueError("input molecule outside declared chemistry vocabulary")
                    result[lane, self.graphs[lane][1][key]] = count
        return result

    def step(self, inputs=None):
        from ._chemistry_kernels import canonical_step

        injection = self._input_counts(inputs)
        self.last_inputs = injection.copy()
        combined = self.host.astype(np.int64)
        mask = injection >= 0
        if self.spec.ports.input_mode == "replenish":
            combined[mask] = injection[mask]
        else:
            combined[mask] += injection[mask]
        self._admit(combined)
        self.inputs.from_numpy(injection.astype(np.int32))
        kernel = cast(Any, canonical_step)
        kernel(
            self.source,
            self.inputs,
            self.target,
            *self.arrays,
            self.expression,
            self.after_input,
            self.after_output,
            self.eligible,
            self.capacity,
            self.demand,
            self.events,
            self.generated,
            self.first_events,
            int(self.spec.ports.input_mode == "replenish"),
            int(self.spec.mode == "regulatory"),
        )
        # Transfer only completed results; source remains the previous complete
        # boundary until all device execution and validation has succeeded.
        host = self.target.to_numpy()
        bits = self.expression.to_numpy()
        events = self.events.to_numpy()
        first_events = self.first_events.to_numpy()
        stages = {
            "after_input": self.after_input.to_numpy(),
            "after_output": self.after_output.to_numpy(),
            "boundary": host,
        }
        if np.any(host < 0):
            raise OverflowError("device molecule counts overflowed")
        self.first_event_counts = first_events
        self.source, self.target = self.target, self.source
        self.host, self.bits, self.last_events, self.stages = host, bits, events, stages
        return self.readouts()

    def readouts(self):
        answer = {}
        genes = self.genotypes.reshape(-1, self.genotypes.shape[-1])
        for readout in self.spec.ports.readouts:
            if readout.gene >= genes.shape[1]:
                raise ValueError("readout gene outside genotype")
            values = np.zeros(len(genes), np.int64)
            stage = self.stages.get(readout.stage, self.host)
            for lane, graph in enumerate(self.graphs):
                if readout.kind in ("expression", "functional_expression"):
                    values[lane] = self.bits[lane, readout.gene]
                    if readout.kind == "functional_expression":
                        values[lane] *= (
                            int(
                                self.model.tables.gene_to_protein[
                                    int(genes[lane, readout.gene]) & 65535
                                ]
                            )
                            > 0
                        )
                elif readout.kind == "protein":
                    protein = readout.protein
                    if protein is None:
                        protein = int(
                            self.model.tables.gene_to_protein[
                                int(genes[lane, readout.gene]) & 65535
                            ]
                        )
                    s = graph[1].get(("P", protein))
                    values[lane] = 0 if s is None else stage[lane, s]
                elif readout.kind == "dimer":
                    if readout.dimer is None:
                        raise ValueError("dimer readout requires a dimer pair")
                    s = graph[1].get(("D", tuple(sorted(readout.dimer))))
                    values[lane] = 0 if s is None else stage[lane, s]
                elif readout.kind == "bound":
                    for s, (kind, key) in enumerate(graph[0]):
                        if kind == "B" and (
                            readout.metabolite is None
                            or key[3] == canonical_metabolite(readout.metabolite)
                        ):
                            if readout.protein is None or key[1] == readout.protein:
                                if readout.dimer is None or (
                                    key[0] == "D" and key[1:3] == tuple(sorted(readout.dimer))
                                ):
                                    values[lane] += stage[lane, s]
                elif readout.kind == "fission":
                    for c, event in enumerate(graph[3]):
                        if event is not None and (
                            readout.metabolite is None
                            or event[0] == canonical_metabolite(readout.metabolite)
                        ):
                            event_counts = (
                                self.first_event_counts
                                if readout.stage == "after_input"
                                else self.last_events
                            )
                            values[lane] += event_counts[lane, c]
            answer[readout.name] = _owned(values.reshape(self.leading))
        return answer

    def inventories(self, stage="boundary"):
        counts = self.stages.get(stage, self.host)
        result = []
        for lane, graph in enumerate(self.graphs):
            state = CellState()
            for s, (kind, key) in enumerate(graph[0]):
                n = int(counts[lane, s])
                if n:
                    {
                        "P": state.proteins,
                        "D": state.dimers,
                        "M": state.metabolites,
                        "B": state.bound,
                    }[kind][key] = n
            result.append(state)
        return result

    def expressions(self):
        values = [
            sum(int(bit) << (len(row) - g - 1) for g, bit in enumerate(row)) for row in self.bits
        ]
        return _owned(
            np.asarray(values, dtype=np.uint64 if self.bits.shape[1] <= 64 else object).reshape(
                self.leading
            )
        )

    def fissions(self):
        return [
            tuple(
                (*event, int(self.last_events[lane, c]))
                for c, event in enumerate(graph[3])
                if event is not None and self.last_events[lane, c]
            )
            for lane, graph in enumerate(self.graphs)
        ]


def advance_batch(genotypes, states, inputs, model, spec=None):
    """One pure batch update returning owned inventories, expression and events.

    Events are compact ``(original,left,right,multiplicity)`` tuples per row.
    """
    spec = spec or ControllerSpec()
    states = [states] if isinstance(states, CellState) else list(states)
    batch = _BatchChemistry(model, genotypes, spec, states)
    batch.step(inputs)
    return batch.inventories(), batch.expressions(), batch.fissions()


from .sessions import SessionBase  # noqa: E402


class ChemistrySession(SessionBase):
    """Own a persistent batched canonical inventory and its update cursor."""

    def __init__(self, model, genotypes, spec=None, limits=None, recording=None, checkpoint=None):
        spec = spec or ControllerSpec()
        genes = validate(genotypes)
        self.model, self.spec = model, spec
        self.genotypes = _owned(genes)
        identity_spec = asdict(spec)
        identity_spec.pop("capacity")
        identity = digest(
            {
                "kind": "chemistry",
                "table_digest": str(model.tables.digest),
                "genotypes": genes.tolist(),
                "controller": identity_spec,
            }
        )
        super().__init__("chemistry", identity, limits, recording, checkpoint)
        self._batch = _BatchChemistry(model, genes, spec)
        self.update = 0
        self._unrecorded_trace = []
        self._trace_bytes = 0
        self.last_readouts = MappingProxyType(self._batch.readouts())
        if checkpoint is not None:
            self.restore_payload(checkpoint.payload["payload"])
        for _ in range(max(0, spec.warmup - self.update)):
            if self.step().status == "partial":
                break

    @property
    def buffer_bytes(self):
        return self._batch.buffer_memory["total"]

    @property
    def buffer_memory(self):
        return MappingProxyType(dict(self._batch.buffer_memory))

    @property
    def inventories(self):
        """Export independent CellStates; edits never change the owned session."""
        return self._batch.inventories()

    @property
    def state(self):
        return MappingProxyType(
            {
                "update": self.update,
                "status": self._status,
                "reason": self._reason,
                "expression": self._batch.expressions(),
                "readouts": MappingProxyType(dict(self.last_readouts)),
                "counters": self.counters,
            }
        )

    def observe(self):
        return self.state

    def step(self, inputs=None):
        was_complete = self._status == "complete"
        if self._pending_trace:
            if not self._flush_pending_trace():
                return self._report(
                    {"chemical_update": self.update}, transition={"advanced": False}
                )
            if was_complete:
                self._status = "complete"
        if self._status in {"complete", "partial", "failed"}:
            return self._report({"chemical_update": self.update}, transition={"advanced": False})
        units = len(self._batch.graphs)
        if not self._allow("chemical_updates", units):
            return self._report({"chemical_update": self.update}, transition={"advanced": False})
        if self._pending_trace and not self._flush_pending_trace():
            return self._report({"chemical_update": self.update}, transition={"advanced": False})
        before = self._batch.host.copy() if self.recording.mode == "full" else None
        if self.recording.mode == "full" and self._record_callback is None:
            # Reserve a conservative JSON representation bound before a step.
            estimate = (
                self._batch.host.size * 5 + self._batch.bits.size + self._batch.last_events.size * 2
            ) * 12 + 4096
            if self._trace_bytes + estimate > self.recording.trace_budget_mib * 1024 * 1024:
                self._status, self._reason = "partial", "trace_budget"
                return self._report(
                    {"chemical_update": self.update}, transition={"advanced": False}
                )
        try:
            readouts = self._batch.step(inputs)
        except OverflowError:
            self._status, self._reason = "partial", "chemistry_capacity"
            return self._report({"chemical_update": self.update}, transition={"advanced": False})
        self.last_readouts = MappingProxyType(readouts)
        self.update += 1
        self._consume("chemical_updates", units)
        self._status, self._reason = "ready", None
        events = [
            {
                "type": "chemical_update",
                "update": self.update,
                "expression": self._batch.expressions().tolist(),
                "fissions": self._batch.fissions(),
            }
        ]
        if self.recording.mode == "full":
            events[0].update(
                {
                    "inputs": self._batch.last_inputs,
                    "species": [tuple(graph[0]) for graph in self._batch.graphs],
                    "before": before,
                    "after_input": self._batch.stages["after_input"],
                    "after_output": self._batch.stages["after_output"],
                    "boundary": self._batch.host,
                    "readouts": dict(readouts),
                }
            )
            if self._record_callback is None:
                import json

                from .records import _jsonable
                from .sessions import owned

                self._unrecorded_trace.extend(owned(events))
                self._trace_bytes += len(json.dumps(_jsonable(events), separators=(",", ":")))
                if self._trace_bytes > self.recording.trace_budget_mib * 1024 * 1024:
                    self._status, self._reason = "partial", "trace_budget"
            elif self._unrecorded_trace:
                events = [*self._unrecorded_trace, *events]
                self._unrecorded_trace = []
                self._trace_bytes = 0
        self._boundary(events)
        return self._report(
            {"chemical_update": self.update}, {"readouts": dict(readouts)}, {"advanced": True}
        )

    def run(self, steps):
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        report = self._report({"chemical_update": self.update}, transition={"advanced": False})
        for _ in range(steps):
            report = self.step()
            if report.status in {"partial", "complete", "failed"}:
                break
        return report

    def snapshot(self, fields=None, include_pending=False):
        result = {
            "state": dict(self.state),
            "genotypes": _owned(self.genotypes),
            "inventories": self.inventories,
            "after_input": self._batch.inventories("after_input"),
            "after_output": self._batch.inventories("after_output"),
            "fissions": self._batch.fissions(),
        }
        if include_pending:
            from .sessions import owned

            result["pending_trace"] = owned(self._pending_trace)
        if fields is not None:
            result = {key: result[key] for key in fields}
        return MappingProxyType(result)

    def drain_trace(self):
        """Transfer retained full rows to an owning parent recorder.

        Every returned row belongs to a completed chemical boundary. A parent
        session retains these rows until its own world boundary is committed.
        """
        from .sessions import owned

        result = owned(self._unrecorded_trace)
        self._unrecorded_trace = []
        self._trace_bytes = 0
        return result

    def _checkpoint_payload(self):
        return self.payload()

    def payload(self):
        """Numerical payload suitable for nesting in a world checkpoint."""
        from .sessions import owned

        return {
            "identity": self.identity,
            "table_digest": str(self.model.tables.digest),
            "ruleset": RULESET,
            "update": self.update,
            "spec": asdict(self.spec),
            "genotypes": self.genotypes.copy(),
            "unrecorded_trace": owned(self._unrecorded_trace),
            "trace_bytes": self._trace_bytes,
            "counts": self._batch.host.copy(),
            "bits": self._batch.bits.copy(),
            "events": self._batch.last_events.copy(),
            "first_events": self._batch.first_event_counts.copy(),
            "after_input": self._batch.stages.get("after_input", self._batch.host).copy(),
            "after_output": self._batch.stages.get("after_output", self._batch.host).copy(),
        }

    def restore_payload(self, payload):
        if (
            payload.get("table_digest") != str(self.model.tables.digest)
            or payload.get("ruleset") != RULESET
        ):
            raise ValueError("chemistry checkpoint table identity mismatch")
        if payload.get("identity") != self.identity:
            raise ValueError("chemistry payload scientific identity mismatch")
        update = payload["update"]
        if type(update) is not int or update < 0:
            raise ValueError("invalid chemical update cursor")
        arrays = {
            key: np.asarray(payload[key])
            for key in ("counts", "bits", "events", "first_events", "after_input", "after_output")
        }
        shapes = {
            "counts": self._batch.host.shape,
            "bits": self._batch.bits.shape,
            "events": self._batch.last_events.shape,
            "first_events": self._batch.last_events.shape,
            "after_input": self._batch.host.shape,
            "after_output": self._batch.host.shape,
        }
        for key, array in arrays.items():
            if (
                array.shape != shapes[key]
                or array.dtype.kind not in "iu"
                or np.any(array < 0)
                or np.any(array > _MAX)
            ):
                raise ValueError("invalid chemistry checkpoint arrays")
        if np.any(arrays["bits"] > 1):
            raise ValueError("expression checkpoint contains nonbinary bits")
        self._batch._admit(arrays["counts"])
        self._batch.host = arrays["counts"].astype(np.int32, copy=True)
        self._batch.bits = arrays["bits"].astype(np.int32, copy=True)
        self._batch.last_events = arrays["events"].astype(np.int32, copy=True)
        self._batch.first_event_counts = arrays["first_events"].astype(np.int32, copy=True)
        self._batch.stages = {
            "boundary": self._batch.host,
            "after_input": arrays["after_input"].astype(np.int32, copy=True),
            "after_output": arrays["after_output"].astype(np.int32, copy=True),
        }
        self._batch.source.from_numpy(self._batch.host)
        self._batch.expression.from_numpy(self._batch.bits)
        from .sessions import owned

        self._unrecorded_trace = owned(payload.get("unrecorded_trace", []))
        self._trace_bytes = int(payload.get("trace_bytes", 0))
        self.update = update
        self.last_readouts = MappingProxyType(self._batch.readouts())
