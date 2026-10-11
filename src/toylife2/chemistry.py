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


def _graph(t, genes, spec, extra_states=(), *, admit=None):
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

    def add_bound(complex_):
        bound.add(complex_)
        if len(keys) + len(bound) > spec.capacity:
            raise OverflowError("closed species vocabulary exceeds controller.capacity")
        if admit is not None and len(bound) % 128 == 0:
            admit(len(genes) * 128 + (len(keys) + len(bound)) * (1024 + len(genes) * 48))

    for p in ps:
        perimeter = int(t.protein_perimeter[p])
        for m in sorted(mets):
            mi = t.met_index.get(m)
            if mi is not None and t.protein_met_energy[mi, perimeter] < 0:
                add_bound(("P", p, -1, m))
    for a, b in sorted(pairs):
        di = t.dimer_for(a, b)
        for m in sorted(mets):
            mi = t.met_index.get(m)
            if mi is not None and t.dimer_met_energy[mi, di] < 0:
                add_bound(("D", a, b, m))
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
        if admit is not None and len(reactions) % 128 == 0:
            admit(len(genes) * 128 + len(keys) * (1024 + len(genes) * 48) + len(reactions) * 512)

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
    if admit is not None:
        admit(len(genes) * 128 + len(keys) * (1024 + len(genes) * 48) + len(reactions) * 512)
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
    return (
        tuple(keys),
        MappingProxyType(index),
        tuple(reactions),
        tuple(fissions),
        _owned(promoters),
        _owned(polymers),
        tuple(_items(initial)),
    )


def _kernel_plan(tables, genes, graph):
    """Compile ordered groups once per host graph, independently of lane padding."""
    keys, index, reactions, _, promoters, polymers = graph[:6]
    ends = np.zeros(max(1, len(reactions)), np.int32)
    cursor = 0
    while cursor < len(reactions):
        end = cursor + 1
        while end < len(reactions) and reactions[end][0] == reactions[cursor][0]:
            end += 1
        ends[cursor:end] = end
        cursor = end
    competitors = np.full((len(genes), len(keys) + 1, 2), -1, np.int32)
    promoter_ends = np.zeros((len(genes), len(keys) + 1), np.int32)
    lengths = np.zeros(len(genes), np.int32)
    products = np.full(len(genes), -1, np.int32)
    for gene in range(len(genes)):
        entries = [
            (int(energy), species, int(active))
            for species, (energy, active) in enumerate(promoters[gene])
            if energy != _MAX
        ]
        if polymers[gene] < 0:
            entries.append((int(polymers[gene]), -1, 1))
        entries.sort()
        lengths[gene] = len(entries)
        cursor = 0
        while cursor < len(entries):
            end = cursor + 1
            while end < len(entries) and entries[end][0] == entries[cursor][0]:
                end += 1
            promoter_ends[gene, cursor:end] = end
            cursor = end
        for c, (_, species, active) in enumerate(entries):
            competitors[gene, c] = (species, active)
        protein = int(tables.gene_to_protein[int(genes[gene]) & 65535])
        if protein:
            products[gene] = index[("P", protein)]
    return tuple(_owned(array) for array in (ends, competitors, promoter_ends, lengths, products))


class _BatchChemistry:
    def __init__(self, model, genotypes, spec, states=None):
        self.model, self.spec = model, spec
        self.genotypes = _owned(validate(genotypes))
        self.leading = self.genotypes.shape[:-1]
        genes = self.genotypes.reshape(-1, self.genotypes.shape[-1])
        if states is not None and len(states) != len(genes):
            raise ValueError("one CellState is required per genotype row")
        # Counts in explicit states affect initialization; their complete key also
        # protects the extra vocabulary admitted by advance_batch.
        cache = model.engine._chemistry_cache
        context = digest(asdict(spec))
        self._plans = []
        local = {}
        for i, row in enumerate(genes):
            extra = () if states is None else (states[i],)
            key = (
                str(model.tables.digest),
                context,
                tuple(map(int, row)),
                None if states is None else _checked_state(model.tables, states[i]).key(),
            )
            plan = local.get(key)
            if plan is None:
                plan = cache.get(key)
                if plan is None:

                    def admit(required):
                        model.engine._check_memory(
                            required, message="chemistry graph construction exceeds memory budget"
                        )

                    graph = _graph(model.tables, row, spec, extra, admit=admit)
                    plan = cache.put(key, graph, _kernel_plan(model.tables, row, graph))
                local[key] = plan
            self._plans.append(plan)
        self.graphs = [plan.graph for plan in self._plans]
        self.species = max(1, *(len(g[0]) for g in self.graphs))
        self.candidates = max(1, *(len(g[2]) for g in self.graphs))
        lanes, ng = genes.shape
        self._shape = (lanes, self.species)
        layout = getattr(model.engine, "_chemistry_metadata_layout", "shared")
        unique = list(dict.fromkeys(self._plans)) if layout == "shared" else self._plans
        plan_index = {id(plan): i for i, plan in enumerate(unique)}
        lane_graph = np.asarray([plan_index[id(plan)] for plan in self._plans], np.int32)
        if layout != "shared":
            lane_graph = np.arange(lanes, dtype=np.int32)
        self.metadata_layout = layout
        self._lane_graph_host = lane_graph
        representatives = np.full(len(unique), -1, np.intp)
        for lane, plan in enumerate(lane_graph):
            if representatives[plan] < 0:
                representatives[plan] = lane
        self._representatives = representatives
        metadata = [plan.graph for plan in unique]
        nplans = len(metadata)
        self._readout_plan = self._compile_readouts(metadata, genes, lane_graph)
        static_bytes = nplans * (
            self.species * 8 + self.candidates * 36 + ng * 8 + ng * (self.species + 1) * 12 + 8
        )
        static_bytes += lane_graph.nbytes + sum(a.nbytes for a in self._readout_plan)
        # Static upload overlaps its host staging arrays; dynamic mirrors and
        # admission/input scratch are included before allocating the cohort.
        projected = (
            static_bytes * 2
            + lanes * self.species * 92
            + lanes * self.candidates * 32
            + lanes * ng * 32
            + lanes * (max(1, len(spec.ports.readouts)) * 20 + 12)
        )
        model.engine._check_memory(
            projected, message="persistent chemistry construction exceeds memory budget"
        )
        kinds = np.zeros((nplans, self.species), np.int32)
        release = np.zeros_like(kinds)
        rx = np.full((nplans, self.candidates, 8), -1, np.int32)
        rx[:, :, 0] = _MAX
        products = np.full((nplans, ng), -1, np.int32)
        lengths = np.zeros((nplans, 2), np.int32)
        ends = np.zeros((nplans, self.candidates), np.int32)
        competitors = np.full((nplans, ng, self.species + 1, 2), -1, np.int32)
        promoter_ends = np.zeros((nplans, ng, self.species + 1), np.int32)
        promoter_lengths = np.zeros((nplans, ng), np.int32)
        initial = np.zeros(self._shape, np.int32)
        for plan, (keys, idx, reactions, _, _prom, _poly, _) in enumerate(metadata):
            lengths[plan] = (len(keys), len(reactions))
            for species, (kind, key) in enumerate(keys):
                kinds[plan, species] = {"P": 1, "D": 2, "M": 3, "B": 4}[kind]
                if kind == "B":
                    release[plan, species] = idx[("M", key[3])]
            if reactions:
                rx[plan, : len(reactions)] = reactions
            rx_ends, ordered, ordered_ends, ordered_lengths, gene_products = unique[plan].compiled
            ends[plan, : len(rx_ends)] = rx_ends
            competitors[plan, :, : ordered.shape[1]] = ordered
            promoter_ends[plan, :, : ordered_ends.shape[1]] = ordered_ends
            promoter_lengths[plan] = ordered_lengths
            products[plan] = gene_products
        for lane, graph in enumerate(self.graphs):
            entries = graph[-1] if states is None else _items(states[lane])
            for key, count in entries:
                if count > _MAX:
                    raise OverflowError("molecule counts exceed signed device representation")
                initial[lane, graph[1][key]] = count
        static = (
            kinds,
            release,
            rx,
            products,
            lengths,
            ends,
            competitors,
            promoter_ends,
            promoter_lengths,
            lane_graph,
        )
        # Include all arrays and two state buffers in execution budget before allocation.
        device_bytes = sum(a.nbytes for a in (*static, *self._readout_plan))
        device_bytes += initial.nbytes * 9 + lanes * self.candidates * 24 + lanes * ng * 8
        device_bytes += lanes * max(1, len(spec.ports.readouts)) * 4
        # Lazy mirrors, mass bounds, input preparation, and checked-admission
        # scratch overlap at inspection/fallback. The engine bills host plans.
        host_bytes = initial.nbytes * 14 + lanes * self.candidates * 8 + lanes * ng * 8
        host_bytes += self.genotypes.nbytes * 2 + lanes * ng * 4
        host_bytes += lanes * len(spec.ports.readouts) * 16
        host_bytes += lanes * 12 + representatives.nbytes
        host_bytes += sum(array.nbytes for array in (*static, *self._readout_plan))
        host_bytes = max(host_bytes, projected - device_bytes)
        self.buffer_memory = {
            "device": device_bytes,
            "host": host_bytes,
            "total": device_bytes + host_bytes,
        }
        needed = self.buffer_memory["total"]
        engine = model.engine
        engine._check_memory(needed, message="persistent chemistry exceeds execution memory budget")
        self._admit(initial)
        self.arrays = [engine._i32(a.shape, a) for a in static]
        self._readout_arrays = [engine._i32(a.shape, a) for a in self._readout_plan]
        self._readout_values = engine._i32((lanes, max(1, len(spec.ports.readouts))))
        self.source = engine._i32(initial.shape, initial)
        self.target = engine._i32(initial.shape, np.zeros_like(initial))
        self.inputs = engine._i32(initial.shape)
        bits = np.zeros((lanes, ng), np.int32)
        for g in range(ng):
            bits[:, g] = bool(spec.initial_expression & (1 << (ng - g - 1)))
        self.expression = engine._i32(bits.shape, bits)
        self.after_input = engine._i32(initial.shape, initial)
        self.after_output = engine._i32(initial.shape, initial)
        self.eligible = engine._i32((lanes, self.candidates))
        self.capacity = engine._i32((lanes, self.candidates))
        self.demand = engine._i32(initial.shape)
        self.events = engine._i32((lanes, self.candidates))
        self.generated = engine._i32(initial.shape)
        self.first_events = engine._i32((lanes, self.candidates))
        self._candidate_records = (
            engine._i32(bits.shape, bits),
            engine._i32((lanes, self.candidates)),
            engine._i32((lanes, self.candidates)),
            engine._i32(initial.shape, initial),
            engine._i32(initial.shape, initial),
        )
        self._host = initial
        self._bits = bits
        self._last_events = np.zeros((lanes, self.candidates), np.int32)
        self._first_event_counts = self._last_events.copy()
        self._stages = {"boundary": initial}
        self._dirty = set()
        self._input_slots = {}
        weights = np.zeros(self._shape, np.int64)
        for lane, graph in enumerate(self.graphs):
            for species, (kind, key) in enumerate(graph[0]):
                weights[lane, species] = (
                    1
                    if kind == "P"
                    else 2
                    if kind == "D"
                    else len(key)
                    if kind == "M"
                    else len(key[3]) + (1 if key[0] == "P" else 2)
                )
        self._mass_weights = weights
        self._mass_bound = np.sum(weights * initial, axis=1)

    def _materialise(self, field):
        if field in self._dirty:
            device = {
                "host": self.source,
                "bits": self.expression,
                "last_events": self.events,
                "first_event_counts": self.first_events,
            }
            if field == "stages":
                self._stages = {
                    "boundary": self.host,
                    "after_input": self.after_input.to_numpy(),
                    "after_output": self.after_output.to_numpy(),
                }
            else:
                setattr(self, "_" + field, device[field].to_numpy())
            self._dirty.discard(field)
        return getattr(self, "_" + field)

    @property
    def host(self):
        return self._materialise("host")

    @host.setter
    def host(self, value):
        self._host = value
        self._dirty.discard("host")

    @property
    def bits(self):
        return self._materialise("bits")

    @bits.setter
    def bits(self, value):
        self._bits = value
        self._dirty.discard("bits")

    @property
    def last_events(self):
        return self._materialise("last_events")

    @last_events.setter
    def last_events(self, value):
        self._last_events = value
        self._dirty.discard("last_events")

    @property
    def first_event_counts(self):
        return self._materialise("first_event_counts")

    @first_event_counts.setter
    def first_event_counts(self, value):
        self._first_event_counts = value
        self._dirty.discard("first_event_counts")

    @property
    def stages(self):
        return self._materialise("stages")

    @stages.setter
    def stages(self, value):
        self._stages = value
        self._dirty.discard("stages")

    def _compile_readouts(self, graphs, genes, lane_graph):
        # Bound both Python selection lists and dense arrays before building them.
        readouts = max(1, len(self.spec.ports.readouts))
        width_bound = max(1, *(max(len(g[0]), len(g[2])) for g in graphs))
        self.model.engine._check_memory(
            len(graphs) * readouts * (width_bound * 40 + 128) + readouts * 8,
            message="chemistry readout construction exceeds memory budget",
        )
        selections = []
        descriptors = np.zeros((max(1, len(self.spec.ports.readouts)), 2), np.int32)
        for r, readout in enumerate(self.spec.ports.readouts):
            if readout.gene >= genes.shape[1]:
                raise ValueError("readout gene outside genotype")
            met = None if readout.metabolite is None else canonical_metabolite(readout.metabolite)
            pair = None if readout.dimer is None else tuple(sorted(readout.dimer))
            mode = (
                0
                if readout.kind in ("expression", "functional_expression")
                else 2
                if readout.kind == "fission"
                else 1
            )
            stage = {"boundary": 0, "after_input": 1, "after_output": 2}[readout.stage]
            descriptors[r] = (mode, stage)
            rows = []
            for plan, graph in enumerate(graphs):
                keys, index, _, events = graph[:4]
                gene = int(genes[self._representatives[plan], readout.gene])
                protein = int(self.model.tables.gene_to_protein[gene & 65535])
                indices = []
                if mode == 0:
                    if readout.kind == "expression" or protein > 0:
                        indices = [readout.gene]
                elif readout.kind == "protein":
                    species = index.get(
                        ("P", protein if readout.protein is None else readout.protein)
                    )
                    if species is not None:
                        indices = [species]
                elif readout.kind == "dimer":
                    if pair is None:
                        raise ValueError("dimer readout requires a dimer pair")
                    species = index.get(("D", pair))
                    if species is not None:
                        indices = [species]
                elif readout.kind == "bound":
                    indices = [
                        s
                        for s, (kind, key) in enumerate(keys)
                        if kind == "B"
                        and (met is None or key[3] == met)
                        and (readout.protein is None or key[1] == readout.protein)
                        and (pair is None or (key[0] == "D" and key[1:3] == pair))
                    ]
                elif mode == 2:
                    indices = [
                        c
                        for c, event in enumerate(events)
                        if event is not None and (met is None or event[0] == met)
                    ]
                rows.append(indices)
            selections.append(rows)
        width = max(1, *(len(row) for rows in selections for row in rows))
        indices = np.full((len(graphs), len(descriptors), width), -1, np.int32)
        sizes = np.zeros(indices.shape[:2], np.int32)
        for r, rows in enumerate(selections):
            for plan, row in enumerate(rows):
                sizes[plan, r] = len(row)
                indices[plan, r, : len(row)] = row
        return indices, sizes, descriptors

    def _admit(self, counts):
        # Host int64 admission gives a provable i32 bound for molecule creation,
        # both phases' fragment generation, and every simultaneous demand sum.
        bound = counts.astype(np.int64).sum(axis=1) + self.genotypes.shape[-1]
        factor = max(16, self.candidates * 16)
        if np.any(bound > _MAX // factor):
            raise OverflowError("molecule inventory exceeds checked device count/demand capacity")

    def _input_counts(self, inputs):
        lanes = len(self.graphs)
        result = np.full(self._shape, -1, np.int64)
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
                slots = self._input_slots.get(key)
                if slots is None:
                    if any(key not in graph[1] for graph in self.graphs):
                        raise ValueError("input molecule outside declared chemistry vocabulary")
                    slots = np.asarray([graph[1][key] for graph in self.graphs], np.intp)
                    self._input_slots[key] = slots
                result[np.arange(lanes), slots] = values
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

    def _safe_steps(self, injection, requested=1, pulse=False):
        added = np.sum(np.maximum(injection, 0) * self._mass_weights, axis=1)
        bound = self._mass_bound.copy()
        maximum = _MAX // max(16, self.candidates * 16)
        steps = 0
        for tick in range(requested):
            extra = added if tick == 0 or not pulse else 0
            if np.any(bound + extra + self.genotypes.shape[-1] > maximum):
                break
            bound += extra + self.genotypes.shape[-1]
            steps += 1
        return steps, bound

    def step(self, inputs=None, *, steps=1, pulse=False):
        from ._chemistry_kernels import canonical_step

        injection = self._input_counts(inputs)
        safe, bound = self._safe_steps(injection, steps, pulse)
        if safe < steps:
            if steps > 1:
                raise ValueError("chemistry block requires a proven admission bound")
            combined = self.host.astype(np.int64)
            mask = injection >= 0
            if self.spec.ports.input_mode == "replenish":
                combined[mask] = injection[mask]
            else:
                combined[mask] += injection[mask]
            self._admit(combined)
            bound = np.sum(combined * self._mass_weights, axis=1) + self.genotypes.shape[-1]
        self.last_inputs = injection
        self.inputs.from_numpy(injection.astype(np.int32))
        static = self.arrays
        bits, events, first, before, after = self._candidate_records
        cast(Any, canonical_step)(
            self.source,
            self.inputs,
            self.target,
            *static[:4],
            self.expression,
            bits,
            before,
            after,
            self.eligible,
            self.capacity,
            self.demand,
            events,
            self.generated,
            first,
            *static[4:],
            steps,
            int(pulse),
            int(self.spec.ports.input_mode == "replenish"),
            int(self.spec.mode == "regulatory"),
        )
        # Readout dispatch and transfer are part of the transaction. Completed
        # counts and diagnostics remain untouched until all results succeed.
        readouts = self.readouts((self.target, bits, events, first, before, after))
        old_records = (
            self.expression,
            self.events,
            self.first_events,
            self.after_input,
            self.after_output,
        )
        self.source, self.target = self.target, self.source
        self.expression, self.events, self.first_events, self.after_input, self.after_output = (
            self._candidate_records
        )
        self._candidate_records = old_records
        self._mass_bound = bound
        self._dirty.update(("host", "bits", "last_events", "first_event_counts", "stages"))
        return readouts

    def readouts(self, devices=None):
        from ._chemistry_kernels import compact_readouts

        boundary, bits, events, first, before, after = (
            (
                self.source,
                self.expression,
                self.events,
                self.first_events,
                self.after_input,
                self.after_output,
            )
            if devices is None
            else devices
        )
        cast(Any, compact_readouts)(
            boundary,
            before,
            after,
            bits,
            events,
            first,
            self.arrays[-1],
            *self._readout_arrays,
            self._readout_values,
        )
        values = self._readout_values.to_numpy()
        return {
            r.name: _owned(values[:, i].astype(np.int64).reshape(self.leading))
            for i, r in enumerate(self.spec.ports.readouts)
        }

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


class _RetainedChemistry:
    """Reusable device copy of a committed or warmed chemistry boundary."""

    def __init__(self, session):
        self.session = session
        batch = session._batch
        self.shapes = (
            batch._shape,
            batch._bits.shape,
            (batch._shape[0], batch.candidates),
            (batch._shape[0], batch.candidates),
            batch._shape,
            batch._shape,
        )
        # Device retention, lazy host materialisation and export's owning copies.
        self.buffer_bytes = sum(int(np.prod(shape)) * 4 for shape in self.shapes) * 3
        self.buffer_bytes += session.genotypes.nbytes * 2 + batch._shape[0] * 8
        self.buffers = [session.model.engine._i32(shape) for shape in self.shapes]
        self._view = None
        self.capture()

    @staticmethod
    def _devices(batch):
        return (
            batch.source,
            batch.expression,
            batch.events,
            batch.first_events,
            batch.after_input,
            batch.after_output,
        )

    def capture(self):
        from ._chemistry_kernels import copy_chemistry_state
        from .sessions import owned

        batch = self.session._batch
        cast(Any, copy_chemistry_state)(*self._devices(batch), *self.buffers)
        self.update = self.session.update
        self.mass_bound = batch._mass_bound.copy()
        self.trace = owned(self.session._unrecorded_trace)
        self.trace_bytes = self.session._trace_bytes
        self._view = None

    def restore(self):
        from ._chemistry_kernels import copy_chemistry_state
        from .sessions import owned

        batch = self.session._batch
        cast(Any, copy_chemistry_state)(*self.buffers, *self._devices(batch))
        batch._mass_bound = self.mass_bound.copy()
        batch._dirty.update(("host", "bits", "last_events", "first_event_counts", "stages"))
        self.session.update = self.update
        self.session._unrecorded_trace = owned(self.trace)
        self.session._trace_bytes = self.trace_bytes
        # Reset-per-decision previously created a fresh child session before
        # restoring its warmed payload. Keep those child-clock semantics.
        self.session._used = dict.fromkeys(self.session._used, 0)
        self.session._total = dict.fromkeys(self.session._total, 0)
        self.session._boundaries = 0
        self.session._status, self.session._reason = "ready", None
        self.session.last_readouts = MappingProxyType(batch.readouts())

    def view(self):
        if self._view is None:
            session = self.session
            self._view = {
                "identity": session.identity,
                "table_digest": str(session.model.tables.digest),
                "ruleset": RULESET,
                "spec": asdict(session.spec),
                "update": self.update,
                "genotypes": session.genotypes,
                "unrecorded_trace": self.trace,
                "trace_bytes": self.trace_bytes,
                **{
                    name: buffer.to_numpy()
                    for name, buffer in zip(
                        ("counts", "bits", "events", "first_events", "after_input", "after_output"),
                        self.buffers,
                        strict=True,
                    )
                },
            }
        return self._view

    def payload(self):
        from .sessions import owned

        return owned(self.view())


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
        self.run(max(0, spec.warmup - self.update))

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
        events = []
        if self.recording.mode == "full":
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
        remaining = steps
        while remaining:
            report = self._step_block(None, remaining)
            remaining -= int((report.transition or {}).get("updates", 1))
            if report.status in {"partial", "complete", "failed"}:
                break
        return report

    def _step_block(self, inputs, maximum, *, pulse=False):
        """Advance a proven prefix; recording and uncertain admission use step()."""
        count = min(maximum, getattr(self.model.engine, "_chemistry_block_size", 1))
        units = self._batch._shape[0]
        limit = self.limits.max_chemical_updates
        if limit is not None:
            count = min(count, max(0, (limit - self._used["chemical_updates"]) // units))
        if (
            count <= 1
            or self.recording.mode == "full"
            or self._record_callback is not None
            or self._status != "ready"
            or self._pending_trace
        ):
            return self.step(inputs)
        injection = self._batch._input_counts(inputs)
        count, _ = self._batch._safe_steps(injection, count, pulse)
        if count <= 1:
            return self.step(inputs)
        readouts = self._batch.step(inputs, steps=count, pulse=pulse)
        self.last_readouts = MappingProxyType(readouts)
        self.update += count
        self._consume("chemical_updates", units * count)
        self._boundaries += count
        return self._report(
            {"chemical_update": self.update},
            {"readouts": dict(readouts)},
            {"advanced": True, "updates": count},
        )

    def snapshot(self, fields=None, include_pending=False):
        from .sessions import owned

        factories = {
            "state": lambda: dict(self.state),
            "genotypes": lambda: _owned(self.genotypes),
            "inventories": lambda: self.inventories,
            "after_input": lambda: self._batch.inventories("after_input"),
            "after_output": lambda: self._batch.inventories("after_output"),
            "fissions": lambda: self._batch.fissions(),
        }
        # Expression construction overlaps its source and immutable owning copy.
        sizes = {name: 0 for name in factories}
        sizes["state"] = self._batch._shape[0] * 16
        sizes["genotypes"] = self.genotypes.nbytes
        if include_pending:
            factories["pending_trace"] = lambda: owned(self._pending_trace)
            sizes["pending_trace"] = self.model.engine._snapshot_bytes(self._pending_trace)
        selected = tuple(factories) if fields is None else tuple(fields)
        required = sum(sizes[name] for name in selected)
        self.model.engine._check_snapshot_memory(self, required)
        return MappingProxyType({name: factories[name]() for name in selected})

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

    def _payload_view(self):
        """Borrow the numerical tree so the parent can admit its owning copy."""
        return {
            "identity": self.identity,
            "table_digest": str(self.model.tables.digest),
            "ruleset": RULESET,
            "update": self.update,
            "spec": asdict(self.spec),
            "genotypes": self.genotypes,
            "unrecorded_trace": self._unrecorded_trace,
            "trace_bytes": self._trace_bytes,
            "counts": self._batch.host,
            "bits": self._batch.bits,
            "events": self._batch.last_events,
            "first_events": self._batch.first_event_counts,
            "after_input": self._batch.stages.get("after_input", self._batch.host),
            "after_output": self._batch.stages.get("after_output", self._batch.host),
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
        self._batch.events.from_numpy(self._batch.last_events)
        self._batch.first_events.from_numpy(self._batch.first_event_counts)
        self._batch.after_input.from_numpy(self._batch.stages["after_input"])
        self._batch.after_output.from_numpy(self._batch.stages["after_output"])
        self._batch._mass_bound = np.sum(self._batch.host * self._batch._mass_weights, axis=1)
        from .sessions import owned

        self._unrecorded_trace = owned(payload.get("unrecorded_trace", []))
        self._trace_bytes = int(payload.get("trace_bytes", 0))
        self.update = update
        self.last_readouts = MappingProxyType(self._batch.readouts())
