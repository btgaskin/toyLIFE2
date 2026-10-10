from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from toylife2 import Model
from toylife2.chemistry import ChemistrySession, _BatchChemistry, advance_batch
from toylife2.protocols import ControllerSpec, PortSpec, ReadoutSpec, WorkLimits
from toylife2.reference import CellState, ReferenceModel
from toylife2.specs import ExecutionSpec


@pytest.fixture(scope="module")
def model():
    return Model(execution=ExecutionSpec(backend="cpu"))


def test_heterogeneous_multispecies_batch_matches_canonical_steps(model):
    rng = np.random.default_rng(142)
    genes = rng.integers(0, 1 << 20, size=(2, 10, 4), dtype=np.int32)
    states = []
    for row in genes.reshape(-1, 4):
        ps = [int(model.tables.gene_to_protein[int(g) & 65535]) for g in row]
        ps = [p for p in ps if p]
        state = CellState(
            proteins=Counter({p: int(rng.integers(1, 7)) for p in ps}),
            metabolites=Counter({"00000001": 3, "00101101": 2}),
        )
        if len(ps) > 1:
            pair = tuple(sorted(ps[:2]))
            if model.tables.dimer_for(*pair) >= 0:
                state.dimers[pair] = 2
                state.bound[("D", *pair, "00000001")] = 1
        states.append(state)
    spec = ControllerSpec(ports=PortSpec(metabolites=("00000001", "00101101")))
    oracle = ReferenceModel(model.tables)
    for _ in range(5):
        actual, expressions, events = advance_batch(genes, states, None, model, spec)
        assert expressions.shape == (2, 10)
        for lane, (row, state) in enumerate(zip(genes.reshape(-1, 4), states, strict=True)):
            expected = oracle.step(row, state)
            assert actual[lane].key() == expected.state.key()
            assert expressions.flat[lane] == expected.expression
            expanded = Counter({})
            for original, left, right, n in events[lane]:
                expanded[(original, left, right)] += n
            assert expanded == Counter(expected.fissions)
        states = actual


def test_fissions_exact_multiplicity_and_phase_readouts(model):
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 475)[0])
    genes = np.asarray([(5 << 16) | coding], np.int32)
    ports = PortSpec(
        readouts=(
            ReadoutSpec("first", "fission", stage="after_input"),
            ReadoutSpec("all", "fission"),
            ReadoutSpec("held", "bound", stage="after_input"),
        )
    )
    spec = ControllerSpec(ports=ports, initial_bound=(("D", 344, 348, "00000001", 2),))
    session = ChemistrySession(model, genes, spec)
    session.step()
    # Newly translated breaker acts in phase two, retaining the exact fragments.
    assert session.last_readouts["first"] == 0
    assert session.last_readouts["all"] == 1
    assert session.last_readouts["held"] == 2
    assert session.inventories[0].metabolites == Counter({"0000": 1, "0001": 1, "00000001": 1})
    spec = replace(spec, initial_proteins=((475, 2),))
    session = ChemistrySession(model, [0], spec)
    session.step()
    assert session.last_readouts["all"] == 2
    assert session.last_readouts["first"] == 2
    assert session.inventories[0].metabolites == Counter({"0000": 2, "0001": 2})


def test_candidate_eligibility_is_phase_start_snapshot(model):
    # P47 can form a homodimer, but the new dimer may only bind a metabolite
    # in the next chemistry phase when it is present at that phase's start.
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 47)[0])
    genes = np.asarray([coding, coding], np.int32)
    state = CellState(proteins=Counter({47: 2}), metabolites=Counter({"00000001": 2}))
    actual, expression, _ = advance_batch(genes, [state], None, model)
    expected = ReferenceModel(model.tables).step(genes, state)
    assert actual[0].key() == expected.state.key()
    assert expression == expected.expression


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_energy_reaction_demand_is_simultaneous(model, monkeypatch, copies, reverse):
    # These two real dimers have equal total energy and compete for protein 21.
    initial = CellState(proteins=Counter({21: copies, 2579: 1, 2623: 1}))
    oracle = ReferenceModel(model.tables)
    candidates = oracle._binding_candidates(initial)
    assert len(candidates) == 2
    assert [c.energy for c in candidates] == [-281, -281]
    if reverse:
        original = oracle._binding_candidates
        monkeypatch.setattr(oracle, "_binding_candidates",
                            lambda state: list(reversed(original(state))))
    expected, events = oracle.react(initial)
    assert not events
    if copies == 1:
        assert expected.key() == initial.key()
    else:
        assert not expected.proteins
        assert expected.dimers == Counter({(21, 2579): 1, (21, 2623): 1})

    genes = np.zeros((1, 3), np.int32)
    batch = _BatchChemistry(model, genes, ControllerSpec(), [initial])
    if reverse:
        reactions = batch.arrays[2].to_numpy()
        group = np.flatnonzero(reactions[0, :, 0] == -281)
        assert len(group) == 2
        reactions[0, group] = reactions[0, group[::-1]]
        batch.arrays[2].from_numpy(reactions)
    batch.step()
    assert batch.inventories("after_input")[0].key() == expected.key()
    full_step = oracle.step(genes[0], initial)
    assert batch.inventories()[0].key() == full_step.state.key()
    assert batch.expressions()[0] == full_step.expression
    assert initial.proteins == Counter({21: copies, 2579: 1, 2623: 1})


def test_distinct_protein_promoter_tie_preserves_inventory(model):
    # Bind distinct proteins at -20.0, stronger than polymerase at -13.9.
    # Both protein candidates leave competition; polymerase can then activate.
    a, b, promoter = 47, 1303, 14
    tables = model.tables
    assert tables.dimer_for(a, b) == -1
    perimeters = tables.protein_perimeter[[a, b]]
    assert perimeters[0] != perimeters[1]
    energy = tables.protein_promoter_energy.copy()
    pol = tables.protein_promoter_pol.copy()
    for protein, perimeter in zip((a, b), perimeters, strict=True):
        energy[promoter, perimeter] = -200 - int(tables.protein_energy[protein])
        pol[promoter, perimeter] = False
    tied = replace(tables, protein_promoter_energy=energy, protein_promoter_pol=pol)
    altered = Model(tables=tied, execution=model.execution)
    oracle = ReferenceModel(tied)
    coding = [int(np.flatnonzero(tied.gene_to_protein == protein)[0]) for protein in (a, b)]
    genes = np.asarray([(promoter << 16) | code for code in coding], np.int32)
    initial = CellState(proteins=Counter({a: 1, b: 1}))
    before = initial.key()
    assert not oracle.promoter_active(promoter, CellState(proteins=Counter({a: 1})))
    assert not oracle.promoter_active(promoter, CellState(proteins=Counter({b: 1})))
    assert oracle.promoter_response(genes, initial.proteins) == 3
    assert initial.key() == before
    assert altered.regulatory(genes).values[3] == 3

    batch = _BatchChemistry(altered, genes[None], ControllerSpec(), [initial])
    batch.step()
    assert batch.inventories("after_input")[0].key() == before
    assert batch.expressions()[0] == 3
    assert batch.inventories()[0].key() == oracle.step(genes, initial).state.key()
    assert initial.key() == before


def test_session_inputs_owned_snapshots_and_checkpoint_resume(model):
    genes = np.asarray([[907050, 1048465, 769001], [1005689, 644498, 941775]], np.int32)
    spec = ControllerSpec(
        ports=PortSpec(metabolites=("00010001", "00000011")), initial_expression=7
    )
    whole = ChemistrySession(model, genes, spec)
    split = ChemistrySession(model, genes, spec, WorkLimits(max_chemical_updates=4))
    pulses = {"00010001": np.asarray([2, 1]), "00000011": 3}
    for _ in range(2):
        whole.step(pulses)
        split.step(pulses)
    before = split.inventories
    assert split.step().status == "partial"
    assert split.update == 2
    cp = split.checkpoint()
    resumed = ChemistrySession(model, genes, spec, checkpoint=cp)
    for _ in range(3):
        whole.step(pulses)
        resumed.step(pulses)
    assert [s.key() for s in whole.inventories] == [s.key() for s in resumed.inventories]
    np.testing.assert_array_equal(whole.state["expression"], resumed.state["expression"])
    assert resumed.counters["total"]["chemical_updates"] == 10
    before[0].proteins[47] = 900
    assert split.inventories[0].proteins[47] != 900
    with pytest.raises(ValueError):
        split.state["expression"][0] = 1
    with pytest.raises(TypeError):
        split.state["update"] = 4


def test_regulatory_requested_states_never_enumerate_truth_table(model, monkeypatch):
    genes = np.asarray([[907050, 1048465, 769001]], np.int32)
    monkeypatch.setattr(model.engine, "regulatory", lambda *args: pytest.fail("implicit table"))
    spec = ControllerSpec(mode="regulatory", initial_expression=5, ports=PortSpec(metabolites=()))
    session = ChemistrySession(model, genes, spec)
    reference = ReferenceModel(model.tables)
    value = 5
    for _ in range(8):
        value = reference.expression(genes[0], reference._regulatory_state(genes[0], value))
        session.step()
        assert session.state["expression"][0] == value


def test_checked_overflow_preserves_previous_complete_boundary(model):
    spec = ControllerSpec(ports=PortSpec(proteins=(47,), max_dose=2**31 - 1))
    session = ChemistrySession(model, [0], spec)
    before = session.payload()
    report = session.step({("P", 47): 2**31 - 1})
    assert report.status == "partial" and report.reason == "chemistry_capacity"
    assert not report.transition["advanced"]
    np.testing.assert_array_equal(session.payload()["counts"], before["counts"])
    assert session.update == 0
    with pytest.raises(OverflowError):
        ChemistrySession(model, [0], replace(spec, capacity=1))


def test_warmup_obeys_chemical_limits_and_zero_steps(model):
    session = ChemistrySession(
        model, [0], ControllerSpec(warmup=4), WorkLimits(max_chemical_updates=2)
    )
    assert session.update == 2 and session.state["status"] == "partial"
    assert session.run(0).cursors["chemical_update"] == 2
    with pytest.raises(ValueError):
        session.run(-1)


def test_complete_initial_input_vocabulary(model):
    state = CellState(
        proteins=Counter({475: 2}),
        dimers=Counter({(344, 348): 1}),
        metabolites=Counter({"00000001": 3}),
        bound=Counter({("D", 344, 348, "00000001"): 2, ("P", 475, -1, "00000001"): 1}),
    )
    spec = ControllerSpec(ports=PortSpec(proteins=(475,), dimers=((344, 348),)))
    actual, expression, events = advance_batch([0], [state], CellState(), model, spec)
    expected = ReferenceModel(model.tables).step(np.asarray([0]), state)
    assert actual[0].key() == expected.state.key()
    assert expression == expected.expression
    assert sum(e[3] for e in events[0]) == len(expected.fissions)


def test_thousand_varied_inventories_scalar_parity(model):
    rng = np.random.default_rng(736212)
    genes = rng.integers(0, 1 << 20, size=(1024, 4), dtype=np.int32)
    oracle = ReferenceModel(model.tables)
    states = []
    for row in genes:
        state = CellState()
        for p in model.tables.gene_to_protein[row & 65535]:
            if p:
                state.proteins[int(p)] += int(rng.integers(1, 8))
        state.proteins[475] += int(rng.integers(1, 8))
        state.dimers[(344, 348)] += int(rng.integers(0, 4))
        for met in ("00000001", "00010001", "00101101"):
            state.metabolites[met] = int(rng.integers(0, 5))
        state.bound[("D", 344, 348, "00000001")] = int(rng.integers(0, 5))
        state.bound[("P", 475, -1, "00000001")] = int(rng.integers(0, 3))
        states.append(state)
    spec = ControllerSpec(ports=PortSpec(metabolites=("00000001", "00010001", "00101101")))
    actual, expressions, events = advance_batch(genes, states, None, model, spec)
    for lane, (row, source) in enumerate(zip(genes, states, strict=True)):
        expected = oracle.step(row, source)
        assert actual[lane].key() == expected.state.key()
        assert int(expressions[lane]) == expected.expression
        compact = Counter()
        for original, left, right, n in events[lane]:
            compact[(original, left, right)] += n
        assert compact == Counter(expected.fissions)


def test_full_warmup_trace_and_capacity_independent_resume(model):
    from toylife2.specs import RecordingSpec

    spec = ControllerSpec(warmup=3)
    session = ChemistrySession(model, [0], spec, recording=RecordingSpec(mode="full"))
    assert len(session.payload()["unrecorded_trace"]) == 3
    cp = session.checkpoint()
    resumed = ChemistrySession(
        model,
        [0],
        replace(spec, capacity=8192),
        recording=RecordingSpec(mode="full"),
        checkpoint=cp,
    )
    assert resumed.update == 3
    assert resumed.identity == session.identity
    assert len(resumed.payload()["unrecorded_trace"]) == 3
    captured = []

    def recorder(checkpoint, events):
        captured.extend(events)
        return True

    resumed._record_callback = recorder
    resumed.step()
    assert len(captured) == 4
    for event in captured:
        assert all(
            key in event
            for key in ("inputs", "before", "after_input", "after_output", "boundary", "readouts")
        )


@pytest.mark.parametrize("input_mode", ["pulse", "replenish"])
def test_repeated_full_vocabulary_input_replacement_and_release(model, input_mode):
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 475)[0])
    genes = np.asarray([[(5 << 16) | coding], [0]], np.int32)
    spec = ControllerSpec(
        ports=PortSpec(proteins=(475,), dimers=((344, 348),), input_mode=input_mode),
        initial_bound=(("D", 344, 348, "00000001", 1),),
        initial_metabolites=(("00000001", 2),),
    )
    session = ChemistrySession(model, genes, spec)
    oracle = ReferenceModel(model.tables)
    states = session.inventories
    pulses = {
        ("P", 475): [2, 1],
        ("D", (344, 348)): [1, 2],
        "00000001": [3, 2],
        ("B", ("D", 344, 348, "00000001")): [2, 1],
    }
    for _ in range(5):
        expected = []
        for lane, source in enumerate(states):
            source = source.copy()
            for molecule, counts in pulses.items():
                kind, key = molecule if isinstance(molecule, tuple) else ("M", molecule)
                pool = {
                    "P": source.proteins,
                    "D": source.dimers,
                    "M": source.metabolites,
                    "B": source.bound,
                }[kind]
                if input_mode == "replenish":
                    pool[key] = counts[lane]
                else:
                    pool[key] += counts[lane]
            expected.append(oracle.step(genes[lane], source))
        session.step(pulses)
        assert [s.key() for s in session.inventories] == [s.state.key() for s in expected]
        np.testing.assert_array_equal(session.state["expression"], [s.expression for s in expected])
        states = session.inventories


def test_large_gene_requested_state_full_checkpoint_has_numeric_arrays(model):
    from toylife2.specs import RecordingSpec

    genes = np.zeros((1, 80), np.int32)
    spec = ControllerSpec(
        mode="regulatory", initial_expression=(1 << 79) + 1, ports=PortSpec(metabolites=())
    )
    session = ChemistrySession(model, genes, spec, recording=RecordingSpec(mode="full"))
    session.step()
    cp = session.checkpoint()
    assert all(not array.dtype.hasobject for array in cp.arrays.values())
    resumed = ChemistrySession(model, genes, spec, checkpoint=cp)
    assert resumed.state["expression"][0] == session.state["expression"][0]


def test_cheap_buffer_estimate_and_explicit_table_identity(model):
    session = ChemistrySession(model, [[907050, 1048465]])
    assert session.buffer_bytes == session.buffer_memory["host"] + session.buffer_memory["device"]
    payload = session.checkpoint().payload["payload"]
    assert payload["table_digest"] == model.tables.digest
    payload["table_digest"] = "different"
    with pytest.raises(ValueError, match="table identity"):
        session.restore_payload(payload)


def test_partial_constructor_warmup_resumes_with_new_quota(model):
    spec = ControllerSpec(warmup=4)
    session = ChemistrySession(model, [0], spec, limits=WorkLimits(max_chemical_updates=2))
    assert session.update == 2 and session.state["status"] == "partial"
    resumed = ChemistrySession(
        model, [0], spec, limits=WorkLimits(max_chemical_updates=2), checkpoint=session.checkpoint()
    )
    assert resumed.update == 4
    assert resumed.counters["total"]["chemical_updates"] == 4
    assert resumed.counters["invocation"]["chemical_updates"] == 2
