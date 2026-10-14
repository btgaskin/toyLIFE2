from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from toylife2.genotypes import encode
from toylife2.reference import CellState, ReferenceModel, _Candidate


@pytest.fixture(scope="module")
def model(canonical_tables) -> ReferenceModel:
    return ReferenceModel(canonical_tables)


def test_same_energy_survivors_commit_simultaneously(model, monkeypatch):
    candidates = [
        _Candidate(-100, (("P", 1, 1), ("P", 2, 1)), ("D", (1, 2))),
        _Candidate(-100, (("P", 1, 1), ("P", 3, 1)), ("D", (1, 3))),
    ]
    monkeypatch.setattr(model, "_binding_candidates", lambda state: candidates)
    state, events = model.react(CellState(proteins=Counter({1: 2, 2: 1, 3: 1})))
    assert state.dimers == Counter({(1, 2): 1, (1, 3): 1})
    assert not state.proteins
    assert not events


def test_all_off_is_a_real_regulatory_attractor(model: ReferenceModel) -> None:
    genotype = encode("0" * 40)
    np.testing.assert_array_equal(model.regulatory(genotype), np.zeros(4, dtype=np.uint64))
    assert model.attractor(genotype) == (0,)
    values, status = model.metabolic(genotype, ["0000", "00000001"])
    np.testing.assert_array_equal(values, [0, 0])
    assert list(status) == ["complete", "complete"]


def test_two_copies_form_one_homodimer(model: ReferenceModel) -> None:
    assert model.tables.dimer_for(47, 47) >= 0
    after, fissions = model.react(CellState(proteins=Counter({47: 2})))
    assert not fissions
    assert after.dimers[(47, 47)] == 1
    assert not after.proteins


def test_polymerase_tie_annihilates_both_candidates(model: ReferenceModel) -> None:
    tables = model.tables
    promoter = 5
    protein = 47
    perimeter = int(tables.protein_perimeter[protein])
    altered_energy = tables.protein_promoter_energy.copy()
    altered_pol = tables.protein_promoter_pol.copy()
    altered_energy[promoter, perimeter] = int(tables.polymerase_energy[promoter]) - int(
        tables.protein_energy[protein])
    altered_pol[promoter, perimeter] = False
    altered = ReferenceModel(replace(tables, protein_promoter_energy=altered_energy,
                                     protein_promoter_pol=altered_pol))
    assert altered.promoter_active(promoter, CellState())
    assert not altered.promoter_active(promoter, CellState(proteins=Counter({protein: 1})))


def test_dimer_bound_metabolite_fissions_with_correct_fragments(model: ReferenceModel) -> None:
    state = CellState(proteins=Counter({470: 1}),
                      bound=Counter({("D", 344, 348, "00000001"): 1}))
    after, fissions = model.react(state)
    assert fissions == (("00000001", "0000", "0001"),)
    assert after.metabolites == Counter({"0000": 1, "0001": 1})
    assert not after.bound
    step = model.step(encode("0" * 40), state)
    assert step.fissions == fissions
    assert step.state.metabolites == after.metabolites


def test_repeated_fissions_preserve_fragment_multiplicity(model: ReferenceModel) -> None:
    # Protein 475 has no 475/475 dimer and can break this D+M complex twice.
    state = CellState(proteins=Counter({475: 2}),
                      bound=Counter({("D", 344, 348, "00000001"): 2}))
    result = model.step(encode("0" * 40), state)
    assert result.fissions == (("00000001", "0000", "0001"),) * 2
    assert result.state.metabolites == Counter({"0000": 2, "0001": 2})


def test_new_expression_reacts_in_second_phase(model: ReferenceModel) -> None:
    coding = int(np.flatnonzero(model.tables.gene_to_protein == 475)[0])
    genotype = encode("0101" + format(coding, "016b"))
    state = CellState(bound=Counter({("D", 344, 348, "00000001"): 1}))
    result = model.step(genotype, state)
    assert result.expression == 1
    assert result.fissions == (("00000001", "0000", "0001"),)
    assert result.state.metabolites == Counter({"0000": 1, "0001": 1})


def test_preformed_ab_survives_an_extra_a(model: ReferenceModel) -> None:
    # These protein IDs have both an AB and an AA table entry. The descriptor
    # evaluates the already formed AB plus free A as the state-four inventory.
    genes_for: dict[int, int] = {}
    for coding, protein in enumerate(model.tables.gene_to_protein):
        genes_for.setdefault(int(protein), coding)
    a, b = 47, 264
    assert model.tables.dimer_for(a, b) >= 0
    assert model.tables.dimer_for(a, a) >= 0
    genotype = np.asarray([(3 << 16) | genes_for[a],
                           (3 << 16) | genes_for[b]], dtype=np.uint32)
    descriptor = model.spatial_descriptor(genotype)
    expected = model.functional_output(genotype,
        model.promoter_response(genotype, {a: 1}, {(a, b): 1}))
    assert int(descriptor[4]) == expected
    assert int(descriptor[4]) == 0
    assert int(descriptor[5]) == 3
    assert int(descriptor[6]) == 1


def test_step_cap_reports_incomplete(model: ReferenceModel) -> None:
    genotype = encode("0" * 40)
    state = CellState(bound=Counter({("D", 344, 348, "00000001"): 1}))
    assert model._assay_phase(genotype, state, 1) == "incomplete"
    assert model._assay_phase(genotype, state, 2) == "intact"
