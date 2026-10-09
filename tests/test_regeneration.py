from __future__ import annotations

from toylife2.regeneration import (
    contact_energy,
    enumerate_compact_folds,
    validate_binding_regeneration,
    validate_regeneration,
)
from toylife2.tables import Tables


def test_independent_fold_and_attachment_regeneration() -> None:
    report = validate_regeneration(Tables.load())
    assert report.folds == 38
    assert report.functional_sequences == 52549
    assert report.distinct_perimeters == 379
    assert report.energy_mismatches == 0
    assert report.functional_mismatches == 0
    assert report.perimeter_identity_mismatches == 0
    assert report.attachment_rows_checked > 1_500_000
    assert report.attachment_mismatches == 0


def test_contact_geometry() -> None:
    assert len(enumerate_compact_folds()) == 38
    assert contact_energy(15, 1) == -29
    assert contact_energy(0, 0) == 0


def test_independent_ligand_regeneration() -> None:
    report = validate_binding_regeneration(Tables.load())
    assert report.dimer_bonds_checked == 20207
    assert report.dimer_bond_mismatches == 0
    assert report.protein_promoter_checked == 16 * 379
    assert report.protein_promoter_mismatches == 0
    assert report.protein_metabolite_checked == 274 * 379
    assert report.protein_metabolite_mismatches == 0
    # The supplied Pol annotations disagree with the adjacent-site rule for
    # three promoter/perimeter combinations, all on perimeter 375.
    assert report.protein_promoter_pol_mismatches == ((10, 375), (12, 375), (14, 375))
