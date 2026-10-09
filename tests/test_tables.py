from __future__ import annotations

import numpy as np
import pytest

from toylife2.tables import Tables, canonical_metabolite


@pytest.fixture(scope="module")
def tables() -> Tables:
    return Tables.load()


def test_published_universe_and_perimeter_factorization(tables: Tables) -> None:
    assert tables.gene_to_protein.shape == (65536,)
    assert tables.protein_energy.shape == (2711,)
    assert tables.perimeter_sides.shape == (379, 8)
    assert tables.dimer_energy.shape == (20207,)
    assert len(tables.met_strings) == 274
    assert np.count_nonzero(tables.met_unbreakable) == 60
    assert tables.protein_promoter_energy.shape == (16, 379)
    assert tables.protein_met_energy.shape == (274, 379)
    assert canonical_metabolite("10000000") == "00000001"
    assert not tables.gene_to_protein.flags.writeable
    with pytest.raises(ValueError):
        tables.protein_energy[1] = 0


def test_geometry_retains_both_breaking_constituents(tables: Tables) -> None:
    assert tables.breaking_energy.shape[1:] == (379, 2)
    met = tables.met_index["00000001"]
    dimer = 1904
    assert tables.dimer_met_breakable[met, dimer]
    sequence = int(tables.dimer_met_seq[met, dimer])
    assert any(energy < 0 for _, energy in tables.breaking_options(sequence, 40))


def test_cache_roundtrip(tmp_path) -> None:
    first = Tables.load(cache_dir=tmp_path)
    second = Tables.load(cache_dir=tmp_path)
    assert first.digest == second.digest
    np.testing.assert_array_equal(first.breaking_energy, second.breaking_energy)
    assert second.met_index["0000"] == 0
