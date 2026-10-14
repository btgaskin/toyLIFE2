from __future__ import annotations

import numpy as np
import pytest

from toylife2._trajectory import _reachable_cycles, _unpack, metabolic_reachable
from toylife2.genotypes import encode
from toylife2.reference import ReferenceModel
from toylife2.runtime import Engine
from toylife2.specs import ExecutionSpec
from toylife2.tables import Tables

pytestmark = pytest.mark.device


@pytest.fixture(scope="module")
def tables(canonical_tables) -> Tables:
    return canonical_tables


def _engine(tables: Tables, tmp_path, *, max_steps: int = 16) -> Engine:
    return Engine(tables, ExecutionSpec(batch_size=4, max_steps=max_steps,
                                        memory_budget_mib=512,
                                        cache_dir=str(tmp_path / "qd")))


def test_reachable_assay_matches_scalar_across_gene_counts(tables: Tables, tmp_path) -> None:
    engine = _engine(tables, tmp_path)
    oracle = ReferenceModel(tables)
    environment = ["00000001", "01010101", "11111111"]
    rng = np.random.default_rng(55)
    for n_genes in (2, 3, 4, 8):
        genotypes = rng.integers(0, 1 << 20, (5, n_genes), dtype=np.uint32)
        result = metabolic_reachable(engine, genotypes, environment)
        assert result.values.shape == (5, len(environment))
        for index, genotype in enumerate(genotypes):
            expected, status = oracle.metabolic(genotype, environment, max_steps=16)
            np.testing.assert_array_equal(result.values[index], expected)
            np.testing.assert_array_equal(result.status[index], status)


def test_reachable_assay_includes_real_cleavage(tables: Tables, tmp_path) -> None:
    engine = _engine(tables, tmp_path)
    genotypes = np.asarray([[907050, 1048465, 769001],
                            [1005689, 644498, 941775]], dtype=np.uint32)
    result = metabolic_reachable(engine, genotypes, ["00010001", "00000011"])
    np.testing.assert_array_equal(result.values, [[1, 0], [0, 0]])
    assert result.complete


def test_engine_routes_five_gene_positive_assay(tables: Tables, tmp_path) -> None:
    engine = _engine(tables, tmp_path)
    genotype = np.asarray([907050, 1048465, 769001, 0, 0], dtype=np.uint32)
    environment = ["00010001", "00000011"]
    result = engine.metabolic(genotype, environment)
    expected, status = ReferenceModel(tables).metabolic(
        genotype, environment, max_steps=16)
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(result.status, status)
    assert int(result.values[0]) == 1


def test_reachable_assay_supports_32_gene_expression(tables: Tables, tmp_path) -> None:
    engine = Engine(tables, ExecutionSpec(batch_size=4, max_steps=100_000,
                                         memory_budget_mib=256,
                                         cache_dir=str(tmp_path / "qd")))
    result = metabolic_reachable(engine, encode("0" * (20 * 32)), ["00000001"])
    np.testing.assert_array_equal(result.values, [0])
    assert result.status.tolist() == ["complete"]
    assert result.metadata["regulatory_mode"] == "reachable_multiword"


def test_regulatory_cap_remains_incomplete(tables: Tables, tmp_path) -> None:
    engine = _engine(tables, tmp_path, max_steps=1)
    result = metabolic_reachable(engine, encode("0" * 40), ["00000001"])
    np.testing.assert_array_equal(result.values, [0])
    assert result.status.tolist() == ["incomplete"]


def test_high_expression_bit_uses_multiword_state(tables: Tables, tmp_path) -> None:
    engine = _engine(tables, tmp_path)
    genotype = encode("0101" + "0" * 16 + "0" * (20 * 31))
    cycles, incomplete = _reachable_cycles(engine, genotype[None, :],
                                           engine._regulatory_tables())
    assert not incomplete[0]
    assert len(cycles[0]) == 1
    np.testing.assert_array_equal(_unpack(cycles[0][0], 32), [1] + [0] * 31)
    result = metabolic_reachable(engine, genotype, ["00000001"])
    assert result.status.tolist() == ["complete"]


def test_large_species_bound_uses_constant_memory_cycle_proof(
    tables: Tables, tmp_path,
) -> None:
    engine = Engine(tables, ExecutionSpec(batch_size=2, max_steps=100_000,
                                         memory_budget_mib=256,
                                         cache_dir=str(tmp_path / "qd")))
    genotype = np.asarray([1005689, 644498, 941775] + [0] * 17,
                          dtype=np.uint32)
    result = engine.metabolic(genotype, ["00000011"])
    assert result.values.tolist() == [0]
    assert result.status.tolist() == ["complete"]
